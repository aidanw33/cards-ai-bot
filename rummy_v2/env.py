"""Multi-agent RL environment for Speed Rummy round 1.

PettingZoo-AEC-style API. One agent acts per `step()`; `agent_selection`
rotates with the engine's phase machine. Action space is 186 discrete
actions, with a per-state legality mask exposed in `infos[agent]["action_mask"]`.

Observations are POV-rotated: the selected agent always sees themselves at
obs-relative seat 0, with subsequent seats following turn order.

`GoDown` is auto-applied by the env (see `ENV_DESIGN.md` §4.1) — it is not
part of the agent's action space. Whenever the active player enters
`POST_DRAW` with a hand satisfying `find_any_go_down`, the env applies it
before yielding control back to the agent.

This module avoids a hard dependency on `pettingzoo`. The class implements
the AEC subset that PPO + self-play needs: `reset`, `step`, `last`, `observe`,
`action_mask`, plus the `agent_selection`, `agents`, `rewards`,
`_cumulative_rewards`, `terminations`, `truncations`, `infos` attributes.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from rummy_v2.actions import (
    Action,
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    LayOff,
    PassBuy,
    find_any_go_down,
    is_legal,
)
from rummy_v2.cards import Card, Rank
from rummy_v2.engine import apply_action
from rummy_v2.state import (
    MAX_WILDS_PER_MELD,
    NUM_PLAYERS,
    SETS_TO_GO_DOWN,
    GameState,
    Phase,
    PlayerState,
    initial_state,
)


# --- Action space layout ------------------------------------------------------

ACTION_DRAW_DECK: int = 0
ACTION_DRAW_DISCARD: int = 1
ACTION_BUY: int = 2
ACTION_PASS_BUY: int = 3
ACTION_DISCARD_BASE: int = 4
NUM_RANK_BUCKETS: int = 14
ACTION_LAYOFF_BASE: int = ACTION_DISCARD_BASE + NUM_RANK_BUCKETS  # 18
NUM_DOWN_PILE_SLOTS: int = NUM_PLAYERS * SETS_TO_GO_DOWN  # 12
NUM_LAYOFF_ACTIONS: int = NUM_RANK_BUCKETS * NUM_DOWN_PILE_SLOTS  # 168
ACTION_SPACE_SIZE: int = ACTION_LAYOFF_BASE + NUM_LAYOFF_ACTIONS  # 186


# --- Observation layout -------------------------------------------------------

OBS_HAND_DIM: int = NUM_RANK_BUCKETS  # 14
_NUM_NONJOKER_RANKS: int = 13  # ranks 2..A
OBS_DOWN_PILE_DIM: int = _NUM_NONJOKER_RANKS + 2  # one-hot rank + count + wilds = 15
OBS_DOWN_PILES_DIM: int = OBS_DOWN_PILE_DIM * NUM_DOWN_PILE_SLOTS  # 180
OBS_DISCARD_TOP_DIM: int = NUM_RANK_BUCKETS  # 14
# Per-rank counts of every card in the discard pile (full history, not just
# the top). Lets the agent infer which ranks are dead — humans card-count this
# implicitly. Added in v2 of the obs (was missing in v1).
OBS_DISCARD_HISTORY_DIM: int = NUM_RANK_BUCKETS  # 14
OBS_PER_PLAYER_DIM: int = 4  # hand_size, buys_used, num_down_piles, is_active
OBS_PLAYERS_DIM: int = OBS_PER_PLAYER_DIM * NUM_PLAYERS  # 24
OBS_PHASE_DIM: int = 4
OBS_PILE_SIZES_DIM: int = 2
OBS_BUY_OFFER_DIM: int = NUM_PLAYERS  # 6
# Per-opponent (5 of them) one-hot rank of the last card they pulled from the
# discard pile — either via DrawDiscard or via Buy. Persistent until the same
# opponent picks up another discard. Lets the agent infer which ranks each
# opponent is collecting (the strongest "they want this rank" signal short of
# seeing their hand).
OBS_OPPONENT_PICKUP_DIM: int = (NUM_PLAYERS - 1) * NUM_RANK_BUCKETS  # 70
OBS_DIM: int = (
    OBS_HAND_DIM
    + OBS_DOWN_PILES_DIM
    + OBS_DISCARD_TOP_DIM
    + OBS_DISCARD_HISTORY_DIM
    + OBS_PLAYERS_DIM
    + OBS_PHASE_DIM
    + OBS_PILE_SIZES_DIM
    + OBS_BUY_OFFER_DIM
    + OBS_OPPONENT_PICKUP_DIM
)  # 328


# --- Reward magnitudes (see ENV_DESIGN.md §5) ---------------------------------
#
# v2 reward shape (2026-05-02): aligned with the *cumulative-score* objective
# of the multi-round game rather than the per-round win objective. The game is
# 6 rounds; lowest cumulative score wins. Each round's score = sum of point
# values left in hand (winner = 0). All shaping rewards are proportional to
# `card.point_value` so the gradient signal tracks game-score directly,
# making "buy speculatively to chase a win" economically irrational.

REWARD_PER_STEP: float = 0.0
# `GOING_DOWN` and `LAYOFF` are kept at 0 as flat bonuses; the per-point
# variants below give the variable-magnitude version that depends on the
# point-value of the cards involved.
REWARD_GOING_DOWN: float = 0.0
REWARD_PER_LAYOFF: float = 0.0
REWARD_GOING_DOWN_PER_POINT: float = 0.05
REWARD_PER_LAYOFF_PER_POINT: float = 0.05
REWARD_PER_BUY: float = -0.5
REWARD_TERMINAL_WIN: float = 0.0
REWARD_TERMINAL_HAND_POINT: float = -0.05


# --- Helpers ------------------------------------------------------------------


def card_to_rank_idx(card: Card) -> int:
    """Map a card to its action-space rank bucket.

    Wild cards (jokers + black 2s) all bucket to index 0. Non-wild ranks
    `Rank.TWO..Rank.ACE` map to indices 1..13 (`rank.value - 1`).
    """
    if card.is_wild:
        return 0
    return card.rank.value - 1


_PHASE_INDEX: dict[Phase, int] = {
    Phase.AWAITING_DRAW: 0,
    Phase.BUY_CASCADE: 1,
    Phase.POST_DRAW: 2,
    Phase.ROUND_OVER: 3,
}


def _agent_id(seat: int) -> str:
    return f"player_{seat}"


def _agent_seat(agent_id: str) -> int:
    return int(agent_id.split("_")[1])


# --- Observation encoding -----------------------------------------------------


def encode_observation(
    state: GameState,
    viewer: int,
    last_pickup_rank: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    """Encode `state` into a 328-dim float32 vector from `viewer`'s POV.

    Player indices in the observation are rotated so that `viewer` is obs-seat
    0 and subsequent seats follow turn order. Hand-content is only filled for
    `viewer`; opponents' hands are revealed only via metadata (size, buys,
    down-pile count).

    `last_pickup_rank` is an env-tracked side-channel (length NUM_PLAYERS, in
    actual seat order, with None for "no pickup yet"); it populates the
    opponent-pickup obs section. When called without it (e.g. from tests
    operating directly on a `GameState`), that section is all-zeros.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    cursor = 0

    # 1. Viewer's hand: per-rank counts (incl. wild bucket).
    for c in state.players[viewer].hand:
        obs[cursor + card_to_rank_idx(c)] += 1.0
    cursor += OBS_HAND_DIM

    # 2. Down piles, in viewer-relative seat order.
    for offset in range(NUM_PLAYERS):
        seat = (viewer + offset) % NUM_PLAYERS
        player = state.players[seat]
        for pile_slot in range(SETS_TO_GO_DOWN):
            base = cursor + (offset * SETS_TO_GO_DOWN + pile_slot) * OBS_DOWN_PILE_DIM
            if pile_slot < len(player.down_piles):
                pile = player.down_piles[pile_slot]
                # one-hot rank (TWO=2 -> dim 0, ..., ACE=14 -> dim 12)
                obs[base + (pile.rank.value - 2)] = 1.0
                obs[base + _NUM_NONJOKER_RANKS] = float(len(pile.cards))
                obs[base + _NUM_NONJOKER_RANKS + 1] = float(pile.num_wilds)
    cursor += OBS_DOWN_PILES_DIM

    # 3. Discard top: one-hot rank bucket (all-zero if empty).
    top = state.top_discard
    if top is not None:
        obs[cursor + card_to_rank_idx(top)] = 1.0
    cursor += OBS_DISCARD_TOP_DIM

    # 3b. Discard pile history: per-rank counts across the full pile (incl.
    # the top). Enables card-counting of which ranks are dead.
    for c in state.discard_pile:
        obs[cursor + card_to_rank_idx(c)] += 1.0
    cursor += OBS_DISCARD_HISTORY_DIM

    # 4. Per-player metadata in viewer-relative order.
    for offset in range(NUM_PLAYERS):
        seat = (viewer + offset) % NUM_PLAYERS
        player = state.players[seat]
        base = cursor + offset * OBS_PER_PLAYER_DIM
        obs[base + 0] = float(len(player.hand))
        obs[base + 1] = float(player.buys_used)
        obs[base + 2] = float(len(player.down_piles))
        obs[base + 3] = 1.0 if seat == state.active_player else 0.0
    cursor += OBS_PLAYERS_DIM

    # 5. Phase one-hot.
    obs[cursor + _PHASE_INDEX[state.phase]] = 1.0
    cursor += OBS_PHASE_DIM

    # 6. Pile sizes.
    obs[cursor + 0] = float(len(state.draw_pile))
    obs[cursor + 1] = float(len(state.discard_pile))
    cursor += OBS_PILE_SIZES_DIM

    # 7. Buy-offer pointer (viewer-relative one-hot, only meaningful in BUY_CASCADE).
    if state.phase is Phase.BUY_CASCADE:
        offset = (state.buy_offer_to - viewer) % NUM_PLAYERS
        obs[cursor + offset] = 1.0
    cursor += OBS_BUY_OFFER_DIM

    # 8. Per-opponent last discard pickup (viewer-rotated, viewer's own seat
    # skipped — they already know their own pickups via their hand contents).
    if last_pickup_rank is not None:
        for offset in range(1, NUM_PLAYERS):
            seat = (viewer + offset) % NUM_PLAYERS
            rank_idx = last_pickup_rank[seat]
            if rank_idx is not None:
                base = cursor + (offset - 1) * NUM_RANK_BUCKETS
                obs[base + rank_idx] = 1.0
    cursor += OBS_OPPONENT_PICKUP_DIM

    assert cursor == OBS_DIM
    return obs


# --- Action mask --------------------------------------------------------------


def compute_action_mask(state: GameState) -> np.ndarray:
    """Return the 186-dim boolean action mask for the current decider."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    if state.phase is Phase.ROUND_OVER:
        return mask

    if state.phase is Phase.AWAITING_DRAW:
        mask[ACTION_DRAW_DECK] = True
        if state.top_discard is not None:
            mask[ACTION_DRAW_DISCARD] = True
        return mask

    if state.phase is Phase.BUY_CASCADE:
        mask[ACTION_PASS_BUY] = True
        if is_legal(state, Buy()):
            mask[ACTION_BUY] = True
        return mask

    # POST_DRAW
    actor = state.players[state.active_player]

    # Discard: one slot per rank bucket present in hand.
    buckets_present: set[int] = set()
    for c in actor.hand:
        buckets_present.add(card_to_rank_idx(c))
    for rank_idx in buckets_present:
        mask[ACTION_DISCARD_BASE + rank_idx] = True

    # LayOff: only if down and hand size >=2 (per RULES.md §6).
    if actor.is_down and len(actor.hand) >= 2:
        has_wild = any(c.is_wild for c in actor.hand)
        non_wild_ranks_in_hand = {
            c.rank for c in actor.hand if not c.is_wild
        }
        for offset in range(NUM_PLAYERS):
            seat = (state.active_player + offset) % NUM_PLAYERS
            target = state.players[seat]
            for pile_slot, pile in enumerate(target.down_piles):
                pile_action_slot = offset * SETS_TO_GO_DOWN + pile_slot
                # Wild lay-off: legal if pile has < 2 wilds AND hand has any wild.
                if pile.num_wilds < MAX_WILDS_PER_MELD and has_wild:
                    idx = ACTION_LAYOFF_BASE + 0 * NUM_DOWN_PILE_SLOTS + pile_action_slot
                    mask[idx] = True
                # Same-rank non-wild lay-off.
                if pile.rank in non_wild_ranks_in_hand:
                    rank_idx = pile.rank.value - 1
                    idx = ACTION_LAYOFF_BASE + rank_idx * NUM_DOWN_PILE_SLOTS + pile_action_slot
                    mask[idx] = True

    return mask


# --- Action encoding (engine-side -> flat) ------------------------------------


def engine_action_to_flat(state: GameState, action: Action) -> int:
    """Translate an engine `Action` to the flat env action index.

    Used by harnesses that compute actions in engine terms (random-play,
    eval, scripted opponents) and feed them through the env. `GoDown` is not
    representable — the env auto-applies it, so it should never be produced
    against an env-driven state.
    """
    if isinstance(action, DrawDeck):
        return ACTION_DRAW_DECK
    if isinstance(action, DrawDiscard):
        return ACTION_DRAW_DISCARD
    if isinstance(action, Buy):
        return ACTION_BUY
    if isinstance(action, PassBuy):
        return ACTION_PASS_BUY
    if isinstance(action, Discard):
        return ACTION_DISCARD_BASE + card_to_rank_idx(action.card)
    if isinstance(action, LayOff):
        target_offset = (action.target_player - state.active_player) % NUM_PLAYERS
        pile_action_slot = target_offset * SETS_TO_GO_DOWN + action.pile_index
        rank_idx = card_to_rank_idx(action.card)
        return ACTION_LAYOFF_BASE + rank_idx * NUM_DOWN_PILE_SLOTS + pile_action_slot
    raise ValueError(f"engine_action_to_flat: don't know how to encode {action!r}")


# --- Action decoding ----------------------------------------------------------


def _pick_card_from_hand(actor: PlayerState, rank_idx: int) -> Card | None:
    """Pick any card matching the rank bucket; returns None if no match.

    rank_idx 0 picks any wild. rank_idx 1..13 picks any non-wild card whose
    rank value equals rank_idx + 1.
    """
    if rank_idx == 0:
        for c in actor.hand:
            if c.is_wild:
                return c
        return None
    target_rank = Rank(rank_idx + 1)
    for c in actor.hand:
        if not c.is_wild and c.rank is target_rank:
            return c
    return None


def _decode_action(state: GameState, action: int) -> Action | None:
    """Convert flat action index to engine `Action`. Returns None if the
    decoded action has no representable engine equivalent in this state.

    LayOff target seats are interpreted relative to the active player (which
    matches the agent_selection in POST_DRAW).
    """
    if not 0 <= action < ACTION_SPACE_SIZE:
        return None

    if action == ACTION_DRAW_DECK:
        return DrawDeck()
    if action == ACTION_DRAW_DISCARD:
        return DrawDiscard()
    if action == ACTION_BUY:
        return Buy()
    if action == ACTION_PASS_BUY:
        return PassBuy()

    actor = state.players[state.active_player]

    if ACTION_DISCARD_BASE <= action < ACTION_LAYOFF_BASE:
        rank_idx = action - ACTION_DISCARD_BASE
        card = _pick_card_from_hand(actor, rank_idx)
        if card is None:
            return None
        return Discard(card=card)

    layoff_offset = action - ACTION_LAYOFF_BASE
    rank_idx = layoff_offset // NUM_DOWN_PILE_SLOTS
    pile_action_slot = layoff_offset % NUM_DOWN_PILE_SLOTS
    target_offset = pile_action_slot // SETS_TO_GO_DOWN
    pile_within = pile_action_slot % SETS_TO_GO_DOWN

    target_seat = (state.active_player + target_offset) % NUM_PLAYERS
    target = state.players[target_seat]
    if pile_within >= len(target.down_piles):
        return None

    card = _pick_card_from_hand(actor, rank_idx)
    if card is None:
        return None
    return LayOff(card=card, target_player=target_seat, pile_index=pile_within)


# --- Env class ----------------------------------------------------------------


class SpeedRummyEnv:
    """PettingZoo-AEC-style env for Speed Rummy round 1.

    Usage:

        env = SpeedRummyEnv(seed=42)
        env.reset()
        while not all(env.terminations.values()):
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            mask = info["action_mask"]
            action = pick_action(obs, mask)
            env.step(action)

    `step()` raises `ValueError` if the action is not legal under the current
    mask. After `terminations[agent]` becomes True, the trainer should call
    `reset()` to begin a new round (we do not implement post-terminal
    iteration over dead agents).
    """

    metadata: dict[str, Any] = {
        "name": "speed_rummy_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        seed: int | None = None,
        expose_opponent_pickup: bool = True,
    ) -> None:
        self.possible_agents: list[str] = [_agent_id(i) for i in range(NUM_PLAYERS)]
        self._initial_seed: int | None = seed
        self._rng: random.Random = random.Random(seed)
        # If False, the opponent-pickup obs section is always zeros (the
        # underlying tracking still happens, but the agent doesn't see it).
        # Used for ablating the feature without changing the network shape.
        self._expose_opponent_pickup: bool = expose_opponent_pickup
        self._state: GameState
        self.agents: list[str]
        self.agent_selection: str
        self.rewards: dict[str, float]
        self._cumulative_rewards: dict[str, float]
        self.terminations: dict[str, bool]
        self.truncations: dict[str, bool]
        self.infos: dict[str, dict[str, Any]]
        # Side-channel: per-player rank-bucket index of the last card they
        # picked up from the discard pile (via DrawDiscard or Buy). None means
        # no pickup yet this episode. Updated in step(); reset in reset().
        self._last_pickup_rank: list[int | None]
        self.reset(seed=seed)

    # -- lifecycle -------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._initial_seed = seed
        self._rng = random.Random(self._initial_seed)

        self._state = initial_state(self._rng)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._last_pickup_rank = [None] * NUM_PLAYERS

        # The initial state is AWAITING_DRAW for the active player; no
        # auto-go-down possible (auto-go-down only fires in POST_DRAW).
        self.agent_selection = self._select_agent()
        self._refresh_mask()

    # -- AEC API ---------------------------------------------------------------

    def observe(self, agent: str) -> np.ndarray:
        pickup = (
            tuple(self._last_pickup_rank)
            if self._expose_opponent_pickup
            else None
        )
        return encode_observation(
            self._state,
            viewer=_agent_seat(agent),
            last_pickup_rank=pickup,
        )

    def action_mask(self, agent: str) -> np.ndarray:
        # Mask is meaningful only for the current decider; we still return a
        # mask shaped from the engine's POV so callers don't crash.
        if agent != self.agent_selection:
            return np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        return compute_action_mask(self._state)

    def last(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        agent = self.agent_selection
        return (
            self.observe(agent),
            float(self._cumulative_rewards[agent]),
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: int | None) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            # Episode is over; trainer should reset(). We accept the call as
            # a no-op so naive iteration loops don't crash.
            return

        if action is None:
            raise ValueError("None action while agent is not terminated")

        # Reset acting agent's accumulated reward — they're consuming via this step.
        self._cumulative_rewards[agent] = 0.0
        # Per-step rewards for everyone start at 0.
        self.rewards = {a: 0.0 for a in self.agents}

        engine_action = _decode_action(self._state, action)
        if engine_action is None or not is_legal(self._state, engine_action):
            raise ValueError(
                f"action {action} not legal in phase {self._state.phase.name}"
            )

        # Per-step idle penalty (now 0 by default — kept as a hook).
        self.rewards[agent] += REWARD_PER_STEP
        if isinstance(engine_action, Buy):
            self.rewards[agent] += REWARD_PER_BUY
        elif isinstance(engine_action, LayOff):
            # Variable bonus = 0.05 × point_value, aligned with the score
            # reduction from removing this card from hand.
            self.rewards[agent] += REWARD_PER_LAYOFF + (
                REWARD_PER_LAYOFF_PER_POINT * float(engine_action.card.point_value)
            )

        # Record discard-pile pickups for the opponent-modeling obs feature.
        # Capture BEFORE applying the action — the engine pops the discard top.
        if isinstance(engine_action, DrawDiscard):
            top = self._state.top_discard
            if top is not None:
                self._last_pickup_rank[self._state.active_player] = (
                    card_to_rank_idx(top)
                )
        elif isinstance(engine_action, Buy):
            top = self._state.top_discard
            if top is not None:
                self._last_pickup_rank[self._state.buy_offer_to] = (
                    card_to_rank_idx(top)
                )

        self._state = apply_action(self._state, engine_action, self._rng)

        # Auto go-down for the active player if newly possible.
        self._maybe_auto_go_down()

        # Terminal handling.
        if self._state.phase is Phase.ROUND_OVER:
            self._apply_terminal_rewards()
            for a in self.agents:
                self.terminations[a] = True

        # Accumulate per-step into per-agent totals.
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]

        # Rotate agent_selection (no-op if terminal — stays at last decider).
        if not self.terminations[agent]:
            self.agent_selection = self._select_agent()
        self._refresh_mask()

    # -- internals -------------------------------------------------------------

    def _select_agent(self) -> str:
        if self._state.phase is Phase.BUY_CASCADE:
            return _agent_id(self._state.buy_offer_to)
        return _agent_id(self._state.active_player)

    def _maybe_auto_go_down(self) -> None:
        if self._state.phase is not Phase.POST_DRAW:
            return
        actor_idx = self._state.active_player
        actor = self._state.players[actor_idx]
        if actor.is_down:
            return
        godown = find_any_go_down(actor.hand)
        if godown is None:
            return
        self._state = apply_action(self._state, godown, self._rng)
        # Bonus = 0.05 × total point value of the 6 melded cards. Aligned
        # with the score reduction (those cards no longer count toward
        # terminal hand_score).
        meld_value = sum(c.point_value for c in godown.set1) + sum(
            c.point_value for c in godown.set2
        )
        self.rewards[_agent_id(actor_idx)] += REWARD_GOING_DOWN + (
            REWARD_GOING_DOWN_PER_POINT * float(meld_value)
        )

    def _apply_terminal_rewards(self) -> None:
        winner = self._state.round_winner
        scores = self._state.round_scores
        assert winner is not None
        assert scores is not None
        # Unified shape: every player's terminal reward is proportional to
        # their hand score. Winner has score=0 by construction (they emptied
        # their hand) → 0 reward, which is optimal under "minimize score".
        # Losers pay -0.05 × hand_points. REWARD_TERMINAL_WIN is kept as 0.
        for i, score in enumerate(scores):
            agent = _agent_id(i)
            self.rewards[agent] += REWARD_TERMINAL_HAND_POINT * float(score)
            if i == winner:
                self.rewards[agent] += REWARD_TERMINAL_WIN

    def _refresh_mask(self) -> None:
        sel = self.agent_selection
        self.infos[sel] = dict(self.infos.get(sel, {}))
        self.infos[sel]["action_mask"] = compute_action_mask(self._state)


# --- Factory -----------------------------------------------------------------


def make_env(seed: int | None = None) -> SpeedRummyEnv:
    """PettingZoo-style factory."""
    return SpeedRummyEnv(seed=seed)
