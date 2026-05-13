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
from dataclasses import replace
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
# Φ-features (potential-based reward shaping signals exposed as observation):
#   - Φ_self: viewer's current hand_potential (in [0, 6]).
#   - Φ_with_top: hand_potential if viewer added the discard top to hand
#     (== Φ_self when top is None).
#   - Φ_after_discard[k]: hand_potential if viewer discarded any one card
#     from rank-bucket k (== Φ_self when bucket k is empty in hand).
# These give the network the same internal signals `structured` uses for
# its decisions, so BC reduces to argmax/threshold over obs slices instead
# of having to re-derive Φ from raw rank counts through the MLP.
OBS_PHI_SELF_DIM: int = 1
OBS_PHI_WITH_TOP_DIM: int = 1
OBS_PHI_AFTER_DISCARD_DIM: int = NUM_RANK_BUCKETS  # 14
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
    + OBS_PHI_SELF_DIM
    + OBS_PHI_WITH_TOP_DIM
    + OBS_PHI_AFTER_DISCARD_DIM
)  # 344


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
# Buys are no longer flat-penalized; potential-based shaping (below) handles
# the "good buy vs junk buy" distinction by responding to the actual hand
# structure change. A buy that doesn't increase Φ (junk card + random deck
# card) yields ~0 shaping reward; a buy that completes a near-set yields a
# meaningful positive Φ change.
REWARD_PER_BUY: float = 0.0
REWARD_TERMINAL_WIN: float = 0.0
REWARD_TERMINAL_HAND_POINT: float = -0.05

# Potential-based reward shaping coefficient. Per Ng et al. (1999),
# adding Φ(s') − Φ(s) to the per-step reward does not change the optimal
# policy but provides a dense gradient signal toward set-construction.
# `hand_potential` returns a value in [0, 6]; SHAPING_COEF * Δ is the
# shaping reward per agent action.
SHAPING_COEF: float = 0.1


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


# --- Hand potential (reward shaping) -----------------------------------------


def hand_potential(player: PlayerState) -> float:
    """Closeness-to-going-down potential, in [0, 6].

    Round 1 needs 2 sets of 3. We score by:
      - Per-rank natural counts (capped at 3 — extras don't help) plus a
        wild pool. A *set* requires a natural anchor; size 1 is a "stranded
        anchor" and contributes 0 (a singleton is no closer to going down
        than no card at all).
      - Pick the top-2 rank counts as the two candidate-set anchors,
        greedily allocating wilds (fill anchor1 to 3 first, then anchor2).
        Each set scores its size (2 or 3) only if it reached size ≥ 2;
        size-1 sets score 0.
      - Optionality bonus: +0.5 per *additional* pair-or-better rank beyond
        the two anchors. Three pairs gives more paths to going down than
        two pairs (any of three ranks can complete you), and the bonus
        captures that without making the third pair worth as much as
        completing the first two anchors.
      - Total is clipped to 6 (going-down readiness).

    Once the player is down, Φ saturates at 6.0; lay-off shaping is handled
    by REWARD_PER_LAYOFF_PER_POINT, so Φ has nothing further to track. This
    also keeps the auto-go-down transition non-punishing (Φ is monotone
    non-decreasing across the go-down boundary).
    """
    if player.is_down:
        return 6.0

    naturals: dict[Rank, int] = {}
    wilds = 0
    for c in player.hand:
        if c.is_wild:
            wilds += 1
        else:
            naturals[c.rank] = naturals.get(c.rank, 0) + 1

    counts = sorted((min(c, 3) for c in naturals.values()), reverse=True)

    if not counts:
        # No naturals → no anchor → no legal set under standard rummy rules.
        return 0.0

    n1 = counts[0]
    n2 = counts[1] if len(counts) >= 2 else 0

    # Greedy wild allocation: fill s1 (top anchor) toward 3 first, then s2.
    s1 = min(n1 + wilds, 3)
    wilds_left = wilds - (s1 - n1)
    s2 = min(n2 + wilds_left, 3) if n2 >= 1 else 0

    # Singleton fix: a "set" of size 1 is a stranded anchor — contributes 0.
    contrib1 = s1 if s1 >= 2 else 0
    contrib2 = s2 if s2 >= 2 else 0
    main = float(contrib1 + contrib2)

    # Optionality bonus: pair-or-better ranks beyond the two used as anchors
    # add a small in-tier nudge (more paths to completion) but must never
    # push across a tier boundary. Tiers are spaced by 1.0 (4.0 = 2 pairs,
    # 5.0 = triple + pair, 6.0 = ready), so cap the bonus at 0.75 — strictly
    # less than 1.0 — so e.g. five pairs (4.0 + 0.75 = 4.75) never beats
    # triple + pair (5.0).
    extra_pairs = sum(1 for c in counts[2:] if c >= 2)
    bonus = min(0.25 * float(extra_pairs), 0.75)

    return min(main + bonus, 6.0)


# --- Observation encoding -----------------------------------------------------


def encode_observation(
    state: GameState,
    viewer: int,
    last_pickup_rank: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    """Encode `state` into a 344-dim float32 vector from `viewer`'s POV.

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

    # 9. Φ-features for the viewer (gives the network the same signals
    # `structured` uses; see OBS_PHI_*_DIM comments for layout).
    viewer_player = state.players[viewer]
    phi_self = hand_potential(viewer_player)
    obs[cursor] = phi_self
    cursor += OBS_PHI_SELF_DIM

    if top is not None:
        with_top = replace(viewer_player, hand=viewer_player.hand + (top,))
        obs[cursor] = hand_potential(with_top)
    else:
        obs[cursor] = phi_self
    cursor += OBS_PHI_WITH_TOP_DIM

    # For each rank-bucket: Φ if viewer discarded any one card from that
    # bucket. If the bucket is empty in hand, defaults to Φ_self (no change).
    bucket_first_card: dict[int, Card] = {}
    for c in viewer_player.hand:
        bucket = card_to_rank_idx(c)
        if bucket not in bucket_first_card:
            bucket_first_card[bucket] = c
    for bucket in range(NUM_RANK_BUCKETS):
        card_to_remove = bucket_first_card.get(bucket)
        if card_to_remove is None:
            obs[cursor + bucket] = phi_self
        else:
            new_hand = list(viewer_player.hand)
            new_hand.remove(card_to_remove)
            hyp = replace(viewer_player, hand=tuple(new_hand))
            obs[cursor + bucket] = hand_potential(hyp)
    cursor += OBS_PHI_AFTER_DISCARD_DIM

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

        # Capture Φ before the action — only the acting agent's hand can
        # change this step, so we only need their potential. (PassBuy /
        # actions on already-down players give Δ=0 trivially.)
        acting_seat = _agent_seat(agent)
        phi_before = hand_potential(self._state.players[acting_seat])

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

        # Potential-based shaping: reward the change in Φ for the acting
        # agent's hand across this step. Optimal-policy-preserving (Ng 1999).
        phi_after = hand_potential(self._state.players[acting_seat])
        self.rewards[agent] += SHAPING_COEF * (phi_after - phi_before)

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
