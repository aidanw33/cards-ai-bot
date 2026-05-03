"""Tests for `rummy_v2.env`.

Six categories per ENV_DESIGN.md §7:

  1. Shape / type tests on observation and mask.
  2. Mask correctness — every mask-true action is legal in the engine.
  3. Mask completeness — every legal engine action maps to a mask-true slot.
  4. Reward sums — closed-form derivation matches accumulated rewards.
  5. Determinism — same seed -> identical observation streams.
  6. Termination — uniform-over-mask policy reaches ROUND_OVER within 5000 steps.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from rummy_v2.actions import (
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    LayOff,
    PassBuy,
    is_legal,
    legal_actions,
)
from rummy_v2.cards import build_deck
from rummy_v2.random_play import random_action as engine_random_action
from rummy_v2.env import (
    ACTION_BUY,
    ACTION_DISCARD_BASE,
    ACTION_DRAW_DECK,
    ACTION_DRAW_DISCARD,
    ACTION_LAYOFF_BASE,
    ACTION_PASS_BUY,
    ACTION_SPACE_SIZE,
    NUM_DOWN_PILE_SLOTS,
    NUM_RANK_BUCKETS,
    OBS_BUY_OFFER_DIM,
    OBS_DIM,
    OBS_DISCARD_HISTORY_DIM,
    OBS_DISCARD_TOP_DIM,
    OBS_DOWN_PILES_DIM,
    OBS_HAND_DIM,
    OBS_OPPONENT_PICKUP_DIM,
    OBS_PHASE_DIM,
    OBS_PILE_SIZES_DIM,
    OBS_PLAYERS_DIM,
    REWARD_GOING_DOWN,
    REWARD_GOING_DOWN_PER_POINT,
    REWARD_PER_BUY,
    REWARD_PER_LAYOFF,
    REWARD_PER_LAYOFF_PER_POINT,
    REWARD_PER_STEP,
    REWARD_TERMINAL_HAND_POINT,
    REWARD_TERMINAL_WIN,
    SpeedRummyEnv,
    _decode_action,
    card_to_rank_idx,
    compute_action_mask,
    encode_observation,
    engine_action_to_flat,
    make_env,
)
from rummy_v2.state import NUM_PLAYERS, SETS_TO_GO_DOWN, Phase


CANONICAL_DECK = Counter(build_deck())


# --- Shape / type ------------------------------------------------------------


class TestShapes:
    def test_action_space_is_186(self) -> None:
        assert ACTION_SPACE_SIZE == 186

    def test_obs_dim_is_328(self) -> None:
        # 14 hand + 180 down piles + 14 discard top + 14 discard history
        # + 24 player meta + 4 phase + 2 pile sizes + 6 buy offer
        # + 70 opponent-pickup = 328
        assert OBS_DIM == 328

    def test_reset_observation_shape(self) -> None:
        env = SpeedRummyEnv(seed=0)
        obs = env.observe(env.agent_selection)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_reset_mask_shape(self) -> None:
        env = SpeedRummyEnv(seed=0)
        mask = env.action_mask(env.agent_selection)
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == bool

    def test_initial_mask_only_draw_actions(self) -> None:
        env = SpeedRummyEnv(seed=0)
        mask = env.action_mask(env.agent_selection)
        assert mask[ACTION_DRAW_DECK]
        assert mask[ACTION_DRAW_DISCARD]
        assert mask.sum() == 2

    def test_make_env_factory(self) -> None:
        env = make_env(seed=0)
        assert isinstance(env, SpeedRummyEnv)
        assert env.agent_selection in env.agents


class TestDiscardHistoryFeature:
    """Verify the discard-pile per-rank counts feature added in obs v2."""

    HISTORY_OFFSET = OBS_HAND_DIM + OBS_DOWN_PILES_DIM + OBS_DISCARD_TOP_DIM

    def test_history_slot_matches_full_pile_counts(self) -> None:
        """Step a few moves, then verify obs[discard_history] equals the
        per-rank multiset of the entire discard_pile."""
        rng = random.Random(7)
        env = SpeedRummyEnv(seed=7)
        for _ in range(40):
            if all(env.terminations.values()):
                break
            mask = env.action_mask(env.agent_selection)
            action = engine_action_to_flat(
                env._state, engine_random_action(env._state, rng)
            )
            env.step(action)

        obs = env.observe(env.agent_selection)
        history = obs[
            self.HISTORY_OFFSET : self.HISTORY_OFFSET + OBS_DISCARD_HISTORY_DIM
        ]

        expected = np.zeros(NUM_RANK_BUCKETS, dtype=np.float32)
        for c in env._state.discard_pile:
            expected[card_to_rank_idx(c)] += 1.0

        np.testing.assert_array_equal(history, expected)

    def test_history_empty_when_pile_empty(self) -> None:
        """After deck reshuffle the discard pile is briefly empty; the history
        slot should be all zeros for that observation. Manufacture the
        condition directly rather than waiting for it organically."""
        from rummy_v2.state import GameState
        import dataclasses
        env = SpeedRummyEnv(seed=0)
        empty_state = dataclasses.replace(env._state, discard_pile=())
        obs = encode_observation(empty_state, viewer=0)
        history = obs[
            self.HISTORY_OFFSET : self.HISTORY_OFFSET + OBS_DISCARD_HISTORY_DIM
        ]
        assert history.sum() == 0.0

    def test_pickup_offset_is_after_buy_offer(self) -> None:
        """Sanity-check: opponent-pickup section sits at the very end of obs."""
        expected_offset = (
            OBS_HAND_DIM
            + OBS_DOWN_PILES_DIM
            + OBS_DISCARD_TOP_DIM
            + OBS_DISCARD_HISTORY_DIM
            + OBS_PLAYERS_DIM
            + OBS_PHASE_DIM
            + OBS_PILE_SIZES_DIM
            + OBS_BUY_OFFER_DIM
        )
        assert expected_offset + OBS_OPPONENT_PICKUP_DIM == OBS_DIM

    def test_history_grows_monotonically_within_episode(self) -> None:
        """Discard pile only grows (until reshuffle); per-rank history sum
        should not decrease within an episode segment between reshuffles."""
        rng = random.Random(42)
        env = SpeedRummyEnv(seed=42)
        prev_total = 0
        prev_pile_size = 0
        for _ in range(100):
            if all(env.terminations.values()):
                break
            obs = env.observe(env.agent_selection)
            history = obs[
                self.HISTORY_OFFSET : self.HISTORY_OFFSET + OBS_DISCARD_HISTORY_DIM
            ]
            total = int(history.sum())
            pile_size = len(env._state.discard_pile)
            assert total == pile_size
            if pile_size >= prev_pile_size:
                assert total >= prev_total
            prev_total = total
            prev_pile_size = pile_size
            mask = env.action_mask(env.agent_selection)
            action = engine_action_to_flat(
                env._state, engine_random_action(env._state, rng)
            )
            env.step(action)


class TestOpponentPickupFeature:
    """Verify the per-opponent last-discard-pickup feature."""

    PICKUP_OFFSET = OBS_DIM - OBS_OPPONENT_PICKUP_DIM

    def _slot(self, viewer_seat: int, target_seat: int, rank_idx: int) -> int:
        offset = (target_seat - viewer_seat) % NUM_PLAYERS
        assert offset != 0, "viewer's own seat is not in the pickup section"
        return self.PICKUP_OFFSET + (offset - 1) * NUM_RANK_BUCKETS + rank_idx

    def test_no_pickups_at_reset(self) -> None:
        env = SpeedRummyEnv(seed=0)
        for agent in env.agents:
            obs = env.observe(agent)
            section = obs[self.PICKUP_OFFSET : self.PICKUP_OFFSET + OBS_OPPONENT_PICKUP_DIM]
            assert section.sum() == 0.0

    def test_draw_discard_records_pickup(self) -> None:
        env = SpeedRummyEnv(seed=0)
        active_seat = env._state.active_player
        top = env._state.top_discard
        assert top is not None
        expected_rank_idx = card_to_rank_idx(top)

        env.step(ACTION_DRAW_DISCARD)

        # View from any non-active seat — they should see active's pickup.
        viewer_seat = (active_seat + 1) % NUM_PLAYERS
        obs = env.observe(f"player_{viewer_seat}")
        slot = self._slot(viewer_seat, active_seat, expected_rank_idx)
        assert obs[slot] == 1.0

        section = obs[self.PICKUP_OFFSET : self.PICKUP_OFFSET + OBS_OPPONENT_PICKUP_DIM]
        assert section.sum() == 1.0  # exactly one slot set

    def test_buy_records_pickup(self) -> None:
        env = SpeedRummyEnv(seed=0)
        # Force the cascade: active draws from deck.
        env.step(ACTION_DRAW_DECK)
        if env._state.phase is not Phase.BUY_CASCADE:
            pytest.skip("buy cascade not triggered for this seed")

        buyer_seat = env._state.buy_offer_to
        top = env._state.top_discard
        assert top is not None
        expected_rank_idx = card_to_rank_idx(top)

        mask = env.action_mask(env.agent_selection)
        if not mask[ACTION_BUY]:
            pytest.skip("Buy not legal here")
        env.step(ACTION_BUY)

        viewer_seat = 0 if buyer_seat != 0 else 1
        obs = env.observe(f"player_{viewer_seat}")
        slot = self._slot(viewer_seat, buyer_seat, expected_rank_idx)
        assert obs[slot] == 1.0

    def test_draw_deck_does_not_record_pickup(self) -> None:
        """Active drawing from deck (not from discard) — even if the cascade
        ends in everyone passing — should not record a pickup for anyone."""
        env = SpeedRummyEnv(seed=1)
        env.step(ACTION_DRAW_DECK)
        # Pass through any cascade.
        while env._state.phase is Phase.BUY_CASCADE:
            mask = env.action_mask(env.agent_selection)
            if mask[ACTION_PASS_BUY]:
                env.step(ACTION_PASS_BUY)
            else:
                break
        for agent in env.agents:
            obs = env.observe(agent)
            section = obs[self.PICKUP_OFFSET : self.PICKUP_OFFSET + OBS_OPPONENT_PICKUP_DIM]
            assert section.sum() == 0.0

    def test_reset_clears_pickups(self) -> None:
        env = SpeedRummyEnv(seed=0)
        env.step(ACTION_DRAW_DISCARD)
        # Confirm a pickup was recorded for some non-viewer.
        any_pickup = any(
            env.observe(a)[self.PICKUP_OFFSET : self.PICKUP_OFFSET + OBS_OPPONENT_PICKUP_DIM].sum() > 0
            for a in env.agents
        )
        assert any_pickup
        env.reset(seed=0)
        for agent in env.agents:
            obs = env.observe(agent)
            section = obs[self.PICKUP_OFFSET : self.PICKUP_OFFSET + OBS_OPPONENT_PICKUP_DIM]
            assert section.sum() == 0.0

    def test_pickup_persists_across_steps(self) -> None:
        """A recorded pickup should remain in obs until that opponent picks
        up another discard."""
        rng = random.Random(0)
        env = SpeedRummyEnv(seed=0)
        active_seat = env._state.active_player
        top = env._state.top_discard
        assert top is not None
        expected_rank_idx = card_to_rank_idx(top)

        env.step(ACTION_DRAW_DISCARD)

        viewer_seat = (active_seat + 1) % NUM_PLAYERS
        slot = self._slot(viewer_seat, active_seat, expected_rank_idx)

        # Take a few more steps with the smart-action policy. The active
        # seat won't be picking up another discard (they just took theirs);
        # their slot should remain set.
        for _ in range(10):
            if all(env.terminations.values()):
                break
            obs = env.observe(f"player_{viewer_seat}")
            assert obs[slot] == 1.0
            action = engine_action_to_flat(
                env._state, engine_random_action(env._state, rng)
            )
            env.step(action)


# --- Mask correctness / completeness -----------------------------------------


def _uniform_mask_action(mask: np.ndarray, rng: random.Random) -> int:
    legal_indices = np.flatnonzero(mask)
    return int(rng.choice(legal_indices))


def _progress_biased_action(mask: np.ndarray, rng: random.Random) -> int:
    """A policy that mirrors `random_play.py`'s progress bias: when in
    POST_DRAW with lay-offs available, always lay off; otherwise uniform
    over the mask. Pure uniform-over-mask can deadlock by always discarding
    instead of laying off.

    Detected from the mask alone: any True slot at index >= ACTION_LAYOFF_BASE
    means we are in POST_DRAW and a lay-off is legal.
    """
    layoff_slots = np.flatnonzero(mask[ACTION_LAYOFF_BASE:])
    if len(layoff_slots):
        return ACTION_LAYOFF_BASE + int(rng.choice(layoff_slots))
    return _uniform_mask_action(mask, rng)


def _smart_action(env: SpeedRummyEnv, rng: random.Random) -> int:
    """Drive the env using `random_play.random_action` (engine-aware policy).

    Used for tests that need the episode to actually finish. The mask-only
    biased policy can deadlock on some seeds by discarding cards that match
    no pile and refusing buys it should accept.
    """
    return engine_action_to_flat(env._state, engine_random_action(env._state, rng))


def _every_mask_true_action_is_legal(env: SpeedRummyEnv) -> None:
    mask = compute_action_mask(env._state)
    for action_idx in np.flatnonzero(mask):
        engine_action = _decode_action(env._state, int(action_idx))
        assert engine_action is not None, (
            f"mask=True at {action_idx} but decode returned None"
        )
        assert is_legal(env._state, engine_action), (
            f"mask=True at {action_idx} but engine says illegal: {engine_action}"
        )


def _every_legal_engine_action_is_in_mask(env: SpeedRummyEnv) -> None:
    """Every (simple) legal engine action maps to a mask-true slot.

    GoDown is excluded because the env auto-applies it; it should never be
    presented to the agent. We check the simple-action coverage from
    `legal_actions(state)`.
    """
    mask = compute_action_mask(env._state)
    state = env._state
    for action in legal_actions(state):
        if isinstance(action, DrawDeck):
            assert mask[ACTION_DRAW_DECK]
        elif isinstance(action, DrawDiscard):
            assert mask[ACTION_DRAW_DISCARD]
        elif isinstance(action, Buy):
            assert mask[ACTION_BUY]
        elif isinstance(action, PassBuy):
            assert mask[ACTION_PASS_BUY]
        elif isinstance(action, Discard):
            rank_idx = card_to_rank_idx(action.card)
            assert mask[ACTION_DISCARD_BASE + rank_idx]
        elif isinstance(action, LayOff):
            target_offset = (action.target_player - state.active_player) % NUM_PLAYERS
            pile_action_slot = (
                target_offset * SETS_TO_GO_DOWN + action.pile_index
            )
            rank_idx = card_to_rank_idx(action.card)
            slot = ACTION_LAYOFF_BASE + rank_idx * NUM_DOWN_PILE_SLOTS + pile_action_slot
            assert mask[slot], (
                f"legal engine LayOff {action} not represented in mask slot {slot}"
            )


@pytest.mark.parametrize("seed", range(20))
def test_mask_correctness_full_episode(seed: int) -> None:
    """Every mask-true action is legal in the engine, throughout an episode."""
    rng = random.Random(seed * 7919 + 1)
    env = SpeedRummyEnv(seed=seed)
    for _ in range(2000):
        if all(env.terminations.values()):
            return
        _every_mask_true_action_is_legal(env)
        mask = env.action_mask(env.agent_selection)
        assert mask.any(), "no legal action for current decider"
        action = _progress_biased_action(mask, rng)
        env.step(action)


@pytest.mark.parametrize("seed", range(20))
def test_mask_completeness_full_episode(seed: int) -> None:
    """Every legal engine simple-action is reachable through the mask."""
    rng = random.Random(seed * 6991 + 2)
    env = SpeedRummyEnv(seed=seed)
    for _ in range(2000):
        if all(env.terminations.values()):
            return
        _every_legal_engine_action_is_in_mask(env)
        mask = env.action_mask(env.agent_selection)
        action = _progress_biased_action(mask, rng)
        env.step(action)


# --- Determinism --------------------------------------------------------------


def _record_obs_stream(seed: int, n_steps: int) -> list[np.ndarray]:
    rng = random.Random(seed * 31 + 7)
    env = SpeedRummyEnv(seed=seed)
    stream: list[np.ndarray] = []
    for _ in range(n_steps):
        if all(env.terminations.values()):
            break
        stream.append(env.observe(env.agent_selection).copy())
        mask = env.action_mask(env.agent_selection)
        action = _progress_biased_action(mask, rng)
        env.step(action)
    return stream


def test_determinism_same_seed() -> None:
    s1 = _record_obs_stream(seed=42, n_steps=200)
    s2 = _record_obs_stream(seed=42, n_steps=200)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge() -> None:
    s1 = _record_obs_stream(seed=1, n_steps=20)
    s2 = _record_obs_stream(seed=2, n_steps=20)
    # Different seeds should produce at least one observation difference within
    # 20 steps. (Initial deal alone differs.)
    assert any(not np.array_equal(a, b) for a, b in zip(s1, s2))


# --- Termination --------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_termination_smart_policy(seed: int) -> None:
    """An engine-aware policy (`random_play.random_action` adapted to the
    env's flat action space) terminates within 5000 steps."""
    rng = random.Random(seed * 17 + 3)
    env = SpeedRummyEnv(seed=seed)
    for _ in range(5000):
        if all(env.terminations.values()):
            assert env._state.phase is Phase.ROUND_OVER
            return
        env.step(_smart_action(env, rng))
    pytest.fail(f"seed {seed}: did not terminate in 5000 steps")


# --- Auto go-down -------------------------------------------------------------


def test_auto_go_down_fires_when_possible() -> None:
    """Walk a few seeds until somebody reaches POST_DRAW with a hand that
    `find_any_go_down` accepts; assert they're already down by the time the
    env yields control."""
    from rummy_v2.actions import find_any_go_down

    saw_at_least_one = False
    for seed in range(50):
        rng = random.Random(seed * 11 + 5)
        env = SpeedRummyEnv(seed=seed)
        for _ in range(2000):
            if all(env.terminations.values()):
                break
            state = env._state
            if state.phase is Phase.POST_DRAW:
                actor = state.players[state.active_player]
                # Whenever the env presents a POST_DRAW state to the agent, the
                # active player must EITHER already be down (they just laid off
                # or were down already) OR have no possible go-down.
                if not actor.is_down:
                    assert find_any_go_down(actor.hand) is None, (
                        "agent given POST_DRAW state with possible go-down — "
                        "auto-go-down failed to fire"
                    )
                else:
                    saw_at_least_one = True
            mask = env.action_mask(env.agent_selection)
            action = _progress_biased_action(mask, rng)
            env.step(action)
        if saw_at_least_one:
            return
    assert saw_at_least_one, "no down-player observed across 50 seeds — odd"


# --- Reward bookkeeping ------------------------------------------------------


def test_terminal_rewards_are_score_proportional() -> None:
    """Every player's terminal-step reward is exactly -0.05 × hand_score.

    Winner has hand_score=0 by construction (they emptied their hand) so they
    get 0 — which is the optimal outcome under the score-minimization
    objective. Losers pay -0.05 × points.
    """
    rng = random.Random(123)
    env = SpeedRummyEnv(seed=123)
    last_rewards: dict[str, float] = {}
    for _ in range(5000):
        if all(env.terminations.values()):
            break
        env.step(_smart_action(env, rng))
        last_rewards = dict(env.rewards)
    assert env._state.phase is Phase.ROUND_OVER
    scores = env._state.round_scores
    assert scores is not None
    # For non-actors (5 of 6 agents), terminal-step reward is purely the
    # terminal hand-point reward — no other shaping fires for them on that
    # step. The actor's reward also includes any pre-terminal action shaping
    # (e.g. their last Discard adds REWARD_PER_STEP=0). Discard takes no
    # other shaping reward, so for a winner who went out via Discard, their
    # terminal-step reward is also exactly REWARD_TERMINAL_HAND_POINT × 0 = 0.
    for i, score in enumerate(scores):
        agent = f"player_{i}"
        expected = REWARD_TERMINAL_HAND_POINT * float(score)
        assert last_rewards[agent] == pytest.approx(expected), (
            f"player {i} terminal-step reward {last_rewards[agent]} "
            f"!= expected {expected} (score={score})"
        )


def test_layoff_reward_scales_with_card_value() -> None:
    """A LayOff action's per-step reward equals
    REWARD_PER_LAYOFF + REWARD_PER_LAYOFF_PER_POINT × card.point_value.
    """
    rng = random.Random(7)
    env = SpeedRummyEnv(seed=7)
    saw_layoff = False
    for _ in range(2000):
        if all(env.terminations.values()):
            break
        agent = env.agent_selection
        engine_action = engine_random_action(env._state, rng)
        if isinstance(engine_action, LayOff):
            expected = REWARD_PER_LAYOFF + (
                REWARD_PER_LAYOFF_PER_POINT * float(engine_action.card.point_value)
            )
            env.step(engine_action_to_flat(env._state, engine_action))
            assert env.rewards[agent] == pytest.approx(expected), (
                f"layoff reward {env.rewards[agent]} != expected {expected} "
                f"for card with value {engine_action.card.point_value}"
            )
            saw_layoff = True
            break
        env.step(engine_action_to_flat(env._state, engine_action))
    assert saw_layoff, "no LayOff observed in 2000 steps"


def test_going_down_reward_scales_with_meld_value() -> None:
    """When auto-go-down fires, the actor's per-step reward includes
    REWARD_GOING_DOWN_PER_POINT × sum(point_values of melded cards)."""
    from rummy_v2.actions import find_any_go_down

    rng = random.Random(0)
    env = SpeedRummyEnv(seed=0)
    saw_godown = False
    for _ in range(2000):
        if all(env.terminations.values()):
            break
        # Detect: if the active player is in AWAITING_DRAW (about to draw),
        # not yet down, and the post-draw hand will satisfy find_any_go_down.
        # Stepping this draw should trigger auto-go-down; the bonus belongs
        # to the active player.
        seat = env._state.active_player
        agent = f"player_{seat}"
        engine_action = engine_random_action(env._state, rng)
        env.step(engine_action_to_flat(env._state, engine_action))
        # If the active player just went down on this step (auto), their
        # reward should include the per-point bonus.
        if (
            env._state.phase is Phase.POST_DRAW
            and env._state.players[seat].is_down
            and not isinstance(engine_action, (Buy,))
        ):
            piles = env._state.players[seat].down_piles
            if len(piles) == 2 and len(piles[0].cards) == 3 and len(piles[1].cards) == 3:
                meld_value = sum(
                    c.point_value for p in piles for c in p.cards
                )
                expected_godown_part = (
                    REWARD_GOING_DOWN + REWARD_GOING_DOWN_PER_POINT * float(meld_value)
                )
                # The full per-step reward also includes the action's own
                # shaping (DrawDeck/DrawDiscard add nothing). So the reward
                # should be approximately the going-down bonus.
                # Allow some slack: actor's reward must be at least
                # the going-down bonus minus epsilon.
                assert env.rewards[agent] >= expected_godown_part - 1e-6, (
                    f"actor reward {env.rewards[agent]} below expected "
                    f"going-down bonus {expected_godown_part}"
                )
                saw_godown = True
                break
    assert saw_godown, "no auto-go-down observed in 2000 steps"


# --- Card conservation under env stepping ------------------------------------


def _all_cards_in_state(state) -> Counter:
    bag: Counter = Counter()
    bag.update(state.draw_pile)
    bag.update(state.discard_pile)
    for p in state.players:
        bag.update(p.hand)
        for pile in p.down_piles:
            bag.update(pile.cards)
    return bag


@pytest.mark.parametrize("seed", range(10))
def test_card_conservation_under_env(seed: int) -> None:
    rng = random.Random(seed * 13 + 4)
    env = SpeedRummyEnv(seed=seed)
    for _ in range(3000):
        assert _all_cards_in_state(env._state) == CANONICAL_DECK
        if all(env.terminations.values()):
            return
        mask = env.action_mask(env.agent_selection)
        action = _progress_biased_action(mask, rng)
        env.step(action)
