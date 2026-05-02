"""Random-play smoke tests — run many full games under the random policy and
assert engine invariants on every transition.

Invariants checked:

  - Card conservation: at every state, the multiset of all cards across all
    locations equals the canonical 108-card deck.
  - Hand-size bounds: 0 <= hand size <= 18 for every player at every state.
  - Buys monotonic: buys_used never decreases for any player.
  - Active player only changes via Discard.
  - Winner's score is 0; non-winners' scores equal sum(point_value(card in hand)).

Note: the random policy can deadlock (e.g. all-down + all-buys-maxed and no
remaining hand cards match any down pile rank). That's a property of the
policy, not the engine. We tolerate non-termination per seed; we only require
that invariants hold throughout however long the game runs, and that *most*
seeds terminate within MAX_TURNS.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from rummy_v2.actions import Action, Discard
from rummy_v2.cards import build_deck
from rummy_v2.engine import apply_action
from rummy_v2.random_play import random_action
from rummy_v2.state import (
    HAND_SIZE,
    MAX_BUYS_PER_ROUND,
    NUM_PLAYERS,
    GameState,
    Phase,
    initial_state,
)


CANONICAL_DECK = Counter(build_deck())
# Peak hand size occurs mid-turn: 11 starting + 6 from buys + 1 freshly drawn,
# pre-discard. After the discard the hand is back to ≤17.
MAX_HAND_SIZE = HAND_SIZE + 2 * MAX_BUYS_PER_ROUND + 1  # 18


def _all_cards_in(state: GameState) -> Counter:
    bag: Counter = Counter()
    bag.update(state.draw_pile)
    bag.update(state.discard_pile)
    for p in state.players:
        bag.update(p.hand)
        for pile in p.down_piles:
            bag.update(pile.cards)
    return bag


def _make_invariant_checker():
    """Returns an `on_step` callback that asserts engine invariants on every
    transition. Captures previous-state buys and active player to enforce
    monotonicity / change-only-on-discard."""
    state_box: dict[str, GameState | None] = {"prev": None}

    def check(state: GameState, action: Action) -> None:
        # Card conservation
        assert _all_cards_in(state) == CANONICAL_DECK, (
            f"card conservation violated after {action!r}"
        )

        # Hand-size bounds
        for i, p in enumerate(state.players):
            assert 0 <= len(p.hand) <= MAX_HAND_SIZE, (
                f"P{i} hand size {len(p.hand)} out of [0, {MAX_HAND_SIZE}]"
            )
            assert 0 <= p.buys_used <= MAX_BUYS_PER_ROUND

        prev = state_box["prev"]
        if prev is not None:
            # Buys monotonic
            for i in range(NUM_PLAYERS):
                assert state.players[i].buys_used >= prev.players[i].buys_used, (
                    f"P{i} buys_used decreased after {action!r}"
                )

            # Active player only changes via Discard
            if state.active_player != prev.active_player:
                assert isinstance(action, Discard), (
                    f"active_player changed via non-Discard action {action!r}"
                )

        state_box["prev"] = state

    return check


MAX_TURNS = 5000


def _run_with_invariants(seed: int) -> tuple[GameState, bool]:
    """Run one random game; assert engine invariants on every transition.
    Returns (final_state, terminated). `terminated` is True iff phase reached
    ROUND_OVER within MAX_TURNS.
    """
    check = _make_invariant_checker()
    rng = random.Random(seed)
    state = initial_state(rng)
    for _ in range(MAX_TURNS):
        if state.phase is Phase.ROUND_OVER:
            return state, True
        action = random_action(state, rng)
        next_state = apply_action(state, action, rng)
        check(next_state, action)
        state = next_state
    return state, False


@pytest.mark.parametrize("seed", range(50))
def test_random_play_invariants(seed: int) -> None:
    """For each seed: invariants hold throughout; if game terminated, the
    terminal state has correct scoring."""
    state, terminated = _run_with_invariants(seed)
    if terminated:
        assert state.round_winner is not None
        assert state.round_scores is not None
        for i, p in enumerate(state.players):
            if i == state.round_winner:
                assert state.round_scores[i] == 0
                assert len(p.hand) == 0
            else:
                assert state.round_scores[i] == sum(
                    c.point_value for c in p.hand
                )


def test_termination_rate() -> None:
    """At least 80% of 100 random seeds should terminate within MAX_TURNS.
    Non-termination is a policy artifact (random play can deadlock when all
    players are down + maxed buys + no hand cards match any pile rank); we
    only require that it's the minority case."""
    terminations = 0
    for seed in range(100):
        rng = random.Random(seed)
        state = initial_state(rng)
        for _ in range(MAX_TURNS):
            if state.phase is Phase.ROUND_OVER:
                terminations += 1
                break
            action = random_action(state, rng)
            state = apply_action(state, action, rng)
    assert terminations >= 80, f"only {terminations}/100 games terminated"
