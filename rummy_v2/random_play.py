"""Random-action policy and game runner — used by smoke tests and (later)
as a baseline opponent for self-play training.

The random policy is *not* purely uniform-over-legal-actions: that would never
finish, since no-one would ever go down. Instead:

  - In POST_DRAW: if the active player can go down, do it. If already down,
    lay off as many cards as possible. Otherwise pick a random Discard.
  - In every other phase: uniform over legal actions.

This guarantees forward progress and lets us stress-test the engine.
"""

from __future__ import annotations

import random
from typing import Callable

from rummy_v2.actions import (
    Action,
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    LayOff,
    Phase,
    find_any_go_down,
    legal_actions,
)
from rummy_v2.cards import Rank
from rummy_v2.engine import apply_action
from rummy_v2.state import GameState, initial_state


def _down_pile_ranks(state: GameState) -> set[Rank]:
    ranks: set[Rank] = set()
    for p in state.players:
        for pile in p.down_piles:
            ranks.add(pile.rank)
    return ranks


def random_action(state: GameState, rng: random.Random) -> Action:
    """Return one legal action under the progress-biased random policy.

    Biases used (not pure-uniform): always go down when possible, always lay
    off when possible, prefer drawing the discard top only when it matches a
    pile rank (otherwise dead weight just cycles through the system), and
    only buy when the top of the discard matches a pile rank.
    """
    if state.phase is Phase.AWAITING_DRAW:
        top = state.top_discard
        pile_ranks = _down_pile_ranks(state)
        if (
            top is not None
            and (top.is_wild or top.rank in pile_ranks)
        ):
            return DrawDiscard()
        return DrawDeck()

    if state.phase is Phase.BUY_CASCADE:
        top = state.top_discard
        pile_ranks = _down_pile_ranks(state)
        if (
            top is not None
            and (top.is_wild or top.rank in pile_ranks)
            and Buy() in legal_actions(state)
        ):
            return Buy()
        from rummy_v2.actions import PassBuy

        return PassBuy()

    if state.phase is Phase.POST_DRAW:
        actor = state.players[state.active_player]
        if not actor.is_down:
            go = find_any_go_down(actor.hand)
            if go is not None:
                return go
        if actor.is_down:
            lay_offs = [
                a for a in legal_actions(state) if isinstance(a, LayOff)
            ]
            if lay_offs:
                return rng.choice(lay_offs)
        discards = [a for a in legal_actions(state) if isinstance(a, Discard)]
        return rng.choice(discards)

    return rng.choice(legal_actions(state))


def play_random_game(
    seed: int,
    *,
    max_turns: int = 2000,
    on_step: Callable[[GameState, Action], None] | None = None,
) -> GameState:
    """Run one full round under the random policy. Returns the terminal state.

    Raises RuntimeError if the game exceeds `max_turns` transitions — useful
    for catching infinite-loop bugs. `on_step(state, action)` is invoked after
    each transition for tracing.
    """
    rng = random.Random(seed)
    state = initial_state(rng)
    for _ in range(max_turns):
        if state.phase is Phase.ROUND_OVER:
            return state
        action = random_action(state, rng)
        next_state = apply_action(state, action, rng)
        if on_step is not None:
            on_step(next_state, action)
        state = next_state
    raise RuntimeError(f"game did not terminate within {max_turns} transitions")
