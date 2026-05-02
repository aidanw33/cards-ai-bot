"""Engine: applies actions to immutable game states.

The single public entry point is `apply_action(state, action, rng) -> GameState`.
Illegal actions raise ValueError. RNG is only consulted when shuffling the
discard pile back into the draw pile (RULES.md §9); deterministic actions are
RNG-free.

Phase machine (recap):

    AWAITING_DRAW
      ├─ DrawDiscard     ──> POST_DRAW  (active grabs top of discard)
      └─ DrawDeck
          ├─ discard empty  or  no eligible buyer ──> POST_DRAW (active draws)
          └─ otherwise                            ──> BUY_CASCADE

    BUY_CASCADE
      ├─ Buy
      │   ├─ buyer at limit, or discard empty ──> _advance_cascade
      │   └─ otherwise                        ──> stay (same buyer may buy again)
      └─ PassBuy                              ──> _advance_cascade

    _advance_cascade
      ├─ next eligible buyer found ──> stay in BUY_CASCADE w/ new buy_offer_to
      └─ no one left or discard empty ──> active draws from deck ──> POST_DRAW

    POST_DRAW
      ├─ GoDown    ──> stay (player can lay off / discard)
      ├─ LayOff    ──> stay
      └─ Discard
          ├─ hand becomes empty ──> ROUND_OVER (scores populated)
          └─ otherwise          ──> AWAITING_DRAW (next player)
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter

from rummy_v2.actions import (
    Action,
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    GoDown,
    LayOff,
    PassBuy,
    is_legal,
    set_rank,
)
from rummy_v2.cards import Card
from rummy_v2.state import (
    MAX_BUYS_PER_ROUND,
    NUM_PLAYERS,
    DownPile,
    GameState,
    Phase,
    PlayerState,
)


def apply_action(
    state: GameState, action: Action, rng: random.Random
) -> GameState:
    if not is_legal(state, action):
        raise ValueError(f"illegal action {action!r} from phase {state.phase.name}")

    match action:
        case DrawDiscard():
            return _apply_draw_discard(state)
        case DrawDeck():
            return _apply_draw_deck(state, rng)
        case Buy():
            return _apply_buy(state, rng)
        case PassBuy():
            return _apply_pass_buy(state, rng)
        case GoDown(set1=s1, set2=s2):
            return _apply_go_down(state, s1, s2)
        case LayOff(card=c, target_player=tp, pile_index=pi):
            return _apply_lay_off(state, c, tp, pi)
        case Discard(card=c):
            return _apply_discard(state, c)

    raise AssertionError("unreachable")


# --- AWAITING_DRAW ------------------------------------------------------------


def _apply_draw_discard(state: GameState) -> GameState:
    actor_idx = state.active_player
    actor = state.players[actor_idx]
    top = state.discard_pile[-1]
    new_actor = dataclasses.replace(actor, hand=actor.hand + (top,))
    return dataclasses.replace(
        state,
        players=_replace_player(state.players, actor_idx, new_actor),
        discard_pile=state.discard_pile[:-1],
        phase=Phase.POST_DRAW,
    )


def _apply_draw_deck(state: GameState, rng: random.Random) -> GameState:
    if state.discard_pile:
        first_buyer = _next_eligible_buyer(state, state.active_player)
        if first_buyer is not None:
            return dataclasses.replace(
                state, phase=Phase.BUY_CASCADE, buy_offer_to=first_buyer
            )
    return _do_active_deck_draw(state, rng)


def _do_active_deck_draw(state: GameState, rng: random.Random) -> GameState:
    card, new_draw, new_discard = _take_from_deck(
        state.draw_pile, state.discard_pile, rng
    )
    actor = state.players[state.active_player]
    new_actor = dataclasses.replace(actor, hand=actor.hand + (card,))
    return dataclasses.replace(
        state,
        players=_replace_player(state.players, state.active_player, new_actor),
        draw_pile=new_draw,
        discard_pile=new_discard,
        phase=Phase.POST_DRAW,
        buy_offer_to=state.active_player,
    )


# --- BUY_CASCADE --------------------------------------------------------------


def _apply_buy(state: GameState, rng: random.Random) -> GameState:
    buyer_idx = state.buy_offer_to
    buyer = state.players[buyer_idx]
    top = state.discard_pile[-1]
    discard_after_top = state.discard_pile[:-1]

    penalty, new_draw, new_discard = _take_from_deck(
        state.draw_pile, discard_after_top, rng
    )

    new_buyer = dataclasses.replace(
        buyer,
        hand=buyer.hand + (top, penalty),
        buys_used=buyer.buys_used + 1,
    )
    new_state = dataclasses.replace(
        state,
        players=_replace_player(state.players, buyer_idx, new_buyer),
        draw_pile=new_draw,
        discard_pile=new_discard,
    )

    if (
        new_buyer.buys_used >= MAX_BUYS_PER_ROUND
        or not new_state.discard_pile
    ):
        return _advance_cascade(new_state, rng, after=buyer_idx)
    return new_state


def _apply_pass_buy(state: GameState, rng: random.Random) -> GameState:
    return _advance_cascade(state, rng, after=state.buy_offer_to)


def _advance_cascade(
    state: GameState, rng: random.Random, after: int
) -> GameState:
    next_buyer = (
        _next_eligible_buyer(state, after) if state.discard_pile else None
    )
    if next_buyer is None:
        return _do_active_deck_draw(state, rng)
    return dataclasses.replace(state, buy_offer_to=next_buyer)


def _next_eligible_buyer(state: GameState, after: int) -> int | None:
    active = state.active_player
    cand = (after + 1) % NUM_PLAYERS
    while cand != active:
        if state.players[cand].buys_used < MAX_BUYS_PER_ROUND:
            return cand
        cand = (cand + 1) % NUM_PLAYERS
    return None


def _take_from_deck(
    draw_pile: tuple[Card, ...],
    discard_pile: tuple[Card, ...],
    rng: random.Random,
) -> tuple[Card, tuple[Card, ...], tuple[Card, ...]]:
    """Draw one card. If the draw pile is empty, shuffle the *entire* discard
    pile (RULES.md §9) into a new draw pile and continue."""
    if not draw_pile:
        if not discard_pile:
            raise RuntimeError("both draw and discard piles are empty")
        shuffled = list(discard_pile)
        rng.shuffle(shuffled)
        draw_pile = tuple(shuffled)
        discard_pile = ()
    card = draw_pile[-1]
    return card, draw_pile[:-1], discard_pile


# --- POST_DRAW ----------------------------------------------------------------


def _apply_go_down(
    state: GameState,
    set1: tuple[Card, Card, Card],
    set2: tuple[Card, Card, Card],
) -> GameState:
    actor_idx = state.active_player
    actor = state.players[actor_idx]

    needed = Counter(set1) + Counter(set2)
    new_hand = _multiset_subtract(actor.hand, needed)

    pile1 = DownPile(rank=set_rank(set1), cards=set1)
    pile2 = DownPile(rank=set_rank(set2), cards=set2)
    new_actor = dataclasses.replace(
        actor, hand=new_hand, down_piles=(pile1, pile2)
    )
    return dataclasses.replace(
        state, players=_replace_player(state.players, actor_idx, new_actor)
    )


def _apply_lay_off(
    state: GameState, card: Card, target_player: int, pile_index: int
) -> GameState:
    actor_idx = state.active_player
    actor = state.players[actor_idx]
    new_hand = _remove_one(actor.hand, card)

    if actor_idx == target_player:
        existing_piles = list(actor.down_piles)
    else:
        existing_piles = list(state.players[target_player].down_piles)

    old_pile = existing_piles[pile_index]
    existing_piles[pile_index] = DownPile(
        rank=old_pile.rank, cards=old_pile.cards + (card,)
    )
    new_piles = tuple(existing_piles)

    new_players = list(state.players)
    if actor_idx == target_player:
        new_players[actor_idx] = dataclasses.replace(
            actor, hand=new_hand, down_piles=new_piles
        )
    else:
        new_players[actor_idx] = dataclasses.replace(actor, hand=new_hand)
        target = state.players[target_player]
        new_players[target_player] = dataclasses.replace(
            target, down_piles=new_piles
        )

    return dataclasses.replace(state, players=tuple(new_players))


def _apply_discard(state: GameState, card: Card) -> GameState:
    actor_idx = state.active_player
    actor = state.players[actor_idx]
    new_hand = _remove_one(actor.hand, card)
    new_actor = dataclasses.replace(actor, hand=new_hand)
    new_discard = state.discard_pile + (card,)
    new_players = _replace_player(state.players, actor_idx, new_actor)

    if not new_hand:
        scores = tuple(
            0 if i == actor_idx else sum(c.point_value for c in p.hand)
            for i, p in enumerate(new_players)
        )
        return dataclasses.replace(
            state,
            players=new_players,
            discard_pile=new_discard,
            phase=Phase.ROUND_OVER,
            round_winner=actor_idx,
            round_scores=scores,
        )

    next_active = (actor_idx + 1) % NUM_PLAYERS
    return dataclasses.replace(
        state,
        players=new_players,
        discard_pile=new_discard,
        active_player=next_active,
        phase=Phase.AWAITING_DRAW,
        buy_offer_to=next_active,
    )


# --- helpers ------------------------------------------------------------------


def _replace_player(
    players: tuple[PlayerState, ...], idx: int, new: PlayerState
) -> tuple[PlayerState, ...]:
    new_players = list(players)
    new_players[idx] = new
    return tuple(new_players)


def _remove_one(seq: tuple[Card, ...], card: Card) -> tuple[Card, ...]:
    for i, c in enumerate(seq):
        if c == card:
            return seq[:i] + seq[i + 1 :]
    raise ValueError(f"{card} not found in sequence")


def _multiset_subtract(
    seq: tuple[Card, ...], to_remove: Counter[Card]
) -> tuple[Card, ...]:
    """Remove the multiset `to_remove` from `seq`, preserving the relative
    order of surviving elements."""
    remaining = Counter(to_remove)
    result: list[Card] = []
    for c in seq:
        if remaining.get(c, 0) > 0:
            remaining[c] -= 1
        else:
            result.append(c)
    return tuple(result)
