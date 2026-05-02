"""Action variants and legality checks for Speed Rummy round 1.

`Action` is a tagged union of frozen dataclasses. The engine consumes them to
derive new states. `is_legal` and `legal_actions` are the public gating API.

Note on enumeration: `legal_actions` deliberately does NOT enumerate every
possible `GoDown` partition — the search space is combinatorial. Use
`find_any_go_down` to get a single legal go-down (handy for random-play tests
and the CLI's "auto" suggestion).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Union

from rummy_v2.cards import Card, Rank
from rummy_v2.state import (
    MAX_BUYS_PER_ROUND,
    MAX_WILDS_PER_MELD,
    NUM_PLAYERS,
    SET_SIZE,
    GameState,
    Phase,
)


# --- Action variants ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrawDiscard:
    """Active player takes the top of the discard pile as their draw."""


@dataclass(frozen=True, slots=True)
class DrawDeck:
    """Active player commits to drawing from the deck. If the discard pile is
    non-empty, this triggers the buy cascade for non-active players first; the
    deck draw is forced after the cascade resolves."""


@dataclass(frozen=True, slots=True)
class Buy:
    """The current `buy_offer_to` player buys the top of the discard pile,
    incurring a penalty card from the deck."""


@dataclass(frozen=True, slots=True)
class PassBuy:
    """The current `buy_offer_to` player declines the top of the discard pile."""


@dataclass(frozen=True, slots=True)
class GoDown:
    """Active player melds two sets of three. Each set: three cards from hand
    sharing a rank, with up to two wilds permitted as substitutes."""

    set1: tuple[Card, Card, Card]
    set2: tuple[Card, Card, Card]


@dataclass(frozen=True, slots=True)
class LayOff:
    """Active player adds one card from their hand to an existing down pile.
    Only legal once the active player has gone down themselves."""

    card: Card
    target_player: int
    pile_index: int


@dataclass(frozen=True, slots=True)
class Discard:
    """Active player discards one card to end their turn."""

    card: Card


Action = Union[DrawDiscard, DrawDeck, Buy, PassBuy, GoDown, LayOff, Discard]


# --- Set / meld helpers -------------------------------------------------------


def is_valid_set(cards: tuple[Card, ...]) -> bool:
    """A valid set: SET_SIZE cards, ≤MAX_WILDS_PER_MELD wilds, and every
    non-wild card shares a rank."""
    if len(cards) != SET_SIZE:
        return False
    non_wilds = [c for c in cards if not c.is_wild]
    wilds = len(cards) - len(non_wilds)
    if wilds > MAX_WILDS_PER_MELD:
        return False
    if not non_wilds:
        return False  # all wilds — already excluded by max-wilds, but defensive
    rank = non_wilds[0].rank
    return all(c.rank is rank for c in non_wilds)


def set_rank(cards: tuple[Card, ...]) -> Rank:
    """Canonical rank of a (presumed valid) set."""
    for c in cards:
        if not c.is_wild:
            return c.rank
    raise ValueError("set has no non-wild card to determine rank")


def _is_submultiset(needed: Counter[Card], available: Counter[Card]) -> bool:
    return all(available.get(card, 0) >= count for card, count in needed.items())


# --- Legality -----------------------------------------------------------------


def is_legal(state: GameState, action: Action) -> bool:
    """True iff `action` is a legal move from `state`."""
    if state.phase is Phase.ROUND_OVER:
        return False

    match action:
        case DrawDiscard():
            return (
                state.phase is Phase.AWAITING_DRAW and state.top_discard is not None
            )

        case DrawDeck():
            return state.phase is Phase.AWAITING_DRAW

        case Buy():
            if state.phase is not Phase.BUY_CASCADE:
                return False
            buyer = state.players[state.buy_offer_to]
            return (
                state.top_discard is not None
                and buyer.buys_used < MAX_BUYS_PER_ROUND
            )

        case PassBuy():
            return state.phase is Phase.BUY_CASCADE

        case GoDown(set1=s1, set2=s2):
            if state.phase is not Phase.POST_DRAW:
                return False
            actor = state.players[state.active_player]
            if actor.is_down:
                return False
            if not (is_valid_set(s1) and is_valid_set(s2)):
                return False
            needed = Counter(s1) + Counter(s2)
            return _is_submultiset(needed, Counter(actor.hand))

        case LayOff(card=card, target_player=tp, pile_index=pi):
            if state.phase is not Phase.POST_DRAW:
                return False
            actor = state.players[state.active_player]
            if not actor.is_down:
                return False
            # Per RULES.md §7, the active player must always end their turn
            # with a discard — laying off the final card is not allowed.
            if len(actor.hand) < 2:
                return False
            if not 0 <= tp < NUM_PLAYERS:
                return False
            target = state.players[tp]
            if not 0 <= pi < len(target.down_piles):
                return False
            if Counter(actor.hand).get(card, 0) < 1:
                return False
            pile = target.down_piles[pi]
            if card.is_wild:
                return pile.num_wilds < MAX_WILDS_PER_MELD
            return card.rank is pile.rank

        case Discard(card=card):
            if state.phase is not Phase.POST_DRAW:
                return False
            actor = state.players[state.active_player]
            return Counter(actor.hand).get(card, 0) >= 1

    return False


def legal_actions(state: GameState) -> list[Action]:
    """Enumerate the legal "simple" actions from `state` — i.e., all action
    types except `GoDown`. Use `find_any_go_down(actor.hand)` separately to
    decide whether going down is possible.

    Returns an empty list when the round is over.
    """
    if state.phase is Phase.ROUND_OVER:
        return []

    if state.phase is Phase.AWAITING_DRAW:
        actions: list[Action] = [DrawDeck()]
        if state.top_discard is not None:
            actions.append(DrawDiscard())
        return actions

    if state.phase is Phase.BUY_CASCADE:
        actions = [PassBuy()]
        if is_legal(state, Buy()):
            actions.append(Buy())
        return actions

    # POST_DRAW
    actor = state.players[state.active_player]
    actions = []
    hand_unique = set(actor.hand)

    if actor.is_down:
        for tp in range(NUM_PLAYERS):
            for pi in range(len(state.players[tp].down_piles)):
                for card in hand_unique:
                    candidate: Action = LayOff(
                        card=card, target_player=tp, pile_index=pi
                    )
                    if is_legal(state, candidate):
                        actions.append(candidate)

    for card in hand_unique:
        actions.append(Discard(card=card))

    return actions


# --- Go-down search -----------------------------------------------------------


def find_any_go_down(hand: tuple[Card, ...]) -> GoDown | None:
    """Return one legal `GoDown` from `hand`, or None if no valid two-set
    partition exists. Deterministic for a given hand ordering.

    Strategy: group non-wilds by rank, count wilds, then try every assignment
    of (rank, non-wilds-used, wilds-used) for set 1 followed by set 2.
    """
    wilds: list[Card] = [c for c in hand if c.is_wild]
    by_rank: dict[Rank, list[Card]] = {}
    for c in hand:
        if not c.is_wild:
            by_rank.setdefault(c.rank, []).append(c)

    # For each rank, the set "shapes" available given how many non-wilds are
    # left for it (we may have to subtract some after picking set 1).
    def shapes(non_wilds_avail: int) -> list[tuple[int, int]]:
        """Return (n_non_wild, n_wild) pairs that can form a SET_SIZE meld."""
        options: list[tuple[int, int]] = []
        if non_wilds_avail >= 3:
            options.append((3, 0))
        if non_wilds_avail >= 2:
            options.append((2, 1))
        if non_wilds_avail >= 1:
            options.append((1, 2))
        return options

    total_wilds = len(wilds)

    for rank1, nw_list1 in by_rank.items():
        for nw1, w1 in shapes(len(nw_list1)):
            if w1 > total_wilds:
                continue
            wilds_after = total_wilds - w1
            remaining_of_r1 = len(nw_list1) - nw1

            for rank2, nw_list2 in by_rank.items():
                if rank2 is rank1:
                    avail2 = remaining_of_r1
                else:
                    avail2 = len(nw_list2)
                for nw2, w2 in shapes(avail2):
                    if w2 > wilds_after:
                        continue
                    set1 = tuple(nw_list1[:nw1] + wilds[:w1])
                    if rank2 is rank1:
                        set2_nw = nw_list1[nw1 : nw1 + nw2]
                    else:
                        set2_nw = nw_list2[:nw2]
                    set2 = tuple(set2_nw + wilds[w1 : w1 + w2])
                    return GoDown(set1=set1, set2=set2)  # type: ignore[arg-type]
    return None
