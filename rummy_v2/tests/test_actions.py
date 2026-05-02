"""Tests for actions.py — set validity, legality, go-down finder."""

from __future__ import annotations

import pytest

from rummy_v2.actions import (
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    GoDown,
    LayOff,
    PassBuy,
    find_any_go_down,
    is_legal,
    is_valid_set,
    legal_actions,
    set_rank,
)
from rummy_v2.cards import Card, Rank, Suit
from rummy_v2.state import (
    HAND_SIZE,
    MAX_BUYS_PER_ROUND,
    NUM_PLAYERS,
    DownPile,
    GameState,
    Phase,
    PlayerState,
)


# ----------------------- Helpers ----------------------------------------------


def _c(rank: Rank, suit: Suit | None = Suit.HEARTS) -> Card:
    if rank is Rank.JOKER:
        return Card(rank=Rank.JOKER, suit=None)
    return Card(rank=rank, suit=suit)


def _kings_set() -> tuple[Card, Card, Card]:
    return (
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.KING, suit=Suit.SPADES),
        Card(rank=Rank.KING, suit=Suit.DIAMONDS),
    )


def _sevens_set() -> tuple[Card, Card, Card]:
    return (
        Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
        Card(rank=Rank.SEVEN, suit=Suit.SPADES),
        Card(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
    )


def _state(
    *,
    phase: Phase = Phase.AWAITING_DRAW,
    active: int = 0,
    buy_offer_to: int | None = None,
    hand: tuple[Card, ...] = (),
    is_down: bool = False,
    buys_used: int = 0,
    discard: tuple[Card, ...] = (),
    draw: tuple[Card, ...] = (),
    other_down_piles: tuple[DownPile, ...] = (),
) -> GameState:
    """Minimal state factory for tests."""
    actor_piles: tuple[DownPile, ...] = ()
    if is_down:
        actor_piles = (DownPile(rank=Rank.KING, cards=_kings_set()),)
    actor = PlayerState(hand=hand, down_piles=actor_piles, buys_used=buys_used)
    other_with_piles_idx = 1 if active != 1 else 2
    players: list[PlayerState] = []
    for i in range(NUM_PLAYERS):
        if i == active:
            players.append(actor)
        elif i == other_with_piles_idx:
            players.append(
                PlayerState(hand=(), down_piles=other_down_piles, buys_used=0)
            )
        else:
            players.append(PlayerState(hand=(), down_piles=(), buys_used=0))
    return GameState(
        players=tuple(players),
        draw_pile=draw,
        discard_pile=discard,
        active_player=active,
        phase=phase,
        buy_offer_to=active if buy_offer_to is None else buy_offer_to,
    )


# ----------------------- Set validity -----------------------------------------


class TestIsValidSet:
    def test_three_same_rank_off_suit(self) -> None:
        assert is_valid_set(_kings_set())

    def test_two_same_rank_one_wild(self) -> None:
        s = (
            Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
            Card(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
            Card(rank=Rank.JOKER, suit=None),
        )
        assert is_valid_set(s)

    def test_one_non_wild_two_wilds(self) -> None:
        s = (
            Card(rank=Rank.QUEEN, suit=Suit.HEARTS),
            Card(rank=Rank.JOKER, suit=None),
            Card(rank=Rank.TWO, suit=Suit.SPADES),  # black 2 wild
        )
        assert is_valid_set(s)

    def test_three_wilds_invalid(self) -> None:
        s = (
            Card(rank=Rank.JOKER, suit=None),
            Card(rank=Rank.JOKER, suit=None),
            Card(rank=Rank.TWO, suit=Suit.SPADES),
        )
        assert not is_valid_set(s)

    def test_mixed_ranks_invalid(self) -> None:
        s = (
            Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
            Card(rank=Rank.JOKER, suit=None),
        )
        assert not is_valid_set(s)

    def test_wrong_size_invalid(self) -> None:
        assert not is_valid_set(_kings_set()[:2])

    def test_red_two_is_not_a_wild(self) -> None:
        # Red two is rank TWO, non-wild. Two red 2s + one wild is a valid 2s set.
        s = (
            Card(rank=Rank.TWO, suit=Suit.HEARTS),
            Card(rank=Rank.TWO, suit=Suit.DIAMONDS),
            Card(rank=Rank.JOKER, suit=None),
        )
        assert is_valid_set(s)
        assert set_rank(s) is Rank.TWO


# ----------------------- AWAITING_DRAW phase ----------------------------------


class TestDrawPhase:
    def test_draw_deck_always_legal(self) -> None:
        s = _state(phase=Phase.AWAITING_DRAW, draw=(_c(Rank.KING),))
        assert is_legal(s, DrawDeck())

    def test_draw_discard_legal_when_discard_non_empty(self) -> None:
        s = _state(phase=Phase.AWAITING_DRAW, discard=(_c(Rank.KING),))
        assert is_legal(s, DrawDiscard())

    def test_draw_discard_illegal_when_discard_empty(self) -> None:
        s = _state(phase=Phase.AWAITING_DRAW)
        assert not is_legal(s, DrawDiscard())

    def test_buy_illegal_in_draw_phase(self) -> None:
        s = _state(phase=Phase.AWAITING_DRAW, discard=(_c(Rank.KING),))
        assert not is_legal(s, Buy())

    def test_legal_actions_includes_both_when_discard_present(self) -> None:
        s = _state(phase=Phase.AWAITING_DRAW, discard=(_c(Rank.KING),))
        acts = legal_actions(s)
        assert DrawDeck() in acts
        assert DrawDiscard() in acts


# ----------------------- BUY_CASCADE phase ------------------------------------


class TestBuyPhase:
    def test_buy_legal_when_discard_present_and_under_limit(self) -> None:
        s = _state(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=2,
            discard=(_c(Rank.KING),),
        )
        assert is_legal(s, Buy())

    def test_buy_illegal_when_at_buy_limit(self) -> None:
        # Set buyer's buys_used to 3.
        s = _state(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=2,
            discard=(_c(Rank.KING),),
        )
        # Hack: rebuild with buyer having 3 buys.
        new_players = list(s.players)
        new_players[2] = PlayerState(
            hand=(), down_piles=(), buys_used=MAX_BUYS_PER_ROUND
        )
        s2 = GameState(
            players=tuple(new_players),
            draw_pile=s.draw_pile,
            discard_pile=s.discard_pile,
            active_player=s.active_player,
            phase=s.phase,
            buy_offer_to=s.buy_offer_to,
        )
        assert not is_legal(s2, Buy())

    def test_buy_illegal_when_discard_empty(self) -> None:
        s = _state(phase=Phase.BUY_CASCADE, active=0, buy_offer_to=2)
        assert not is_legal(s, Buy())

    def test_pass_buy_always_legal_in_cascade(self) -> None:
        s = _state(phase=Phase.BUY_CASCADE, active=0, buy_offer_to=2)
        assert is_legal(s, PassBuy())

    def test_draw_illegal_in_buy_phase(self) -> None:
        s = _state(phase=Phase.BUY_CASCADE, discard=(_c(Rank.KING),))
        assert not is_legal(s, DrawDeck())
        assert not is_legal(s, DrawDiscard())


# ----------------------- POST_DRAW: GoDown ------------------------------------


class TestGoDown:
    def test_legal_with_two_sets(self) -> None:
        hand = _kings_set() + _sevens_set() + (_c(Rank.JACK, Suit.CLUBS),)
        s = _state(phase=Phase.POST_DRAW, hand=hand)
        action = GoDown(set1=_kings_set(), set2=_sevens_set())
        assert is_legal(s, action)

    def test_illegal_if_already_down(self) -> None:
        hand = _kings_set() + _sevens_set()
        s = _state(phase=Phase.POST_DRAW, hand=hand, is_down=True)
        action = GoDown(set1=_kings_set(), set2=_sevens_set())
        assert not is_legal(s, action)

    def test_illegal_if_card_not_in_hand(self) -> None:
        # Hand has only one set worth of cards; can't form 6-card go-down.
        hand = _kings_set()
        s = _state(phase=Phase.POST_DRAW, hand=hand)
        action = GoDown(set1=_kings_set(), set2=_sevens_set())
        assert not is_legal(s, action)

    def test_illegal_in_wrong_phase(self) -> None:
        hand = _kings_set() + _sevens_set()
        s = _state(phase=Phase.AWAITING_DRAW, hand=hand)
        action = GoDown(set1=_kings_set(), set2=_sevens_set())
        assert not is_legal(s, action)

    def test_illegal_if_set_invalid(self) -> None:
        # set1 has mixed ranks
        bad_set = (
            Card(rank=Rank.KING, suit=Suit.HEARTS),
            Card(rank=Rank.SEVEN, suit=Suit.SPADES),
            Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
        )
        hand = bad_set + _sevens_set()
        s = _state(phase=Phase.POST_DRAW, hand=hand)
        action = GoDown(set1=bad_set, set2=_sevens_set())
        assert not is_legal(s, action)


# ----------------------- POST_DRAW: LayOff ------------------------------------


class TestLayOff:
    def test_legal_matching_rank(self) -> None:
        # Other player has a kings down-pile; actor has a king to lay off.
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())
        # Need >=2 cards in hand: lay-off can't reduce hand to 0.
        hand = (
            Card(rank=Rank.KING, suit=Suit.CLUBS),
            Card(rank=Rank.THREE, suit=Suit.HEARTS),
        )
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.KING, suit=Suit.CLUBS), target_player=1, pile_index=0
        )
        assert is_legal(s, action)

    def test_illegal_when_actor_not_down(self) -> None:
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())
        hand = (Card(rank=Rank.KING, suit=Suit.CLUBS),)
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=False,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.KING, suit=Suit.CLUBS), target_player=1, pile_index=0
        )
        assert not is_legal(s, action)

    def test_illegal_rank_mismatch(self) -> None:
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())
        hand = (Card(rank=Rank.SEVEN, suit=Suit.CLUBS),)
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.SEVEN, suit=Suit.CLUBS), target_player=1, pile_index=0
        )
        assert not is_legal(s, action)

    def test_legal_wild_when_pile_under_wild_limit(self) -> None:
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())  # 0 wilds
        hand = (
            Card(rank=Rank.JOKER, suit=None),
            Card(rank=Rank.THREE, suit=Suit.HEARTS),
        )
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.JOKER, suit=None), target_player=1, pile_index=0
        )
        assert is_legal(s, action)

    def test_illegal_wild_when_pile_at_wild_limit(self) -> None:
        target_pile = DownPile(
            rank=Rank.KING,
            cards=(
                Card(rank=Rank.KING, suit=Suit.HEARTS),
                Card(rank=Rank.JOKER, suit=None),
                Card(rank=Rank.TWO, suit=Suit.SPADES),
            ),
        )
        hand = (Card(rank=Rank.JOKER, suit=None),)
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.JOKER, suit=None), target_player=1, pile_index=0
        )
        assert not is_legal(s, action)

    def test_illegal_card_not_in_hand(self) -> None:
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())
        s = _state(
            phase=Phase.POST_DRAW,
            hand=(),
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.KING, suit=Suit.CLUBS), target_player=1, pile_index=0
        )
        assert not is_legal(s, action)

    def test_illegal_when_lay_off_would_empty_hand(self) -> None:
        # Hand has exactly 1 card; laying off would leave nothing to discard.
        target_pile = DownPile(rank=Rank.KING, cards=_kings_set())
        hand = (Card(rank=Rank.KING, suit=Suit.CLUBS),)
        s = _state(
            phase=Phase.POST_DRAW,
            hand=hand,
            is_down=True,
            other_down_piles=(target_pile,),
        )
        action = LayOff(
            card=Card(rank=Rank.KING, suit=Suit.CLUBS), target_player=1, pile_index=0
        )
        assert not is_legal(s, action)


# ----------------------- POST_DRAW: Discard -----------------------------------


class TestDiscard:
    def test_legal_when_card_in_hand(self) -> None:
        hand = (Card(rank=Rank.SEVEN, suit=Suit.HEARTS),)
        s = _state(phase=Phase.POST_DRAW, hand=hand)
        action = Discard(card=Card(rank=Rank.SEVEN, suit=Suit.HEARTS))
        assert is_legal(s, action)

    def test_illegal_when_card_not_in_hand(self) -> None:
        s = _state(phase=Phase.POST_DRAW, hand=())
        action = Discard(card=Card(rank=Rank.SEVEN, suit=Suit.HEARTS))
        assert not is_legal(s, action)

    def test_illegal_in_wrong_phase(self) -> None:
        hand = (Card(rank=Rank.SEVEN, suit=Suit.HEARTS),)
        s = _state(phase=Phase.AWAITING_DRAW, hand=hand)
        action = Discard(card=Card(rank=Rank.SEVEN, suit=Suit.HEARTS))
        assert not is_legal(s, action)


# ----------------------- ROUND_OVER -------------------------------------------


class TestRoundOver:
    def test_no_actions_legal(self) -> None:
        empty = PlayerState(hand=(), down_piles=(), buys_used=0)
        s = GameState(
            players=tuple(empty for _ in range(NUM_PLAYERS)),
            draw_pile=(),
            discard_pile=(),
            active_player=0,
            phase=Phase.ROUND_OVER,
            buy_offer_to=0,
            round_winner=0,
            round_scores=(0,) * NUM_PLAYERS,
        )
        assert legal_actions(s) == []
        assert not is_legal(s, DrawDeck())


# ----------------------- find_any_go_down -------------------------------------


class TestFindAnyGoDown:
    def test_finds_two_pure_sets(self) -> None:
        hand = _kings_set() + _sevens_set() + (Card(rank=Rank.JACK, suit=Suit.HEARTS),)
        result = find_any_go_down(hand)
        assert result is not None
        assert is_valid_set(result.set1) and is_valid_set(result.set2)

    def test_finds_set_using_wilds(self) -> None:
        hand = (
            Card(rank=Rank.KING, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
            Card(rank=Rank.JOKER, suit=None),
            Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
            Card(rank=Rank.SEVEN, suit=Suit.SPADES),
            Card(rank=Rank.TWO, suit=Suit.SPADES),  # black 2 wild
        )
        result = find_any_go_down(hand)
        assert result is not None
        assert is_valid_set(result.set1) and is_valid_set(result.set2)

    def test_returns_none_when_impossible(self) -> None:
        # All different ranks, no wilds.
        hand = (
            Card(rank=Rank.THREE, suit=Suit.HEARTS),
            Card(rank=Rank.FOUR, suit=Suit.HEARTS),
            Card(rank=Rank.FIVE, suit=Suit.HEARTS),
            Card(rank=Rank.SIX, suit=Suit.HEARTS),
            Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
            Card(rank=Rank.EIGHT, suit=Suit.HEARTS),
        )
        assert find_any_go_down(hand) is None

    def test_returns_none_with_too_few_cards(self) -> None:
        assert find_any_go_down(()) is None

    def test_two_sets_of_same_rank(self) -> None:
        # 6 of the same rank (possible since 8 of each rank exist in 2 decks).
        hand = (
            Card(rank=Rank.KING, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.DIAMONDS),
            Card(rank=Rank.KING, suit=Suit.CLUBS),
            Card(rank=Rank.KING, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        result = find_any_go_down(hand)
        assert result is not None
        assert is_valid_set(result.set1) and is_valid_set(result.set2)
