"""Tests for engine.py — full coverage of the turn machine."""

from __future__ import annotations

import random

import pytest

from rummy_v2.actions import (
    Buy,
    Discard,
    DrawDeck,
    DrawDiscard,
    GoDown,
    LayOff,
    PassBuy,
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
from rummy_v2.engine import apply_action


# --- helpers ------------------------------------------------------------------


def _c(rank: Rank, suit: Suit | None = Suit.HEARTS) -> Card:
    if rank is Rank.JOKER:
        return Card(rank=Rank.JOKER, suit=None)
    return Card(rank=rank, suit=suit)


def _kings() -> tuple[Card, Card, Card]:
    return (
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.KING, suit=Suit.SPADES),
        Card(rank=Rank.KING, suit=Suit.DIAMONDS),
    )


def _sevens() -> tuple[Card, Card, Card]:
    return (
        Card(rank=Rank.SEVEN, suit=Suit.HEARTS),
        Card(rank=Rank.SEVEN, suit=Suit.SPADES),
        Card(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
    )


def _empty_player(buys_used: int = 0) -> PlayerState:
    return PlayerState(hand=(), down_piles=(), buys_used=buys_used)


def _build(
    *,
    phase: Phase,
    active: int = 0,
    buy_offer_to: int | None = None,
    actor_hand: tuple[Card, ...] = (),
    actor_down: tuple[DownPile, ...] = (),
    actor_buys: int = 0,
    others_buys: tuple[int, ...] = (0, 0, 0, 0, 0),
    others_down: dict[int, tuple[DownPile, ...]] | None = None,
    discard: tuple[Card, ...] = (),
    draw: tuple[Card, ...] = (),
) -> GameState:
    """Construct a minimal GameState. Active is at `active`; the other 5
    players are listed in `others_buys` in the seat order *skipping* active.

    others_down maps absolute player index → down piles.
    """
    others_down = others_down or {}
    actor = PlayerState(
        hand=actor_hand, down_piles=actor_down, buys_used=actor_buys
    )

    other_iter = iter(others_buys)
    players: list[PlayerState] = []
    for i in range(NUM_PLAYERS):
        if i == active:
            players.append(actor)
        else:
            buys = next(other_iter)
            piles = others_down.get(i, ())
            players.append(
                PlayerState(hand=(), down_piles=piles, buys_used=buys)
            )

    return GameState(
        players=tuple(players),
        draw_pile=draw,
        discard_pile=discard,
        active_player=active,
        phase=phase,
        buy_offer_to=active if buy_offer_to is None else buy_offer_to,
    )


def _rng() -> random.Random:
    return random.Random(0)


# --- DrawDiscard --------------------------------------------------------------


class TestDrawDiscard:
    def test_top_lands_in_hand(self) -> None:
        top = _c(Rank.KING)
        s = _build(
            phase=Phase.AWAITING_DRAW,
            actor_hand=(_c(Rank.SEVEN),),
            discard=(_c(Rank.THREE), top),
        )
        s2 = apply_action(s, DrawDiscard(), _rng())
        assert top in s2.players[0].hand
        assert len(s2.players[0].hand) == 2

    def test_discard_pile_shrinks(self) -> None:
        s = _build(
            phase=Phase.AWAITING_DRAW,
            discard=(_c(Rank.THREE), _c(Rank.KING)),
        )
        s2 = apply_action(s, DrawDiscard(), _rng())
        assert s2.discard_pile == (_c(Rank.THREE),)

    def test_phase_becomes_post_draw(self) -> None:
        s = _build(phase=Phase.AWAITING_DRAW, discard=(_c(Rank.KING),))
        s2 = apply_action(s, DrawDiscard(), _rng())
        assert s2.phase is Phase.POST_DRAW

    def test_illegal_when_discard_empty(self) -> None:
        s = _build(phase=Phase.AWAITING_DRAW)
        with pytest.raises(ValueError):
            apply_action(s, DrawDiscard(), _rng())


# --- DrawDeck -----------------------------------------------------------------


class TestDrawDeck:
    def test_no_cascade_when_discard_empty(self) -> None:
        s = _build(phase=Phase.AWAITING_DRAW, draw=(_c(Rank.KING),))
        s2 = apply_action(s, DrawDeck(), _rng())
        assert s2.phase is Phase.POST_DRAW
        assert len(s2.players[0].hand) == 1
        assert s2.draw_pile == ()

    def test_cascade_entered_when_discard_present_and_buyers_eligible(
        self,
    ) -> None:
        s = _build(
            phase=Phase.AWAITING_DRAW,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.THREE),),
        )
        s2 = apply_action(s, DrawDeck(), _rng())
        assert s2.phase is Phase.BUY_CASCADE
        # Active is 0; first eligible buyer is 1.
        assert s2.buy_offer_to == 1

    def test_no_cascade_when_all_others_at_limit(self) -> None:
        s = _build(
            phase=Phase.AWAITING_DRAW,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.THREE),),
            others_buys=(MAX_BUYS_PER_ROUND,) * 5,
        )
        s2 = apply_action(s, DrawDeck(), _rng())
        assert s2.phase is Phase.POST_DRAW

    def test_skips_to_first_eligible_buyer(self) -> None:
        # players 1, 2 maxed; player 3 still eligible.
        s = _build(
            phase=Phase.AWAITING_DRAW,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.THREE),),
            others_buys=(MAX_BUYS_PER_ROUND, MAX_BUYS_PER_ROUND, 0, 0, 0),
        )
        s2 = apply_action(s, DrawDeck(), _rng())
        assert s2.phase is Phase.BUY_CASCADE
        assert s2.buy_offer_to == 3


# --- Buy ----------------------------------------------------------------------


class TestBuy:
    def test_top_and_penalty_go_to_hand(self) -> None:
        top = _c(Rank.KING, Suit.SPADES)
        penalty = _c(Rank.QUEEN, Suit.HEARTS)
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.THREE), top),
            draw=(_c(Rank.JACK), penalty),
        )
        s2 = apply_action(s, Buy(), _rng())
        assert top in s2.players[1].hand
        assert penalty in s2.players[1].hand

    def test_buys_used_increments(self) -> None:
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.THREE), _c(Rank.KING)),
            draw=(_c(Rank.JACK), _c(Rank.QUEEN)),
        )
        s2 = apply_action(s, Buy(), _rng())
        assert s2.players[1].buys_used == 1

    def test_stays_on_same_buyer_when_under_limit(self) -> None:
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.THREE), _c(Rank.KING)),
            draw=(_c(Rank.QUEEN),),
        )
        s2 = apply_action(s, Buy(), _rng())
        assert s2.phase is Phase.BUY_CASCADE
        assert s2.buy_offer_to == 1

    def test_advances_when_buyer_hits_limit(self) -> None:
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.THREE), _c(Rank.KING)),
            draw=(_c(Rank.QUEEN),),
            others_buys=(MAX_BUYS_PER_ROUND - 1, 0, 0, 0, 0),
        )
        s2 = apply_action(s, Buy(), _rng())
        # Buyer 1 hit limit; cascade advances to player 2.
        assert s2.phase is Phase.BUY_CASCADE
        assert s2.buy_offer_to == 2

    def test_ends_cascade_when_discard_emptied(self) -> None:
        # Only one card on discard. After buy, discard empty → cascade ends.
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.QUEEN), _c(Rank.JACK)),
        )
        s2 = apply_action(s, Buy(), _rng())
        # Discard now empty → active force-draws → POST_DRAW
        assert s2.phase is Phase.POST_DRAW
        # Active gets the deck draw
        assert len(s2.players[0].hand) == 1


# --- PassBuy ------------------------------------------------------------------


class TestPassBuy:
    def test_advances_to_next_eligible(self) -> None:
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.QUEEN),),
        )
        s2 = apply_action(s, PassBuy(), _rng())
        assert s2.phase is Phase.BUY_CASCADE
        assert s2.buy_offer_to == 2

    def test_skips_maxed_players(self) -> None:
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=1,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.QUEEN),),
            others_buys=(0, MAX_BUYS_PER_ROUND, MAX_BUYS_PER_ROUND, 0, 0),
        )
        s2 = apply_action(s, PassBuy(), _rng())
        # Skipping players 2 and 3 (indexes 2,3 → others_buys[1],others_buys[2]).
        assert s2.phase is Phase.BUY_CASCADE
        assert s2.buy_offer_to == 4

    def test_ends_cascade_after_last_eligible(self) -> None:
        # Active=0; only player 5 is in the cascade and they pass.
        s = _build(
            phase=Phase.BUY_CASCADE,
            active=0,
            buy_offer_to=5,
            discard=(_c(Rank.KING),),
            draw=(_c(Rank.QUEEN),),
        )
        s2 = apply_action(s, PassBuy(), _rng())
        # Cascade ends; active draws from deck → POST_DRAW
        assert s2.phase is Phase.POST_DRAW
        assert len(s2.players[0].hand) == 1


# --- GoDown -------------------------------------------------------------------


class TestGoDown:
    def test_creates_two_down_piles(self) -> None:
        hand = _kings() + _sevens() + (_c(Rank.JACK),)
        s = _build(phase=Phase.POST_DRAW, actor_hand=hand)
        s2 = apply_action(s, GoDown(set1=_kings(), set2=_sevens()), _rng())
        assert len(s2.players[0].down_piles) == 2
        assert s2.players[0].down_piles[0].rank is Rank.KING
        assert s2.players[0].down_piles[1].rank is Rank.SEVEN

    def test_removes_six_cards_from_hand(self) -> None:
        hand = _kings() + _sevens() + (_c(Rank.JACK),)
        s = _build(phase=Phase.POST_DRAW, actor_hand=hand)
        s2 = apply_action(s, GoDown(set1=_kings(), set2=_sevens()), _rng())
        assert s2.players[0].hand == (_c(Rank.JACK),)

    def test_phase_unchanged(self) -> None:
        hand = _kings() + _sevens() + (_c(Rank.JACK),)
        s = _build(phase=Phase.POST_DRAW, actor_hand=hand)
        s2 = apply_action(s, GoDown(set1=_kings(), set2=_sevens()), _rng())
        assert s2.phase is Phase.POST_DRAW

    def test_set_using_wild(self) -> None:
        joker = _c(Rank.JOKER)
        s1 = (_c(Rank.KING), _c(Rank.KING, Suit.SPADES), joker)
        s2_cards = _sevens()
        hand = s1 + s2_cards + (_c(Rank.JACK),)
        s = _build(phase=Phase.POST_DRAW, actor_hand=hand)
        result = apply_action(s, GoDown(set1=s1, set2=s2_cards), _rng())
        # The kings pile has rank=KING and 1 wild
        kings_pile = result.players[0].down_piles[0]
        assert kings_pile.rank is Rank.KING
        assert kings_pile.num_wilds == 1


# --- LayOff -------------------------------------------------------------------


class TestLayOff:
    def test_lay_off_on_other_player(self) -> None:
        target_pile = DownPile(rank=Rank.KING, cards=_kings())
        my_kings_set = _kings()
        my_sevens_set = _sevens()
        extra = _c(Rank.KING, Suit.CLUBS)
        keeper = _c(Rank.THREE, Suit.HEARTS)  # second card so lay-off is legal
        s = _build(
            phase=Phase.POST_DRAW,
            actor_hand=(extra, keeper),
            actor_down=(
                DownPile(rank=Rank.KING, cards=my_kings_set),
                DownPile(rank=Rank.SEVEN, cards=my_sevens_set),
            ),
            others_down={2: (target_pile,)},
        )
        s2 = apply_action(
            s, LayOff(card=extra, target_player=2, pile_index=0), _rng()
        )
        assert extra not in s2.players[0].hand
        assert s2.players[2].down_piles[0].cards[-1] == extra

    def test_lay_off_on_own_pile(self) -> None:
        my_kings = _kings()
        my_sevens = _sevens()
        extra = _c(Rank.KING, Suit.CLUBS)
        keeper = _c(Rank.THREE, Suit.HEARTS)
        s = _build(
            phase=Phase.POST_DRAW,
            actor_hand=(extra, keeper),
            actor_down=(
                DownPile(rank=Rank.KING, cards=my_kings),
                DownPile(rank=Rank.SEVEN, cards=my_sevens),
            ),
        )
        s2 = apply_action(
            s, LayOff(card=extra, target_player=0, pile_index=0), _rng()
        )
        assert extra not in s2.players[0].hand
        assert s2.players[0].down_piles[0].cards[-1] == extra
        assert s2.players[0].down_piles[1] == DownPile(
            rank=Rank.SEVEN, cards=my_sevens
        )

    def test_lay_off_wild_increments_wild_count(self) -> None:
        my_kings = _kings()
        my_sevens = _sevens()
        joker = _c(Rank.JOKER)
        keeper = _c(Rank.THREE, Suit.HEARTS)
        s = _build(
            phase=Phase.POST_DRAW,
            actor_hand=(joker, keeper),
            actor_down=(
                DownPile(rank=Rank.KING, cards=my_kings),
                DownPile(rank=Rank.SEVEN, cards=my_sevens),
            ),
        )
        s2 = apply_action(
            s, LayOff(card=joker, target_player=0, pile_index=0), _rng()
        )
        assert s2.players[0].down_piles[0].num_wilds == 1


# --- Discard ------------------------------------------------------------------


class TestDiscard:
    def test_advances_to_next_player(self) -> None:
        card = _c(Rank.KING)
        s = _build(phase=Phase.POST_DRAW, actor_hand=(card, _c(Rank.SEVEN)))
        s2 = apply_action(s, Discard(card=card), _rng())
        assert s2.active_player == 1
        assert s2.phase is Phase.AWAITING_DRAW
        assert s2.buy_offer_to == 1
        assert s2.discard_pile[-1] == card

    def test_card_leaves_hand(self) -> None:
        card = _c(Rank.KING)
        s = _build(phase=Phase.POST_DRAW, actor_hand=(card, _c(Rank.SEVEN)))
        s2 = apply_action(s, Discard(card=card), _rng())
        assert s2.players[0].hand == (_c(Rank.SEVEN),)

    def test_round_over_when_hand_empties(self) -> None:
        last = _c(Rank.JACK)
        s = _build(
            phase=Phase.POST_DRAW,
            actor_hand=(last,),
            actor_down=(
                DownPile(rank=Rank.KING, cards=_kings()),
                DownPile(rank=Rank.SEVEN, cards=_sevens()),
            ),
        )
        s2 = apply_action(s, Discard(card=last), _rng())
        assert s2.phase is Phase.ROUND_OVER
        assert s2.round_winner == 0
        assert s2.round_scores is not None
        assert s2.round_scores[0] == 0  # winner

    def test_round_scoring_sums_hand_point_values(self) -> None:
        # Build a state where:
        # - active (0) is about to discard their last card and go out
        # - player 1 holds: KH (10), 7H (5), JK (20) → 35
        # - others empty
        last = _c(Rank.JACK)
        kh = Card(rank=Rank.KING, suit=Suit.HEARTS)
        seven = Card(rank=Rank.SEVEN, suit=Suit.HEARTS)
        joker = Card(rank=Rank.JOKER, suit=None)

        actor = PlayerState(
            hand=(last,),
            down_piles=(
                DownPile(rank=Rank.KING, cards=_kings()),
                DownPile(rank=Rank.SEVEN, cards=_sevens()),
            ),
            buys_used=0,
        )
        p1 = PlayerState(hand=(kh, seven, joker), down_piles=(), buys_used=0)
        others = [PlayerState(hand=(), down_piles=(), buys_used=0)] * 4
        players = (actor, p1, *others)

        s = GameState(
            players=players,
            draw_pile=(),
            discard_pile=(),
            active_player=0,
            phase=Phase.POST_DRAW,
            buy_offer_to=0,
        )
        s2 = apply_action(s, Discard(card=last), _rng())
        assert s2.round_scores == (0, 35, 0, 0, 0, 0)


# --- Deck reshuffle on exhaustion ---------------------------------------------


class TestDeckReshuffle:
    def test_reshuffles_when_drawing_from_empty_deck(self) -> None:
        s = _build(
            phase=Phase.AWAITING_DRAW,
            draw=(),
            discard=(_c(Rank.THREE), _c(Rank.SEVEN), _c(Rank.KING)),
        )
        s2 = apply_action(s, DrawDeck(), _rng())
        # All non-active players have 0 buys → cascade entered. Walk through it.
        # With 3 cards in discard before this DrawDeck, 1 ends up in active hand
        # eventually. But cascade may pre-empt; for this test we want the actual
        # reshuffle path. Use a state where buyers are maxed so cascade is skipped.
        s3 = _build(
            phase=Phase.AWAITING_DRAW,
            draw=(),
            discard=(_c(Rank.THREE), _c(Rank.SEVEN), _c(Rank.KING)),
            others_buys=(MAX_BUYS_PER_ROUND,) * 5,
        )
        s4 = apply_action(s3, DrawDeck(), _rng())
        # All 3 cards reshuffled to draw, top of new draw goes to active hand.
        # Net: hand has 1 card, draw has 2, discard is empty.
        assert s4.phase is Phase.POST_DRAW
        assert len(s4.players[0].hand) == 1
        assert len(s4.draw_pile) == 2
        assert s4.discard_pile == ()


# --- Illegal actions ----------------------------------------------------------


class TestIllegalActions:
    def test_raises_value_error(self) -> None:
        s = _build(phase=Phase.AWAITING_DRAW)
        with pytest.raises(ValueError):
            apply_action(s, Buy(), _rng())
