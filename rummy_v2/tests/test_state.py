"""Tests for state.py — verifies dataclass invariants and the initial dealer."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from rummy_v2.cards import Card, Rank, Suit, build_deck
from rummy_v2.state import (
    HAND_SIZE,
    NUM_PLAYERS,
    DownPile,
    GameState,
    Phase,
    PlayerState,
    initial_state,
)


def _seeded() -> random.Random:
    return random.Random(42)


class TestDownPile:
    def test_no_wilds(self) -> None:
        pile = DownPile(
            rank=Rank.KING,
            cards=(
                Card(rank=Rank.KING, suit=Suit.HEARTS),
                Card(rank=Rank.KING, suit=Suit.SPADES),
                Card(rank=Rank.KING, suit=Suit.DIAMONDS),
            ),
        )
        assert pile.num_wilds == 0

    def test_counts_jokers_and_black_twos_as_wild(self) -> None:
        pile = DownPile(
            rank=Rank.KING,
            cards=(
                Card(rank=Rank.KING, suit=Suit.HEARTS),
                Card(rank=Rank.JOKER, suit=None),
                Card(rank=Rank.TWO, suit=Suit.SPADES),
            ),
        )
        assert pile.num_wilds == 2


class TestPlayerState:
    def test_is_down_false_when_no_piles(self) -> None:
        p = PlayerState(hand=(), down_piles=(), buys_used=0)
        assert not p.is_down

    def test_is_down_true_when_piles_present(self) -> None:
        pile = DownPile(rank=Rank.SEVEN, cards=(Card(rank=Rank.SEVEN, suit=Suit.HEARTS),) * 3)
        p = PlayerState(hand=(), down_piles=(pile,), buys_used=0)
        assert p.is_down


class TestGameStateInvariants:
    def _good_kwargs(self) -> dict:
        empty = PlayerState(hand=(), down_piles=(), buys_used=0)
        return dict(
            players=tuple(empty for _ in range(NUM_PLAYERS)),
            draw_pile=(),
            discard_pile=(),
            active_player=0,
            phase=Phase.AWAITING_DRAW,
            buy_offer_to=0,
        )

    def test_rejects_wrong_player_count(self) -> None:
        kwargs = self._good_kwargs()
        empty = PlayerState(hand=(), down_piles=(), buys_used=0)
        kwargs["players"] = (empty,) * (NUM_PLAYERS - 1)
        with pytest.raises(ValueError):
            GameState(**kwargs)

    def test_rejects_active_player_out_of_range(self) -> None:
        kwargs = self._good_kwargs()
        kwargs["active_player"] = NUM_PLAYERS
        with pytest.raises(ValueError):
            GameState(**kwargs)

    def test_rejects_buy_offer_to_out_of_range(self) -> None:
        kwargs = self._good_kwargs()
        kwargs["buy_offer_to"] = -1
        with pytest.raises(ValueError):
            GameState(**kwargs)

    def test_rejects_partial_round_result(self) -> None:
        kwargs = self._good_kwargs()
        kwargs["round_winner"] = 0
        with pytest.raises(ValueError):
            GameState(**kwargs)

    def test_top_discard_none_when_empty(self) -> None:
        s = GameState(**self._good_kwargs())
        assert s.top_discard is None

    def test_top_discard_returns_last(self) -> None:
        kwargs = self._good_kwargs()
        last = Card(rank=Rank.KING, suit=Suit.HEARTS)
        kwargs["discard_pile"] = (
            Card(rank=Rank.THREE, suit=Suit.CLUBS),
            last,
        )
        s = GameState(**kwargs)
        assert s.top_discard == last


class TestInitialState:
    def test_six_hands_of_eleven(self) -> None:
        s = initial_state(_seeded())
        assert len(s.players) == NUM_PLAYERS
        for p in s.players:
            assert len(p.hand) == HAND_SIZE
            assert p.down_piles == ()
            assert p.buys_used == 0

    def test_one_card_in_discard(self) -> None:
        s = initial_state(_seeded())
        assert len(s.discard_pile) == 1

    def test_draw_pile_has_remaining_cards(self) -> None:
        s = initial_state(_seeded())
        # 108 - 6*11 dealt - 1 flipped = 41
        assert len(s.draw_pile) == 41

    def test_all_108_cards_accounted_for(self) -> None:
        s = initial_state(_seeded())
        all_cards = list(s.draw_pile) + list(s.discard_pile)
        for p in s.players:
            all_cards.extend(p.hand)
        assert len(all_cards) == 108
        # Compare multiset against a fresh deck.
        assert Counter(all_cards) == Counter(build_deck())

    def test_starting_phase_is_awaiting_draw(self) -> None:
        s = initial_state(_seeded())
        assert s.phase is Phase.AWAITING_DRAW

    def test_active_player_after_dealer(self) -> None:
        s = initial_state(_seeded(), dealer=2)
        assert s.active_player == 3
        assert s.buy_offer_to == 3

    def test_dealer_wraps(self) -> None:
        s = initial_state(_seeded(), dealer=NUM_PLAYERS - 1)
        assert s.active_player == 0

    def test_rejects_bad_dealer(self) -> None:
        with pytest.raises(ValueError):
            initial_state(_seeded(), dealer=NUM_PLAYERS)

    def test_seed_reproducibility(self) -> None:
        a = initial_state(random.Random(7))
        b = initial_state(random.Random(7))
        assert a == b

    def test_different_seeds_diverge(self) -> None:
        a = initial_state(random.Random(1))
        b = initial_state(random.Random(2))
        assert a != b
