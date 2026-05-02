"""Tests for cards.py — verifies wild detection, point values, and deck composition."""

from __future__ import annotations

from collections import Counter

import pytest

from rummy_v2.cards import Card, Rank, Suit, build_deck


class TestCardConstruction:
    def test_normal_card_requires_suit(self) -> None:
        with pytest.raises(ValueError):
            Card(rank=Rank.KING, suit=None)

    def test_joker_must_have_no_suit(self) -> None:
        with pytest.raises(ValueError):
            Card(rank=Rank.JOKER, suit=Suit.SPADES)

    def test_joker_with_no_suit_is_valid(self) -> None:
        Card(rank=Rank.JOKER, suit=None)

    def test_cards_with_same_rank_and_suit_are_equal(self) -> None:
        a = Card(rank=Rank.SEVEN, suit=Suit.HEARTS)
        b = Card(rank=Rank.SEVEN, suit=Suit.HEARTS)
        assert a == b
        assert hash(a) == hash(b)


class TestWildDetection:
    def test_joker_is_wild(self) -> None:
        assert Card(rank=Rank.JOKER, suit=None).is_wild

    @pytest.mark.parametrize("suit", [Suit.SPADES, Suit.CLUBS])
    def test_black_two_is_wild(self, suit: Suit) -> None:
        assert Card(rank=Rank.TWO, suit=suit).is_wild

    @pytest.mark.parametrize("suit", [Suit.HEARTS, Suit.DIAMONDS])
    def test_red_two_is_not_wild(self, suit: Suit) -> None:
        assert not Card(rank=Rank.TWO, suit=suit).is_wild

    @pytest.mark.parametrize(
        "rank",
        [Rank.THREE, Rank.SEVEN, Rank.JACK, Rank.KING, Rank.ACE],
    )
    def test_other_ranks_not_wild(self, rank: Rank) -> None:
        for suit in Suit:
            assert not Card(rank=rank, suit=suit).is_wild


class TestPointValues:
    @pytest.mark.parametrize(
        "rank,expected",
        [
            (Rank.THREE, 5),
            (Rank.FOUR, 5),
            (Rank.FIVE, 5),
            (Rank.SIX, 5),
            (Rank.SEVEN, 5),
            (Rank.EIGHT, 10),
            (Rank.NINE, 10),
            (Rank.TEN, 10),
            (Rank.JACK, 10),
            (Rank.QUEEN, 10),
            (Rank.KING, 10),
            (Rank.ACE, 20),
        ],
    )
    def test_non_two_point_values(self, rank: Rank, expected: int) -> None:
        for suit in Suit:
            assert Card(rank=rank, suit=suit).point_value == expected

    def test_red_two_is_five(self) -> None:
        for suit in (Suit.HEARTS, Suit.DIAMONDS):
            assert Card(rank=Rank.TWO, suit=suit).point_value == 5

    def test_black_two_is_twenty(self) -> None:
        for suit in (Suit.SPADES, Suit.CLUBS):
            assert Card(rank=Rank.TWO, suit=suit).point_value == 20

    def test_joker_is_twenty(self) -> None:
        assert Card(rank=Rank.JOKER, suit=None).point_value == 20


class TestDeckComposition:
    def test_total_card_count(self) -> None:
        assert len(build_deck()) == 108

    def test_four_jokers(self) -> None:
        deck = build_deck()
        jokers = [c for c in deck if c.rank is Rank.JOKER]
        assert len(jokers) == 4

    def test_eight_of_each_non_joker_rank(self) -> None:
        deck = build_deck()
        counts = Counter(c.rank for c in deck if c.rank is not Rank.JOKER)
        for rank in Rank:
            if rank is Rank.JOKER:
                continue
            assert counts[rank] == 8, f"expected 8 of {rank}, got {counts[rank]}"

    def test_two_of_each_non_joker_card(self) -> None:
        deck = build_deck()
        non_jokers = [c for c in deck if c.rank is not Rank.JOKER]
        counts = Counter(non_jokers)
        assert all(n == 2 for n in counts.values())
        assert len(counts) == 52

    def test_eight_wild_cards_total(self) -> None:
        deck = build_deck()
        wilds = [c for c in deck if c.is_wild]
        assert len(wilds) == 8

    def test_total_point_value_of_full_deck(self) -> None:
        # Sanity: 8 wilds * 20 + 8 aces * 20 + (3,4,5,6,7 = 5 ranks * 8 cards) * 5 + 2 red twos * 5
        # + (8,9,10,J,Q,K = 6 ranks * 8 cards) * 10
        # = 160 + 160 + 200 + 10 + 480 = 1010
        # Note: there are 4 wild "two" cards (black twos counted in wilds), and 4 non-wild twos (red).
        # Recompute: wilds = 4 jokers + 4 black 2s = 8 cards * 20 = 160
        #            aces = 8 * 20 = 160
        #            red twos = 4 * 5 = 20
        #            ranks 3-7 = 5 ranks * 8 cards * 5 = 200
        #            ranks 8-K = 6 ranks * 8 cards * 10 = 480
        # total = 160 + 160 + 20 + 200 + 480 = 1020
        assert sum(c.point_value for c in build_deck()) == 1020


class TestDisplay:
    def test_face_card_str(self) -> None:
        assert str(Card(rank=Rank.KING, suit=Suit.HEARTS)) == "KH"

    def test_ten_uses_t(self) -> None:
        assert str(Card(rank=Rank.TEN, suit=Suit.SPADES)) == "TS"

    def test_joker_str(self) -> None:
        assert str(Card(rank=Rank.JOKER, suit=None)) == "JK"

    def test_ace_str(self) -> None:
        assert str(Card(rank=Rank.ACE, suit=Suit.DIAMONDS)) == "AD"
