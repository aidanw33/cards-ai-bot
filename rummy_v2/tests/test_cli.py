"""Smoke tests for cli.py — import surface + parse_card + render shape."""

from __future__ import annotations

import random

from rummy_v2.cards import Card, Rank, Suit
from rummy_v2.cli import parse_card, render
from rummy_v2.state import initial_state


class TestParseCard:
    def test_face_card(self) -> None:
        assert parse_card("KH") == Card(rank=Rank.KING, suit=Suit.HEARTS)

    def test_lowercase(self) -> None:
        assert parse_card("kh") == Card(rank=Rank.KING, suit=Suit.HEARTS)

    def test_ten_uses_t(self) -> None:
        assert parse_card("TS") == Card(rank=Rank.TEN, suit=Suit.SPADES)

    def test_joker(self) -> None:
        assert parse_card("JK") == Card(rank=Rank.JOKER, suit=None)
        assert parse_card("joker") == Card(rank=Rank.JOKER, suit=None)

    def test_returns_none_on_garbage(self) -> None:
        assert parse_card("") is None
        assert parse_card("XY") is None
        assert parse_card("KKH") is None
        assert parse_card("3Z") is None


class TestRender:
    def test_renders_initial_state(self) -> None:
        s = initial_state(random.Random(0))
        out = render(s, viewer=s.active_player)
        # Just verify nothing blows up and the output mentions key bits.
        assert "Phase: AWAITING_DRAW" in out
        assert f"Active: P{s.active_player}" in out
        assert "Discard top:" in out
        for i in range(6):
            assert f"P{i}:" in out
