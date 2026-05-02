"""Card primitives for Speed Rummy.

A 108-card shoe: two standard 52-card decks plus 4 jokers (2 per deck).
Wild cards are jokers and the 4 black 2s (2 of spades and 2 of clubs from each deck).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"


class Rank(Enum):
    JOKER = 0
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


_BLACK_SUITS: frozenset[Suit] = frozenset({Suit.SPADES, Suit.CLUBS})

_RANK_DISPLAY: dict[Rank, str] = {
    Rank.JOKER: "JK",
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit | None

    def __post_init__(self) -> None:
        if self.rank is Rank.JOKER and self.suit is not None:
            raise ValueError("jokers must have suit=None")
        if self.rank is not Rank.JOKER and self.suit is None:
            raise ValueError("non-jokers must have a suit")

    @property
    def is_wild(self) -> bool:
        if self.rank is Rank.JOKER:
            return True
        if self.rank is Rank.TWO and self.suit in _BLACK_SUITS:
            return True
        return False

    @property
    def point_value(self) -> int:
        if self.is_wild:
            return 20
        if self.rank is Rank.ACE:
            return 20
        if self.rank.value <= 7:
            return 5
        return 10

    def __str__(self) -> str:
        rank_str = _RANK_DISPLAY[self.rank]
        if self.suit is None:
            return rank_str
        return f"{rank_str}{self.suit.value}"


def build_deck() -> list[Card]:
    """Return the 108-card shoe in canonical (unshuffled) order."""
    cards: list[Card] = []
    for _ in range(2):
        for suit in Suit:
            for rank in Rank:
                if rank is Rank.JOKER:
                    continue
                cards.append(Card(rank=rank, suit=suit))
        cards.append(Card(rank=Rank.JOKER, suit=None))
        cards.append(Card(rank=Rank.JOKER, suit=None))
    return cards
