"""Immutable game-state primitives for Speed Rummy round 1.

State is fully frozen and hashable. The engine derives new states from old
ones; nothing here mutates. Conventions:

- Hands and piles are tuples; the *top* of a pile is the **last** element.
- `players` is indexed 0..NUM_PLAYERS-1. Turn order is index-ascending modulo
  NUM_PLAYERS.
- Random draws / shuffles are driven by an RNG passed to the engine, not stored
  in state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from rummy_v2.cards import Card, Rank, build_deck


NUM_PLAYERS: int = 6
HAND_SIZE: int = 11
SETS_TO_GO_DOWN: int = 2
SET_SIZE: int = 3
MAX_BUYS_PER_ROUND: int = 3
MAX_WILDS_PER_MELD: int = 2


class Phase(Enum):
    """Where in the turn machine we are.

    - AWAITING_DRAW: active player must choose draw source (deck or discard top).
      A choice of deck with a non-empty discard pile transitions to BUY_CASCADE
      first; otherwise straight to POST_DRAW.
    - BUY_CASCADE: top-of-discard is being offered to a non-active player. They
      either BUY (taking it + a penalty card) or PASS. Cascade walks turn order
      and ends when all non-active players have passed or the discard empties.
    - POST_DRAW: active player has the new card; may go down (once per round),
      lay off cards onto any down pile, then must DISCARD to end their turn.
    - ROUND_OVER: a player has gone out; `round_scores` is populated.
    """

    AWAITING_DRAW = "awaiting_draw"
    BUY_CASCADE = "buy_cascade"
    POST_DRAW = "post_draw"
    ROUND_OVER = "round_over"


@dataclass(frozen=True, slots=True)
class DownPile:
    """A face-up meld in front of a player.

    `rank` is the rank wilds substitute for (every non-wild card in the pile
    must match it). `cards` records every card placed in order added; this lets
    us enforce the max-2-wilds rule and reconstruct play history.
    """

    rank: Rank
    cards: tuple[Card, ...]

    @property
    def num_wilds(self) -> int:
        return sum(1 for c in self.cards if c.is_wild)


@dataclass(frozen=True, slots=True)
class PlayerState:
    hand: tuple[Card, ...]
    down_piles: tuple[DownPile, ...]
    buys_used: int

    @property
    def is_down(self) -> bool:
        return len(self.down_piles) > 0


@dataclass(frozen=True, slots=True)
class GameState:
    players: tuple[PlayerState, ...]
    draw_pile: tuple[Card, ...]
    discard_pile: tuple[Card, ...]

    active_player: int
    phase: Phase

    buy_offer_to: int

    round_winner: int | None = None
    round_scores: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.players) != NUM_PLAYERS:
            raise ValueError(
                f"expected {NUM_PLAYERS} players, got {len(self.players)}"
            )
        if not 0 <= self.active_player < NUM_PLAYERS:
            raise ValueError(f"active_player out of range: {self.active_player}")
        if not 0 <= self.buy_offer_to < NUM_PLAYERS:
            raise ValueError(f"buy_offer_to out of range: {self.buy_offer_to}")
        if (self.round_scores is None) != (self.round_winner is None):
            raise ValueError("round_scores and round_winner must be set together")
        if self.round_scores is not None and len(self.round_scores) != NUM_PLAYERS:
            raise ValueError("round_scores must have one entry per player")

    @property
    def top_discard(self) -> Card | None:
        return self.discard_pile[-1] if self.discard_pile else None

    @property
    def is_round_over(self) -> bool:
        return self.phase is Phase.ROUND_OVER


def initial_state(rng: random.Random, dealer: int = NUM_PLAYERS - 1) -> GameState:
    """Deal a fresh round-1 state.

    Player at (dealer + 1) % NUM_PLAYERS becomes the active player. Per RULES.md
    §3, the flipped starter card sits on the discard pile and is treated as if
    it were just discarded — the active player gets first option to take it,
    and the buy cascade can offer it to the rest.
    """
    if not 0 <= dealer < NUM_PLAYERS:
        raise ValueError(f"dealer out of range: {dealer}")

    deck = build_deck()
    rng.shuffle(deck)

    hands = tuple(
        tuple(deck[i * HAND_SIZE : (i + 1) * HAND_SIZE]) for i in range(NUM_PLAYERS)
    )
    after_deal = NUM_PLAYERS * HAND_SIZE
    discard_top = deck[after_deal]
    draw_pile = tuple(deck[after_deal + 1 :])

    players = tuple(
        PlayerState(hand=hands[i], down_piles=(), buys_used=0)
        for i in range(NUM_PLAYERS)
    )
    active = (dealer + 1) % NUM_PLAYERS

    return GameState(
        players=players,
        draw_pile=draw_pile,
        discard_pile=(discard_top,),
        active_player=active,
        phase=Phase.AWAITING_DRAW,
        buy_offer_to=active,
    )
