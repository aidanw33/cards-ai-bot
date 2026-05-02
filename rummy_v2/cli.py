"""Text-mode hot-seat driver for Speed Rummy round 1.

Run as `python -m rummy_v2.cli` (optionally with `--seed N`). The CLI prints
the perspective of whichever player must act next and reads a numeric menu
choice; for actions that need parameters (cards, target piles) it prompts.

This is a play-and-debug tool, not part of the RL training surface.
"""

from __future__ import annotations

import argparse
import random
import sys

from rummy_v2.actions import (
    Action,
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
)
from rummy_v2.cards import Card, Rank, Suit
from rummy_v2.engine import apply_action
from rummy_v2.state import (
    NUM_PLAYERS,
    DownPile,
    GameState,
    Phase,
    initial_state,
)


# --- Card parsing -------------------------------------------------------------


_RANK_BY_CHAR: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}
_SUIT_BY_CHAR: dict[str, Suit] = {
    "S": Suit.SPADES,
    "H": Suit.HEARTS,
    "D": Suit.DIAMONDS,
    "C": Suit.CLUBS,
}


def parse_card(s: str) -> Card | None:
    s = s.strip().upper()
    if s in ("JK", "JOKER"):
        return Card(rank=Rank.JOKER, suit=None)
    if len(s) != 2:
        return None
    rank_char, suit_char = s[0], s[1]
    rank = _RANK_BY_CHAR.get(rank_char)
    suit = _SUIT_BY_CHAR.get(suit_char)
    if rank is None or suit is None:
        return None
    return Card(rank=rank, suit=suit)


# --- Rendering ----------------------------------------------------------------


def _format_hand(hand: tuple[Card, ...]) -> str:
    if not hand:
        return "(empty)"
    return " ".join(str(c) for c in sorted(hand, key=_card_sort_key))


def _card_sort_key(c: Card) -> tuple[int, str]:
    return (c.rank.value, c.suit.value if c.suit else "")


def _format_down_pile(pile: DownPile) -> str:
    return f"[{pile.rank.name[0]}={' '.join(str(c) for c in pile.cards)}]"


def _format_player_summary(state: GameState, idx: int, focus: bool) -> str:
    p = state.players[idx]
    marker = "*" if focus else " "
    down_str = (
        " ".join(_format_down_pile(dp) for dp in p.down_piles)
        if p.down_piles
        else "(none)"
    )
    return (
        f"  {marker} P{idx}: hand={len(p.hand):2d} buys={p.buys_used} "
        f"down={down_str}"
    )


def render(state: GameState, viewer: int) -> str:
    """Render the state from `viewer`'s perspective."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Phase: {state.phase.name}    Active: P{state.active_player}")

    for i in range(NUM_PLAYERS):
        lines.append(_format_player_summary(state, i, focus=(i == viewer)))

    top = state.top_discard
    lines.append(
        f"  Discard top: {top}    pile size: {len(state.discard_pile)}    "
        f"draw size: {len(state.draw_pile)}"
    )

    you = state.players[viewer]
    lines.append(f"  YOU (P{viewer}) hand: {_format_hand(you.hand)}")
    if state.phase is Phase.BUY_CASCADE:
        lines.append(f"  Cascade offer: top discard -> P{state.buy_offer_to}")
    lines.append("=" * 72)
    return "\n".join(lines)


# --- Action prompts -----------------------------------------------------------


def _read_choice(prompt: str, n: int) -> int:
    while True:
        raw = input(f"{prompt} [1-{n}, q to quit] > ").strip().lower()
        if raw == "q":
            sys.exit(0)
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= n:
                return choice
        print(f"  invalid; enter 1-{n} or 'q'")


def _read_card_from(hand: tuple[Card, ...], prompt: str) -> Card:
    while True:
        raw = input(f"{prompt} > ").strip()
        card = parse_card(raw)
        if card is None:
            print("  unrecognized card (e.g. KH, 7S, TC, JK)")
            continue
        if card not in hand:
            print(f"  {card} is not in your hand")
            continue
        return card


def _prompt_awaiting_draw(state: GameState) -> Action:
    options: list[tuple[str, Action]] = [("Draw from deck", DrawDeck())]
    if state.top_discard is not None:
        options.append((f"Take {state.top_discard} from discard", DrawDiscard()))

    print()
    print(f"P{state.active_player} - choose draw source:")
    for i, (label, _) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    choice = _read_choice("Select", len(options))
    return options[choice - 1][1]


def _prompt_buy_cascade(state: GameState) -> Action:
    options: list[tuple[str, Action]] = [("Pass", PassBuy())]
    if is_legal(state, Buy()):
        options.append(
            (
                f"Buy {state.top_discard} (+1 penalty card)",
                Buy(),
            )
        )

    print()
    p = state.players[state.buy_offer_to]
    print(
        f"P{state.buy_offer_to} (buys used {p.buys_used}/3) - "
        f"top of discard: {state.top_discard}"
    )
    for i, (label, _) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    choice = _read_choice("Select", len(options))
    return options[choice - 1][1]


def _prompt_post_draw(state: GameState) -> Action:
    actor = state.players[state.active_player]
    options: list[tuple[str, Action | str]] = []

    if not actor.is_down:
        suggested = find_any_go_down(actor.hand)
        if suggested is not None:
            options.append(
                (
                    f"Go down (auto: {' '.join(str(c) for c in suggested.set1)} "
                    f"+ {' '.join(str(c) for c in suggested.set2)})",
                    suggested,
                )
            )
        options.append(("Go down (manual: pick 6 cards)", "MANUAL_GODOWN"))

    if actor.is_down:
        options.append(("Lay off a card on a down pile", "LAYOFF"))

    options.append(("Discard a card", "DISCARD"))

    print()
    print(f"P{state.active_player} - choose action:")
    for i, (label, _) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    choice = _read_choice("Select", len(options))
    selection = options[choice - 1][1]

    if isinstance(selection, str):
        if selection == "MANUAL_GODOWN":
            return _prompt_manual_godown(state)
        if selection == "LAYOFF":
            return _prompt_layoff(state)
        if selection == "DISCARD":
            return _prompt_discard(state)
    return selection  # type: ignore[return-value]


def _prompt_manual_godown(state: GameState) -> Action:
    actor = state.players[state.active_player]
    while True:
        print("Enter set 1 - three cards from your hand, space-separated:")
        s1 = _read_three_cards(actor.hand)
        print("Enter set 2:")
        # Remove set 1's cards from the available pool when validating set 2.
        remaining = list(actor.hand)
        for c in s1:
            remaining.remove(c)
        s2 = _read_three_cards(tuple(remaining))
        if not is_valid_set(s1):
            print("  set 1 isn't a valid meld; try again")
            continue
        if not is_valid_set(s2):
            print("  set 2 isn't a valid meld; try again")
            continue
        return GoDown(set1=s1, set2=s2)


def _read_three_cards(
    available: tuple[Card, ...],
) -> tuple[Card, Card, Card]:
    while True:
        raw = input("  cards (e.g. 'KH KS JK') > ").strip()
        parts = raw.split()
        if len(parts) != 3:
            print("  need exactly three cards")
            continue
        parsed: list[Card] = []
        bad = False
        for p in parts:
            card = parse_card(p)
            if card is None:
                print(f"  could not parse '{p}'")
                bad = True
                break
            parsed.append(card)
        if bad:
            continue
        # Verify all three are present in `available` (multiset check).
        avail = list(available)
        ok = True
        for c in parsed:
            if c in avail:
                avail.remove(c)
            else:
                print(f"  {c} not available")
                ok = False
                break
        if not ok:
            continue
        return (parsed[0], parsed[1], parsed[2])


def _prompt_layoff(state: GameState) -> Action:
    actor = state.players[state.active_player]
    targets: list[tuple[int, int, DownPile]] = []
    for tp_idx, p in enumerate(state.players):
        for pi, pile in enumerate(p.down_piles):
            targets.append((tp_idx, pi, pile))

    print("Available down piles:")
    for i, (tp, pi, pile) in enumerate(targets, 1):
        print(f"  [{i}] P{tp} pile #{pi}: {_format_down_pile(pile)}")
    pile_choice = _read_choice("Choose target pile", len(targets))
    tp, pi, _pile = targets[pile_choice - 1]

    card = _read_card_from(actor.hand, "Card to lay off")
    return LayOff(card=card, target_player=tp, pile_index=pi)


def _prompt_discard(state: GameState) -> Action:
    actor = state.players[state.active_player]
    card = _read_card_from(actor.hand, "Card to discard")
    return Discard(card=card)


# --- Main loop ----------------------------------------------------------------


def play(seed: int) -> None:
    rng = random.Random(seed)
    state = initial_state(rng)

    while state.phase is not Phase.ROUND_OVER:
        viewer = state.active_player
        if state.phase is Phase.BUY_CASCADE:
            viewer = state.buy_offer_to
        print(render(state, viewer))

        if state.phase is Phase.AWAITING_DRAW:
            action: Action = _prompt_awaiting_draw(state)
        elif state.phase is Phase.BUY_CASCADE:
            action = _prompt_buy_cascade(state)
        elif state.phase is Phase.POST_DRAW:
            action = _prompt_post_draw(state)
        else:
            raise AssertionError(f"unexpected phase: {state.phase}")

        try:
            state = apply_action(state, action, rng)
        except ValueError as e:
            print(f"  illegal: {e}")

    print()
    print("=" * 72)
    print(f"Round over. Winner: P{state.round_winner}")
    assert state.round_scores is not None
    for i, score in enumerate(state.round_scores):
        marker = " (winner)" if i == state.round_winner else ""
        print(f"  P{i}: {score}{marker}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed Rummy hot-seat CLI")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    play(args.seed)


if __name__ == "__main__":
    main()
