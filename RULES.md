# Rules — Speed Rummy (Round 1)

This document specifies the rules for Round 1 of the family card game the bot is being trained to play. It is the source of truth — the implementation in `rummy/` must conform to this document, not the other way around. Future rounds (with different meld goals) are out of scope for now.

## 1. Equipment

- **Deck**: Two standard 52-card decks combined, plus all 4 jokers = **108 cards total**.
- **Wild cards**: The 4 jokers and all 4 black twos (2♠, 2♣ from each deck = 4 cards) are **wild**. There are therefore **8 wild cards** in the deck.
  - A wild card may stand in for any rank and suit.
  - **Maximum 2 wilds per set** (and per run, in later rounds).

## 2. Players

- Standard game: **6 players**.
- The bot is being trained against this 6-player setting. Earlier 2-player simplifications are deprecated.

## 3. Setup

- Shuffle the 108-card deck.
- Deal **11 cards** to each player.
- Place the remaining cards face-down as the **draw pile**.
- Flip the top card of the draw pile face-up to start the **discard pile**.
- Play proceeds clockwise (or in fixed seat order). Player to the dealer's left goes first.
- **First turn note**: the flipped starter card is treated as if it had just been discarded. The first player may take it as their normal draw; if they decline, the buy cascade (§8) offers it to the remaining players before the first player completes their turn.

## 4. Round 1 Goal

- **Go down** with exactly **2 sets of 3** (six cards from your hand laid face-up in front of you).
- A **set** = 3 cards of the same rank, suit irrelevant. Wilds count toward the rank, max 2 per set.
- You may not go down with more than 2 sets in a single action; the third set must be played as lay-offs after going down.

## 5. Turn Structure

A turn proceeds in this fixed order:

1. **Buy phase** (from previous discard) — see §8. Resolves before the active player draws.
2. **Draw**: take the top card of the discard pile (visible) **or** the top card of the draw pile (hidden).
3. **Optionally go down**: if your hand contains 2 valid sets, you may lay them face-up in front of you. Going down is **optional** — you may choose not to even when able.
4. **Optionally lay off** (only if already down, including the same turn you went down) — see §6.
5. **Discard**: place exactly one card on top of the discard pile. This ends your turn.
   - You **must** discard every turn, including the turn you go out. The final card is always discarded — even if it could have been laid off, you simply discard it instead (functionally equivalent for going out).

## 6. Laying Off

- Only available **after you have gone down** (you may lay off on the same turn you went down).
- A lay-off is placing one or more cards from your hand onto **any** down pile (your own or another player's) where the rank matches.
- Cards are added one at a time; lay-offs do not need to come in triplets.
- Wilds may be laid off as long as the destination pile does not already contain 2 wilds. (Wild lay-off behavior matters more in later rounds with runs; in Round 1 it is uncommon but legal.)
- **You cannot lay off your final card.** Per §7, every turn ends with a discard, so you must keep at least one card in hand when laying off. (A lay-off action is only legal when your hand has 2 or more cards.) The path to going out is: lay off down to one card, then discard it.

## 7. Going Out

- A player goes out when they have **0 cards in hand** at the end of their turn.
- Going out always happens via the discard step. Even if the final card could be laid off onto a down pile, the player simply discards it — there is no functional difference for the purpose of going out.
- A player can only realistically go out after they have gone down, since lay-offs (which shrink hand size beyond the +1/−1 net of draw+discard) require being down.
- As soon as a player goes out, the round ends immediately.

## 8. Buying

When a card is discarded, it becomes available to be **bought** by other players before the next turn begins.

- The player whose turn is next (call them N+1) does **not** buy. If they want the discarded card, they simply take it as their normal draw on their upcoming turn (no penalty).
- Players N+2, N+3, … (in turn order) may **buy** the top of the discard pile.
- A buy costs:
  - 1 **penalty card** drawn from the top of the draw pile (added to the buyer's hand).
  - The bought card itself (added to the buyer's hand).
  - Net: hand grows by 2 cards per buy.
- **Buy limit**: each player may buy a maximum of **3 times per round**.
- **Multiple buys allowed per discard event**:
  - A single player may take multiple cards (top, then new top, then new top), each with its own penalty card.
  - After one player takes (or passes), the next eligible player may also buy from what remains.
- Buy resolution order: starting at N+2 and proceeding in turn order; each eligible player chooses how many (0+) to buy from the current top, subject to their per-round limit, before the offer moves on. Once all players have had the chance, the N+1 player begins their normal turn.
- **Maximum hand size**: 11 + (3 buys × 2 cards each) = **17 cards** at the start of a round.

## 9. Draw Pile Exhaustion

- If the draw pile runs out mid-round, shuffle the **entire discard pile** (including the current top card) into a new face-down draw pile.
- The discard pile is now **empty** until the next player completes a turn and discards.
- During this brief window, the buy phase has nothing to act on, and players drawing must take from the (newly reshuffled) draw pile.

## 10. Scoring

When a player goes out, every other player tallies the point value of cards remaining in their hand:

| Card                | Points |
|---------------------|--------|
| 2–7 (non-wild)      | 5      |
| 8–10, J, Q, K       | 10     |
| Ace                 | 20     |
| Joker               | 20     |
| Black 2 (wild)      | 20     |

- Cards already in down piles (your own or others') are worth 0.
- Each player adds their hand total to their cumulative score. The player who went out scores 0 for the round.
- **Lowest cumulative score wins** at the end of the game (game-end condition is out of scope for Round 1 modeling).

## 11. Notes

- **Dealer rotation across rounds**: The starting player rotates each round. Out of scope for Round 1 modeling.
- **Future rounds**: Later rounds introduce runs (sequential same-suit cards) and different meld goals. Wild card behavior on lay-offs becomes more visible in those rounds. Not in scope yet.
