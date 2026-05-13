"""Eval harness — pit policies against each other and tally win rates.

Usage (from repo root):

    python -m rummy_v2.eval --n-games 200
    python -m rummy_v2.eval --n-games 500 --policies progress,progress,progress,progress,progress,uniform
    python -m rummy_v2.eval --n-games 200 --policies greedy,progress,progress,progress,progress,progress

A policy is a callable `f(env, rng) -> flat_action_index`. The 6 seats are
filled by the 6 policies given in `--policies`. Seat assignments rotate
round-robin across games to remove first-mover bias (player to dealer's left
goes first), so the reported win rates reflect the policies, not the seats
they happened to occupy.

Available built-in policies:

  uniform   — uniform over the legal-action mask. The trivial floor; tests
              the env's correctness more than any kind of skill.
  progress  — `random_play.random_action`: takes discard / buys only when
              useful, lays off greedily, otherwise random.
  greedy    — `progress` with one improvement: when forced to discard,
              throws away the highest-point card in hand.

Symmetry sanity check: with all 6 seats running the same policy, each seat
should win ~16.7% by symmetry. Significant deviation signals a bug.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from rummy_v2.actions import Buy, Discard, LayOff, is_legal, legal_actions
from rummy_v2.env import (
    ACTION_BUY,
    ACTION_DRAW_DECK,
    ACTION_DRAW_DISCARD,
    ACTION_PASS_BUY,
    SpeedRummyEnv,
    card_to_rank_idx,
    engine_action_to_flat,
    hand_potential,
)
from rummy_v2.random_play import random_action as _progress_random_action
from rummy_v2.state import NUM_PLAYERS, Phase


Policy = Callable[[SpeedRummyEnv, random.Random], int]


# --- Built-in policies -------------------------------------------------------


def uniform_policy(env: SpeedRummyEnv, rng: random.Random) -> int:
    mask = env.action_mask(env.agent_selection)
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def progress_policy(env: SpeedRummyEnv, rng: random.Random) -> int:
    action = _progress_random_action(env._state, rng)
    return engine_action_to_flat(env._state, action)


def greedy_policy(env: SpeedRummyEnv, rng: random.Random) -> int:
    """Same as `progress` but discards the highest-point card when discarding.

    Lay-offs and draw/buy decisions are unchanged — this only affects the
    *which card* tiebreaker on the discard step.
    """
    state = env._state
    action = _progress_random_action(state, rng)
    if isinstance(action, Discard):
        actor = state.players[state.active_player]
        best = max(actor.hand, key=lambda c: c.point_value)
        action = Discard(card=best)
    return engine_action_to_flat(state, action)


def structured_policy(env: SpeedRummyEnv, rng: random.Random) -> int:
    """Φ-aware heuristic that reasons about own hand structure.

    Improvements over `progress`/`greedy`:
      - Take discard top iff Φ(hand ∪ {top}) > Φ(hand)  — i.e. the top
        actively helps build a near-set, not just whether it matches a pile.
      - Buy iff ΔΦ from the discard top alone ≥ 1.0 — saves buy slots for
        cards that materially advance a near-set (turns 2 → 3, or 1+wild → 3).
      - Discard the rank whose removal causes the smallest Φ drop, with
        highest-point-value tiebreak (dump high cards when structurally
        equivalent — same logic as `greedy` but only after the Φ tiebreak).
      - Lay-off greedily (per-action cumulative; engine handles repeats).

    This is intentionally blind to opponents and the discard pile history:
    those signals are left for PPO to learn on top of a strong structural
    floor.
    """
    state = env._state

    if state.phase is Phase.AWAITING_DRAW:
        actor = state.players[state.active_player]
        top = state.top_discard
        if top is not None:
            phi_now = hand_potential(actor)
            phi_with = hand_potential(replace(actor, hand=actor.hand + (top,)))
            if phi_with > phi_now:
                return ACTION_DRAW_DISCARD
        return ACTION_DRAW_DECK

    if state.phase is Phase.BUY_CASCADE:
        if not is_legal(state, Buy()):
            return ACTION_PASS_BUY
        buyer = state.players[state.buy_offer_to]
        top = state.top_discard
        if top is None:
            return ACTION_PASS_BUY
        phi_now = hand_potential(buyer)
        phi_with = hand_potential(replace(buyer, hand=buyer.hand + (top,)))
        if phi_with - phi_now >= 1.0:
            return ACTION_BUY
        return ACTION_PASS_BUY

    # POST_DRAW: lay off greedily, else pick best discard.
    actor = state.players[state.active_player]
    legal = legal_actions(state)

    lay_offs = [a for a in legal if isinstance(a, LayOff)]
    if lay_offs:
        return engine_action_to_flat(state, rng.choice(lay_offs))

    best_phi = -1.0
    best_pts = -1
    best_action: Discard | None = None
    seen_buckets: set[int] = set()
    for c in actor.hand:
        bucket = card_to_rank_idx(c)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        new_hand = list(actor.hand)
        new_hand.remove(c)
        phi_after = hand_potential(replace(actor, hand=tuple(new_hand)))
        pv = c.point_value
        if phi_after > best_phi or (phi_after == best_phi and pv > best_pts):
            best_phi = phi_after
            best_pts = pv
            best_action = Discard(card=c)

    assert best_action is not None
    return engine_action_to_flat(state, best_action)


POLICIES: dict[str, Policy] = {
    "uniform": uniform_policy,
    "progress": progress_policy,
    "greedy": greedy_policy,
    "structured": structured_policy,
}


# --- Eval runner -------------------------------------------------------------


@dataclass
class EvalResult:
    n_games: int
    policy_names: list[str]
    win_counts: list[int]
    score_sums: list[float]
    score_counts: list[int]
    total_steps: int
    not_terminated: int

    @property
    def win_rates(self) -> list[float]:
        denom = max(1, self.n_games - self.not_terminated)
        return [w / denom for w in self.win_counts]

    @property
    def avg_scores(self) -> list[float]:
        """Per-policy average hand-score *across all games* (winners count as
        score=0). This is the headline number for the multi-round game: lowest
        cumulative score wins. Lower is better."""
        return [
            s / c if c else 0.0
            for s, c in zip(self.score_sums, self.score_counts)
        ]

    @property
    def avg_loser_scores(self) -> list[float]:
        """Per-policy average hand-score conditioned on losing the round.
        Reports how badly a policy loses *when* it loses (not whether it
        wins). Lower is better."""
        return [
            (s_total / max(1, c_total - w))
            if c_total > w
            else 0.0
            for s_total, c_total, w in zip(
                self.score_sums, self.score_counts, self.win_counts
            )
        ]

    @property
    def avg_steps(self) -> float:
        denom = max(1, self.n_games - self.not_terminated)
        return self.total_steps / denom


def run_eval(
    policies: list[Policy],
    policy_names: list[str],
    n_games: int,
    seed: int = 0,
    max_steps: int = 5000,
    expose_opponent_pickup: bool = True,
) -> EvalResult:
    """Play `n_games` rounds with the given seat policies.

    Seat-policy assignments rotate every game (game `g` puts policy `p` in
    seat `(p + g) % NUM_PLAYERS`), so all policies see every seat equally.
    """
    n = len(policies)
    if n != NUM_PLAYERS:
        raise ValueError(f"need {NUM_PLAYERS} policies, got {n}")

    win_counts = [0] * n
    score_sums = [0.0] * n
    score_counts = [0] * n
    total_steps = 0
    not_terminated = 0

    for game_idx in range(n_games):
        rotation = game_idx % NUM_PLAYERS
        env = SpeedRummyEnv(
            seed=seed + game_idx,
            expose_opponent_pickup=expose_opponent_pickup,
        )
        rng = random.Random(seed * 9973 + game_idx + 1)

        steps = 0
        while not all(env.terminations.values()) and steps < max_steps:
            seat = int(env.agent_selection.split("_")[1])
            policy_idx = (seat - rotation) % NUM_PLAYERS
            action = policies[policy_idx](env, rng)
            env.step(action)
            steps += 1

        if env._state.phase is not Phase.ROUND_OVER:
            not_terminated += 1
            continue

        total_steps += steps
        winner_seat = env._state.round_winner
        assert winner_seat is not None
        winning_policy_idx = (winner_seat - rotation) % NUM_PLAYERS
        win_counts[winning_policy_idx] += 1

        scores = env._state.round_scores
        assert scores is not None
        for seat in range(NUM_PLAYERS):
            policy_idx = (seat - rotation) % NUM_PLAYERS
            score_sums[policy_idx] += float(scores[seat])
            score_counts[policy_idx] += 1

    return EvalResult(
        n_games=n_games,
        policy_names=policy_names,
        win_counts=win_counts,
        score_sums=score_sums,
        score_counts=score_counts,
        total_steps=total_steps,
        not_terminated=not_terminated,
    )


def format_result(result: EvalResult) -> str:
    completed = result.n_games - result.not_terminated
    lines = [
        f"6-way Speed Rummy eval - {completed}/{result.n_games} games completed",
        "",
        "Headline metric is *avg-score* (lower is better). The game's",
        "actual objective is minimum cumulative score across all 6 rounds.",
        "Wins are useful (score=0 for that round) but not the only good",
        "outcome.",
        "",
        f"  seat-pol  {'policy':<10} {'wins':>5} {'win-rate':>9}"
        f" {'avg-score':>10} {'when-lost':>10}",
    ]
    for i, (name, wins, rate, avg, lost_avg) in enumerate(
        zip(
            result.policy_names,
            result.win_counts,
            result.win_rates,
            result.avg_scores,
            result.avg_loser_scores,
        )
    ):
        lines.append(
            f"  P{i}        {name:<10} {wins:>5} {rate*100:>8.1f}%"
            f" {avg:>10.1f} {lost_avg:>10.1f}"
        )
    lines.append("")
    lines.append(f"avg game length: {result.avg_steps:.1f} steps")
    if result.not_terminated:
        lines.append(
            f"  warning: {result.not_terminated} games did not terminate within max_steps"
        )
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------


def _parse_policies(spec: str) -> tuple[list[str], list[Policy]]:
    names = [s.strip() for s in spec.split(",")]
    if len(names) != NUM_PLAYERS:
        raise ValueError(
            f"need {NUM_PLAYERS} comma-separated policy names, got {len(names)}"
        )
    for n in names:
        if n not in POLICIES:
            raise ValueError(
                f"unknown policy {n!r}; available: {sorted(POLICIES)}"
            )
    return names, [POLICIES[n] for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval harness for Speed Rummy policies"
    )
    parser.add_argument(
        "--policies",
        default=",".join(["progress"] * NUM_PLAYERS),
        help=(
            f"Comma-separated policies, one per seat ({NUM_PLAYERS} total). "
            f"Available: {sorted(POLICIES)}. "
            "Default: 6x progress (the symmetric baseline)."
        ),
    )
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    names, policies = _parse_policies(args.policies)
    result = run_eval(
        policies, names, args.n_games, seed=args.seed, max_steps=args.max_steps
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
