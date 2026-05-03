"""Tests for `rummy_v2.eval`.

Two flavors:
  - Symmetry: 6 identical policies should split wins ~uniformly. With N=120
    games, each seat expects 20 wins; we accept anything in [5, 35] (very
    loose, this is a sanity check, not a precise test).
  - Asymmetry: a uniform-random policy at one seat among 5 progress players
    should be measurably worse — its win rate should be well below the 1/6
    null.
"""

from __future__ import annotations

import pytest

from rummy_v2.eval import (
    POLICIES,
    EvalResult,
    greedy_policy,
    progress_policy,
    run_eval,
    uniform_policy,
)
from rummy_v2.state import NUM_PLAYERS


def _all_progress() -> tuple[list[str], list]:
    names = ["progress"] * NUM_PLAYERS
    policies = [POLICIES[n] for n in names]
    return names, policies


def test_runs_at_all() -> None:
    names, policies = _all_progress()
    result = run_eval(policies, names, n_games=10, seed=0)
    assert result.n_games == 10
    assert sum(result.win_counts) + result.not_terminated == 10


def test_symmetry_six_progress() -> None:
    """All six progress policies should have roughly equal win rates."""
    names, policies = _all_progress()
    result = run_eval(policies, names, n_games=120, seed=0)
    # Loose bounds: each seat should win at least 5, at most 35 (expected 20).
    for i, w in enumerate(result.win_counts):
        assert 5 <= w <= 35, (
            f"seat {i} ({names[i]}) won {w}/120 — outside loose symmetry range "
            f"[5, 35]; possible position bias"
        )


def test_uniform_loses_to_progress() -> None:
    """A single uniform-random seat among 5 progress players should win
    significantly less than 1/6. (Uniform discards greedy lay-off
    opportunities and takes the discard pile when it shouldn't.)"""
    names = ["uniform"] + ["progress"] * (NUM_PLAYERS - 1)
    policies = [POLICIES[n] for n in names]
    result = run_eval(policies, names, n_games=120, seed=42)
    completed = result.n_games - result.not_terminated
    if completed < 60:
        pytest.skip("too many non-terminations to evaluate")
    uniform_win_rate = result.win_counts[0] / completed
    # Expect uniform to win much less than 1/6 (~16.7%).
    assert uniform_win_rate < 0.10, (
        f"uniform won {uniform_win_rate:.1%} vs 5 progress — expected < 10%"
    )


def test_eval_result_properties() -> None:
    names, policies = _all_progress()
    result = run_eval(policies, names, n_games=20, seed=0)
    completed = result.n_games - result.not_terminated
    if completed == 0:
        pytest.skip("no completed games")
    assert sum(result.win_counts) == completed
    # Win rates should sum to 1 (over completed games).
    assert sum(result.win_rates) == pytest.approx(1.0)
    # Avg scores: at minimum the winner has 0; non-winners have positive scores.
    assert result.avg_steps > 0


def test_greedy_at_least_competitive_with_progress() -> None:
    """A greedy seat among 5 progress should win at least the symmetric 1/6
    minus reasonable noise. Greedy is not a strictly stronger policy (the
    discard heuristic can backfire), but it should not be markedly worse."""
    names = ["greedy"] + ["progress"] * (NUM_PLAYERS - 1)
    policies = [POLICIES[n] for n in names]
    result = run_eval(policies, names, n_games=120, seed=7)
    completed = result.n_games - result.not_terminated
    if completed < 60:
        pytest.skip("too many non-terminations to evaluate")
    greedy_win_rate = result.win_counts[0] / completed
    # Loose bound: greedy should win at least 8% (vs symmetric 16.7%).
    # If greedy were *much* worse than progress, we'd want to know.
    assert greedy_win_rate >= 0.08, (
        f"greedy won {greedy_win_rate:.1%} vs 5 progress — surprisingly weak"
    )
