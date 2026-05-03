"""Live training monitor.

Reads `runs/<name>/metrics.jsonl` (line-buffered, written by `train.py`) and
re-renders a matplotlib dashboard every few seconds. Run this in a *separate*
terminal alongside the training process — the training process keeps writing,
this process keeps re-reading.

Usage:
    python -m rummy_v2.monitor runs/run6_long
    python -m rummy_v2.monitor runs/run6_long --refresh 2.0

Close the window or Ctrl+C to exit.

The four panels:
  - Win rate vs progress  : the headline metric. Symmetric baseline at 16.7%.
  - Mean episode return   : per-agent return averaged across the rollout.
  - Policy loss & entropy : PPO clip loss (red, left axis) and policy entropy
                            (green, right axis). Entropy collapse without
                            improvement = exploration ran out.
  - Rollout trunc rate    : fraction of episodes force-ended at
                            `max_episode_steps`. Should be near 0 against a
                            progress opponent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    bc_records: list[dict] = []
    ppo_records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tail-end partial write
            if r.get("phase") == "bc":
                bc_records.append(r)
            elif "update" in r:
                ppo_records.append(r)
    return bc_records, ppo_records


def _redraw(fig, axes, bc, ppo) -> None:
    ax_score, ax_win, ax_loss, ax_ret = axes.flatten()
    for ax in axes.flatten():
        ax.clear()

    if not ppo:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, "waiting for first update...", ha="center", va="center",
                    transform=ax.transAxes)
        return

    steps = [r["steps_done"] for r in ppo]
    bc_win = (bc[0]["bc_post_win_rate"] * 100) if bc and "bc_post_win_rate" in bc[0] else None
    bc_score = bc[0].get("bc_post_avg_score") if bc else None
    bc_prog_score = bc[0].get("bc_post_progress_avg_score") if bc else None
    eval_records = [r for r in ppo if "neural_win_rate" in r]

    # 1. AVG SCORE (headline metric — lower is better).
    score_records = [r for r in ppo if "neural_avg_score" in r]
    if score_records:
        score_steps = [r["steps_done"] for r in score_records]
        neural_scores = [r["neural_avg_score"] for r in score_records]
        prog_scores = [r["progress_avg_score"] for r in score_records]
        ax_score.plot(score_steps, neural_scores, "b-o", markersize=4, label="neural")
        ax_score.plot(score_steps, prog_scores, "orange", linestyle="--", linewidth=1,
                      label="progress (in same eval)")
    if bc_score is not None:
        ax_score.axhline(y=bc_score, color="green", linestyle=":", linewidth=1,
                         label=f"post-BC neural ({bc_score:.1f})")
    if bc_prog_score is not None:
        ax_score.axhline(y=bc_prog_score, color="gray", linestyle=":", linewidth=1,
                         label=f"post-BC progress ({bc_prog_score:.1f})")
    ax_score.set_xlabel("trainee transitions")
    ax_score.set_ylabel("avg score (lower is better)")
    ax_score.set_title("Avg score vs progress (HEADLINE)")
    ax_score.legend(loc="best", fontsize=8)
    ax_score.grid(True, alpha=0.3)
    ax_score.invert_yaxis()  # lower = up = better

    # 2. WIN RATE (secondary).
    if eval_records:
        eval_steps = [r["steps_done"] for r in eval_records]
        wins = [r["neural_win_rate"] * 100 for r in eval_records]
        ax_win.plot(eval_steps, wins, "b-o", markersize=4)
    ax_win.axhline(y=16.7, color="gray", linestyle="--", linewidth=1,
                   label="baseline (16.7%)")
    if bc_win is not None:
        ax_win.axhline(y=bc_win, color="green", linestyle=":", linewidth=1,
                       label=f"post-BC ({bc_win:.1f}%)")
    ax_win.set_xlabel("trainee transitions")
    ax_win.set_ylabel("win rate (%)")
    ax_win.set_title("Win rate vs progress (secondary)")
    ax_win.legend(loc="lower right", fontsize=8)
    ax_win.grid(True, alpha=0.3)

    # 3. Policy loss + entropy (twin axes).
    pi_loss = [r["policy_loss"] for r in ppo]
    entropy = [r["entropy"] for r in ppo]
    ax_loss.plot(steps, pi_loss, "r-", label="policy loss")
    ax_loss.set_xlabel("trainee transitions")
    ax_loss.set_ylabel("policy loss", color="r")
    ax_loss.tick_params(axis="y", labelcolor="r")
    ax_loss.set_title("Policy loss (red) & entropy (green)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss2 = ax_loss.twinx()
    ax_loss2.plot(steps, entropy, "g-")
    ax_loss2.set_ylabel("entropy", color="g")
    ax_loss2.tick_params(axis="y", labelcolor="g")

    # 4. Episode return (informational — doesn't directly correspond to game score).
    returns = [r["mean_episode_return"] for r in ppo]
    ax_ret.plot(steps, returns, "b-")
    ax_ret.set_xlabel("trainee transitions")
    ax_ret.set_ylabel("mean episode return")
    ax_ret.set_title("Mean episode return (informational)")
    ax_ret.grid(True, alpha=0.3)

    # Header status text.
    latest = ppo[-1]
    status_parts = [
        f"update {latest['update']}",
        f"steps {int(latest['steps_done']):,}",
        f"{latest.get('steps_per_sec', 0):.0f}/s",
    ]
    if score_records:
        status_parts.append(f"latest score: {score_records[-1]['neural_avg_score']:.1f}")
    if eval_records:
        status_parts.append(f"win: {eval_records[-1]['neural_win_rate']*100:.1f}%")
    fig.suptitle(" | ".join(status_parts), fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=str, help="path to runs/<name>/")
    parser.add_argument("--refresh", type=float, default=2.0,
                        help="redraw period in seconds (default: 2.0)")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    metrics_path = run_path / "metrics.jsonl"
    print(f"[monitor] watching {metrics_path}")
    print(f"[monitor] refresh every {args.refresh}s — close window or Ctrl+C to exit")

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    while plt.fignum_exists(fig.number):
        bc, ppo = _read_metrics(metrics_path)
        _redraw(fig, axes, bc, ppo)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        try:
            plt.pause(args.refresh)
        except Exception:
            break


if __name__ == "__main__":
    main()
