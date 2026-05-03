"""PPO self-play trainer for Speed Rummy.

Architecture:
- One policy (`PolicyNet`) plays all 6 seats — parameter-shared self-play.
- Rollout collector walks the AEC env, recording one transition per agent per
  decision. Each agent's reward is the cumulative reward earned between their
  consecutive actions (matches PettingZoo's AEC reward semantics).
- GAE-Lambda advantages are computed per agent stream, then flattened.
- Standard PPO clip update with policy / value / entropy losses, masked logits.
- Periodic evaluation against `progress` baselines via `rummy_v2.eval`.

Run output goes to `runs/<run-name>/`:
    config.json          hyperparameters and meta
    metrics.jsonl        one JSON line per PPO update
    checkpoints/*.pt     periodic model snapshots
    final.pt             final model

Usage:
    python -m rummy_v2.train --total-steps 200000 --eval-every 5
    python -m rummy_v2.train --total-steps 1000000 --rollout-steps 8192

Quick smoke run:
    python -m rummy_v2.train --total-steps 8000 --rollout-steps 1024 \\
        --eval-every 2 --eval-games 30
"""

from __future__ import annotations

import argparse
import json
import random as pyrandom
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rummy_v2.env import (
    ACTION_SPACE_SIZE,
    OBS_DIM,
    REWARD_TERMINAL_HAND_POINT,
    SpeedRummyEnv,
)
from rummy_v2.eval import POLICIES, run_eval
from rummy_v2.policy import PolicyNet, make_neural_policy
from rummy_v2.state import NUM_PLAYERS, Phase


# --- Config ------------------------------------------------------------------


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    rollout_steps: int = 4096
    minibatch_size: int = 256
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 0.5
    hidden: int = 256
    seed: int = 0
    eval_every: int = 5
    eval_games: int = 50
    checkpoint_every: int = 25
    run_name: str | None = None
    runs_root: str = "runs"
    # If an episode runs longer than this many env steps without anybody going
    # out, force-terminate it. Each player's terminal reward is computed from
    # their hand points (the standard loser penalty) — there is no winner.
    # Without this cap, a policy that collapses to "never lay off" produces
    # zero terminal transitions and PPO never sees the +10/-h.p. signal.
    max_episode_steps: int = 800
    # Opponent policy. "self" means parameter-shared self-play (trainee plays
    # all 6 seats). Any name from `rummy_v2.eval.POLICIES` makes 5 fixed
    # opponents play that policy while the trainee plays one rotating seat.
    # Pure self-play is unstable from random init; vs-progress gives a much
    # cleaner training signal until the trainee can beat the baseline.
    opponent: str = "progress"
    # Behavior-cloning warm start: before PPO, train the network via
    # supervised cross-entropy on (obs, mask, action) triples sampled from
    # `bc_teacher`-vs-`bc_teacher` play. This skips the cold-start phase where
    # PPO struggles to find the +10 win signal among ~17% wins.
    bc_pretrain: bool = True
    bc_teacher: str = "progress"
    bc_dataset_size: int = 20_000
    bc_epochs: int = 3
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    # Whether to expose the opponent-pickup obs feature. If False, the obs
    # section is forced to zeros (network shape unchanged, info ablated).
    expose_opponent_pickup: bool = True


# --- Rollout collection ------------------------------------------------------


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    agent_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)


def collect_rollout(
    env: SpeedRummyEnv,
    policy: PolicyNet,
    n_steps: int,
    device: str = "cpu",
    max_episode_steps: int = 800,
    opponent_policy_fn=None,
    rng: pyrandom.Random | None = None,
) -> tuple[RolloutBuffer, dict[str, float]]:
    """Collect at least `n_steps` closed transitions for the trainee.

    If `opponent_policy_fn` is None, runs parameter-shared self-play (trainee
    plays all 6 seats; every step is a recorded transition). Otherwise runs
    asymmetric play: the trainee plays one rotating seat per episode, while
    the other 5 seats are driven by `opponent_policy_fn(env, rng) -> int`.
    Only trainee transitions are recorded.

    Each trainee transition is closed when (a) the trainee is selected to act
    again, with reward = `_cumulative_rewards[trainee_agent]` since its
    previous action, or (b) the episode ends. Pending transitions at rollout
    end are dropped (≤ 6 per rollout).
    """
    buf = RolloutBuffer()
    pending: dict[str, dict] = {}
    n_episodes = 0
    n_truncations = 0
    episode_returns: dict[str, list[float]] = defaultdict(list)
    running_episode_returns: dict[str, float] = defaultdict(float)
    episode_env_steps = 0
    if rng is None:
        rng = pyrandom.Random(0)
    self_play = opponent_policy_fn is None
    trainee_seat = 0  # rotates per episode below; only used in opponent mode

    def _close(agent: str, reward: float, done: bool, next_obs=None, next_mask=None):
        rec = pending.pop(agent, None)
        if rec is None:
            return
        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                no_t = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
                nm_t = torch.from_numpy(next_mask).bool().unsqueeze(0).to(device)
                _, nv = policy.forward(no_t, nm_t)
                next_value = float(nv.item())
        buf.obs.append(rec["obs"])
        buf.masks.append(rec["mask"])
        buf.actions.append(rec["action"])
        buf.log_probs.append(rec["log_prob"])
        buf.values.append(rec["value"])
        buf.rewards.append(reward)
        buf.dones.append(done)
        buf.next_values.append(next_value)
        buf.agent_ids.append(agent)

    def _end_episode(natural: bool) -> None:
        nonlocal n_episodes, n_truncations, episode_env_steps, trainee_seat
        if not natural:
            for seat in range(NUM_PLAYERS):
                hand_points = sum(
                    c.point_value for c in env._state.players[seat].hand
                )
                env._cumulative_rewards[f"player_{seat}"] += (
                    REWARD_TERMINAL_HAND_POINT * float(hand_points)
                )
        for agent in list(pending.keys()):
            final_reward = float(env._cumulative_rewards[agent])
            running_episode_returns[agent] += final_reward
            _close(agent, final_reward, done=True)
        for agent, total in running_episode_returns.items():
            episode_returns[agent].append(total)
        running_episode_returns.clear()
        n_episodes += 1
        if not natural:
            n_truncations += 1
        env.reset()
        episode_env_steps = 0
        if not self_play:
            trainee_seat = (trainee_seat + 1) % NUM_PLAYERS

    while len(buf) < n_steps:
        if all(env.terminations.values()):
            _end_episode(natural=True)
            continue

        if episode_env_steps >= max_episode_steps:
            _end_episode(natural=False)
            continue

        agent = env.agent_selection
        seat = int(agent.split("_")[1])
        is_trainee = self_play or (seat == trainee_seat)

        if is_trainee:
            obs = env.observe(agent)
            mask = env.action_mask(agent)
            if agent in pending:
                reward = float(env._cumulative_rewards[agent])
                running_episode_returns[agent] += reward
                _close(agent, reward, done=False, next_obs=obs, next_mask=mask)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
                action, log_prob, _entropy, value = policy.get_action_and_value(
                    obs_t, mask_t
                )
            a = int(action.item())
            pending[agent] = {
                "obs": obs.copy(),
                "mask": mask.copy(),
                "action": a,
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            }
        else:
            assert opponent_policy_fn is not None
            a = opponent_policy_fn(env, rng)

        env.step(a)
        episode_env_steps += 1

    all_returns = [r for rs in episode_returns.values() for r in rs]
    stats = {
        "n_episodes": float(n_episodes),
        "n_truncations": float(n_truncations),
        "trunc_rate": float(n_truncations / max(n_episodes, 1)),
        "mean_episode_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "rollout_size": float(len(buf)),
    }
    return buf, stats


# --- GAE ---------------------------------------------------------------------


def compute_gae(
    buf: RolloutBuffer, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-agent GAE-Lambda. Returns (advantages, returns) aligned to `buf`."""
    advantages = np.zeros(len(buf), dtype=np.float32)
    returns = np.zeros(len(buf), dtype=np.float32)
    by_agent: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(buf.agent_ids):
        by_agent[a].append(i)

    for indices in by_agent.values():
        last_gae = 0.0
        # Indices are in chronological collection order for this agent.
        for j in reversed(range(len(indices))):
            i = indices[j]
            r = buf.rewards[i]
            v = buf.values[i]
            nv = buf.next_values[i]
            d = 1.0 if buf.dones[i] else 0.0
            delta = r + gamma * nv * (1.0 - d) - v
            last_gae = delta + gamma * lam * (1.0 - d) * last_gae
            advantages[i] = last_gae
            returns[i] = advantages[i] + v
    return advantages, returns


# --- PPO update --------------------------------------------------------------


def ppo_update(
    policy: PolicyNet,
    optimizer: optim.Optimizer,
    buf: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: TrainConfig,
    device: str = "cpu",
) -> dict[str, float]:
    n = len(buf)
    obs = torch.from_numpy(np.stack(buf.obs)).float().to(device)
    masks = torch.from_numpy(np.stack(buf.masks)).bool().to(device)
    actions = torch.tensor(buf.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buf.log_probs, dtype=torch.float32, device=device)
    advs = torch.from_numpy(advantages).to(device)
    rets = torch.from_numpy(returns).to(device)

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    losses_p: list[float] = []
    losses_v: list[float] = []
    losses_e: list[float] = []
    clip_fracs: list[float] = []

    indices = np.arange(n)
    for _epoch in range(cfg.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]
            mb_t = torch.from_numpy(mb).long().to(device)
            _, log_prob, entropy, value = policy.get_action_and_value(
                obs[mb_t], masks[mb_t], action=actions[mb_t]
            )
            ratio = torch.exp(log_prob - old_log_probs[mb_t])
            surr1 = ratio * advs[mb_t]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advs[mb_t]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, rets[mb_t])
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            losses_p.append(float(policy_loss.item()))
            losses_v.append(float(value_loss.item()))
            losses_e.append(float(entropy.mean().item()))
            with torch.no_grad():
                clip_fracs.append(
                    float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())
                )

    return {
        "policy_loss": float(np.mean(losses_p)),
        "value_loss": float(np.mean(losses_v)),
        "entropy": float(np.mean(losses_e)),
        "clip_frac": float(np.mean(clip_fracs)),
    }


# --- Behavior cloning warm start ---------------------------------------------


def collect_bc_dataset(
    teacher_name: str,
    n_transitions: int,
    seed: int = 0,
    expose_opponent_pickup: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out `teacher_name` vs itself across all 6 seats, recording
    (obs, mask, action) triples for every decision until `n_transitions` are
    captured. The teacher is chosen from `rummy_v2.eval.POLICIES`.
    """
    if teacher_name not in POLICIES:
        raise ValueError(
            f"unknown bc_teacher={teacher_name!r}; valid: {sorted(POLICIES)}"
        )
    teacher = POLICIES[teacher_name]

    env = SpeedRummyEnv(seed=seed, expose_opponent_pickup=expose_opponent_pickup)
    rng = pyrandom.Random(seed * 7349 + 11)

    obs_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    act_buf: list[int] = []

    while len(obs_buf) < n_transitions:
        if all(env.terminations.values()):
            env.reset()
            continue
        obs = env.observe(env.agent_selection).copy()
        mask = env.action_mask(env.agent_selection).copy()
        a = teacher(env, rng)
        obs_buf.append(obs)
        mask_buf.append(mask)
        act_buf.append(a)
        env.step(a)

    return (
        np.stack(obs_buf),
        np.stack(mask_buf),
        np.asarray(act_buf, dtype=np.int64),
    )


def behavior_clone(
    net: PolicyNet,
    obs: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
) -> list[float]:
    """Cross-entropy supervised training of the policy head."""
    obs_t = torch.from_numpy(obs).float().to(device)
    masks_t = torch.from_numpy(masks).bool().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    n = len(actions_t)
    losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, _ = net(obs_t[idx], masks_t[idx])
            loss = F.cross_entropy(logits, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))
    return losses


# --- Eval-during-training ----------------------------------------------------


def evaluate_against_progress(
    policy: PolicyNet,
    n_games: int,
    seed: int,
    device: str = "cpu",
    expose_opponent_pickup: bool = True,
) -> dict[str, float]:
    """Pit the trained policy against 5 `progress` baselines (rotated seats)."""
    neural = make_neural_policy(policy, device=device, deterministic=False)
    policies = [neural] + [POLICIES["progress"]] * (NUM_PLAYERS - 1)
    names = ["neural"] + ["progress"] * (NUM_PLAYERS - 1)
    result = run_eval(
        policies, names, n_games=n_games, seed=seed,
        expose_opponent_pickup=expose_opponent_pickup,
    )
    completed = result.n_games - result.not_terminated
    return {
        "neural_win_rate": result.win_rates[0],
        "neural_avg_score": result.avg_scores[0],  # headline: lower is better
        "neural_avg_loser_score": result.avg_loser_scores[0],
        "progress_avg_score": float(np.mean(result.avg_scores[1:])),
        "progress_avg_win_rate": float(np.mean(result.win_rates[1:])),
        "completed": float(completed),
        "not_terminated": float(result.not_terminated),
    }


# --- Main loop ---------------------------------------------------------------


def train(cfg: TrainConfig) -> Path:
    run_name = cfg.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.runs_root) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    pyrandom.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cpu"
    policy = PolicyNet(hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    env = SpeedRummyEnv(
        seed=cfg.seed,
        expose_opponent_pickup=cfg.expose_opponent_pickup,
    )
    rollout_rng = pyrandom.Random(cfg.seed * 7919 + 1)

    if cfg.opponent == "self":
        opponent_fn = None
    elif cfg.opponent in POLICIES:
        opponent_fn = POLICIES[cfg.opponent]
    else:
        raise ValueError(
            f"unknown --opponent={cfg.opponent!r}; "
            f"valid: 'self' or {sorted(POLICIES)}"
        )

    metrics_path = run_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w")

    if cfg.bc_pretrain:
        print(
            f"[bc] collecting {cfg.bc_dataset_size} transitions from "
            f"{cfg.bc_teacher}-vs-{cfg.bc_teacher} play..."
        )
        bc_obs, bc_masks, bc_acts = collect_bc_dataset(
            cfg.bc_teacher, cfg.bc_dataset_size, seed=cfg.seed + 1,
            expose_opponent_pickup=cfg.expose_opponent_pickup,
        )
        print(
            f"[bc] training: {cfg.bc_epochs} epochs, batch={cfg.bc_batch_size}, "
            f"lr={cfg.bc_lr}"
        )
        bc_losses = behavior_clone(
            policy,
            bc_obs,
            bc_masks,
            bc_acts,
            epochs=cfg.bc_epochs,
            batch_size=cfg.bc_batch_size,
            lr=cfg.bc_lr,
            device=device,
        )
        for i, l in enumerate(bc_losses):
            print(f"  [bc epoch {i+1}/{cfg.bc_epochs}] loss={l:.4f}")
        bc_eval = evaluate_against_progress(
            policy, n_games=50, seed=99_999, device=device
        )
        print(
            f"[bc] post-BC: "
            f"avg_score={bc_eval['neural_avg_score']:.1f} "
            f"(progress avg={bc_eval['progress_avg_score']:.1f}; "
            f"lower is better)  "
            f"win={bc_eval['neural_win_rate']*100:.1f}%"
        )
        torch.save(policy.state_dict(), run_dir / "checkpoints" / "policy_bc.pt")
        metrics_f.write(
            json.dumps(
                {
                    "phase": "bc",
                    "bc_losses": bc_losses,
                    "bc_post_win_rate": bc_eval["neural_win_rate"],
                    "bc_post_avg_score": bc_eval["neural_avg_score"],
                    "bc_post_progress_avg_score": bc_eval["progress_avg_score"],
                }
            )
            + "\n"
        )
        metrics_f.flush()
        # Re-init optimizer fresh for PPO (separate hyperparams).
        optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    total_steps_done = 0
    update_idx = 0
    start_time = time.time()

    print(
        f"[train] run_dir={run_dir}, total_steps={cfg.total_steps}, "
        f"rollout_steps={cfg.rollout_steps}, opponent={cfg.opponent}"
    )

    while total_steps_done < cfg.total_steps:
        buf, rollout_stats = collect_rollout(
            env,
            policy,
            cfg.rollout_steps,
            device=device,
            max_episode_steps=cfg.max_episode_steps,
            opponent_policy_fn=opponent_fn,
            rng=rollout_rng,
        )
        advantages, returns = compute_gae(buf, cfg.gamma, cfg.gae_lambda)
        update_stats = ppo_update(
            policy, optimizer, buf, advantages, returns, cfg, device=device
        )
        total_steps_done += len(buf)
        update_idx += 1

        elapsed = time.time() - start_time
        steps_per_sec = total_steps_done / max(elapsed, 1e-6)
        record: dict[str, float | int | str] = {
            "update": update_idx,
            "steps_done": total_steps_done,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            **rollout_stats,
            **update_stats,
        }

        if update_idx % cfg.eval_every == 0:
            eval_stats = evaluate_against_progress(
                policy, n_games=cfg.eval_games, seed=10_000 + update_idx,
                device=device,
                expose_opponent_pickup=cfg.expose_opponent_pickup,
            )
            record.update(eval_stats)
            print(
                f"[update {update_idx:4d}] steps={total_steps_done:>8} "
                f"({steps_per_sec:.0f}/s)  "
                f"ep_ret={record['mean_episode_return']:+.2f}  "
                f"trunc={record['trunc_rate']*100:.0f}%  "
                f"pi_loss={record['policy_loss']:+.3f}  "
                f"H={record['entropy']:.2f}  "
                f"avg_score={eval_stats['neural_avg_score']:.1f} "
                f"(prog={eval_stats['progress_avg_score']:.1f})  "
                f"win={eval_stats['neural_win_rate']*100:.1f}%"
            )
        else:
            print(
                f"[update {update_idx:4d}] steps={total_steps_done:>8} "
                f"({steps_per_sec:.0f}/s)  "
                f"ep_ret={record['mean_episode_return']:+.2f}  "
                f"trunc={record['trunc_rate']*100:.0f}%  "
                f"pi_loss={record['policy_loss']:+.3f}  "
                f"H={record['entropy']:.2f}"
            )

        metrics_f.write(json.dumps(record) + "\n")
        metrics_f.flush()

        if update_idx % cfg.checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"policy_{update_idx:04d}.pt"
            torch.save(policy.state_dict(), ckpt_path)

    metrics_f.close()
    final_path = run_dir / "final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"[train] done. final model: {final_path}")
    return run_dir


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO self-play trainer for Speed Rummy")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--max-episode-steps", type=int, default=800)
    p.add_argument(
        "--opponent",
        type=str,
        default="progress",
        help="Opponent policy: 'self' (parameter-shared self-play) or any "
             "name from rummy_v2.eval.POLICIES (default: progress).",
    )
    p.add_argument("--no-bc", action="store_true", help="Skip BC warm-start.")
    p.add_argument(
        "--no-pickup",
        action="store_true",
        help="Ablate the opponent-pickup obs feature (zeros that section).",
    )
    p.add_argument("--bc-teacher", type=str, default="progress")
    p.add_argument("--bc-dataset-size", type=int, default=20_000)
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args_dict = vars(args)
    no_bc = args_dict.pop("no_bc", False)
    no_pickup = args_dict.pop("no_pickup", False)
    cfg = TrainConfig(**args_dict)
    cfg.bc_pretrain = not no_bc
    cfg.expose_opponent_pickup = not no_pickup
    train(cfg)


if __name__ == "__main__":
    main()
