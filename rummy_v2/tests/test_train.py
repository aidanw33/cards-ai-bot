"""Smoke tests for `rummy_v2.train` and `rummy_v2.policy`.

Trains for a tiny number of steps to verify the pipeline runs without errors
and produces sane outputs. Does NOT verify the trained policy is *good* —
that's the point of `eval.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from rummy_v2.env import ACTION_SPACE_SIZE, OBS_DIM, SpeedRummyEnv
from rummy_v2.policy import PolicyNet, make_neural_policy
from rummy_v2.train import (
    RolloutBuffer,
    TrainConfig,
    collect_rollout,
    compute_gae,
    ppo_update,
    train,
)
from torch import optim


class TestPolicyNet:
    def test_forward_shape(self) -> None:
        net = PolicyNet()
        obs = torch.zeros(3, OBS_DIM)
        mask = torch.ones(3, ACTION_SPACE_SIZE, dtype=torch.bool)
        logits, value = net(obs, mask)
        assert logits.shape == (3, ACTION_SPACE_SIZE)
        assert value.shape == (3,)

    def test_mask_zeros_illegal_logits(self) -> None:
        net = PolicyNet()
        obs = torch.zeros(1, OBS_DIM)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0, 5] = True  # only one legal action
        logits, _ = net(obs, mask)
        # Action 5 should have a finite logit; others should be -inf-ish.
        assert torch.isfinite(logits[0, 5])
        assert (logits[0, :5] < -1e8).all()
        assert (logits[0, 6:] < -1e8).all()

    def test_sampled_action_respects_mask(self) -> None:
        net = PolicyNet()
        obs = torch.zeros(1, OBS_DIM)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0, 42] = True
        for _ in range(10):
            action, _, _, _ = net.get_action_and_value(obs, mask)
            assert int(action.item()) == 42

    def test_neural_policy_returns_legal_action(self) -> None:
        net = PolicyNet()
        env = SpeedRummyEnv(seed=0)
        rng = random.Random(0)
        policy_fn = make_neural_policy(net)
        for _ in range(50):
            if all(env.terminations.values()):
                break
            action = policy_fn(env, rng)
            mask = env.action_mask(env.agent_selection)
            assert mask[action], f"neural policy chose illegal action {action}"
            env.step(action)


class TestRollout:
    def test_collect_rollout_returns_target_size(self) -> None:
        net = PolicyNet()
        env = SpeedRummyEnv(seed=0)
        buf, stats = collect_rollout(env, net, n_steps=512)
        # Buffer is at least n_steps, possibly slightly more if episode-end
        # closure adds a few extras.
        assert len(buf) >= 512
        assert stats["rollout_size"] == float(len(buf))

    def test_collect_rollout_buffer_consistency(self) -> None:
        net = PolicyNet()
        env = SpeedRummyEnv(seed=0)
        buf, _ = collect_rollout(env, net, n_steps=512)
        n = len(buf)
        assert len(buf.obs) == n
        assert len(buf.masks) == n
        assert len(buf.actions) == n
        assert len(buf.log_probs) == n
        assert len(buf.values) == n
        assert len(buf.rewards) == n
        assert len(buf.dones) == n
        assert len(buf.next_values) == n
        assert len(buf.agent_ids) == n


class TestGAE:
    def test_gae_shape(self) -> None:
        net = PolicyNet()
        env = SpeedRummyEnv(seed=0)
        buf, _ = collect_rollout(env, net, n_steps=256)
        adv, ret = compute_gae(buf, gamma=0.99, lam=0.95)
        assert adv.shape == (len(buf),)
        assert ret.shape == (len(buf),)

    def test_gae_terminal_resets(self) -> None:
        """At a done=True transition, advantage = reward + 0 - value."""
        buf = RolloutBuffer()
        buf.obs.append(np.zeros(OBS_DIM, dtype=np.float32))
        buf.masks.append(np.ones(ACTION_SPACE_SIZE, dtype=bool))
        buf.actions.append(0)
        buf.log_probs.append(0.0)
        buf.values.append(2.0)
        buf.rewards.append(5.0)
        buf.dones.append(True)
        buf.next_values.append(0.0)
        buf.agent_ids.append("player_0")
        adv, ret = compute_gae(buf, gamma=0.99, lam=0.95)
        # delta = 5 + 0.99 * 0 * (1-1) - 2 = 3.0
        # last_gae = 3.0 + ... * 0 (no next) = 3.0
        assert adv[0] == 3.0
        assert ret[0] == 3.0 + 2.0


class TestPPOUpdate:
    def test_ppo_update_runs(self) -> None:
        net = PolicyNet()
        env = SpeedRummyEnv(seed=0)
        buf, _ = collect_rollout(env, net, n_steps=256)
        adv, ret = compute_gae(buf, gamma=0.99, lam=0.95)
        cfg = TrainConfig(minibatch_size=64, update_epochs=2)
        opt = optim.Adam(net.parameters(), lr=3e-4)
        stats = ppo_update(net, opt, buf, adv, ret, cfg)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["value_loss"])


class TestTrainLoop:
    def test_train_smoke(self, tmp_path: Path) -> None:
        cfg = TrainConfig(
            total_steps=512,
            rollout_steps=256,
            minibatch_size=64,
            update_epochs=1,
            eval_every=100,  # skip eval during smoke
            checkpoint_every=100,
            run_name="pytest_smoke",
            runs_root=str(tmp_path),
            bc_pretrain=False,  # skip BC for speed
        )
        run_dir = train(cfg)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "final.pt").exists()

        with (run_dir / "metrics.jsonl").open() as f:
            records = [json.loads(line) for line in f]
        assert len(records) >= 1
        ppo_records = [r for r in records if r.get("phase") != "bc"]
        assert len(ppo_records) >= 1
        for r in ppo_records:
            assert "policy_loss" in r
            assert np.isfinite(r["policy_loss"])

        # Loaded model should still be a valid policy.
        net = PolicyNet()
        net.load_state_dict(torch.load(run_dir / "final.pt"))
        env = SpeedRummyEnv(seed=999)
        policy_fn = make_neural_policy(net)
        for _ in range(20):
            if all(env.terminations.values()):
                break
            action = policy_fn(env, random.Random(0))
            assert env.action_mask(env.agent_selection)[action]
            env.step(action)
