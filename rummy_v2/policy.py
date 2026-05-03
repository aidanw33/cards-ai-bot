"""Policy network for Speed Rummy + adapter into the eval harness.

`PolicyNet` is a 2-layer MLP with separate policy (logits) and value heads,
sharing a trunk. The action distribution is masked at the logit level
(illegal actions get logit = -inf), so sampled actions are guaranteed legal.

`make_neural_policy(net)` wraps a trained network into the `Policy` callable
shape used by `rummy_v2.eval` so trained agents can be benchmarked against
the existing baselines via the same harness.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from rummy_v2.env import (
    ACTION_SPACE_SIZE,
    OBS_DIM,
    SpeedRummyEnv,
)


_NEG_INF: float = -1e9


class PolicyNet(nn.Module):
    """Shared-trunk MLP policy + value network.

    Forward returns (masked_logits, value). Always pass the boolean legality
    mask (True = legal): illegal logits are clamped to a large negative number
    so sampling and softmax both ignore them.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        # Standard PPO init: small policy head, larger trunk gain.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        masked_logits = logits.masked_fill(~mask, _NEG_INF)
        value = self.value_head(h).squeeze(-1)
        return masked_logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_logits, value = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=masked_logits)
        if action is None:
            if deterministic:
                action = masked_logits.argmax(dim=-1)
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# --- Adapter into eval harness -----------------------------------------------


def make_neural_policy(
    net: PolicyNet,
    device: str = "cpu",
    deterministic: bool = False,
) -> Callable[[SpeedRummyEnv, random.Random], int]:
    """Wrap a `PolicyNet` into the `Policy` shape used by `rummy_v2.eval`.

    The `rng` argument is unused — sampling is from the network's
    `Categorical` distribution (or argmax if `deterministic=True`). The signature
    is preserved for interchangeability with the engine-side baselines.
    """

    def policy(env: SpeedRummyEnv, rng: random.Random) -> int:
        obs = env.observe(env.agent_selection)
        mask = env.action_mask(env.agent_selection)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
            action, _, _, _ = net.get_action_and_value(
                obs_t, mask_t, deterministic=deterministic
            )
        return int(action.item())

    return policy
