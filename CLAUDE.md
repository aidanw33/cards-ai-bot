# Claude orientation — cards-ai-bot

Goal: train an AI agent to play **Speed Rummy**, a 6-player rummy variant the user's family plays. Round 1 only is in scope (2 sets of 3); later rounds with runs are deferred.

## Layout

- `RULES.md` — **source of truth** for the game. Match implementation to this doc, not vice versa.
- `rummy_v2/` — clean rewrite. **Edit here.**
- `rummy/`, `gui/` — **legacy 2-player code from 2025. Do not edit.** Kept only as reference until rummy_v2 reaches feature parity, then will be deleted.

## rummy_v2 module map

| Module | Purpose |
| --- | --- |
| `cards.py` | `Card`, `Rank`, `Suit`, wild detection, point values, `build_deck()` (108 cards) |
| `state.py` | Frozen `GameState` / `PlayerState` / `DownPile`, `Phase` enum, `initial_state()` |
| `actions.py` | Action variants (`DrawDeck`, `DrawDiscard`, `Buy`, `PassBuy`, `GoDown`, `LayOff`, `Discard`), `is_legal`, `legal_actions`, `find_any_go_down` |
| `engine.py` | `apply_action(state, action, rng) -> GameState` — the single transition function |
| `cli.py` | Hot-seat terminal play: `python -m rummy_v2.cli [--seed N]` |
| `random_play.py` | Progress-biased random policy + game runner; used by smoke tests and as a future RL baseline |
| `env.py` | PettingZoo-AEC-style multi-agent env. 244-dim obs, 186-dim action space (no GoDown — auto-applied), per-state action mask, dense rewards. See `ENV_DESIGN.md` for rationale |
| `eval.py` | Eval harness: pits policies against each other in 6-seat games with rotated seats. Built-in baselines: `uniform`, `progress`, `greedy` |
| `policy.py` | `PolicyNet` (MLP with masked-logit policy head + value head) and `make_neural_policy()` adapter into the eval harness |
| `train.py` | PPO trainer with BC warm-start, action mask, vs-fixed-opponent or self-play, GAE, eval-during-training, checkpointing to `runs/<name>/` |
| `tests/` | pytest suites; **269 tests** currently |

## Design rules

- **Immutable state.** `GameState` and friends are frozen dataclasses. Engine returns new states via `dataclasses.replace`; never mutate.
- **Phase machine.** `Phase.AWAITING_DRAW → BUY_CASCADE? → POST_DRAW → (next player or ROUND_OVER)`. See the docstring at the top of `engine.py` for the full diagram.
- **Action mask split.** `legal_actions(state)` returns every legal action *except* `GoDown` (combinatorial). Use `find_any_go_down(hand)` to test go-down feasibility; the env will need its own enumeration when wired up.
- **RNG out of state.** `random.Random` is passed to `apply_action`; only consulted on reshuffle. State remains hashable.
- **Type hints + Python 3.11+.** `from __future__ import annotations`, `slots=True` on dataclasses, written to be `mypy --strict` clean.

## Running things

```bash
python -m pytest rummy_v2/                     # full suite
python -m pytest rummy_v2/tests/test_smoke.py  # invariants on 50 random games
python -m rummy_v2.cli --seed 42               # play hot-seat
```

## Training results so far (2026-05-02)

First trainer run (`runs/run5_bc_lowlr`): BC warm-start (3 epochs on 20k progress-vs-progress transitions) + 100k PPO trainee transitions vs progress, lr=1e-4, entropy_coef=0.05. Final agent eval (3 × 500 games): **~13-14% win rate vs 5 progress** — within noise of (slightly below) the 16.7% symmetric baseline. Avg loser score ~38 (lowest at the table) — agent loses gracefully.

Lessons learned:
- **Pure self-play from random init does not converge** (0-4% win rate, policy collapses to non-terminating loop). Vs-fixed-opponent training is far more stable.
- **BC warm-start is a strong floor** — 3 epochs of supervised cross-entropy on `progress` traces gets the policy to ~16% win rate immediately.
- **PPO can destabilize BC** if LR is too high. lr=1e-4 (vs default 3e-4) keeps BC roughly intact while letting PPO refine.
- **Episode-length truncation** in the rollout (apply natural loser penalty after 800 env steps) is essential; without it, stuck self-play episodes never produce terminal transitions and PPO sees no win signal.

## Pending work (next sessions)

- Beat the 16.7% baseline meaningfully. Options: longer PPO training (current runs are ~5 min on CPU); KL regularization toward the BC policy; staged training (BC → vs-progress → vs-greedy → self-play); reward shaping de-emphasizing going-down/lay-off bonuses in favor of the +10 win signal.
- Self-play with opponent pool of past-self snapshots (currently `--opponent self` works but is unstable from cold; should be tried *after* BC + vs-progress stabilizes a strong base policy).

## Why this rewrite exists

The legacy `rummy/` code was a 2-player simplification with a sparse-reward, lossy-state-encoding env. The agent never learned. See `git log` and conversation memory for the full diagnosis. **The bottleneck was environment quality, not the model** — that's the lesson driving rummy_v2.
