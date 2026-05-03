# Env design — `rummy_v2/env.py`

This is a design doc, not a spec. It records the choices that shape the RL environment, the alternatives I rejected, and the questions where I want pushback before writing code. Once we agree on the decisions marked **DECIDE**, I'll implement and the answers will become the docstring of `env.py`.

The legacy bot failed because the environment was bad (sparse reward, lossy state, automatic go-down). This doc is mostly an exercise in *not repeating* those mistakes.

---

## 1. Goal

Wrap the `rummy_v2` engine in a multi-agent RL environment suitable for self-play training of a single shared policy. Round 1 only, 6 players. The wrapper must:

- expose a clean observation that contains everything a skilled player would know (and nothing they wouldn't),
- expose a tractable action space with a correct legal-action mask,
- give dense-enough reward signal that learning has gradient before the first win,
- run fast enough that we can do millions of steps in self-play,
- be deterministic given a seed, for debugging.

---

## 2. Multi-agent framing

The phase machine already has exactly one decider at any moment (active player during draw/post-draw, `buy_offer_to` player during cascade). That maps cleanly onto **PettingZoo's AECEnv** (Agent Environment Cycle) — turn-based, one agent acts per `step()`.

**DECIDE 1 — framework choice**
- **(A) PettingZoo AECEnv (recommended).** Industry standard for turn-based multi-agent. Lets us reuse `pettingzoo.utils` wrappers (action mask, observation flattening). PPO trainers like SB3-Contrib and CleanRL have known patterns for it.
- **(B) Custom Gym-style env with `info["active_agent"]`.** Simpler, no dependency, but we'd reinvent agent rotation and many trainers don't speak it natively.

I lean (A). The dependency is light (`pettingzoo` is pure Python).

For training: one shared policy plays all 6 seats. Observations are POV-relative ("you are player 0, opponents are 1..5 in turn order from your left").

---

## 3. Observation space

What a human player knows on their turn:
- their own hand (rank + suit identities)
- every player's down piles (public, including own)
- top of discard pile (and its rank/wildness)
- discard pile size, draw pile size
- every player's hand size
- every player's `buys_used`
- whose turn it is, what phase, who's being offered the buy
- (arguably) the full discard pile history — humans can in principle remember it

What they *don't* know:
- contents of other players' hands
- order of cards in the draw pile

### 3.1 Encoding: per-rank counts vs one-hot multiset vs card IDs

**Round 1 has a structural simplification**: sets are rank-only. Suit is irrelevant for legality. This argues strongly for a **per-rank count encoding** as the base unit.

- **Hand**: 14-dim vector — count of each non-wild rank (2..A = 13 dims) + count of wilds (1 dim). Total = 14 ints in [0, ~8].
- **Each down pile**: (rank_one_hot[13], num_cards[1], num_wilds[1]) = 15 dims. Up to `6 players × 2 piles = 12` piles. Pad with zeros, mask absent piles.
- **Discard top**: same 14-dim card encoding (one-hot rank-or-wild).
- **Player metadata** (per player, including self): hand_size[1], buys_used[1], num_down_piles[1], is_active[1] = 4 dims × 6 players = 24.
- **Phase**: one-hot of the 4 phases = 4 dims.
- **Pile sizes**: draw_pile_size[1], discard_pile_size[1] = 2 dims.
- **Buy cascade pointer**: who's being offered (one-hot relative to viewer, 6 dims).

Total observation: roughly **14 + 12×15 + 14 + 24 + 4 + 2 + 6 ≈ 244 floats**. Small, fast.

### 3.2 What we lose with per-rank counts

- Card *identity* is gone — we can't distinguish "the 7♥ I drew last turn" from "the other 7♥". For Round 1 this is fine because sets don't care about suit.
- We lose discard pile history below the top. Humans can in principle remember which 7s have been discarded; the agent cannot. This is a deliberate cap on the agent's memory — we can revisit if it becomes a ceiling.

### 3.3 Alternatives considered

- **108-dim one-hot multiset** of the hand: redundant for Round 1 (suit doesn't matter), and identical card copies would need to be summed anyway.
- **Variable-length card-ID list**: requires a transformer/set encoder. Overkill for the information available.
- **Append full discard pile history**: doable, costs another ~108 dims if we one-hot it. Defer.

**DECIDE 2 — observation encoding**
I recommend per-rank counts as above. Two sub-questions:
- **2a**: include full discard history, or top-only? (Recommend top-only for v1.)
- **2b**: include the *current player's POV index* explicitly, or always rotate so viewer is index 0? (Recommend rotate — simpler, removes a symmetry the model would otherwise have to learn.)

---

## 4. Action space

This is the hard part. The actions are:

| Action | Cardinality |
| --- | --- |
| `DrawDeck` | 1 |
| `DrawDiscard` | 1 |
| `Buy` | 1 |
| `PassBuy` | 1 |
| `Discard(card)` | up to 14 (one per rank, wilds as one bucket) |
| `LayOff(card, target_player, pile_index)` | (14 ranks) × (≤12 destination piles) ≈ 168 |
| `GoDown(set1, set2)` | **combinatorial — the problem** |

### 4.1 GoDown — handled automatically by the env

`GoDown` is **not** an agent-controlled action. The env wrapper applies it automatically: whenever the active player enters `POST_DRAW` with a hand that satisfies `find_any_go_down`, the env immediately applies that `GoDown` action *before* yielding control back to the agent. The agent's POST_DRAW decision becomes simply: lay off (if down) or discard.

The auto-go-down choice was made on domain grounds: in actual Speed Rummy play, the situations where you'd want to delay going down once you can are vanishingly rare. The complexity cost of exposing the choice (combinatorial action space, pointer-network or canonical-set-picker logic, mask blow-up) doesn't earn its keep for this round. If a stronger v2 agent later plateaus and we suspect go-down timing is the ceiling, we can revisit.

Alternatives considered and rejected:
- **Two-headed factored action** (macro "go down y/n" + canonical-set picker): adds a binary action and a tie-break heuristic for almost no behavioral diversity.
- **Variable enumerated go-down slots** (Pointer-Network-style scoring over valid `(set1, set2)` pairs): meaningful engineering — pad to ~64 slots, train per-action embeddings — without strategic upside in Round 1.

### 4.2 LayOff factoring

168 raw `(card_rank, target_pile)` slots is fine — small enough for a flat head, sparse mask handles legality. We'll lay off **one card per LayOff action**; the agent may chain multiple lay-offs in the same `POST_DRAW` phase across multiple `step()` calls.

### 4.3 Discard

Discard by rank (14 slots: 13 non-wild ranks + 1 wild bucket). If multiple cards of the rank are in hand, the engine picks one (the order doesn't matter — they're equivalent for game state). Already true at the engine level since suit is irrelevant in Round 1.

### 4.4 Total flat action space

| Range | Action |
| --- | --- |
| 0 | `DrawDeck` |
| 1 | `DrawDiscard` |
| 2 | `Buy` |
| 3 | `PassBuy` |
| 4..17 | `Discard(rank=k)` (14 slots) |
| 18..185 | `LayOff(rank=k, pile=p)` (14×12 slots) |

Total: **186 discrete actions**. Mask is a 186-dim bool vector returned in `info["action_mask"]`. `GoDown` is not represented in the action space — see §4.1.

**DECIDE 4 — discard granularity**
Discard-by-rank (14 slots) vs discard-by-card-id (108 slots, mostly masked). For Round 1, by-rank is information-equivalent and simpler. Recommend by-rank.

---

## 5. Reward shaping

Sparse reward (only +1 for going out) is what killed the legacy agent. We need dense signal that points the policy toward going-out without distorting the optimum.

Proposal — sum these per step:

| Event | Reward |
| --- | --- |
| Going down (first time this round) | **+5.0** |
| Each card laid off | **+0.5** |
| Each buy made | **-0.3** (buys cost penalty cards; discourage spam) |
| Per-step idle penalty | **-0.01** (gentle bias toward speed) |
| Terminal: round won (you went out) | **+10.0** |
| Terminal: round lost | **-0.05 × your hand point total** |

Notes:
- Hand point total at terminal can be high (a 17-card hand of 10s = 170 → -8.5). Going out is +10 — a player with a moderately bad hand prefers losing slightly rather than aggressively buying to try to come back, which matches human intuition.
- Per-step penalty is small enough that it won't dominate the going-down bonus but discourages infinite stalling.
- Going-down bonus is one-shot per round — `info["went_down_this_step"]` flag from the engine.
- The `-0.3` buy penalty is the most arguable number. Buys are sometimes correct (catching a key wild) — too high a penalty kills that option. We may need to tune.

**DECIDE 5 — reward magnitudes**
The numbers above are first-pass intuitions, not derived. I'd start with these and tune on early training curves. Object now if any look obviously wrong (e.g. you'd want a much larger going-out bonus, or zero per-step penalty).

---

## 6. API surface (pettingzoo AECEnv)

```python
class SpeedRummyEnv(AECEnv):
    metadata = {"name": "speed_rummy_v0", "is_parallelizable": False}

    def __init__(self, seed: int | None = None): ...
    def reset(self, seed: int | None = None) -> None: ...
    def observe(self, agent: str) -> np.ndarray: ...      # 244-dim float32
    def step(self, action: int) -> None: ...              # advances one decision
    def action_mask(self, agent: str) -> np.ndarray: ...  # 187-dim bool
    def render(self) -> str: ...                          # reuse cli.render()

    # PettingZoo-required: agent_selection, rewards, terminations, truncations, infos
```

Per AECEnv convention: `agent_selection` exposes who acts next; `step()` consumes that agent's action and rotates internally. `info["action_mask"]` carries the legal-action mask (SB3-Contrib's MaskablePPO reads this).

**DECIDE 6 — env factory shape**
Single class `SpeedRummyEnv`, or factory function `make_env(...)` per PettingZoo idiom? Recommend factory + class, both exported.

---

## 7. Testing plan

Before any training:
1. **Type / shape tests**: observation is `(244,) float32`, mask is `(187,) bool`, both shapes constant across all reachable states.
2. **Mask correctness**: for 100 random rollouts, every action sampled from the mask is legal in the underlying engine (no engine `is_legal=False` ever fires).
3. **Mask completeness**: for 100 random states, every legal engine action maps to a mask-true slot.
4. **Reward sums**: sum of all per-step rewards across a full episode equals a closed-form value derived from final hand points, buys made, lay-offs, etc. Catches double-counting.
5. **Determinism**: `seed=42` produces byte-identical observation streams across runs.
6. **Episode termination**: 100 random rollouts under a uniform-over-mask policy all terminate within 2000 steps. (Random_play already shows this works at the engine level.)

---

## 8. Out of scope (deferred)

- Full discard-pile history in obs (revisit if agent plateaus).
- AlphaZero-style perfect-info rollouts. Speed Rummy has hidden state (opponents' hands, deck order); MCTS would need belief sampling — meaningful infra investment for unclear gain over PPO+selfplay.
- Opponent modeling / belief over opponent hands as an explicit obs feature.
- Round 2+ rules (runs, different meld goals).
- Curriculum (e.g. start with 2 players, scale up). The legacy bot already showed 2-player ≠ 6-player; we go to 6 immediately.

---

## 9. Decisions to confirm

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | Framework: PettingZoo AECEnv vs custom | **PettingZoo AECEnv** ✅ confirmed |
| 2a | Discard history in obs? | **No (top-only) for v1** ✅ confirmed |
| 2b | Rotate POV so viewer = index 0? | **Yes** ✅ confirmed |
| 3 | GoDown action | **Auto (env-applied, not in action space)** ✅ confirmed (domain call: rare in practice to want to delay) |
| 4 | Discard granularity: by-rank vs by-card | **By-rank (14 slots)** ✅ confirmed |
| 5 | Reward magnitudes | **Start with table in §5, tune on curves** ✅ confirmed |
| 6 | Env factory shape | **Class + `make_env()` factory** ✅ confirmed |

All decisions confirmed. Implementing `env.py` + tests next.
