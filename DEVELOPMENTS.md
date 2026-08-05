# Kaggriculture — development notes

Working document for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition. Game breakdown, architecture walkthrough, how to run everything,
and a dated log of what we tried.

- **Entry deadline** 2026-09-23 · **Final submission** 2026-09-30
- **Prizes** $50,000 — $5,000 each to the **top 10**
- **Limits** 5 submissions/day, only the latest 2 are scored

---

## 1. The game

Two players, two **separate** 10×10 farms, 30 days × 24 turns = **720 turns**.
Most coins in the bank at the end wins. Unsold stock scores nothing.

You control one farmer plus farm hands you re-hire each day. Each unit takes
**one action per turn**. You may also issue up to **10 market orders per turn**.

### What actually couples the two players

Almost nothing. Farms are independent; the only shared object is the **market**.
A competitor's [seat-swap experiment](https://www.kaggle.com/competitions/kaggriculture/discussion/731152)
got bit-identical banks from either seat, because the engine quotes both players
against the same pre-commit inventory — there is no first-mover advantage.

This is the single most important strategic fact in the project: it makes
Kaggriculture ~95% a **single-agent scheduling problem** and only ~5% a game.
It is why we optimise *mean final bank* rather than win-rate (§5), and why
self-play RL would buy very little (§7).

### Economics

Value density per tile-day, and how the price reacts to flooding the market:

| Resource | Yield/tile/day | Base $ | $/tile/day | Glut behaviour (`above_func`/`target`) |
|---|---|---|---|---|
| Melon | 0.55 | 250 | **137** | `sq` / 3.60 — $1 floor at +T. Brutal. |
| Milk | 0.50 | 160 | 80 | `linear` / 1.60 — $1 at +T |
| Wool | 0.33 | 200 | 66 | `sq` / 3.20 — $1 at +T |
| **Egg** | 1.00 | 50 | **50** | `log` / 0.20 — only $50→$40. **Resilient** |
| Strawberry | 0.24 | 120 | 29 | `linear` / 1.60 — $1 at +T |
| Carrot | 0.75 | 35 | 26 | `sqrt` / 0.70 |
| Wheat | 0.80 | 25 | 20 | `log` / 0.20 — resilient |
| Tomato | 0.33 | 60 | 20 | `sqrt` / 0.60 |

Melon looks dominant on raw value and is a trap if you dump it: `P(I0+T) = $1`.
The real lesson is **dump staples, drip premiums**.

Things the tables above don't show, which matter a lot:

- **`CARE` doubles goose output.** The bonus banks at end of day when fed *and*
  cared for, and is paid out on the next production tick. Goose interval is 1
  day, so steady state is 2 eggs/day, not 1.
- **Every animal yields 1 free fertilizer per day** regardless of care, and
  fertilizer bases at $100. `COLLECT_FERTILIZER` is one action for ~$100.
- **Wheat is a cost centre for livestock.** Every animal eats 1 wheat/day. Town
  shops constantly drain wheat, and wheat's *scarcity* curve is steep
  (`sqrt`/0.80), so its price climbs all season while egg prices sag. Buying
  feed late-game can be value-destroying — hence `wheat_buy_max_price`.
- **Labour is cheap but not free.** Hire cost is `fib(n)` per day, resetting
  daily: 10 hands = 143 coins, but 16 hands = 2,583 and 20 hands = 17,710. The
  marginal hand stops paying for itself somewhere in the mid-teens.
- **The shed caps at 100 items and overflow is silently discarded.**
- **`SELL` draws only from the shed**, so the pipeline is
  harvest → unit inventory → shed (via `DROP` or the end-of-day auto-drop) → sell.
- **`FEED` and `FERTILIZE` consume from the unit's inventory**, not the shed, so
  a unit must fetch wheat before it can feed anything.

### Engine gotchas that cost us real time

The competition docs disagree with the engine in several places; the host
confirmed **"engine is the source of truth"**
([discussion/732450](https://www.kaggle.com/competitions/kaggriculture/discussion/732450)).
Everything in `agent/rules.py` is transcribed from the engine source and
verified by `tests/test_parity.py`.

1. **Atomic PLANT validation** — the single worst trap. The engine computes
   ```python
   blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}
   ```
   If more units issue `PLANT WHEAT` in a turn than you hold wheat seeds,
   **every one of them is silently dropped**. With 7 units and 2 seeds the farm
   deadlocks permanently — nothing about the state changes to break the tie.
   Cost us a full run that scored 88 coins. `_build_tasks` now caps plant tasks
   at the seed count.
2. **`__file__` does not exist inside an agent.** `kaggle_environments` loads
   agents via `exec(compile(raw, path), {})`. Touching `__file__` raises
   NameError, which the loader swallows into `InvalidArgument` — the submission
   fails its validation episode. See `_agent_dir()` in `agent/main.py`.
3. **The file loader takes the *last callable defined*,** not the one named
   `agent`. Define `agent` last, import nothing after it.
4. A fresh seed starts at `consecutive_unwatered = 1`, so a seed planted and not
   watered the same day becomes a weed that night. No grace period.
5. Melon caps at age 10; ages 11–12 are dead turns.
6. Wheat/carrot only reach their listed max yield **with fertilizer**.
7. The shed is not a tile. Access tiles at `boardSize=10` are (4,4), (5,4),
   (4,5), (5,5) — and three of them are **locked** until you buy land, while
   `PICKUP` no-ops on a locked tile.
8. Sales at the $1 floor do not add market supply, so the floor stays responsive.

### Hard limits

| | |
|---|---|
| `actTimeout` | **1 s/turn** (plus a 60 s bank for the whole episode) |
| `runTimeout` | 1200 s per episode |
| Submission box | 2 vCPU, ~12 GiB RAM |
| Our own ceiling | p99 turn < 300 ms, enforced by `tests/test_contract.py` |

The field currently uses ~0.005 ms of that 1 s budget. Per-turn forward
simulation is wide-open ground (§7, milestone 6).

---

## 2. Why this is not an RL project

The original brief assumed RL + PufferLib. The research said otherwise:

- **Every** RL winner of a `kaggle_environments` competition rewrote the
  simulator first — Lux S1, Lux S3, Orbit Wars 1st (Rust, ~2,400 B200-hours,
  15B steps), Orbit Wars 3rd (JAX). Nobody trains against the Python env.
- `kaggriculture.py` is 1,063 lines of fiddly dict mutation. A port is 2–4 weeks
  with **silent** divergence risk, against a 7-week clock.
- ~99% of actions are forced maintenance (water, feed, walk). The real decisions
  number a few hundred per episode. Model-free RL would spend its sample budget
  rediscovering "water the plant".
- The Halite IV winner (ex-DeepMind) abandoned deep RL on exactly this problem
  shape: variable unit count, long episodes, dynamic opponent pool.
- Compute is not the constraint. ~1.3 s/episode means CEM's ~10⁴ evaluations
  cost about **$2** on Modal, against the real engine, with zero divergence risk.

So: heuristic policy → CEM parameter search → per-turn forward-sim planner.
**PufferLib is not used and no simulator is rewritten.**

---

## 3. Architecture

```
agent/            the submission (flat imports — Kaggle unpacks it flat)
  main.py         entry point; `agent` defined LAST; robust dir discovery
  policy.py       the brain: tasks -> greedy unit assignment -> market orders
  params.py       Params dataclass + SEARCH_SPACE (41 tunable scalars)
  market.py       price-curve port + sell scheduling
  rules.py        constants mirrored from engine source
  params.json     tuned params, written by CEM (absent = use defaults)

sim/              local evaluation
  harness.py      one definition of "play an episode"
  run.py          single episode + replay dump
  trace.py        per-day X-ray — the main debugging view
  arena.py        holdout matrix, both seats, W&B logging

search/
  cem.py          cross-entropy method over Params
  modal_app.py    fan-out on Modal

obs/wandb_setup.py  one place that decides how we talk to W&B
tests/              engine parity + submission contract + timing
```

### The policy is deliberately stateless

Every turn the whole plan is recomputed from `obs`; nothing is remembered
between turns. It costs ~1 ms against a 1000 ms budget and buys two things:
no possible desync between what we believe and what the engine did, and any
turn is reproducible from its observation alone. On a game with this many
silent no-ops, that is worth far more than the microseconds.

Each turn:

1. **Scan** — what do I own, what does each tile need.
2. **Build tasks** — every job worth doing, each with a priority and an
   optional required item (`FEED` needs wheat *in hand*).
3. **Assign** — greedy: each unit takes its best `priority − distance × penalty`.
   A Hungarian assignment is possible; greedy is within noise and far more
   readable.
4. **Market** — hire, buy land, buy livestock/seed/feed, then sell.

Fetch trips are first-class tasks. Without them, goods bought into the shed
never reach a unit's hands, because a unit only wanders to the shed when it has
nothing else to do — which on a busy farm is never.

---

## 4. Setup

```bash
make setup                      # uv venv (3.12) + pinned deps
```

`kaggle-environments` requires Python ≥3.11; the system `python3` here is
3.10.4, so the venv is explicitly 3.12.

**Kaggle** — accept the rules on the website, then put an API token in
`~/.kaggle/access_token` (`chmod 600`). Verify with `make status`.

**W&B** — `~/.netrc` already authenticates the Python SDK. The MCP server needs
the key as a header, so export it for MCP use:

```bash
export WANDB_API_KEY=...   # .mcp.json references ${WANDB_API_KEY}, never the literal
```

**Modal** — `modal secret create wandb WANDB_API_KEY=...` so workers can log.

---

## 5. Running things

| Command | What it does |
|---|---|
| `make play` | one episode vs `starter`, prints both banks |
| `make trace` | **per-day X-ray** — cash, tiles, shed, prices. Start here when debugging |
| `make arena` | holdout matrix vs frozen opponents, both seats |
| `make arena WANDB=1` | same, logged to W&B |
| `make test` | engine parity + submission contract |
| `make check` | everything, including the timing test — gate before submitting |
| `make search` | small local CEM |
| `make search-modal` | the real CEM, fanned out on Modal |
| `make bundle` | build `submission.tar.gz` (runs `make check` first) |
| `make submit CONFIRM=1 M="..."` | submit — refuses without `CONFIRM=1` |
| `make check-engine` | diff our pinned engine against upstream master |

### Two metrics, two jobs

- **`mean_bank`** — near-deterministic and nearly opponent-independent, so it
  has much lower variance than a win/loss bit. **This is what CEM optimises**,
  and it is why a handful of seeds per candidate is enough.
- **`win_rate`** — what the ladder actually scores. Used only as the final gate
  before a submission, never for tuning; at these sample sizes it is too noisy
  to steer on.

### Observability

Everything logs to W&B project `farmermaxxing`, via `obs/wandb_setup.py`:

- `job_type="arena"` — one run per evaluation sweep, with a per-episode
  `wandb.Table` (opponent, seed, seat, bank, opp_bank, win, status) and summary
  metrics `mean_bank` / `median_bank` / `min_bank` / `stderr` / `win_rate` / `errors`.
- `job_type="cem"` — one run per search, logging per generation:
  `best_bank`, `elite_mean_bank`, `pop_mean_bank`, `best_overall`. The winning
  `params.json` is attached as a versioned artifact.
- Runs are grouped (`--group`) so a search and its evaluations line up.
- `WANDB_MODE=disabled` turns it all off; tests set this automatically.
- Container defaults (`WANDB_CACHE_DIR=/tmp`, `WANDB_DISABLE_GIT`,
  `WANDB_SILENT`) are applied centrally, and runs always open as context
  managers so a preempted Modal container still calls `finish()`.

---

## 6. Experiment log

### 2026-08-05 — project set up, baseline agent working

Scaffold, engine pinned at `kaggle-environments==1.32.4` (verified identical to
upstream master via `make check-engine`).

Baseline built and debugged through `make trace`. Progression on seed 0 vs
`starter`, each step a distinct bug found in the day-by-day view:

| Fix | Final bank |
|---|---|
| first working version | 88 |
| gate livestock purchases on cash + housing | 12,330 |
| cap PLANT tasks at seeds held (atomic-validation deadlock) | 12,330 → real fix |
| reclaim weeds; stop hoarding wheat; cap feed price | **24,895** |

Bugs found, in order of damage:

1. **Bankruptcy spiral.** Bought 9 geese (2,700 of 3,000) on day 1, leaving
   nothing to hire hands. With no hands nothing was watered or fed, so crops
   weeded over and the geese starved. Fixed with `animal_cash_reserve` plus a
   rule never to hold more livestock than there are empty structures.
2. **Plant deadlock** (the atomic-validation trap above) — scored 88 coins.
3. **Weeds never cleared.** `prio_dig` was the lowest priority, so 41 of 75
   tiles ended the season as dead land. Raised to 45.
4. **Shed pinned at its 100 cap** by 47 hoarded wheat + 44 unsold fertilizer, so
   egg production was being discarded. Cut `wheat_reserve_days` to 1.3 and
   dropped the fertilizer sell floor.
5. **Feed cost exceeded egg revenue** after ~day 22: wheat had climbed to $55
   while eggs sagged to $41. Added `wheat_buy_max_price`.

Baseline result — **100% win rate**, mean bank 25,418 ± 921 over 12 episodes
(2 opponents × 3 seeds × 2 seats). ~1.3 s/episode.

Note this only clears the built-in `starter` and `pass` agents, which are a very
low bar (`starter` farms a single carrot tile and never hires). It says the
agent works, not that it is competitive.

### Next

- CEM over the 41 searchable params, then re-gate on a held-out seed set.
- Freeze the tuned agent as an arena opponent so later candidates face something
  real rather than only the built-ins.
- Milestone 6: per-turn forward-sim planner using the unused 1 s/turn.
  **First step is to test the hypothesis** that `kaggle_environments` is
  importable inside the submission sandbox, which would allow exact rollouts.
