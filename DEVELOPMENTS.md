# Kaggriculture, development notes

Working document for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition. Game breakdown, architecture walkthrough, how to run everything,
and a dated log of what we tried.

- **Entry deadline** 2026-09-23 · **Final submission** 2026-09-30
- **Prizes** $50,000, $5,000 each to the **top 10**
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
against the same pre-commit inventory, there is no first-mover advantage.

This is the single most important strategic fact in the project: it makes
Kaggriculture ~95% a **single-agent scheduling problem** and only ~5% a game.
It is why we optimise *mean final bank* rather than win-rate (§5), and why
self-play RL would buy very little (§7).

### Economics

Value density per tile-day, and how the price reacts to flooding the market:

| Resource | Yield/tile/day | Base $ | $/tile/day | Glut behaviour (`above_func`/`target`) |
|---|---|---|---|---|
| Melon | 0.55 | 250 | **137** | `sq` / 3.60, $1 floor at +T. Brutal. |
| Milk | 0.50 | 160 | 80 | `linear` / 1.60, $1 at +T |
| Wool | 0.33 | 200 | 66 | `sq` / 3.20, $1 at +T |
| **Egg** | 1.00 | 50 | **50** | `log` / 0.20, only $50→$40. **Resilient** |
| Strawberry | 0.24 | 120 | 29 | `linear` / 1.60, $1 at +T |
| Carrot | 0.75 | 35 | 26 | `sqrt` / 0.70 |
| Wheat | 0.80 | 25 | 20 | `log` / 0.20, resilient |
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
  feed late-game can be value-destroying, hence `wheat_buy_max_price`.
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

1. **Atomic PLANT validation**, the single worst trap. The engine computes
   ```python
   blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}
   ```
   If more units issue `PLANT WHEAT` in a turn than you hold wheat seeds,
   **every one of them is silently dropped**. With 7 units and 2 seeds the farm
   deadlocks permanently, nothing about the state changes to break the tie.
   Cost us a full run that scored 88 coins. `_build_tasks` now caps plant tasks
   at the seed count.
2. **`__file__` does not exist inside an agent.** `kaggle_environments` loads
   agents via `exec(compile(raw, path), {})`. Touching `__file__` raises
   NameError, which the loader swallows into `InvalidArgument`, the submission
   fails its validation episode. See `_agent_dir()` in `agent/main.py`.
3. **The file loader takes the *last callable defined*,** not the one named
   `agent`. Define `agent` last, import nothing after it.
4. A fresh seed starts at `consecutive_unwatered = 1`, so a seed planted and not
   watered the same day becomes a weed that night. No grace period.
5. Melon caps at age 10; ages 11-12 are dead turns.
6. Wheat/carrot only reach their listed max yield **with fertilizer**.
7. The shed is not a tile. Access tiles at `boardSize=10` are (4,4), (5,4),
   (4,5), (5,5), and three of them are **locked** until you buy land, while
   `PICKUP` no-ops on a locked tile.
8. Sales at the $1 floor do not add market supply, so the floor stays responsive.

### Hard limits

| | |
|---|---|
| `actTimeout` | **1 s/turn** (plus a 60 s bank for the whole episode) |
| `runTimeout` | 1200 s per episode |
| Submission box | 2 vCPU, ~12 GiB RAM |
| Our own ceiling | p99 turn < 50 ms, enforced by `tests/test_contract.py` |

The field uses ~0.005 ms of that 1 s budget and we use ~0.25 ms, i.e. 0.03%.
Per-turn forward simulation is wide-open ground (issue #10).

---

## 2. Why this is not an RL project

The original brief assumed RL + PufferLib. The research said otherwise:

- **Every** RL winner of a `kaggle_environments` competition rewrote the
  simulator first, Lux S1, Lux S3, Orbit Wars 1st (Rust, ~2,400 B200-hours,
  15B steps), Orbit Wars 3rd (JAX). Nobody trains against the Python env.
- `kaggriculture.py` is 1,063 lines of fiddly dict mutation. A port is 2-4 weeks
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
agent/            the submission (flat imports, Kaggle unpacks it flat)
  main.py         entry point; `agent` defined LAST; dir discovery that survives exec()
  policy.py       the brain: tasks -> greedy unit assignment -> market orders
  params.py       Params dataclass (46 fields) + SEARCH_SPACE (60 dims: 36 float, 24 int)
  market.py       price-curve port + sell scheduling
  rules.py        constants mirrored from engine source
  params.json     tuned params, written by CEM (absent = use defaults)

sim/              local evaluation
  harness.py      one definition of "play an episode"
  fastplay.py     the same episode without replay bookkeeping (3.3x), for search
  run.py          single episode + replay dump
  trace.py        per-day X-ray, the main debugging view
  arena.py        holdout matrix, both seats, W&B logging
  census.py       land accounting + per-product sales mix
  gate.py         the promotion gate; reads the CHAMPION file, defaults to POOL=top
  opponents.py    the pool: builtins, frozen snapshots, band/top tapes
  tape.py         a recorded ladder seat, replayable as an opponent
  ladder.py       read the ladder: verify / census / fetch / clone / sync /
                  calibrate / mine
  ledger.py       every submission, its local claim, and what the ladder said
  mix.py          our REALISED tile mix beside a ladder opponent's, per day

search/
  cem.py          cross-entropy method over Params
  archetypes.py   constructed restart portfolios, incl. `metabuild` from a replay
  modal_app.py    fan-out on Modal

obs/wandb_setup.py  one place that decides how we talk to W&B
tests/              validity only: parity, contract, tarball, seats, crash-safety,
                    timing, and the search's own statistics. NO score assertions.

CHAMPION            the snapshot a candidate must beat (moves on a ladder rating)
ledger.json         the durable submission record
AGENTS.md           the operating contract; CLAUDE.md just imports it
```

### The policy is deliberately stateless

Every turn the whole plan is recomputed from `obs`; nothing is remembered
between turns. It costs ~1 ms against a 1000 ms budget and buys two things:
no possible desync between what we believe and what the engine did, and any
turn is reproducible from its observation alone. On a game with this many
silent no-ops, that is worth far more than the microseconds.

Each turn:

1. **Scan**, what do I own, what does each tile need.
2. **Build tasks**, every job worth doing, each with a priority and an
   optional required item (`FEED` needs wheat *in hand*).
3. **Assign**, greedy: each unit takes its best `priority − distance × penalty`.
   A Hungarian assignment is possible; greedy is within noise and far more
   readable.
4. **Market**, hire, buy land, buy livestock/seed/feed, then sell.

Fetch trips are first-class tasks. Without them, goods bought into the shed
never reach a unit's hands, because a unit only wanders to the shed when it has
nothing else to do, which on a busy farm is never.

---

## 4. Setup

```bash
make setup                      # uv venv (3.12) + pinned deps
```

`kaggle-environments` requires Python ≥3.11; the system `python3` here is
3.10.4, so the venv is explicitly 3.12.

**Kaggle**, accept the rules on the website, then put an API token in
`~/.kaggle/access_token` (`chmod 600`). Verify with `make status`.

**W&B**, `~/.netrc` already authenticates the Python SDK. The MCP server needs
the key as a header, so export it for MCP use:

```bash
export WANDB_API_KEY=...   # .mcp.json references ${WANDB_API_KEY}, never the literal
```

**Modal**, `modal secret create wandb WANDB_API_KEY=...` so workers can log.

---

## 5. Running things

| Command | What it does |
|---|---|
| `make play` | one episode vs `starter`, prints both banks |
| `make trace` | **per-day X-ray**, cash, tiles, shed, prices. Start here when debugging |
| `make arena` | holdout matrix vs the full opponent pool, both seats |
| `make arena WANDB=1` | same, logged to W&B |
| `make gate` | promotion gate: four checks, exits non-zero on failure |
| `make promote FROM=runs/.../best_params.json` | move a search result into `agent/params.json` |
| `make freeze NAME=v1-cem` | snapshot the current params into the opponent pool |
| `make test` | engine parity + submission contract |
| `make check` | everything, including the timing test, gate before submitting |
| `make search` | small local CEM |
| `make search-modal` | the real CEM, fanned out on Modal |
| `make bundle` | build `submission.tar.gz` (runs `make check` first) |
| `make submit CONFIRM=1 M="..."` | submit, refuses without `CONFIRM=1` |
| `make check-engine` | diff our pinned engine against upstream master |

### The promotion workflow

A search never writes into the tracked tree. Results land in
`runs/<group>/best_params.json`, which is gitignored, and moving one into the
agent is deliberate:

```
make promote FROM=runs/cem-modal-full-v2/best_params.json
make gate                     # four checks against the frozen pool
make freeze NAME=v2-cem       # only if the gate passed
```

The gate exits non-zero unless all four pass: no errored episodes, mean beats
the champion by more than the combined standard error, the floor does not
regress beyond tolerance, and there is no losing record against any single
opponent in the pool.

The floor check is not decoration. A livestock variant measured mean 51,131
against a champion's 50,588 while its worst seed fell from 48,542 to 40,173.
Rating moves on win and loss, so that trade buys average points at the cost of
matches.

### Two metrics, two jobs

- **`mean_bank`**, near-deterministic and nearly opponent-independent, so it
  has much lower variance than a win/loss bit. **This is what CEM optimises**,
  and it is why a handful of seeds per candidate is enough.
- **`win_rate`**, what the ladder actually scores. Used only as the final gate
  before a submission, never for tuning; at these sample sizes it is too noisy
  to steer on.

### Observability

Everything logs to W&B project `farmermaxxing`, via `obs/wandb_setup.py`:

- `job_type="arena"`, one run per evaluation sweep, with a per-episode
  `wandb.Table` (opponent, seed, seat, bank, opp_bank, win, status) and summary
  metrics `mean_bank` / `median_bank` / `min_bank` / `stderr` / `win_rate` / `errors`.
- `job_type="cem"`, one run per search, logging per generation:
  `train_best_bank`, `train_elite_mean_bank`, `train_pop_mean_bank`,
  `holdout_best_bank`, `holdout_win_rate`, `generalisation_gap`,
  `best_holdout_overall`, `worst_opponent_margin`, and `vs/<label>/*` per
  reference opponent. Run summary adds `clean_bank`, `clean_selection_score`,
  `selection_bias` and `heldout_opponents`. The winning `params.json` is
  attached as a versioned artifact.
- Runs are grouped (`--group`) so a search and its evaluations line up.
- `WANDB_MODE=disabled` turns it all off; tests set this automatically.
- Container defaults (`WANDB_CACHE_DIR=/tmp`, `WANDB_DISABLE_GIT`,
  `WANDB_SILENT`) are applied centrally, and runs always open as context
  managers so a preempted Modal container still calls `finish()`.

---

## 6. Experiment log

### 2026-08-05, project set up, baseline agent working

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
2. **Plant deadlock** (the atomic-validation trap above), scored 88 coins.
3. **Weeds never cleared.** `prio_dig` was the lowest priority, so 41 of 75
   tiles ended the season as dead land. Raised to 45.
4. **Shed pinned at its 100 cap** by 47 hoarded wheat + 44 unsold fertilizer, so
   egg production was being discarded. Cut `wheat_reserve_days` to 1.3 and
   dropped the fertilizer sell floor.
5. **Feed cost exceeded egg revenue** after ~day 22: wheat had climbed to $55
   while eggs sagged to $41. Added `wheat_buy_max_price`.

Baseline result, **100% win rate**, mean bank 25,418 ± 921 over 12 episodes
(2 opponents × 3 seeds × 2 seats). ~1.3 s/episode.

Note this only clears the built-in `starter` and `pass` agents, which are a very
low bar (`starter` farms a single carrot tile and never hires). It says the
agent works, not that it is competitive.

### 2026-08-06, v10: the late multiplier was dead code, and the pool was the wrong pool

Started from the ladder rather than from the search. We sit at **rank 871 of
2,260, rating 866.2**, and the shape of the field matters: the ratings are
bimodal, ~1,300 teams below 1,000 and a long tail to 3,047. Small rating moves
near 866 are worth many ranks (rank 850 is 880.4, rank 900 is 837.3).

**Our banks are strongly anti-correlated with opponent strength.** Over the 46
real episodes our two active submissions played:

| opponent rating | n | our bank | their bank | win rate |
|---|---|---|---|---|
| < 750 | 13 | 102,773 | 64,893 | **100%** |
| 750-850 | 12 | 99,314 | 71,395 | 75% |
| 850-1000 | 15 | 93,806 | 100,240 | 27% |
| 1000+ | 5 | 74,752 | 124,492 | 20% |

Monotone, so not noise. Reading the replays: against a strong opponent milk goes
160 -> 7 and melon 250 -> 31, and our realised milk price falls from $273 to $83.

**The find.** The per-day census against a rank-49 opponent showed our land use
falling to **31% at day 27 (52 of 75 tiles idle)** while they held 100% and
rotated 42 strawberry into 38 wheat. The cause was mechanical: `mix_switch_day`
was **28**, but wheat's own plant cutoff (`first_yield_day` 2 +
`plant_cutoff_slack` 4) blocks planting after day **24**. The late-game
multiplier fired four days after the only crop it could act on had stopped being
plantable. It was dead code.

Setting `mix_switch_day` 16 and the late wheat multiplier to 6 is v10:
mean **+4,500 (11 sigma paired)**, worst seed 48,827 -> 54,012, land use
61.3% -> 69.1%. Submitted.

> **Reconciled 2026-08-17: docs/6 is the correct record and this paragraph is
> not.** Re-run on `--opponents real` over 8 clean seeds and both seats, now
> that the pool actually contains the champion, the delta is **+4,133** against
> docs/6's +4,138, with the worst seed **45,519 -> 50,195** matching docs/6's
> table exactly and land use 61.0% -> 69.0% matching its 61.3% -> 69.1%. The
> figures quoted above -- 48,827 -> 54,012 at 11 sigma -- reproduce on no pool
> and should be read as a transcription error.

**Five refutations, all measured against the ladder tapes and the frozen pool:**

1. *Shift melon to strawberry/wool.* The meta sells 308 strawberry and 212 wool
   and we sell 131 and 0, and melon has the most brutal glut curve in the game
   (`sq`/3.60). Cutting melon costs **-49,283 margin**; strawberry is already
   pinned at its bound. Melon is our main earner and the probe was wrong.
2. *We are labour-limited.* The meta fields 13-14 hands from day 15, we field 9,
   and `hands_late` 12/15/18 all scored identically because `hire_turns` 1 caps
   the farm at MAX_MARKET_ORDERS. Raising both together reaches 16 hands and
   **loses 37k of margin** -- the hires crowd out the turn's SELL orders and the
   fib cost compounds.
3. *A separate cutoff for the 2-day crops holds the endgame land.* Implemented
   `plant_cutoff_slack_fast`; worse on both pools at every value (**-1,001 to
   -2,500 margin**). The idle tail really is correct. Reverted.
4. *A CEM run warm-started on v10 improves it.* 14 generations x 160, ranked on
   margin against tapes + frozen: nine of fourteen generations rejected, and the
   winner fails the gate (floor 54,012 -> 46,757, win 30.2% -> 15.6%).
5. *Higher-bank archetypes are a better hedge.* `target_wheat_tiles` 9 gains
   11,500 of bank in a coordinate sweep and loses **-3.31 sigma of margin**;
   wheat 5 loses -2.67 sigma. Bank and margin come apart exactly as in docs/5.

**A better instrument: the band pool.** All three existing pools sit in the
wrong place. Our frozen lineage banks 60-75k, the meta tapes bank ~190k, and the
opponents Kaggle actually matches us with -- rated 840-1050 -- bank 100-123k.
`sim.ladder clone` now holds six tapes cut from the opponent seats of six real
episodes we played today, all verified non-degenerate on unseen seeds.

**And two cautions about it.**

*Sample size.* On 6 seeds x 1 seat the band pool said v11 beat v10 by +4,274
margin and 42% vs 36% wins, which read as the gate mismeasuring. On 14 seeds x
2 seats, paired, that collapses to **+394 at 0.20 sigma with wins 2.4%
*worse***. The gate was right. A new instrument that immediately overturns an
old verdict deserves the larger sample before it is believed.

*Opponent overfitting.* A CEM run trained on four of the six band tapes (v12)
looked like the best candidate of the day, at +3,812 margin and +9.5% wins
overall. Split by opponent it is **+10,188 (3.96 sigma) on the four it trained
against and -8,940 (-1.89 sigma) on the two held out**, and on the independent
`real` pool it drops to 18.8% against v5 and 31.2% against v8. A four-opponent
training pool is small enough to memorise. Hold opponents out, not just seeds.

**Submitted, two of three slots:**

| | agent | band pool wins | margin | worst | why |
|---|---|---|---|---|---|
| 1 | **v10** | **37.5%** | -3,215 | 53,042 | the champion; +4,138 at 13 sigma over v8 |
| 2 | **v11** | 35.1% | -2,821 | 43,681 | diverse hedge, replaces v8 (25.6%, -14.15 sigma) |

v11 is a statistical tie with v10 but a genuinely different basin -- 2 crops per
turn, no late melon, premium sell floors raised (melon 0.49 -> 0.75) so it drips
rather than dumps, rival-aware selling on. The second slot previously held v8,
which the band pool puts 14 sigma behind v10, so the swap is a strict upgrade
even though v11 does not beat the champion. Third slot deliberately unused: no
remaining candidate measured better than v10 on any pool.

**Where the gap actually is.** After v10 we bank ~88k against the meta tapes and
they bank ~140k, and we win 0% of those. Every coordinate around v10 is at a
local optimum -- the best single-parameter move in a 65-variant sweep was
+2,661, and two CEM runs from that basin produced nothing promotable. Parameter
search on this policy looks exhausted. The remaining gap is structural: the meta
holds 100% land use with 0 weeds from day 15 to day 27 on the same 75 tiles
where we manage 69%, and closing that is an intra-day scheduling problem
(#78, #30), not a tuning one.

### 2026-08-17, what the ladder said, eleven days later

Nothing was submitted between 2026-08-06 and this entry: **eleven idle days, or
about 55 unused submissions.** The gap matters because the ladder is the only
ground truth we have, and it was free.

**The v10 promotion cost 108 rating points.**

| version | submission | local claim | pool it was measured on | ladder |
|---|---|---|---|---|
| v2 | 55255699 | 83,586 holdout bank | `starter` | 667.6 |
| v3 | 55257006 | 109,606 clean bank | `starter` | **645.3** |
| v4 | 55261512 | 137,686 clean bank | `starter` | 774.7 |
| v5 | 55287313 | 90,842 clean bank | mixture | 832.3 |
| v8 | 55290443 | worst seed 52,764, 100% vs frozen | `frozen` | **853.9** |
| v10 | 55294423 | **+4,138 at 13 sigma** | `real` | **746.1** |
| v11 | 55294870 | -2,821 margin, 35.1% wins | `band` | 730.2 |

The rank that follows from the two currently-scored slots is **2455 of 4883**, at
a score of 746.1 against a median of 749 and a top-ten cutoff of ~2963. Every one
of these numbers had to be recovered by re-querying Kaggle, which is why
`ledger.json` and `sim/ledger.py` now exist.

**The obvious reading of that table is wrong, and the correct one is worse.**
The obvious reading is "the local gate lied about v10". It did not. Re-run on a
`resolve_pool` that is what it claims to be, v10 beats v8 uniformly:

| pool | v10 mean | v8 mean | v10 win | v8 win | v10 beats v8 on margin |
|---|---|---|---|---|---|
| `top`, 4 clean seeds | 87,625 | 81,726 | **0.0%** | **0.0%** | all 4 opponents |
| `band`, 6 clean seeds | 90,260 | 86,244 | 36.1% | 29.2% | all 6 opponents |

Bank, floor, win rate, and per-opponent margin all favour v10, on both pools,
without exception. The ladder column that says otherwise is **confounded by
convergence time**: a submission restarts at 600 and climbs, and v8 held a slot
for 2h45m — perhaps fifteen episodes, still rising — while v10 has had eleven
days to settle. Comparing a partially-climbed rating against a converged one is
the error. v8's 853.9 is an artifact until it is re-submitted and left alone,
which costs one slot and 36 hours and is worth doing precisely because this
belief would otherwise misdirect every comparison downstream of it.

So the honest lesson is not "distrust the gate". It is this: **both agents win
0% against the top band, at a margin of about -53,000, and the entire six-version
lineage has been trading bank deltas of ±5,000.** A ladder scores wins. No
sequence of ±5,000 improvements flips a match you are losing by 53,000, which is
why five promotions in a row moved the rating by nothing much in either
direction. The gate was measuring a real quantity that cannot reach the goal.

*A broken pool underneath the broken promotion.* `resolve_pool("real")` takes
`frozen_names()[-2:]`, and `frozen_names()` sorts **alphabetically**, so
`v10-wheatfix` sorts second and the reigning champion was never in the pool.
Every `--opponents real` measurement from 2026-08-06 onward, including v10's own
13-sigma gate, ran against v5 and v8 standing in for "recent".

**The band pool is well calibrated; it was the wrong target.** Against the six
`band-*` tapes the champion goes **3W-3L**, which is exactly a 746 rating. The
instrument is not lying. But it was used as the optimisation target, and a 50%
win rate against a 900-rated band has a ceiling of about 900. Nothing has ever
been optimised against the 3200 band that the prizes are in.

**The gap to the top is sales volume, and it is 5x.** Champion vs the four
`meta-*` tapes, `fast_play(metrics=True)`, at each tape's own seed:

| | our bank | their bank | our land use | theirs | our units sold | theirs | our products | theirs |
|---|---|---|---|---|---|---|---|---|
| meta-a | 92,943 | 144,814 | 0.65 | 0.80 | 839 | **4,427** | 5 | 9 |
| meta-b | 58,091 | 121,692 | 0.70 | 0.80 | 850 | **4,348** | 5 | 9 |
| meta-c | 75,133 | 128,132 | 0.69 | 0.79 | 872 | **4,348** | 5 | 9 |
| meta-e | 65,475 | 120,400 | 0.68 | 0.80 | 848 | **3,547** | 5 | 9 |

0W-4L. Land use differs by 1.2x. **Volume differs by 5x**, and per-unit realised
price runs the other way: ~$111/unit for us against ~$33 for them. They win on
throughput, not on price.

`agent/params.json` says why, arithmetically:

```
target_melon_tiles 17   target_strawberry_tiles 12   target_wheat_tiles 2
target_carrot_tiles 0   target_tomato_tiles 0        target_geese 0
sell_floor_frac: MELON 0.486  MILK 0.314  FERTILIZER 0.323   <- dumped
                 CARROT 1.237 STRAWBERRY 1.279 WOOL 1.001    <- hoarded
```

17x0.55 + 12x0.24 + 2x0.80 = **13.8 units/tile-day/quadrant**, which is ~850
units a season and matches the census exactly. We grow the two lowest-yield
crops, on the two most punishing glut curves (melon `sq`/3.60 to a $1 floor,
strawberry `linear`), dump those below half base price, and hoard the resilient
ones. **Zero geese**, when egg is 1.00/tile/day on a `log` curve that only moves
$50 to $40 and `CARE` doubles goose output.

This is issue #48's measurement, unacted on for twelve days: 7 of 9 products end
*above* base price with inventory below I0, we sell 3 of 9, and the only two
discounted products are melon and fertilizer — the two we concentrate on. It
also reframes the most expensive decision in the project: `target_wheat_tiles: 9`
gave the **highest bank in a 65-variant sweep (+11,500)** and was rejected at
-3.31 sigma on margin measured against a pool of our own lineage.

So the standing conclusion from docs/6 — "parameter search on this policy is
exhausted, the rest is intra-day scheduling" — is **half right**. Scheduling is
worth ~1.2x of land use. The portfolio is worth ~5x of volume, and it was never
searched, because every bound and every judge pointed the other way.

**Four silent bugs, all in the instruments.** None raised; all produced complete
episodes with plausible numbers.

1. `resolve_pool("real")` took `frozen_names()[-2:]` and `frozen_names()` sorted
   **lexicographically**, so `v10-wheatfix` sorted second and the reigning
   champion was never in the pool. Every `--opponents real` run after v10 was
   frozen -- including v10's own 13-sigma gate -- ran against v5 and v8.
2. `make gate` with no `CHAMPION=` compared against **`Params()` dataclass
   defaults**, the hand-tuned baseline that banks 24,895. Every candidate passes
   that. `sim.gate` now reads a tracked `CHAMPION` file and refuses to guess.
3. **`TapeAgent` was a no-op on the slow path.** `kaggle_environments` truncates
   agent arguments with `agent.__code__.co_argcount`; a class instance has no
   `__code__`, so `__call__(obs)` was invoked with two arguments, the TypeError
   was swallowed by `Agent.act`'s own `except Exception` into a no-op action, and
   the seat still finished `DONE`. On seed 20000 vs meta-a, `fast_play` gave
   `[105,504, 151,737]` and `harness.play` gave `[114,521, 3,000]`.
4. **`submission.tar.gz` shipped parameters that were not `agent/params.json`.**
   Nothing had ever validated the archive. Finding it *recovered v11's
   parameters*, which the project had written off as unrecoverable, and they are
   now frozen as `v11-hedge`.

**Two refutations of our own new hypotheses**, both worth recording because both
looked obvious:

- *SELL is starved by the 10-order market budget.* No. `sell_order_floor` swept
  0..6 makes units sold **fall** (858 -> 645) and bank fall with it; the slots
  come out of BUY_SEED and HIRE and cost more production than the extra SELL
  orders move. We end a season with ~9 unsold units, so we already sell nearly
  everything we grow. The gap is **production**, not order slots, not sale
  scheduling, and not the endgame.
- *Transcribing the meta's tile counts is enough.* No. `search/archetypes.py
  metabuild` sets exactly the build above and reaches **30% land use**, the worst
  of the five archetypes (`diversified` 40.5%, `premium` 45.4%, `staples` 45.0%,
  `livestock` 43.5%). Our policy cannot service 40 strawberry tiles. Whatever
  closes the gap is upstream of the portfolio.

**The pool is now current.** `make refresh-tapes DATE=2026-08-16` mints tapes from
the day's episode dataset, keeping only seats that banked >=110,000 and survive
three unseen seeds. Four came through, including the present leader (カワシギ,
3228.5) and Utkarsh #2; they bank 161,000-187,000 on fresh seeds. Against the
eight-tape `top` pool over 4 clean seeds and both seats, v10 wins **0 of 8** at
margins of -52,658 to **-83,112**, and the mix is:

```
       units   WHEA  CARR  TOMA  STRA  MELO   EGG  MILK  WOOL  FERT
us       865    131     0     0   107   157     0   302     0   167
them   3,748  1,176    10     2   584   188     4   564   351   869
```

Wheat 9x, fertilizer 5x, strawberry 5x, milk 2x, wool from zero. Fertilizer is
the cheapest of those to attack: every animal yields one free per day regardless
of care, it bases at $100, and `COLLECT_FERTILIZER` is a single action.

So the live question is narrower than it was this morning: **we grow ~865 units a
season on 69% of our land and they grow ~3,750 on 80% of theirs, with the same
number of plant actions per day.** That is roughly 4x the output per productive
tile-day, and it is not order slots, not sale scheduling and not the target
vector. The remaining candidates are yield per harvest -- fertilizer coverage,
watering discipline, and crop cycle length -- and intra-day scheduling (#30,
#40). #40 in particular is now the highest-value open issue: nobody has ever
counted where the ~200 unit-actions a day actually go, and every invalid action
in this engine is a silent no-op.

### 2026-08-17, v12: the first champion selected against the top of the ladder

CEM, population 96 x 40 generations, against the eight-tape `top` pool with **two
opponents held out**, ranked on **margin**, warm-started from v10 at spread 0.22.
Submission **55577866**. ~246,000 episodes on Modal.

**The bound widening was load-bearing.** `target_strawberry_tiles` had a ceiling
of 12 with v10 pinned exactly on it; raised to 20, the search moved to **15**
within the run. Alongside it: `plant_crops_per_turn` 1 -> 3, `target_geese` 0 ->
1 (the first geese any champion has run), `target_cows` 4 -> 5, and sell floors
down almost across the board -- **WOOL 1.001 -> 0.827**, so wool can trade at all
for the first time; MELON 0.486 -> 0.365; WHEAT 0.366 -> 0.266; MILK 0.314 ->
0.226. `rival_supply_urgency` 0.0 -> 0.198. Hands 4/8/9 -> 5/10/10.

Gated against v10 on 8 clean seeds and both seats, paired:

| | top pool (8 opp) | band pool (6 opp) |
|---|---|---|
| mean bank | 83,438 vs 83,227 (**+211**, 1 sigma 1,495) | 92,338 vs 92,037 (**+301**, 1 sigma 2,191) |
| **win rate** | 0.0% vs 0.0% | **43.8% vs 33.3%** |
| worst seed | **56,912 vs 43,915** | 54,415 vs 50,195 |
| held-out opponents | **+6,394 over 32 cells, 1 sigma 3,206** | n/a |
| products sold | **6 vs 5** (eggs, 99 units) | 6 vs 5 |

**The gate failed, and was overridden.** Two of five checks failed. "Mean beats
champion by >1 sigma" failed at +211 -- flat bank, which is deliberately not what
this was selected on, and the same quantity that gained +4,133 for v10 while
costing 108 rating points. "Beats every opponent in the pool" failed because we
win 0% against the top band, which is unachievable for anything we can currently
build; that check is calibrated for a pool of our own lineage. Recorded here
rather than quietly relaxed, as v8's override was.

What *did* move is the ladder-relevant part: **+10.5 points of win rate on the
band pool** we are actually matched into, a **30% higher floor** on the top pool,
better margin against all four current top tapes (+4,005 to +14,263) and five of
six band tapes, and the new held-out-opponent check passing at 2 sigma -- so this
is generalisation, not memorisation of the six it trained on.

Two honest caveats. It is **worse against the four stale `meta-*` tapes** (-1,262
to -1,642 each), which date from 2026-08-05 when the leader was 3047. And the
shop-unlock confound is high: omega^2 0.52 against the champion's 0.43, so a large
share of the bank spread tracks which shop unlocked first. The partial answer is
that the win-rate and floor gains reproduce across two pools with disjoint
opponents, which is the reproduction that warning asks for; the bank gain does not
reproduce and is not claimed.

Submitting evicted v10 (741.7) from the scored pair, leaving v11 (730.2) as the
floor until v12 converges. Standing at submission time: **rank 2478 of 4915, score
741.7**, median 748.5, top-10 cutoff 2957.0.

### Next

The loop, not the list: **measure against the 3200 band, submit daily, record
what the ladder says.** See `AGENTS.md` for the operating rules.

- **Read v12's rating after ~36h** (`make ladder-sync`). If it converges above
  741.7, move `CHAMPION` to `v12-topband` -- and note that this will be the first
  promotion in the project's history decided by the ladder rather than by a local
  gate. If it lands below, the band-pool win rate does not predict rating either,
  and `make calibrate` will have earned its keep twice.
- **Fix the gate's fourth check.** "No losing record against any single opponent"
  cannot pass against a pool we lose to 100% of the time, so on `POOL=top` it
  fails identically for every candidate and carries no information. It should
  compare win rate *against the champion's* per-opponent win rate, not against
  0.5.

- **Slot discipline.** One anchor we believe in, one challenger per day. A fresh
  submission restarts at 600 and needs ~24-48h to converge, so a rating younger
  than that is not evidence. Every submission gets a `ledger.json` row the same
  day.
- **Throughput rebuild** (the only workstream that can move the score): mint a
  top-band tape pool from the current top 20, widen the bounds that cap volume
  (`target_wheat_tiles`, `target_carrot_tiles`, `target_geese`,
  `target_tomato_tiles`, `sell_floor_frac`), add a `volume` archetype, reserve
  SELL slots in the 10-order market budget, then search from there against the
  top band with opponents held out.
- **Calibration as a first-class instrument.** `make calibrate` joins ledger
  local-claims against realised ratings. If the gate does not predict the
  ladder, the gate criterion changes — that is the point of measuring it.
- Deferred, and still open: the search-methodology backlog (#65/#67/#68/#69) and
  Milestone 6, the per-turn forward-sim planner using the unused 1 s/turn (#10).
  We use 0.03% of the per-turn budget. **Its first step is still a probe
  submission** testing whether `kaggle_environments` imports inside the
  submission sandbox, which would allow exact rollouts.
