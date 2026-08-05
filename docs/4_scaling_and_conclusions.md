# 4. Scaling the search, and where the project stands

**Date:** 2026-08-05
**Milestone:** infrastructure scaling, multi-restart sweep, v4 promoted and submitted
**Status:** all background work stopped; three submissions made; 19 issues open

This is the closing document for the session. It records what was built, what
the numbers actually mean, the mistakes that were instructive, and what someone
picking this up next should do first.

---

## Where the agent ended up

| Version | Clean bank vs `starter` | Ladder rating | Notes |
|---|---|---|---|
| hand-tuned baseline | 24,895 | not submitted | iteration 1 |
| v2 | 83,586 (selection) | 667.6 | first submission |
| v3 | 109,606 | 745.5 | beat v2 on the ladder on equal episodes |
| **v4** | **137,684** | 600 (new) | submitted, validating |

v4 is a **5.5x** improvement on the hand-tuned baseline, and **+26%** on the
previously deployed v3.

The v2 versus v3 comparison is the cleanest external evidence the project has.
Both played 14 episodes; v3 finished at 745.5 and v2 at 667.6. Local evaluation
predicted v3 would win 91.7% head to head, and the ladder agreed in direction.
That is one of very few times a local prediction was confirmed by an independent
measurement rather than by another local measurement.

---

## The scaling work

Modal caps this account at **100 concurrent containers** and it cannot be
raised, so search throughput had to come from somewhere else. Four changes, each
measured rather than assumed:

| Change | Effect |
|---|---|
| Hold the Modal app open across a search | 15 min/generation to ~2.5 min |
| Fan out per episode, not per candidate | removed a ~34s/generation floor |
| Warm container pool (`min_containers`, `@modal.enter()`) | removed cold start at all 40 barriers |
| Call the engine `interpreter()` directly (`fast_play`) | 1,455ms to 448ms per episode |
| 8 worker processes per container | 5.82x on the same container cap |

End to end: a generation went from **150s at population 64** to **15s at
population 256**, which is 2.34 to 0.059 seconds per candidate, about **40x**.

Two of those deserve the detail.

**`fast_play` (3.3x).** Profiling one episode showed 30.7M function calls
dominated by `copy.deepcopy` (~2.4s cumulative) and `structify` (~1.6s). None of
it is game logic: `core.py` deep-copies the whole state every step to build
replay history, re-validates actions against a JSON schema, re-structifies, and
wraps each step in stdout capture. Scoring a candidate needs none of that.

The important design choice was calling the engine's **own** `interpreter()`
rather than writing a simulator. A rewrite carries silent divergence risk: if
our copy of the economy drifts, the search optimises a fiction and the only
symptom is a slowly sinking ladder rank weeks later. Reusing the interpreter
keeps the logic byte-identical to the code that scores the ladder, and parity
against `env.run()` is asserted in tests rather than claimed.

**Cores per container (5.8x).** The cap counts containers, not cores, so eight
worker processes inside one container multiply throughput without needing a
limit increase. It must be processes rather than threads because the workload is
pure Python and GIL-bound. Benchmarked:

| cpu | eps/sec | speedup | results identical |
|---|---|---|---|
| 1 | 0.87 | 1.00x | yes |
| 4 | 2.96 | 3.39x | yes |
| 8 | **5.09** | **5.82x** | yes |
| 16 | 5.11 | 5.84x | yes |

It plateaus flat at 8, so the host offers about 8 usable cores and `cpu=16`
doubles the bill for 0.4% more work.

This is a **wall-clock win, not a cost win**: `cpu=8` bills 8x per
container-second for 5.8x the work, so cost per episode rises about 38%. The
full sweep is roughly $13 either way and wall clock is the constraint that
cannot be bought around, so the trade is right, but it should be known rather
than discovered on a bill.

**What did not work.** Running six searches in parallel was actively
counterproductive: they split the same 100 containers six ways and none
completed a generation for ten minutes. Searches are sequential now, each with
the whole pool.

---

## The multi-restart sweep

Five searches, 30 generations, population 384, 8 seeds each, from deliberately
different starting portfolios. Clean-seed results with selection bias measured
per run:

| Run | Selection | **Clean** | Clean worst | Bias |
|---|---|---|---|---|
| champion | 144,542 | **137,684** | 87,991 | 5.0% |
| livestock | 145,220 | 136,974 | 94,388 | 6.0% |
| premium | 145,324 | 132,152 | **104,169** | 10.0% |
| diversified | 141,582 | 129,383 | 100,542 | 9.4% |
| staples | 142,612 | 120,656 | 69,064 | 18.2% |
| deployed v3 | 125,911 | 109,606 | 54,371 | 14.9% |

**Selection seeds ranked `premium` first; clean seeds rank it third.** That
reordering is the whole justification for the three-set split: train fits,
holdout selects, clean reports once. Without it we would have promoted the wrong
run and reported a number inflated by 10%.

All five basins converged into a 120k to 138k band, which is itself informative:
the landscape has a broad high plateau rather than one narrow peak, and the
starting portfolio matters less after 30 generations than it did after 12.

---

## What the numbers actually mean

This is the part worth carrying forward, because local evaluation has been
optimistic in three compounding ways, and each was found by measurement rather
than suspicion.

**1. Weak opponents (~25%).** Ladder replays showed real opponents banking 54k
to 62k while local evaluation claimed 83k for the same agent.

**2. Selection bias (5% to 18%).** Holdout was used to pick the champion once
per generation across many generations, which makes it a selection set. Now
measured per run rather than argued about.

**3. Market coupling (up to 3x).** The largest and last-found. The gate now
reports the full spread for one agent:

| Opponent | Our bank |
|---|---|
| starter | 141,397 |
| random | 139,292 |
| pass | 124,334 |
| v2-cem | 92,961 |
| v1-warmbase | 80,138 |
| **v3-fixed** | **46,454** |

Same agent, same seeds, a **3x** range depending only on who is on the other
side. Farms are independent, so all of that difference is the shared market.
Local numbers against weak opponents do not merely overstate: they describe a
different economy.

---

## The market finding

End of season, market inventory relative to the I0 equilibrium:

| Product | End price | vs base | We sold | Inventory vs I0 |
|---|---|---|---|---|
| WHEAT | 55 | **220%** | **0** | -927 |
| STRAWBERRY | 321 | **268%** | 16 | -574 |
| EGG | 70 | 140% | **0** | -338 |
| WOOL | 252 | 126% | **0** | -428 |
| CARROT | 42 | 120% | **0** | -410 |
| MELON | 202 | 81% | 209 | +69 |
| FERTILIZER | 42 | 42% | 292 | +292 |

**The market is starved, not saturated.** Town demand drains inventory faster
than two players supply it, so seven of nine products end above base. The
champion at the time produced three of them. The only two it discounted were the
two it concentrated on.

This overturned the working assumption that the market was the limit and that
concentrating on high value-density crops was correct, and it explains why
several plausible experiments moved nothing: raising crop targets 2.5x, freeing
labour through routing, and adding hands all pushed on axes that were not
binding.

---

## Mistakes worth recording

Four had real teeth, and all four produced *plausible numbers* rather than
errors, which is the dangerous shape.

**The routing hypothesis was wrong, and confidently so.** 60.7% of actions were
MOVE, so cutting travel looked like it would double productive work. Three
implementations, all worse. The best of them did exactly what it was designed to
do, cutting MOVE to 56.1% and lifting productive work to 43.2%, and lost 22k of
bank. The premise was checkable beforehand: 1.65 moves per productive action is
near-optimal for scattered tiles, only 33% of capacity did work, and
`distance_penalty` was already tuned to 13.0 of 15, meaning CEM had solved
locality already. 60.7% MOVE was slack, not waste.

**Fixing bugs made the score worse, correctly.** Repairing the crop-starvation
and hiring-cap bugs dropped the then-champion from 82,759 to 66,620, because its
parameters had been tuned against the bugs and encoded them. `target_carrot_tiles`
was free to be 13 precisely because it never took effect. Keeping the bugs to
protect the number would have made every future search optimise a fiction. The
re-search recovered past the old value by generation 2.

**`fast_play` silently disabled every frozen opponent.** It resolved string
opponents but passed a `Params` instance through uncalled, so each call raised
and the except clause turned it into a no-op. Found by noticing the gate
reported the candidate banking *exactly* 124,334 against four different
opponents. Four opponents cannot give identical results. Searches were
unaffected, since they pass a string; gate comparisons were wrong. The test
suite covered only string opponents, which is exactly why the other path broke.

**Measuring against a dirty tree.** Two archetype scoring runs were taken with
reverted routing code still present, because `git checkout main` carries
uncommitted changes across rather than discarding them. The champion read 38,326
instead of 115,402. Caught because the number was implausible, not because
anything warned.

The common thread: every one of these was found by an inconsistency in a number,
not by a failure. The tooling that made that possible (the per-day `trace` view,
the per-opponent gate breakdown, parity assertions, the score-pin test) earned
its cost several times over.

---

## What exists now

```
agent/     the submission: stateless policy, engine constants mirrored from source
sim/       harness, fastplay, per-day trace, arena, promotion gate, opponent pool
search/    CEM with train/holdout/clean splits, archetypes, Modal fan-out, benchmarks
obs/       shared W&B setup with signal handling
tests/     44 tests: engine parity, submission contract, timing, flat layout,
           fast_play parity, gate decision rule, score regression
docs/      1 setup and baseline, 2 search and gating, 3 market starvation, 4 this
```

Promotion workflow, which is now enforced rather than remembered:

```
make promote FROM=runs/<group>/best_params.json
make gate                     # four checks, exits non-zero on failure
make freeze NAME=v5-...       # only if the gate passed
make submit CONFIRM=1 M="..."
```

The gate checks: no errored episodes, mean beats the champion by more than the
combined standard error, the floor does not regress beyond tolerance, and no
losing record against any single opponent. The floor check exists because a
livestock variant once scored mean 51,131 against a champion's 50,588 while its
worst seed fell from 48,542 to 40,173, and rating moves on win and loss.

---

## What to do next

Ranked by expected value, with reasoning rather than enthusiasm.

**1. Hedge the second submission slot (#45).** v3 and v4 are the same lineage,
so the two active slots are two correlated samples of one strategy. `premium`
scored 132,152 clean with a much better floor (104,169 against v4's 87,991).
Given the final standings come from a single Bradley-Terry tournament, a
deliberately different archetype in the second slot is worth more than a near
duplicate. Cheap: the parameters already exist.

**2. Expand what the policy can express, before improving the optimiser.** The
sweep showed all five basins converging into the same band, which suggests the
parameterisation is the ceiling rather than the search. The agent currently
cannot vary its crop mix over time (targets are static for the whole season),
cannot react to the opponent (whose farm is fully public and entirely unused),
cannot time sales beyond a fixed price floor, and does not plan a day. Each of
those is a dimension no optimiser can search because it does not exist.

**3. Endgame liquidation solver (#43).** Unsold stock scores zero and premium
goods floor hard, so dumping into the last two days is measurably wrong. The
price curve is closed-form and the remaining turns are known, so this is a small
exact optimisation rather than a search. It costs market orders, not unit
actions, so it competes with nothing.

**4. Opponent-aware selling (#42, #11).** The 3x market-coupling spread is the
strongest argument in the project for this. Their maturing crops are visible;
selling ahead of their supply is the difference between base price and the $1
floor on melon.

**5. Only then, better search.** Successive halving on seeds is nearly free and
multiplies candidate throughput 3-4x. CMA-ES would fix the specific blindness
that made diversification unreachable, since CEM keeps per-dimension variance
with no covariance and diversifying requires correlated moves. MAP-Elites suits
the multi-basin structure and would produce the diverse opponent pool the
Bradley-Terry endgame rewards as a by-product.

RL remains the wrong tool for unit control, for the reasons in `docs/1`. The one
place a learned component could genuinely win is the **sell policy**, which is
low-dimensional and sequential under an uncertain price process. That is a much
better target than unit control, and `fast_play` has now made the sample cost
tractable.

---

## Honest summary

The agent improved 5.5x over the session and every promotion cleared a gate that
could have rejected it, twice did. The infrastructure is about 40x faster and
the measurement discipline is considerably better than it started: three
independent sources of optimism in local evaluation were found and quantified
rather than assumed away.

What remains unresolved is the gap to the leader. First place banks over 170k;
v4 banks 137,684 on clean seeds against `starter` and 46,454 against our own
strongest frozen opponent. Which of those numbers is comparable to 170k is not
known, and finding out matters more than another search: if the leader's figure
is also measured against a weak opponent, the real gap is smaller than it looks;
if it is against strong opposition, it is much larger. The ladder is the only
instrument that can settle it, which is an argument for reading replays
(#41) before running more searches.
