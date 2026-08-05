# 2. Parameter search on Modal, and learning to trust the result

**Date:** 2026-08-05
**Milestone covered:** 5 of the plan (cross entropy search), plus the evaluation
machinery that makes its output trustworthy
**Status:** search complete, champion frozen as `v2-cem`, gate passed, submitted 2026-08-05

---

## What this milestone set out to do

Take the working baseline from iteration 1 and tune its 41 parameters by cross
entropy search against the real engine, fanned out on Modal, with every run
tracked in W&B.

The larger goal was not really the number. It was to be able to believe the
number. Iteration 1 ended with a 100% win rate against opponents that farm a
single carrot tile, which proves the agent runs and nothing else.

---

## The methodology fix that had to come first

The original CEM selected its best candidate using the same seeds it optimised
on. With 41 free parameters and six seeds, that value measures fit, not
generalisation.

This mattered specifically because the next step was a much larger search. A
bigger search fits harder, so without a split the reported improvement and the
amount of overfitting would have moved together and been impossible to separate.
The final leaderboard is a single Bradley-Terry tournament over episodes nobody
has seen, which is exactly the setting where overfit parameters come apart.

The fix (issue #7, PR #12):

- train and holdout seed sets are disjoint, holdout offset by 10,000
- the population is scored on train, then the elites are re-scored on holdout
- the generation champion is chosen on **holdout**
- `generalisation_gap` (train best minus holdout best) is logged every generation

Only the elites get re-scored, so the added cost is about 25% of a generation.

That last metric earned its place during the run. See "the gen 4 scare" below.

---

## Search results

Warm-started from a 50,091 holdout champion. Population 64, 6 train seeds, 8
holdout seeds, both seats, on Modal.

| Generation | Train best | Holdout best | Gap |
|---|---|---|---|
| hand-tuned baseline | | 24,895 | |
| 0 | 49,683 | 51,444 | -1,760 |
| 1 | 59,346 | 60,245 | -900 |
| 2 | 61,801 | 62,161 | -360 |
| 3 | 63,062 | 64,305 | -1,243 |
| 4 | 69,391 | 64,570 | +4,821 |
| 5 | 69,255 | 68,689 | +567 |
| 6 | 69,769 | 69,822 | -52 |
| 7 | 76,659 | 75,398 | +1,261 |
| 8 | 73,769 | 72,523 | +1,246 |
| 9 | 79,655 | 76,938 | +2,717 |
| 10 | 77,374 | 80,467 | -3,093 |
| 11 | 82,257 | 80,187 | +2,070 |
| 12 | 82,336 | 79,489 | +2,848 |
| 13 | 83,849 | 82,652 | +1,197 |
| 14 | 83,787 | **83,586** | +201 |
| 15 | 83,633 | 81,934 | +1,699 |

Final: **83,586 holdout**, 3.4x the hand-tuned baseline of 24,895, in about 40
minutes of wall clock and roughly $2 of Modal CPU.

Compute was never the constraint here; wall clock was, which is what made the
app lifecycle bug below worth fixing rather than tolerating.

### Two false alarms, and why watching beat reacting

For four generations the gap sat negative, meaning holdout scored *above* train.
At generation 4 it flipped to +4,821 while holdout barely moved, 64,305 to
64,570. That is the shape of overfitting: train pulling away while unseen seeds
stall. Generation 5 closed the gap to +567 and holdout jumped to 68,689, so it
was noise.

It happened again, and more convincingly, at generations 10 to 12. Train climbed
77,374 to 82,257 to 82,336 while holdout went 80,467 to 80,187 to 79,489 and the
gap widened to +2,848. Three consecutive generations of the textbook pattern.
Stopping there was the obvious call.

Generations 13 and 14 then produced the two best holdout scores of the entire
run, 82,652 and 83,586, with the gap narrowing to +201. The plateau was a
plateau, not a ceiling.

The lesson is not that the gap metric is useless. It is that a three-generation
trend in a stochastic search is still a small sample, and the cost of being
wrong was asymmetric: letting it finish cost about eight minutes, and stopping
early would have cost the best result. Selection is on holdout throughout, so
an overfit tail can never promote a worse agent, which is what made waiting
cheap.

---

## What the search found that contradicted the plan

Iteration 1 opened with geese as the best steady earner. The reasoning: `CARE`
doubles output to 2 eggs per day, every animal drops a free fertilizer worth
about $100, and egg has a flat glut curve (`log`, 0.20) so the price only sags
from $50 to $40 under heavy selling.

The search set `target_geese: 0`.

A direct ablation on the champion parameters, holdout seeds, confirms it is not
a search artefact:

| Variant | Mean bank | Worst |
|---|---|---|
| champion (geese = 0) | 50,588 | 48,542 |
| +8 geese | 20,534 | 16,362 |
| +8 geese, care off | 15,958 | 13,757 |
| +6 cows | 51,131 | 40,173 |
| +6 sheep | 26,800 | 18,825 |

Eight geese cost 60% of the bank. What the original reasoning missed is that the
per-animal revenue was never the binding question. Each animal needs about four
actions per day (feed, care, harvest, collect fertilizer) plus a wheat fetch and
the travel to reach it, and eight coops occupy eight tiles that melon would use
at roughly $137 per tile-day. The animal has to beat that opportunity cost, and
it does not.

`CARE` does work exactly as the mechanics suggested. Turning it off makes things
worse still, 20,534 down to 15,958. The mechanic was understood correctly and
the conclusion drawn from it was wrong, which is a useful distinction: reasoning
about a mechanic in isolation says nothing about whether to use it.

Cows are the interesting case. On the mean they slightly beat the champion,
51,131 against 50,588, and on the floor they are far worse, 40,173 against
48,542. Rating moves on win and loss, so a variant that raises the average while
lowering the worst case is losing matches to buy points that do not count. That
single row is why the promotion gate checks the floor.

---

## The evaluation machinery

Three additions, each prompted by a specific way of being fooled.

**Frozen opponent pool (issue #8, PR #15).** Beating `starter` proves nothing.
Promoted parameter sets are now snapshotted into `sim/opponents/` and stay in the
pool permanently. Keeping every past champion rather than only the latest matters
because a candidate can beat the current best while collapsing against an older
archetype, and the final tournament is played against everyone. Arena reports a
per-opponent breakdown, since an aggregate hides exactly that case.

**Promotion gate (issue #18, PR #19).** Reporting numbers and leaving the
judgement to whoever reads them is where a regression gets waved through. The
gate runs candidate and champion identically on holdout seeds against the full
pool and exits non-zero unless four checks pass: no errored episodes, mean beats
champion by more than the combined standard error, floor does not regress beyond
tolerance, and no losing record against any single opponent.

The margin check exists because at eight seeds the standard error runs near
1,000 coins, so a 500-coin gain is not evidence of anything. The floor check
exists because of the cow row above.

**Warm start (issue #16, PR #17).** Searches always restarted from dataclass
defaults, so an interrupted run lost everything and a follow-up run re-explored
ground already covered. `--init-params` centres the distribution on an existing
set, with a narrower initial spread (0.10 versus 0.25) so a warm start refines
rather than wanders back out.

---

## Worth flagging

### The Modal app was re-created on every call

`score_population` opened `with app.run():` per invocation, re-uploading mounts
twice per generation. Measured effect: about 15 minutes per generation against
roughly one minute of actual episode compute (64 candidates x 12 episodes fanned
across 200 containers at 1.3 seconds each). A 14-generation search would have
taken hours.

What is worth recording is how it was found. Nothing failed. The search ran
correctly and would have finished. It only surfaced from comparing observed wall
clock against the arithmetic of what the work should cost, which is a check
worth running on any fan-out that feels slow but is not erroring. Fixed in PR
#15, generations went to about 2.5 minutes. Issue #14.

### Searches wrote straight into the tracked working tree

CEM wrote its running best directly to `agent/params.json`. Two concrete failures
followed, both mine:

1. I copied a warm-start base into `agent/params.json` while a search was
   running, clobbering its saved best.
2. `git add -A` during a search committed whatever intermediate parameter set
   happened to be on disk, which is how an untuned-but-not-default parameter file
   ended up in PR #15.

Searches now write to `runs/<group>/best_params.json`, which is gitignored, and
promotion into `agent/params.json` is an explicit `make promote FROM=...` step.
Issue #21, PR #22.

### The flat submission layout was never tested

`test_contract.py` runs `agent/main.py` in place from the repo root, where
`agent/` is importable and the working directory is helpful. Kaggle unpacks a
submission flat into `/kaggle_simulations/agent/` with no package, and execs
`main.py` with no `__file__`. Iteration 1's worst bug was exactly that shape, and
its symptom is a failed validation episode rather than a bad score, so it costs a
submission slot and a day of rating convergence to diagnose.

The old test could not have caught a regression that only appears in the real
layout. New tests build the flat layout in a tmpdir, run it from a foreign
working directory, assert no relative imports crept in, and assert that a
`params.json` beside `main.py` is actually loaded rather than silently falling
back to defaults. Issue #20, PR #22.

### A test that measured the wrong thing

The gate's floor check was first tested by starving a candidate of labour and
asserting its worst case regressed. It failed: at 240 steps a labour-starved
agent simply hoards its starting cash, so its floor came out higher.

The test was measuring agent behaviour at an arbitrary episode length rather than
the decision rule. `decide()` was extracted as a pure function of summary
statistics and tested with synthetic numbers covering each rejection path,
including the exact cow figures from the ablation. Faster, deterministic, and it
actually tests the rule.

---

## The gate verdict

New champion against the previous one, holdout seeds, full pool, both seats:

```
                     mean        worst      win
candidate          78,937       55,707  100.0%
champion           47,239       35,408   87.5%
delta             +31,698 (1 sigma = 1,384)

per opponent (candidate):
  pass                   bank      82,492   win 100.0%
  random                 bank      82,647   win 100.0%
  starter                bank      83,599   win 100.0%
  v1-warmbase            bank      67,010   win 100.0%

GATE PASSED (all four checks)
```

A 23 sigma margin, and the floor rose from 35,408 to 55,707 rather than being
traded away for the mean.

One number in there is more interesting than the verdict. The candidate banks
about 83,000 against the built-in opponents and 67,010 against `v1-warmbase`, a
20% drop. That is the market coupling appearing for the first time in a
measurement. A capable opponent selling into the same market moves prices
against us, and the weak built-ins never did enough business to show it.

It also means every number produced before the pool existed was measured against
opponents too weak to apply the pressure the ladder will. The absolute figures
are optimistic; the comparisons between them are still sound, because every
candidate was measured the same way.

## Open questions

1. Cows deserve a proper look. The ablation put them at mean-positive and
   floor-negative with only four seeds, which is not enough to conclude anything.
   The search has them available and has not taken them.
2. Every number here is still against built-in opponents. The frozen pool exists
   so the next gate is against our own champions, but the real calibration is the
   ladder.
3. The forward-simulation planner (issue #10) is untouched, and it is where the
   structural edge is: one second per turn, of which the field reportedly uses
   0.005 milliseconds. The first step is a probe to test whether
   `kaggle_environments` imports inside the submission sandbox, which would allow
   exact rollouts.
4. Sell planning still assumes we are the only seller (issue #11). The opponent's
   farm is public, so their production capacity is observable even though their
   shed is not.

---

## Submitted

Submission 55255699, 2026-08-05. Validation episode COMPLETED, so the flat
layout, the `_agent_dir()` fallback and the `params.json` load all work in the
real sandbox. That was the one failure mode local testing was built to rule out,
because it fails as an error rather than a low score.

Initial rating is 600, which is the default every submission starts at rather
than a measurement. Ranked episodes accumulate from here and the rating is
reported to converge slowly, so the meaningful number is days away, not minutes.
Top of the leaderboard is currently 3105.6.

## Next

Use the ladder rating to calibrate how far the local numbers carry, then work
the backlog in issues #25 to #31. The first question is #25: local evaluation
overstates bank by about 20% because the opponents used during tuning were too
weak to move market prices.
