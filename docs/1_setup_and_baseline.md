# 1. Project setup and a working baseline

**Date:** 2026-08-05
**Milestones covered:** 1 to 4 of the plan (scaffold, engine parity, baseline policy, submission readiness)
**Status:** baseline agent working, not yet submitted

---

## What this milestone set out to do

Stand up the project for the Kaggle Kaggriculture competition and get an agent
that plays a full 720 turn episode without erroring, beats the built-in
opponents, and is ready to submit.

Kaggriculture is a two player farming simulation. Each player has a separate
10x10 farm, the season runs 30 days of 24 turns, and whoever holds the most
coins at the end wins. Skill rating moves on win, loss or tie only, not on the
size of the margin.

---

## The decision that shaped everything else

The original brief assumed this was a reinforcement learning problem and asked
for PufferLib as a submodule. Research before writing any code said otherwise,
and the plan changed as a result.

Three findings drove it:

**The game is barely interactive.** Farms are fully separate. The only shared
object is the market. A competitor ran the same seeds from both seats and got
bit identical bank balances, because the engine quotes both players against the
same pre-commit inventory, so there is no first mover edge. That makes this
about 95% a single agent scheduling problem. Self play buys very little.

**Every RL winner of a `kaggle_environments` competition rewrote the simulator
first.** Lux S1, Lux S3, Orbit Wars 1st (Rust, roughly 2,400 B200 hours across
15 billion steps), Orbit Wars 3rd (JAX). Nobody trains against the Python
environment. Porting `kaggriculture.py`, 1,063 lines of dict mutation with
fiddly end of day ordering and a per unit market loop, is a two to four week job
where mistakes fail silently. The entry deadline is 2026-09-23.

**Compute is not the bottleneck.** An episode takes about 1.3 seconds on the
real engine. Cross entropy search needs on the order of 10,000 evaluations,
which costs roughly $2 on Modal and carries no risk of our simulator disagreeing
with the one that scores the ladder.

So the approach is a parameterised heuristic policy, tuned by cross entropy
search against the real engine, with a per turn forward simulation planner
planned for later. PufferLib is not used and no simulator is rewritten.

The unclaimed opening we are aiming at: agents get one second per turn and the
field reportedly uses about 0.005 milliseconds of it. The current top meta is
replaying frozen action sequences that do not react to market state at all.

---

## What got built

```
agent/     main.py, policy.py, params.py, market.py, rules.py
sim/       harness.py, run.py, trace.py, arena.py
search/    cem.py, modal_app.py
obs/       wandb_setup.py
tests/     test_parity.py, test_contract.py
```

The policy is stateless. Every turn the whole plan is recomputed from the
observation and nothing is remembered between turns. That costs about a
millisecond against a 1000 millisecond budget, and it removes any chance of the
agent's beliefs drifting out of sync with what the engine actually did. In a
game where every invalid action is a silent no-op, that tradeoff is worth
taking. It also means any turn can be reproduced from its observation alone.

Each turn runs four steps: scan the farm, build a task list where each job
carries a priority and an optional required item, assign units greedily by
`priority - distance * penalty`, then emit market orders.

---

## Worth flagging

### The atomic PLANT deadlock

This is the one that cost the most time. The engine validates planting
atomically per crop per turn:

```python
blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}
```

If more units issue `PLANT WHEAT` in a single turn than you hold wheat seeds,
every one of those actions is silently dropped. With seven units and two seeds,
nothing is planted.

The reason it is worse than a normal bug is that it is a permanent deadlock, not
a stall. Nothing about the state changes to break the tie, so the next turn
reproduces the identical situation. The farm sat frozen from day 2 to day 28 and
the episode finished with 88 coins.

It is also invisible from the outside. There is no error, no warning, no failed
action count. The only reason it got found is the per day trace view showing
tile counts that never moved. Filed as issue #1.

### `__file__` does not exist inside a submitted agent

`kaggle_environments` loads agents with `exec(compile(raw, path), {})`. The
globals dict is empty, so any module level reference to `__file__` raises
NameError, which the loader converts into `InvalidArgument`.

This would not have shown up as a bad score. It would have failed the validation
episode on submission, burning a submission slot and a day of rating convergence
to work out why. The contract test caught it locally first.

The fix walks a chain of fallbacks. `compile()` keeps the real path on the code
object even when `__file__` is unset, so `sys._getframe(1).f_code.co_filename`
recovers it, with `/kaggle_simulations/agent` and the working directory behind
that. Filed as issue #2.

### The bankruptcy spiral

The first working version spent 2,700 of its 3,000 starting coins on nine geese
on day one, before building any coops and with no wheat to feed them.

What followed is worth recording because the failure was indirect. No cash left
meant no hands could be hired. Hiring is cheap (ten hands cost 143 coins) but it
is not free, and at zero cash it stops entirely. With one unit on the farm,
nothing got watered or fed. Crops turned to weeds, animals starved after two
unfed days and are unrecoverable. Final bank: 0.

The geese also never got placed, because `PLACE` needs the animal in a unit's
inventory and units were only fetching things when idle, which on a busy farm
never happens. Fetch trips became first class tasks. Filed as issue #3.

### Weeds compound

`prio_dig` started as the lowest priority, so clearing weeds never won an
assignment against feeding or watering. Weeds arrive from two sources: random
spawns at 0.005 per tile per day, and, dominantly, plants that died from a
missed watering.

By day 29, 41 of 75 unlocked tiles were weeds. More than half the farm was dead
land. The loop feeds itself: fewer tiles means less production, less cash, fewer
hands, more missed watering, more weeds. Filed as issue #4.

### The shed cap silently discards production

`shedCapacity` is 100 and overflow is thrown away with no signal. We were
holding 47 wheat as animal feed reserve plus 44 unsold fertilizer, pinning the
shed at exactly 100, so egg production was being discarded every night. Filed as
issue #5.

### Feed cost overtakes animal revenue

Wheat and egg prices move in opposite directions across a season. Wheat has a
steep scarcity curve (`sqrt`, 0.80) and town shops drain it constantly, so it
climbed from $25 to $55. Egg has a flat glut curve (`log`, 0.20), so our own
selling pushed it from $50 down to $41.

Past roughly day 22 a goose ate more value in wheat than it laid in eggs, and
the bank balance actually fell from 15,807 to 10,548. Filed as issue #6, with
the deeper question (grow wheat to self supply, or keep fewer animals) left to
the parameter search.

---

## Numbers

Progression on seed 0 against the built-in `starter` agent, each step a distinct
bug found through the per day trace:

| Change | Final bank |
|---|---|
| first working version | 88 |
| gate livestock on cash and housing | 12,330 |
| cap PLANT tasks at seeds held | fixed the deadlock |
| reclaim weeds, stop hoarding wheat, cap feed price | 24,895 |

Arena result across 12 episodes (2 opponents x 3 seeds x both seats):

```
mean bank   :       25,418  +/- 921
median bank :       24,360
worst bank  :       21,120
win rate    :       100.0%
```

Episode cost is about 1.3 seconds, which matches the estimate the plan was built
on.

**This number should not be over-read.** The built-in opponents are a very low
bar. `starter` farms a single carrot tile, never hires a hand and never expands.
Public notebooks reportedly score around 2670 to 2800 on the ladder, which is
rank 13 to 35, and the top ten sit around 2810 to 3105. Beating `starter` says
the agent runs correctly, not that it is competitive.

---

## Verification in place

- 20 tests passing. `test_parity.py` compares our copy of the crop, animal and
  market tables against the engine's, and sweeps the price curve across plus and
  minus 3T for all nine products. If Kaggle changes the economy and we do not
  notice, every sell decision quietly starts optimising the wrong market.
- `test_contract.py` asserts `agent` is the last callable defined, runs a full
  720 turn episode, and enforces p99 turn time under 300 ms.
- `make check-engine` diffs the pinned engine against upstream master. Four
  commits landed in the 72 hours before this project started, so this is not
  hypothetical. Currently identical to master at `kaggle-environments==1.32.4`.
- Modal verified end to end by running real episodes in a container and
  returning stats.
- Kaggle CLI authenticated, competition already entered.

---

## Open questions going into the next milestone

1. Does the tuned parameter set generalise, or is it fitting the seeds? The
   first search had no train and holdout split, which is why that landed as
   issue #7 and PR #12 before any large search was run.
2. Is livestock actually worth it? A partial search drifted to `target_geese: 1`
   and `target_carrot_tiles: 16`, which contradicts the opening prior that geese
   are the best steady earner. That result was unvalidated, so it is a question
   rather than a finding.
3. Can the agent import `kaggle_environments` inside the submission sandbox? If
   so, the forward simulation planner can roll out exactly, with no divergence.
   This is a hypothesis to test with a probe, not an assumption to build on.

---

## Next

Full cross entropy search on Modal with the holdout split, then gate the result
on unseen seeds before proposing a submission.
