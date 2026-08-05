# 3. A wrong hypothesis, and what measuring it revealed

**Date:** 2026-08-05
**Milestone:** Phase 1 of the push toward 170k
**Status:** hypothesis refuted, real constraint identified, multi-restart search running

---

## What this milestone set out to do

The leader banks over 170k. We were at roughly 118k against `starter`. The plan
was to close that with routing: 60.7% of all our unit-actions were MOVE, so
cutting travel looked like it would roughly double productive work.

It did not survive contact with measurement. The value of the milestone is the
refutation, so this document leads with that rather than burying it.

---

## The hypothesis

Instrumenting a full season gave:

| Action | Count | Share |
|---|---|---|
| MOVE | 4,319 | 60.7% |
| WATER | 658 | 9.2% |
| PICKUP | 564 | 7.9% |
| FEED / CARE / COLLECT | 984 | 13.8% |
| HARVEST | 227 | 3.2% |
| PASS | 202 | 2.8% |
| PLANT / DIG / BUILD / PLACE | 158 | 2.2% |

Two thirds walking, roughly a quarter working. The reasoning was that
`_assign` is global and stateless, so a unit crosses the board for a marginally
better task and crosses back when priorities shift. Watering makes the case
concrete: one water per plant per day, so a unit sweeping a contiguous block
pays about one move per plant and a unit picking globally pays three or four.

## Three implementations, all worse

| Approach | Mean bank | MOVE share |
|---|---|---|
| greedy (current) | 118,752 | 61.1% |
| zoned, partition the task list | 34,840 | 69.8% |
| zoned, partition the board by rows | 96,336 | **56.1%** |
| zoned, serpentine equal-area | 30,240 | 62.9% |

The first attempt failed for a reason worth recording: partitioning the *task
list* seemed natural, but the task list changes every turn as jobs complete, so
chunk boundaries moved and units were reassigned to different patches turn after
turn. That is commuting with extra steps, and it scored worse than doing
nothing.

Partitioning the board fixed the churn. The row version then did exactly what it
was designed to do, cutting MOVE to 56.1% and lifting productive work to 43.2%,
and still lost 22k of bank.

That is the cleanest possible refutation. The intervention achieved its stated
mechanism and the objective got worse, which means the mechanism was not the one
that mattered.

## Why the premise was wrong

Two numbers, both available before any code was written:

| | |
|---|---|
| moves per productive action | **1.65** |
| productive share of action capacity | **33%** |

1.65 moves per job on scattered tiles across a 10x10 board is close to the floor
for that layout. And only a third of the 264 daily unit-actions do work, so
labour was never scarce. Reclaiming travel turns buys idle time, not output.

The champion parameters said so directly: `distance_penalty` was tuned to
**13.0 out of a 15 maximum**. CEM had already solved locality. Zoned routing
adds nothing on top of greedy-with-a-strong-distance-penalty, because that
already *is* local routing.

60.7% MOVE was not waste. It was slack. `PLANT` runs 2.8 times per day, so the
binding constraint sits well upstream of labour.

---

## The real constraint

End of season, champion params, seed 20000, final bank 117,918:

| Product | Base | End price | vs base | Units we sold | Market inv vs I0 |
|---|---|---|---|---|---|
| WHEAT | 25 | 55 | **220%** | **0** | -927 |
| STRAWBERRY | 120 | 321 | **268%** | 16 | -574 |
| TOMATO | 60 | 86 | 143% | 50 | -216 |
| EGG | 50 | 70 | 140% | **0** | -338 |
| MILK | 160 | 222 | 139% | 359 | -51 |
| WOOL | 200 | 252 | 126% | **0** | -428 |
| CARROT | 35 | 42 | 120% | **0** | -410 |
| MELON | 250 | 202 | 81% | 209 | +69 |
| FERTILIZER | 100 | 42 | 42% | 292 | +292 |

**Seven of nine products end the season priced above base, with market inventory
below the equilibrium I0.** Town shops and the town centre consume faster than
two players supply, and town demand grows monotonically: shops unlock every
three days and the centre scales to 2x after day 10 and 4x after day 20.

We sell three products. We sell zero wheat, carrot, egg and wool, and those
trade at 120% to 220% of base. The only two products we have driven to a
discount are the two we concentrate on.

The mechanism is not subtle. The champion carries `target_geese: 0`,
`target_sheep: 0` and zero wheat tiles, so it produces no eggs, no wool and no
wheat at all. It is a narrow melon, milk and fertilizer operation selling into a
market that is paying a premium for the four things it does not make.

### This explains every failed experiment at once

- Raising crop targets 2.5x: 116,076 against 117,858. More of the *same*
  products floods those products and earns nothing.
- Zoned routing: freed labour that had no profitable use.
- 14 hands: 12,756. Fib-priced coins spent on work that was not revenue-limited.

All three pushed on an axis that was not binding.

### It also invalidates an earlier conclusion

The livestock ablations in iteration 2 changed one parameter from a local
optimum and concluded animals were bad. Adding sheep without pastures, without
the wheat to feed them, without labour allocated to service them and with a wool
sell floor of 1.28 (which refuses to trade at any realistic price) was never a
test of livestock. It was a test of adding cost without capability.

Wool ending at 126% of base with zero units sold says that question is open.

---

## Why the search never found this

Every CEM run since v2 warm-started from the previous champion with a narrowing
Gaussian, so all of them explored one basin.

Diversifying is not a small perturbation. Adding sheep requires pastures, wheat
to feed them, labour to service them, and a wool floor low enough to trade.
Every single-parameter step toward that basin is worse than the local optimum,
which is exactly the shape a local search cannot cross.

So the starting points have to be constructed rather than discovered.
`search/archetypes.py` defines four coherent portfolios, each with the sell
floors and feed policy that portfolio needs:

| Archetype | Untuned mean bank |
|---|---|
| champion (16 generations of tuning) | 115,402 |
| premium | 51,402 |
| staples | 41,518 |
| diversified | 41,495 |
| livestock | 10,496 |

Untuned they are all far behind, which is expected. The question each run
answers is where its basin tunes to, not where it starts.

---

## Process failures worth recording

**Measuring against a dirty tree.** Two archetype scoring runs were taken with
the failed routing code still in the working tree, because `git checkout main`
carries uncommitted changes across rather than discarding them. The champion
read 38,326 instead of 115,402. It was caught because the number was
implausible, not because anything warned. Everything above is re-measured on a
clean tree, and `git checkout -- <files>` is the discard that was needed.

**A default argument eaten by the framework.** An instrumentation helper written
as `def spy(obs, _pol=pol)` silently received the environment configuration as
its second argument, because `kaggle_environments` passes `(obs, config)` to any
callable whose `co_argcount` is 2, and default parameters count. The fix is a
closure with exactly one parameter.

---

## Open questions

1. Does any archetype beat 115,402 after tuning? If none do, the diversification
   thesis is wrong and the market table needs a different reading.
2. If diversification wins, how much of the gap to 170k does it close?
3. The market inventory figures are from a game against `starter`, which
   produces almost nothing. Against a real opponent both players drain and
   supply the market, so the starvation may be milder. The ladder replays can
   answer this and have not been checked for it yet.

---

## Next

Four searches running from the archetypes above, on Modal, logged to W&B under
`cem-arch-*`. Whichever basin wins gets gated against the current champion on
clean seeds.
