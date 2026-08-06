# 5. Reading the ladder

The previous document ended on an admission: *"First place banks over 170k; v4
banks 137,684 on clean seeds against `starter` and 46,454 against our own
strongest frozen opponent. Which of those numbers is comparable to 170k is not
known, and finding out matters more than another search."*

This one answers that, and the answer invalidates most of what came before it.
It also records four refuted hypotheses, three bugs in our own measurement
apparatus, and one change that produced the first champion to beat every
opponent in the pool.

---

## The number that was comparable to nothing

Ladder replays carry their seed at `info.seed` — `configuration.seed` is null,
which is why nobody had noticed. Both farms are public in the shared
observation, and both seats' actions are recorded. So any ladder episode can be
re-run locally, exactly.

`python -m sim.ladder verify` replays an episode from its recorded actions and
checks the final banks. **Seven of seven reproduce to the coin.** That check is
the load-bearing one: an off-by-one in step alignment would still produce a
complete episode with entirely plausible banks, and every derived number would
be quietly wrong.

With that in hand, the comparison is direct. The top of the ladder is a scripted
meta playing near-mirror games, so the comparable statistic is the
**mirror-match bank**:

| | bank |
|---|---|
| ladder mirror matches, 2026-08-05 | 152,469 / 151,880 · 148,032 / 146,390 · 97,459 / 89,971 · 97,160 / 93,586 |
| **v5 mirror match, 4 seeds** | **74,918** (min 59,455, max 97,184) |

We were at roughly half the strong pairings. Every number this project has
quoted against `starter` — v4's 137,684, v5's 128,071 — was never comparable to
anything on the ladder. The rule now is: **quote the mirror-match bank, or the
bank against the strongest frozen opponent. Never the `starter` number.**

## The meta, read off the replays

Both seats byte-identical, confirming the open-loop scripted strategy the forum
describes:

| day | quadrants | land in use | portfolio |
|---|---|---|---|
| 3 | 1 | **92%** | wheat 11, melon 8, cow 3, sheep 1 |
| 9 | 2 | 82% | strawberry 12, melon 11, wheat 7, sheep 6, cow 5 |
| 15 | 3 | 88% | **strawberry 40**, melon 8, cow 8, sheep 6 |
| 27 | 3 | **92%** | **wheat 31, strawberry 22**, cow 8, sheep 6 |

They rotate the crop mix late — 40 strawberry to 31 wheat between days 21 and 27,
because wheat first-yields in two days and fills a tail that ten-day crops
cannot. They hold 85–92% land use through the mid-game. They never buy the
fourth quadrant. They tolerate 13–15 weeds rather than spending late actions
digging. They run broke on purpose: $217 on day 3.

---

## The finding that changed the project

Every CEM search since v2 trained against `starter`, the weakest agent
available. That is not a magnitude bias. Measured on clean seeds against the v4
champion:

| variant | vs `starter` | vs `v3-fixed` |
|---|---|---|
| champion (melon monoculture) | **137,052** | 46,285 |
| +8 wheat tiles/quadrant | 120,669 | **53,946** |

All three diversifying variants tested lose against `starter` and win against
`v3-fixed`, on the mean and on the worst seed. **The gradient points the
opposite way.** The melon monoculture was never a discovery; it is what an
uncontested market rewards.

Fixing it produced v5: mean 79,533 → 90,842 on the mixture pool, worst seed
24,205 → 33,768, and +42.8% against `v3-fixed`. It gained **6.6% worse** against
`starter`, exactly as predicted.

The part that was not predicted: with no policy change at all, the search
independently found what two open issues had been asking for.

| | v4 | v5 |
|---|---|---|
| land in use | 58.9% | **67.7%** |
| distinct products sold | 4.0 | **5.0** |
| unsold stock at end | 17 | **3** |

The parameterisation was not the ceiling. The objective was.

On the ladder, v4 scored 774.7 and v5 scored **823.4**.

## Bank and wins come apart

Two candidates then beat the entire opponent pool and lost to the incumbent. v6
banked 81,623 against v5 — the second-highest number in its per-opponent column —
and won **0%** of those matches.

That is the objective again. The population was ranked on mean bank; the ladder
scores wins. Those separate exactly when both players trade into one market: a
candidate can earn more coins overall while that specific opponent earns more
still.

| | mean bank | margin vs v5 |
|---|---|---|
| v5 | 79,502 | +0 (self) |
| v7 | **86,573** | −2,887 |
| v4 | 61,403 | −9,246 |

Ranking on **margin** — my bank minus theirs, per cell — fixed it. Margin is also
the paired statistic *within* an episode: a market shock lifts both banks and
cancels in the difference. v8 wins **100% against all eight frozen opponents**
and lifts the worst clean seed from 33,768 to 52,764.

v8 does not pass the gate. Its mean bank is a statistical tie (−384 against 1σ of
1,652), so the second check fails. It was promoted deliberately and on explicit
instruction, and the override is recorded in the commit rather than being
laundered through a relaxed threshold.

---

## Four refuted hypotheses

**The endgame liquidation solver was a non-problem.** Splitting a sale into
chunks gains nothing — the engine prices per-unit *inside* one order, so 200
melon in one `SELL` and eight orders of 25 return identical revenue. Town drain
across the entire final two days is 16 melon and 17 milk. And there is nothing to
liquidate: the champion ends the season with a mean of **3 unsold units**.

**"We trail because we fill land slower" is false.** Multi-crop planting closes
the day-3 gap exactly — land use 56% → 88%, matching the meta's 92% — and costs
money against *every* opponent, on margin as well as bank. Not the usual
`starter`-disagrees pattern. The meta reaches 92% on day 3 and banks 150k, but
the two are not causally linked for our policy: every extra plant is a daily
watering task competing with feeding and harvesting.

**The late-season collapse was mostly a v4 problem.** "Utilisation collapses from
day 19" was measured on v4. v5 holds 61% at day 25 where v4 had fallen to 48%.
The recoverable window is roughly days 26–27, not 19–29, and days 28–29 staying
low is *correct* — wheat needs two days to first yield, so a tile planted on day
28 cannot pay back.

**The opening book is not currently justified.** Five of the meta's seven opening
moves are now expressible as parameters. A sixth — buying land on day 0 — was
reachable only at the exact upper bound of `land_buy_empty_max`, which is a bounds
problem, since a fresh farm has exactly 25 empty tiles and the bound was 25. Only
one genuine gap remains: our policy is a per-turn priority match with no notion of
an intra-day schedule. That is a large build against a single remaining
justification, and its main motivation was the fill-rate hypothesis above, which
is refuted.

---

## Three bugs in our own instruments

**The gate discarded 80% of its statistical power.** It used
`sqrt(se_a² + se_b²)`, the standard error for *independent* means, while running
both agents on identical cells. The opponent main effect — roughly 128k against
`starter` down to 59k against a frozen champion — is identical for both agents
and cancels exactly in a paired difference. Measured on a real gate run, 79.7% of
variance was between cells. The paired test is **2.2× tighter**, and v5 goes from
2.44σ to 5.41σ.

**CEM's Gaussian mean escaped the box.** `sample()` draws unbounded and only
`unflatten()` clips, so `refit()` was fitting to raw out-of-range values. For any
parameter whose optimum sits at a bound, the mean marched further outside every
generation — simulated over 30 generations, a mean of −124 against a bound of 0,
with **100% of draws clipping to the same value**. A population of duplicates
differing only by their noise draw. The champion had 22 of 45 parameters on a
bound.

**Three integer parameters were unsearchable.** The variance floor was a fraction
of range, so once the elites agreed, the probability of ever changing
`fertilize_enabled` was 0.0000%, `hire_turns` 0.0000%, `liquidate_days` 0.0001%.
The mechanism whose docstring says it exists so "a parameter that collapses early
can still recover" did not do that for the parameters that needed it most.

A fourth, in a feature rather than the search: opponent-aware selling shipped
half blind. Rival crop tiles carry `planted_day`, not `age`, so
`tile.get("age", 0)` made every crop look freshly sown and no crop ever entered
the lookahead window — the signal reported animals only. It measured as
worthless and was nearly recorded as a fifth refutation. With the age computed
correctly the same rival shows 25 melon units of incoming supply that were
invisible.

**The tell, in three of the four cases, was an effect suspiciously close to
zero.** A refutation and a broken instrument look identical from the outside.

---

## What is left

The mirror-match gap is the honest scoreboard: ~75k for v5 against 146k–152k for
strong ladder pairs. Nothing measured here closes it.

Ranked by what the evidence supports:

1. **The search's own statistics** — racing on episode allocation and shrinkage
   before ranking both cost zero extra episodes and both reduce the winner's
   curse rather than adding to it (#65, #67). Training seeds are still a fixed
   block for all 30 generations, which is seed-set overfitting (#68).
2. **Bounds.** Seven task priorities sit at exactly 0.0, which is the search
   trying to say "never do this" against a floor of 0. The box is wrong in at
   least that many places (#73, #70).
3. **Population shape.** 384 × 40 generations is ~380 × D, and the literature we
   were drawing on is benchmarked at 10⁶ × D. We are generation-starved, not
   population-starved (#69).
4. **The opening book**, if and only if intra-day scheduling turns out to be
   worth the build (#78).

A standing hazard for all of it: the daily shop unlock is drawn from the same RNG
stream as weed spawns, *after* one draw per empty unlocked tile across **both**
farms. Same seed, different agent, different economy — verified on three seeds.
Any change to land use silently changes which products the town drains, and a
search on fixed seeds can bank a favourable unlock that does not travel. The gate
now reports a shop-order confound statistic for exactly this reason.
