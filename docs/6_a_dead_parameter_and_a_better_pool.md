# 6. A dead parameter, and a better pool

docs/5 ended by naming the honest scoreboard — "~75k for v5 against 146k–152k
for strong ladder pairs" — and ranking four directions by what the evidence
supported. This document took none of them. It started from the ladder's own
episode list instead, and the first thing that list said was that we had been
measuring the wrong quantity again.

It records one fix worth 13 sigma, six refutations, one new instrument, and two
ways that instrument lied before it was trusted.

---

## Where we actually stood

Rank **871 of 2,260**, rating **866.2**, against a leader at 3,047.8. The shape
of the field matters more than the rank: ratings are bimodal, with ~1,300 teams
below 1,000 and a long thin tail above 2,000. Near 866 the ratings are packed —
rank 850 is 880.4 and rank 900 is 837.3 — so small rating moves are worth many
places.

The 46 episodes our two active submissions had played, bucketed by opponent
rating:

| opponent rating | n | our bank | their bank | win rate |
|---|---|---|---|---|
| < 750 | 13 | 102,773 | 64,893 | **100%** |
| 750–850 | 12 | 99,314 | 71,395 | 75% |
| 850–1000 | 15 | 93,806 | 100,240 | 27% |
| 1000+ | 5 | 74,752 | 124,492 | 20% |

Monotone across four buckets, so not noise. Our bank is not a property of our
farm; it is a property of who we are playing. Reading the replays gives the
channel: against a strong opponent milk goes 160 → 7 and melon 250 → 31, and our
realised milk price falls from $273 to $83.

That is docs/5's market-coupling finding again, but it now has a magnitude
attached to the only bucket that matters. At rating 866 the matchmaker gives us
the 850–1000 band, we lose it 73% of the time, and we lose it by ~6,400 coins.

## The dead parameter

The per-day census of a loss to LuoXingda (rank 49):

```
      us                              them
d15   util  63%  empty 20  weeds 8    util 100%  empty 0  str42 mel12 cow8 whe7 she6
d21   util  59%  empty 28             util 100%  empty 0  str42 mel12 cow8 whe7 she6
d27   util  31%  empty 52             util 100%  empty 0  whe38 str23 cow8 she6
```

Fifty-two idle tiles on day 27 while holding $51,590. The meta rotates 42
strawberry into 38 wheat and never idles a tile.

The cause is four lines apart in the source and took an hour to see:

```python
mix_switch_day = 28                          # agent/params.json
# agent/policy.py, _crop_scores
if CROPS[crop]["first_yield_day"] + p.plant_cutoff_slack > days_left:
    continue                                 # WHEAT: 2 + 4 = 6  ->  blocked after day 24
```

Wheat is the only crop the late multiplier can act on, and the switch fires four
days after wheat stops being plantable. **`late_target_mult` was dead code.**
The mechanism docs/4 designed and #60 asked for had never once run.

Nothing about a dead parameter looks broken from outside, which is why five
searches sampled `mix_switch_day` across its full range and none of them found
this. Every draw above ~24 produces byte-identical episodes. The fitness surface
is flat there, and the CEM mean drifts wherever noise takes it — a flat region
and a converged optimum are indistinguishable in the logs.

Switching at day 16 with a 6× late wheat multiplier is **v10**:

| | mean | worst | land use |
|---|---|---|---|
| v8 | 78,744 | 45,519 | 61.3% |
| **v10** | **83,243** | **50,195** | **69.1%** |

+4,138 at **13 sigma** paired, on the `real` pool over 8 clean seeds and both
seats.

## Six refutations

Every other hypothesis this session generated was wrong, and measuring them was
most of the work. All figures are margin deltas against the variant's own
baseline, paired on identical cells.

| hypothesis | evidence for trying it | result |
|---|---|---|
| Shift melon → strawberry | meta sells 308 strawberry to our 131; melon has the worst glut curve in the game (`sq`/3.60) and crashes to $31 | **−49,283** — melon is our main earner, and strawberry was already pinned at its bound |
| Add sheep for wool | meta sells 212–258 wool at $237, above base; we sell **zero** | **−37,070** — feeding and care cost more actions than the wool returns |
| We are labour-limited | meta fields 13–14 hands from day 15, we field 9; `hands_late` 12/15/18 all scored *identically* | **−36,959** — `hire_turns` > 1 does lift the MAX_MARKET_ORDERS cap, and the extra hires then crowd out the turn's SELL orders while the fib cost compounds |
| Per-crop cutoff for the 2-day crops | wheat can pay until day 27; one shared slack stops it at 24 | **−1,001 to −2,500** at every value, both pools — the idle tail is correct |
| More wheat tiles overall | `target_wheat_tiles` 9 gave the **highest bank in a 65-variant sweep**, +11,500 | **−3.31 sigma** on margin — bank and margin come apart exactly as docs/5 found |
| A CEM run from the v10 basin | it is what worked five times before | nine of fourteen generations rejected; the winner fails the gate (floor 50,195 → 46,757, win 30.2% → 15.6%) |

The identical scores in row three are worth dwelling on. Three different values
of `hands_late` producing the same number to the coin is the same tell as the
dead parameter: a coordinate sweep reports a flat coordinate and a saturated
constraint the same way. The constraint was `MAX_MARKET_ORDERS`, and finding it
required reading why the numbers matched rather than which was largest.

## A better pool, and two ways it lied

Every opponent pool we own sits in the wrong place:

| pool | banks | what it represents |
|---|---|---|
| frozen lineage (v1–v8) | 60–75k | our own past selves |
| meta tapes | ~190k | the top of the ladder |
| **the 840–1050 band** | **100–123k** | **who Kaggle actually matches us with** |

`sim/tapes/band-*.json` are six tapes cloned from the opponent seats of six real
episodes played that day, all verified non-degenerate on unseen seeds. The
justification is direct: v8 beats the entire frozen lineage and sits **14.15
sigma** behind v10 on the band pool.

It then produced two false results before it produced a true one.

**It lied about sample size first.** On 6 seeds × 1 seat it reported v11 beating
v10 by +4,274 margin and 42% to 36% on wins — which read as the promotion gate
mismeasuring, since the gate had just failed v11. On 14 seeds × 2 seats, paired,
that collapses to **+394 at 0.20 sigma, with wins 2.4% *worse***. The gate was
right. A new instrument whose first act is to overturn an old verdict is the
case for more samples, not fewer.

**Then it lied about opponents.** A CEM run trained on four of the six band
tapes looked like the best candidate of the day: +3,812 margin, +9.5% wins.
Split by opponent:

| split | margin delta | sigma | win delta |
|---|---|---|---|
| trained-on (4 tapes) | **+10,188** | 3.96 | +18.8% |
| **held out (2 tapes)** | **−8,940** | **−1.89** | **−8.9%** |

Four opponents is small enough to memorise. On the independent `real` pool the
same candidate drops to 18.8% against v5 and 31.2% against v8. **Hold opponents
out, not just seeds** — the seed splits this project has run since docs/2 do not
catch this failure at all.

## What shipped

| slot | agent | band wins | margin | worst | why |
|---|---|---|---|---|---|
| 1 | **v10** | **37.5%** | −3,215 | 53,042 | the fix above |
| 2 | **v11** | 35.1% | −2,821 | 43,681 | diverse hedge; replaces v8 at 25.6% |

v11 is a statistical tie with v10 (+394, 0.20 sigma) but a genuinely different
basin — two crops per turn, no late melon, premium sell floors raised so it
drips rather than dumps, rival-aware selling on. It does not beat the champion,
and it does not need to: the slot it takes held v8, which the band pool puts
14.15 sigma behind v10. The third daily submission was left unused, because
nothing else measured better than v10 on any pool and spending it would have
evicted a verified agent.

## Where the gap is now

Against the meta tapes v10 banks ~88k to their ~140k and wins **0%**. The best
single-coordinate move in a 65-variant sweep was +2,661. Two CEM runs from this
basin produced nothing promotable. Parameter search on this policy is, on the
present evidence, exhausted.

What the census keeps pointing at is not a parameter. The meta holds **100% land
use with 0 weeds from day 15 through day 27** on the same 75 tiles where v10
manages 69% and carries 3–7 weeds. It does that with 13–14 hands where our own
attempt at 14 hands lost 37k, which means the difference is not how many actions
they have but how few they waste — travel, and the order jobs are done within a
day. That is #78 and #30, and it is a build, not a tuning run.

The counter-argument recorded in docs/5 still stands and is now the main open
question: the one time we tried to close a land-use gap directly (multi-crop
planting, to match the meta's 92% on day 3) it closed the gap exactly and cost
money against every opponent. Land use is a symptom that has twice been mistaken
for the disease. The third attempt should predict the coin value of a scheduling
change before building it.
