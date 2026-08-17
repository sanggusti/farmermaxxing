# farmermaxxing — how to work on this

Agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition. This file is the operating contract. `DEVELOPMENTS.md`
has the game breakdown, the engine gotchas and the dated experiment log;
`docs/1..6` are the milestone reports. Read this one first.

## Where we stand

| | |
|---|---|
| Rank | **2455 / 4883**, score 746.1, team `an Expired Engineer` |
| Leaderboard | leader 3228 · top-10 cutoff ~2963 · 10th pct 2310 · **median 749** |
| Submissions used | 7, in the project's whole life. None between 2026-08-06 and 2026-08-17 |
| Budget | 5/day, latest 2 scored. Entry deadline 2026-09-23, final 2026-09-30 |
| Against the prize band | **0 wins in 8**, margins -53,000 to -83,000 |
| Sales volume | we sell ~865 units of 5 products; they sell ~3,750 of 9 |

## The rules

**1. The objective is ladder rating.** Not mean bank, not margin against our own
snapshots. A local number is a *hypothesis* about the ladder until `ledger.json`
says otherwise. Six promotions were each justified against a different local
yardstick and the project never once checked whether the yardstick predicted the
rating. Run `make calibrate` before believing a gate.

**2. Never move the anchor on local evidence.** `CHAMPION` names the snapshot
every candidate must beat, and it changes only on a **converged** ladder rating.
A submission restarts at 600 and needs ~24-48h to settle; a younger rating is
not a measurement. On 2026-08-06 a locally-dominant candidate replaced the
incumbent and the recorded outcome was -108 points — and the comparison is
*still unresolved*, because the incumbent only ever held a slot for 2h45m.

**3. Slot discipline: one anchor, one challenger per day.** An unused slot is a
measurement we chose not to take, and eleven idle days cost ~55 of them.

**4. Record every submission the same day, with its pool.** `make submit`
requires `POOL=` and `METRIC=`. A bank figure without its opponent is not
comparable to anything: the same agent banks 102,773 against sub-750 opponents
and 74,752 against 1000+. This is not bookkeeping — it is the only input to
calibration, and its absence is how v10's result was lost for eleven days.

**5. Bank and wins come apart.** v6 banked 81,623 against v5 and won 0% of those
matches. Against the prize band, v10 banks +4,133 more than v8 and both win 0 of
8. A ladder scores wins; no sequence of ±5,000 bank improvements flips a match
you are losing by 53,000. Quote a win rate next to every bank delta.

**6. Never gate against `Params()` defaults or a pool of our own lineage.**
`make gate` defaults to `POOL=top` and to the `CHAMPION` file for this reason.
Pass `TRAINED_ON=` so the held-out-opponent check can run: a search trained on 4
of 6 band tapes scored +10,188 (3.96σ) on those four and **-8,940 on the two held
out**, and every instrument called it the day's best candidate.

**7. "An effect suspiciously close to zero" is the tell.** It was the tell in
three of four instrument bugs, and in two more found on 2026-08-17. Bugs in this
repo do not raise; they produce complete episodes with plausible numbers.

**8. The engine is the source of truth**, confirmed by the host. Everything in
`agent/rules.py` is transcribed from engine source and pinned by
`tests/test_parity.py`. And the daily shop unlock draws from the same RNG stream
as weed spawns, one draw per empty unlocked tile across **both** farms — so no
land-use change is ever isolated. `sim.gate` reports the confound; heed it.

## Tests guard validity. The tape pool judges score.

`pytest` is 91 tests in ~6s and none of them can tell you whether the agent got
better. That is deliberate as of 2026-08-17. `tests/expected_scores.json` used to
be the only score signal and it was measured against `starter`, on the CEM
*selection* seeds, at 0.5% tolerance in both directions, excluded from the
default run. It changed 6 times, all 6 in "Promote vN" commits, and never caught
a regression. It is gone, along with four tests that pinned behaviour later
measured as harmful.

So: **do not add a test that asserts a score or a default parameter value.** If
a change would make the agent better, no test should have to be re-recorded.
Judge score with `make gate` / `make meta-gap` / `make mix`.

What tests do guard: engine parity, the submission contract, the flat layout, the
real tarball, both seats, crash-safety, timing, and the *statistics* in
`sim/gate.py`, `search/league.py` and `search/cem.py`. Those last 30 encode
instruments that reported progress we had not made; they matter more in a
score-driven regime, not less.

## The loop

```bash
make ladder-sync     # ratings into ledger.json, our rank, the band benchmarks
make calibrate       # has local evidence ever predicted the rating?
make mix             # our REALISED tile mix beside the opponent's, per day
make meta-gap        # champion vs the prize band, decomposed by product
make gate POOL=top TRAINED_ON=tape:a,tape:b
make refresh-tapes DATE=2026-08-16   # remint the top-band pool
make preflight       # everything that must hold before spending a slot
make submit CONFIRM=1 V=v12 M="..." POOL=top METRIC=margin VALUE=n
```

Slash commands wrap these: `/compete-status`, `/compete-candidate`,
`/compete-submit`.

Which instrument answers which question:

| question | tool |
|---|---|
| where are we, and what should today's move be | `make ladder-sync`, `/compete-status` |
| is this candidate worth a slot | `make gate POOL=top`, `/compete-candidate` |
| *why* is it better or worse | `make mix` (realised tiles), `make meta-gap` (products) |
| does our local evidence mean anything | `make calibrate` |
| is the pool still current | `make refresh-tapes DATE=<yesterday>` |
| will this submission be rejected | `make preflight` |

`kaggle/kaggriculture-episodes-<date>` publishes ~700 full replays **every day**,
current and free. `sim.ladder` fetches, verifies, censuses and clones them. The
top of the ladder has sat flat at ~3200 since 2026-08-12, so the target is both
strong and stationary.

## Where the gap actually is

Measured 2026-08-17, `make mix` on seed 20000 against replay 90044961's seat
(both seats banked ~152,000):

```
day    us                                util   them                                  util
  3    mel6 str4 cow4 whe2                64%   whe11 mel8 cow3 she1                    92%
 15    mel18 str17 cow12 whe4             68%   str40 mel8 whe4 cow8 she6               88%
 27    whe14 cow12 str7                   44%   whe30 str21 cow8 she6                   92%
```

Three differences, largest first. **Strawberry 40 against our 17** — they hold it
through the whole middle of the season. **Melon 8 against our 18**, and they still
sell more melon than we do from under half the tiles. **Six sheep against none.**

And the sales mix, `make gate POOL=top` over 8 current top tapes, 4 clean seeds,
both seats:

```
       units   WHEA  CARR  TOMA  STRA  MELO   EGG  MILK  WOOL  FERT
us       865    131     0     0   107   157     0   302     0   167
them   3,748  1,176    10     2   584   188     4   564   351   869
```

Wheat 9x, fertilizer 5x, strawberry 5x, milk 2x, wool from zero. **Fertilizer is
the cheapest of those to fix**: every animal yields one free per day regardless of
care, it bases at $100, and `COLLECT_FERTILIZER` is one action.

And the reason our targets don't produce our mix: **targets are per-quadrant and
v10's sum to 105 tiles of a 75-tile farm** (melon 17 + strawberry 12 + wheat 2,
times three quadrants, plus 12 structures). So `Policy._wanted_crop`'s shortfall
arbitration decides the mix, not the targets, and melon wins it permanently by
having the largest one. Every target sweep the project ran was moving a number
the code then overruled — which is a decent explanation for six mutually
confusing refutations, including "shift melon to strawberry costs -49,283" and
"strawberry is already at its bound" (the bound was 12; the meta runs 13.3 per
quadrant; it is now 20).

## Refuted. Do not re-litigate without new evidence.

From `docs/5` and `docs/6`: endgame liquidation (a non-problem — the engine prices
per-unit inside one order); fill-rate as the cause (closing the day-3 land-use
gap directly costs money against every opponent); melon → strawberry as a target
move (-49,283); sheep for wool as an addition to *this* labour budget (-37,070);
"we are labour-limited" (`hands_late` 12/15/18 score identically); per-crop
cutoffs for 2-day crops; the mid-season half of the tile collapse (dead code,
fixed in v10).

Added 2026-08-17:

- **SELL is not starved by the 10-order budget.** `sell_order_floor` swept 0..6
  makes units sold *fall* (858 → 645) and bank fall with it. We end the season
  with ~9 unsold units, so we already sell nearly everything we grow. The gap is
  **production**, not order slots or sale scheduling.
- **Transcribing the meta's tile counts is not enough.** `search/archetypes.py
  metabuild` sets exactly the build above and reaches 30% land use, the worst of
  the five archetypes. Our policy cannot service 40 strawberry tiles. Whatever
  closes the gap is upstream of the portfolio.

## Bugs found on 2026-08-17, for the pattern

All four were silent. None raised. All produced plausible numbers.

1. **`resolve_pool("real")` excluded the reigning champion.** `frozen_names()`
   sorted lexicographically, so `v10-wheatfix` sorted second and `[-2:]` never
   picked it. Every `--opponents real` run after v10 was frozen — including
   v10's own 13σ gate — measured against v5 and v8.
2. **`make gate` with no `CHAMPION=` compared against `Params()` defaults**, the
   hand-tuned baseline that banks 24,895. Every candidate passes that.
3. **`TapeAgent` was a no-op on the slow path.** `kaggle_environments` truncates
   agent args via `agent.__code__.co_argcount`; a class instance has no
   `__code__`, so `__call__(obs)` was invoked with two arguments, the TypeError
   was swallowed into a no-op action, and the seat still finished `DONE`.
   `fast_play` gave `[105,504, 151,737]`; `harness.play` gave `[114,521, 3,000]`.
4. **`submission.tar.gz` shipped parameters that were not `agent/params.json`.**
   Nothing had ever validated the tarball. Finding it recovered v11's parameters,
   which the project had recorded as unrecoverable.

## Hard limits

`actTimeout` 1 s/turn (plus a 60 s episode bank); `runTimeout` 1200 s; 2 vCPU and
~12 GiB in the submission box. We use **0.03%** of the per-turn allowance and the
top of the ladder replays frozen open-loop sequences, so per-turn lookahead is
wide-open ground — issue #10, still open, and its first step is still a probe
submission testing whether `kaggle_environments` imports inside the sandbox.

## Conventions

- `agent/` modules are **flat** (`from policy import ...`) because Kaggle unpacks
  a submission flat. `agent` must be the **last callable defined** in `main.py`,
  and `__file__` does not exist there.
- Adding a module to `agent/` means adding it to `AGENT_FILES` in the `Makefile`;
  `tests/test_submission.py` enforces that the Makefile matches `agent/*.py`.
- `agent.main.agent` swallows exceptions into a PASS turn. `FM_STRICT=1` makes it
  raise, and `tests/conftest.py` sets it.
- Docstrings here carry the *measurement that motivated the code*, with numbers.
  Match that. A comment saying what the code does is worth little; one saying
  which experiment would otherwise be repeated is worth a lot.
- `runs/`, `wandb/` and `replays/` are gitignored. `ledger.json`, `CHAMPION`,
  `sim/opponents/*.json` and `sim/tapes/*.json` are tracked — they are the
  durable record.
- Search runs sequentially. Six parallel searches split the Modal container pool
  and none completed a generation for ten minutes. Cost is not the constraint
  (~$2-13 a run); wall clock is.
