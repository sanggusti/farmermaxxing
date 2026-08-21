---
description: Judge a candidate against the prize band, with opponents held out
argument-hint: [path to params.json] [TRAINED_ON=tape:a,tape:b]
allowed-tools: Bash(make gate:*), Bash(make mix:*), Bash(make meta-gap:*), Bash(.venv/bin/python -m sim.gate:*), Bash(.venv/bin/python -m sim.mix:*), Read
---

Judge the candidate at $1 (default `agent/params.json`). Report a verdict, not
a number dump.

Run, in order:

1. `make gate POOL=top` — against the prize band, versus the snapshot named in
   the `CHAMPION` file. Pass `TRAINED_ON=` with whatever opponents the search
   trained on; without it the held-out check cannot run and you must say so.
2. `make gate POOL=band` — against our own matchmaking band, where wins are
   actually contestable. The prize band currently returns 0% either way, so this
   is where a win-rate change shows up.
3. `make mix` — the realised per-day tile mix beside the opponent's.

Then report:

- **The five gate checks**, and which failed.
- **Win rate on `band`**, candidate versus champion. This is the quantity the
  ladder scores. A bank delta with an unchanged win rate has never moved our
  rating.
- **The sales mix**: units total and how many of the 9 products have nonzero
  volume. Reference, `POOL=top` on 2026-08-17 over 8 current top tapes: we sell
  **865 units of 5 products**, they sell **3,748 of 9** — wheat 131 against
  1,176, fertilizer 167 against 869, strawberry 107 against 584, wool 0 against
  351.
- **The realised mix versus the target**, and whether they agree. `params.json`
  targets are per-quadrant and v10's sum to 105 tiles of a 75-tile farm, so a
  target is a request that `Policy._wanted_crop` may overrule. If the realised
  mix does not look like the target, say which one the search was actually
  moving.
- **The memorisation gap** if `TRAINED_ON=` was given.

State plainly whether this is worth a submission slot. "Banks more" is not
sufficient: v10 banked +4,133 over v8 on `real` and lost rating.
