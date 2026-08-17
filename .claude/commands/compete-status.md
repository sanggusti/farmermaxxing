---
description: Where we stand on the ladder, and what to do about it today
allowed-tools: Bash(make ladder-sync), Bash(make calibrate), Bash(make ledger), Bash(git log:*), Read
---

Report the competition state and recommend today's action. Run:

1. `make ladder-sync` — ratings into `ledger.json`, our rank, and the band benchmarks.
2. `make calibrate` — whether our local evidence has predicted the ladder.

Then answer these, in this order, briefly:

- **Which agents are in the two scored slots, and how old are their ratings?**
  A submission restarts at 600 and needs ~24-48h to converge. A rating younger
  than that is not evidence; say so rather than reading it.
- **Has the anchor been beaten by the challenger?** If the challenger has a
  converged rating above the anchor's, say to update `CHAMPION` and promote it.
- **How many submissions are left today?** 5/day. Entry deadline 2026-09-23,
  final 2026-09-30. An unused slot is a measurement we chose not to take.
- **What is the gap to the band above us?** Rank, plus the 10th-percentile and
  top-10 cutoff from the sync output.

Do not run a search, do not gate, and do not submit. This command reports.

If `ledger.json` has a submission with no `local` claim, flag it: a submission
whose local justification was never written down cannot be calibrated, and
that is precisely how v10's -108 point result went unnoticed for eleven days.
