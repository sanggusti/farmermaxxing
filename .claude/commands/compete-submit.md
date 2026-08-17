---
description: Preflight, then submit the day's challenger and record it in the ledger
argument-hint: <version> "<what changed>"
allowed-tools: Bash(make preflight), Bash(make bundle), Bash(make ladder-sync), Bash(.venv/bin/python -m sim.ledger:*), Bash(git status:*), Read
---

Submit `agent/params.json` as version $1 with message $2. A submission is
outward-facing and spends one of five daily slots, so:

**Ask the user to confirm before running the submit itself.** Show them what
they are about to spend the slot on: the version, the message, the local claim
and the pool it was measured on, and which currently-scored agent it will
displace (Kaggle scores the latest two, so a new submission pushes out the older
of the pair — say which one by name).

Steps:

1. `make preflight` — parity, contract, flat layout, the real tarball, both
   seats, crash-safety, and timing against a ladder opponent. If anything fails,
   stop. A rejected submission costs the slot and a day to discover.
2. `make ladder-sync` — confirm what is in the two slots right now and how old
   their ratings are. If the challenger slot holds something whose rating has
   not converged (<24h), say so: replacing it throws away the measurement we
   were paying for.
3. Confirm with the user.
4. `make submit CONFIRM=1 V=$1 M=$2 POOL=<pool> METRIC=<metric> VALUE=<n>` with
   the pool and metric from the gate run that justified this candidate.
5. Record the row it prints, using the submission id from the output:
   `.venv/bin/python -m sim.ledger add --id <ID> --version $1 --params
   agent/params.json --pool <pool> --metric <metric> --value <n> --note $2`
6. Set roles: `sim.ledger role --id <ID> --role challenger`, and confirm the
   anchor is still marked `anchor`.

**Slot discipline.** One anchor we believe in, one challenger per day. Never
replace the anchor on local evidence — only on a converged ladder rating. The
one time that rule was broken, a locally-dominant agent replaced the incumbent
and the recorded outcome was -108 points, and the comparison is still unresolved
because the incumbent was never given time to converge.
