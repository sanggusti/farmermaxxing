"""Evaluate a parameter set against frozen opponents. The promotion gate.

    python -m sim.arena seeds=12 'opponents="starter,pass"' wandb=true

Configuration composes from configs/arena.yaml (issue #98); comma-separated
pool specs need quoting because Hydra's override grammar reserves the comma.

Every matchup is played from BOTH seats. The engine quotes both players against
the same pre-commit market inventory, so seats are nearly symmetric -- but weed
spawns are drawn from one RNG stream in seat order, so they are not exactly, and
playing both removes that bias for free.

Two metrics, used for different jobs:

  mean_bank -- near-deterministic and nearly opponent-independent, so it has far
               lower variance than win-rate. This is what CEM optimises.
  win_rate  -- what the ladder actually scores. Used as the final gate before a
               submission, never for tuning (too noisy at this sample size).
"""

import os
import statistics
import sys

from sim.harness import play, make_agent
from sim.fastplay import fast_play
from sim.census import PRODUCTS as CENSUS_PRODUCTS
from sim.opponents import resolve_pool
from params import Params
from obs import wandb_setup


def evaluate(params, opponents, seeds, steps=720, on_episode=None, labels=None,
             fast=True, metrics=False):
    """Play params against each opponent across seeds, from both seats.

    `fast` uses sim.fastplay, which skips kaggle_environments' replay
    bookkeeping and is ~3.3x quicker for identical results (see
    tests/test_fastplay.py). Set fast=False when a replay is needed.

    `opponents` may hold built-in names or Params instances. `labels` gives the
    display name for each; without it, Params opponents are all labelled the
    same and per-opponent breakdowns become useless.
    """
    if labels is None:
        labels = [o if isinstance(o, str) else f"params-{i}"
                  for i, o in enumerate(opponents)]

    rows = []
    for opp, label in zip(opponents, labels):
        for seed in seeds:
            for seat in (0, 1):
                me = make_agent(params)
                a, b = (me, opp) if seat == 0 else (opp, me)
                if fast:
                    r = fast_play(a, b, seed=seed, steps=steps, metrics=metrics)
                else:
                    r = play(a, b, seed=seed, steps=steps)

                my_bank = r["banks"][seat]
                their_bank = r["banks"][1 - seat]
                row = {
                    "opponent": label,
                    "seed": seed,
                    "seat": seat,
                    "bank": my_bank,
                    "opp_bank": their_bank,
                    "win": 1 if my_bank > their_bank else (0 if my_bank < their_bank else 0.5),
                    "status": r["statuses"][seat],
                }
                # Land/action census for OUR seat, merged flat so summarise()
                # can average them without knowing the names.
                if "metrics" in r:
                    row.update(r["metrics"][seat])
                    # ...plus the opponent's sales mix, under an `opp_` prefix.
                    # Only the mix, not the whole census: what the opponent does
                    # with its land is its business, but what it SELLS is the one
                    # comparison that located the real gap -- ~850 units of 5
                    # products for us against ~4,000 of 9 for the top of the
                    # ladder, on the same board.
                    row.update({f"opp_sell_units_{p}":
                                r["metrics"][1 - seat].get(f"sell_units_{p}", 0)
                                for p in CENSUS_PRODUCTS})
                rows.append(row)
                if on_episode:
                    on_episode(row)
    return rows


def per_opponent(rows):
    """Win rate and mean bank broken down by opponent.

    An aggregate can hide a regression: beating the built-ins by more while
    losing to the reigning champion nets out to roughly no change.
    """
    out = {}
    for row in rows:
        out.setdefault(row["opponent"], []).append(row)
    return {
        name: {
            "n": len(rs),
            "mean_bank": statistics.mean([r["bank"] for r in rs]),
            "win_rate": statistics.mean([r["win"] for r in rs]),
            # Margin, because bank and wins come apart: v6 banked 81,623
            # against v5 and won 0% of those matches. A per-opponent bank with
            # no margin beside it cannot distinguish the two.
            "mean_margin": statistics.mean([r["bank"] - r["opp_bank"] for r in rs]),
            "min_bank": min(r["bank"] for r in rs),
        }
        for name, rs in out.items()
    }


# Census keys carried through evaluate() and averaged by summarise(). Listed
# explicitly rather than inferred from the rows so a partially-populated batch
# (an old Modal container returning rows without them) degrades to a missing
# key instead of a silently wrong average over the subset that has it.
CENSUS_KEYS = (
    "productive_tile_day_frac",
    "weed_tile_day_frac",
    "idle_structure_tile_day_frac",
    "mean_unlocked_tiles",
    "max_quadrants",
    "plant_actions_per_day",
    "products_sold_distinct",
    "sell_units_total",
    "end_shed_units",
) + tuple(f"sell_units_{p}" for p in CENSUS_PRODUCTS)

# Not averaged -- a mean of hashes is meaningless. Carried per-row so an
# analysis can ask whether a bank advantage tracks the shop order, which would
# make it an RNG artefact rather than a strategy. See sim.census.record_town.
CONFOUND_KEYS = ("shop_order_hash",)


def opponent_mix(rows):
    """Mean per-product units the OPPONENTS sold, across these episodes.

    Returns {} when the rows carry no census (metrics=False), so callers can
    skip the comparison rather than print a row of zeros that reads as "the
    opponent sold nothing".
    """
    key = f"opp_sell_units_{CENSUS_PRODUCTS[0]}"
    if not rows or key not in rows[0]:
        return {}
    return {p: statistics.mean([r.get(f"opp_sell_units_{p}", 0) for r in rows])
            for p in CENSUS_PRODUCTS}


def shop_order_confound(rows):
    """How much of the bank spread is explained by which shop unlocked first.

    The engine picks the daily shop unlock from the same RNG stream that spawns
    weeds, and weeds draw once per empty unlocked tile across BOTH farms. So an
    agent that changes how it fills land changes its own shop order, and with it
    which products the town drains all season.

    That is a confound, not a strategy: on the ladder the opponent's land-fill
    profile is different, so a favourable unlock found on fixed seeds does not
    travel.

    Grouped on the FIRST shop only (8 possible values). Grouping on the whole
    order produces near-unique groups, and a variance ratio over singleton
    groups is ~1 no matter what the data says.

    Returns omega squared -- the between-group share of variance with the bias
    from having several groups removed, so it is comparable across runs with
    different group counts. Clamped at 0; negative means the grouping explains
    less than chance. `min_group` is reported because omega squared is only
    worth reading when the groups have several episodes each.
    """
    groups = {}
    for r in rows:
        if "shop_first" not in r:
            return None
        groups.setdefault(r["shop_first"], []).append(r["bank"])
    banks = [b for g in groups.values() for b in g]
    n, k = len(banks), len(groups)
    out = {"n": n, "n_groups": k,
           "min_group": min((len(g) for g in groups.values()), default=0)}
    if n < 3 or k < 2 or k >= n:
        return {**out, "omega_sq": 0.0}

    grand = statistics.mean(banks)
    ss_total = sum((b - grand) ** 2 for b in banks)
    ss_between = sum(len(g) * (statistics.mean(g) - grand) ** 2
                     for g in groups.values())
    if ss_total <= 0:
        return {**out, "omega_sq": 0.0}
    ms_within = (ss_total - ss_between) / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return {**out, "omega_sq": max(0.0, omega)}


def summarise(rows):
    banks = [r["bank"] for r in rows]
    wins = [r["win"] for r in rows]
    errors = sum(1 for r in rows if r["status"] != "DONE")
    out = {
        "n": len(rows),
        "mean_bank": statistics.mean(banks),
        "median_bank": statistics.median(banks),
        "min_bank": min(banks),
        "stderr": statistics.stdev(banks) / len(banks) ** 0.5 if len(banks) > 1 else 0.0,
        "win_rate": statistics.mean(wins),
        "errors": errors,
    }
    for key in CENSUS_KEYS:
        vals = [r[key] for r in rows if key in r]
        if len(vals) == len(rows) and vals:
            out[f"mean_{key}"] = statistics.mean(vals)
    return out


def main(cfg):
    """Run the sweep described by `cfg` (composed from configs/arena.yaml).

    Exits via sys.exit rather than returning the code: hydra.main discards
    return values, and errored episodes must fail the invocation.
    """
    params = Params.from_json(cfg.params) if cfg.params else Params()
    opponents, labels = resolve_pool(cfg.opponents)
    seeds = list(range(cfg.seeds))

    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"

    from dataclasses import asdict
    with wandb_setup.start("arena", config=asdict(params), group=cfg.group,
                           tags=["arena"]) as run:
        table = wandb_setup.table(
            ["opponent", "seed", "seat", "bank", "opp_bank", "win", "status"])

        def record(row):
            if table is not None:
                table.add_data(*[row[c] for c in
                                 ("opponent", "seed", "seat", "bank",
                                  "opp_bank", "win", "status")])

        rows = evaluate(params, opponents, seeds, cfg.steps,
                        on_episode=record, labels=labels, metrics=True)
        stats = summarise(rows)

        breakdown = per_opponent(rows)
        for k, v in stats.items():
            run.summary[k] = v
        for name, b in breakdown.items():
            run.summary[f"vs_{name}/mean_bank"] = b["mean_bank"]
            run.summary[f"vs_{name}/win_rate"] = b["win_rate"]
        if table is not None:
            run.log({"episodes": table})

    print(f"episodes    : {stats['n']}  ({len(opponents)} opponents x "
          f"{len(seeds)} seeds x 2 seats)")
    print(f"mean bank   : {stats['mean_bank']:>12,.0f}  +/- {stats['stderr']:,.0f}")
    print(f"median bank : {stats['median_bank']:>12,.0f}")
    print(f"worst bank  : {stats['min_bank']:>12,.0f}")
    print(f"win rate    : {stats['win_rate']:>12.1%}")
    if "mean_productive_tile_day_frac" in stats:
        print(f"land in use : {stats['mean_productive_tile_day_frac']:>12.1%}"
              f"   (weeds {stats['mean_weed_tile_day_frac']:.1%},"
              f" idle structures {stats['mean_idle_structure_tile_day_frac']:.1%})")
        print(f"quadrants   : {stats['mean_max_quadrants']:>12.2f}"
              f"   plants/day {stats['mean_plant_actions_per_day']:.1f}")
        print(f"products    : {stats['mean_products_sold_distinct']:>12.1f} of 9"
              f"   unsold at end {stats['mean_end_shed_units']:.0f}")
    print("per opponent:")
    for name, b in sorted(breakdown.items()):
        print(f"  {name:<22} bank {b['mean_bank']:>11,.0f}   win {b['win_rate']:>6.1%}   n={b['n']}")
    if stats["errors"]:
        print(f"ERRORS      : {stats['errors']} episodes did not finish cleanly")
        sys.exit(1)


if __name__ == "__main__":
    # Deferred decoration: importing this module (search.cem imports
    # CENSUS_KEYS) must never require hydra -- only running it does.
    import hydra
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hydra.main(config_path=os.path.join(_REPO, "configs"),
               config_name="arena", version_base=None)(main)()
