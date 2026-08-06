"""Search the small discrete switches by enumeration instead of by Gaussian.

Three integer parameters have ranges so short that a Gaussian cannot explore
them reliably even after the variance floor was fixed: `fertilize_enabled`
(0-1), `hire_turns` (1-4) and `liquidate_days` (1-6). Before the fix in refit()
they were frozen outright -- the probability of ever changing value once the
elites agreed was 0.0000%, 0.0000% and 0.0001%.

That matters more than three parameters would suggest, because these are
*regime* switches rather than magnitudes:

  fertilize_enabled  whether fertiliser is a field input or a product to sell
  hire_turns         whether the hand target above 10 is reachable at all,
                     since market orders are capped at 10 per turn
  liquidate_days     how long the endgame dump runs, and the whole shed is
                     worth nothing unsold

Flipping one changes what the *other* forty-two parameters should be. That is
the shape a local search cannot cross, and it is a plausible answer to the open
question in docs/4 about whether the five archetype restarts converged into one
band because the parameterisation is the ceiling or because the search could not
move. A restart re-rolls these; a generation could not.

There are only 48 combinations, so they can simply be enumerated. This module
produces one starting `Params` per combination, for short searches run in
parallel, rather than hoping a Gaussian finds them.

    python -m search.switches --base agent/params.json --out runs/switches
"""

import argparse
import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Flat agent modules, same reason as everywhere else: Kaggle unpacks the
# submission flat. Set the path here so `python -m search.switches` works on its
# own, which is the bug that shipped in PR #15 for sim/opponents.py.
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))

from dataclasses import replace                      # noqa: E402

from params import Params, SEARCH_SPACE             # noqa: E402

# Ranges are read from SEARCH_SPACE rather than written out, so widening a bound
# there cannot silently leave this enumeration behind.
SWITCHES = ("fertilize_enabled", "hire_turns", "liquidate_days")


def switch_values(name):
    lo, hi, kind = SEARCH_SPACE[name]
    if kind != "i":
        raise TypeError(f"{name} is not an integer dimension")
    return list(range(int(lo), int(hi) + 1))


def combinations():
    """Every assignment of the discrete switches, as a list of dicts."""
    grids = [[(n, v) for v in switch_values(n)] for n in SWITCHES]
    return [dict(combo) for combo in itertools.product(*grids)]


def label(combo):
    """Short filesystem-safe name, e.g. 'fert0-hire2-liq1'."""
    return (f"fert{combo['fertilize_enabled']}"
            f"-hire{combo['hire_turns']}"
            f"-liq{combo['liquidate_days']}")


def build(base=None):
    """(label, Params) for every switch combination, from one starting point.

    Only the switches differ. Everything else is inherited, so a difference in
    outcome is attributable to the switch rather than to a whole hand-built
    portfolio -- which is what separates this from search/archetypes.py.
    """
    base = base or Params()
    return [(label(c), replace(base, **c)) for c in combinations()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="params.json to vary; defaults to hand-set defaults")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "switches"))
    args = ap.parse_args()

    base = Params.from_json(args.base) if args.base else Params()
    os.makedirs(args.out, exist_ok=True)

    built = build(base)
    for name, params in built:
        params.to_json(os.path.join(args.out, f"{name}.json"))

    print(f"wrote {len(built)} starting points to {args.out}")
    print(f"switches: {', '.join(SWITCHES)}")
    for name in SWITCHES:
        print(f"  {name:<20} {switch_values(name)}")
    print()
    print("Run a short search from each, then compare on CLEAN seeds only --")
    print("picking the best of 48 on selection seeds would move the winner's")
    print("curse onto the comparison, with 48 draws to maximise over.")


if __name__ == "__main__":
    main()
