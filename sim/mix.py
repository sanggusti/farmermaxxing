"""What we actually build, next to what the top of the ladder actually builds.

The gap this closes. On 2026-08-17 the project could read the meta's realised
tile mix off a replay (`sim.ladder census`) but could only read its OWN mix off
`agent/params.json` -- and those are not the same kind of object:

    our crop targets, v10, at 3 quadrants
      melon 17 -> 51 absolute
      strawberry 12 -> 36
      wheat 2 -> 6
      cows/geese/sheep -> 12
      TOTAL DEMAND 105 tiles, against a 75-tile farm

Targets are per-quadrant and multiply by owned land, so the portfolio asks for
105 tiles of a 75-tile farm. Which means **the targets do not decide the mix**;
`Policy._wanted_crop`'s shortfall arbitration does, and melon wins it because
melon has the largest target and therefore the largest absolute shortfall,
permanently. Every target sweep the project has run was moving a number that
the code then overruled -- which is a decent explanation for why six of them
produced results nobody could interpret, including "shift melon to strawberry
costs -49,283" and "strawberry is already at its bound".

For comparison, the meta's realised build on episode 90044961 (both seats
banking ~152,000):

    d 3  wheat 11  melon  8  cow 3  sheep 1     util 92%
    d15  strawberry 40  melon  8  wheat  4  cow 8  sheep 6   util 88%
    d27  wheat 31  strawberry 22  cow 8  sheep 6             util 92%

Eight melon tiles, not fifty-one. Forty strawberry. Six sheep, which we do not
run at all. And they sell more melon than we do (279 units to our 160) from a
sixth of the tiles.

    python -m sim.mix                                  # vs the top pool
    python -m sim.mix --opponent tape:meta-a --seed 20000
    python -m sim.mix --params sim/opponents/v8-margin.json
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))

from sim.census import PRODUCTS                                  # noqa: E402
from sim.ladder import census_steps                              # noqa: E402
from sim.opponents import resolve_pool                           # noqa: E402

DAYS = (3, 9, 15, 21, 27, 29)


def realised_mix(params, opponent, seed, steps=720, days=DAYS):
    """Play one episode and census BOTH farms per day.

    Uses `sim.harness.play`, not `sim.fastplay`, because the census needs the
    step history and fast_play exists precisely to not keep it. One episode is
    ~1.5 s, which is fine for an instrument that runs on demand rather than
    inside a search loop.
    """
    from sim.harness import play, make_agent

    r = play(make_agent(params), opponent, seed=seed, steps=steps)
    env = r["env"]
    return census_steps(env.steps, r["banks"],
                        ["us", getattr(opponent, "name", "opponent")],
                        seed, days=days), r


def _mix_line(s):
    return " ".join(f"{k[:3].lower()}{v}" for k, v in
                    sorted(s["mix"].items(), key=lambda kv: -kv[1])[:6])


def report(params, opponents, labels, seed, steps=720):
    print(f"seed {seed}, {steps} steps\n")
    for opp, label in zip(opponents, labels):
        c, r = realised_mix(params, opp, seed, steps)
        us, them = r["banks"][0], r["banks"][1]
        verdict = "WIN " if us > them else "loss"
        print(f"vs {label}   {verdict}  us {us:>10,.0f}   them {them:>10,.0f}"
              f"   margin {us - them:>+11,.0f}")
        print(f"   {'day':>4}  {'':<3} {'$':>9}  {'q':>2} {'util':>5} "
              f"{'empty':>5} {'weed':>4}  realised mix")
        for day, seats in c["days"].items():
            for p, s in enumerate(seats):
                who = "us " if p == 0 else "them"
                print(f"   {day:>4}  {who:<3} {s['money']:>9,.0f}  "
                      f"{s['quadrants']:>2} {s['util']:>5.0%} "
                      f"{s['empty']:>5} {s['weeds']:>4}  {_mix_line(s)}")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sim.mix",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--params", default=os.path.join(REPO, "agent", "params.json"))
    ap.add_argument("--opponents", default="top",
                    help="pool spec; defaults to the prize band")
    ap.add_argument("--opponent", default=None, help="shorthand for a single one")
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args(argv)

    from params import Params

    spec = args.opponent or args.opponents
    opponents, labels = resolve_pool(spec)
    report(Params.from_json(args.params), opponents, labels, args.seed, args.steps)

    print("Read this as: which tiles are actually growing what, on both farms,")
    print("on the same board. A target in params.json is a request, not a mix --")
    print(f"v10 requests 105 tiles of a 75-tile farm, so {PRODUCTS[4].lower()} wins the")
    print("shortfall arbitration by default rather than by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
