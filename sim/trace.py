"""Per-day X-ray of an episode. The main debugging tool.

    python -m sim.trace --steps 240 --player 0

Prints one row per day: cash, what is on the tiles, what is in the shed, and
the market prices we are actually facing. Almost every bug shows up here as a
number that stops moving or a resource that never converts into cash.
"""

import argparse
from collections import Counter

from sim.harness import play, make_agent
from sim.census import TURNS_PER_DAY, tile_stats, tile_summary, utilisation
from params import Params


def fmt(counter, keys=None):
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    if keys:
        items = [(k, v) for k, v in items if k in keys]
    return " ".join(f"{k}:{v}" for k, v in items if v) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    params = Params.from_json(args.params) if args.params else Params()
    result = play(make_agent(params), args.opponent, seed=args.seed,
                  steps=args.steps, debug=True)
    steps = result["env"].steps
    p = args.player

    print(f"{'day':>3} {'cash':>8} {'hands':>5} {'unlk':>4} {'use':>4}  "
          f"{'tiles':<44} {'shed':<34} prices")
    print("-" * 143)
    util_days, unlocked_days = 0, 0
    for i in range(0, len(steps), TURNS_PER_DAY):
        s = steps[i]
        obs0 = s[0].observation
        farm = obs0["farms"][p]
        private = s[p].observation["private"]
        day = obs0["day"]
        prices = obs0["market"]["prices"]
        shown = {k: prices[k] for k in ("WHEAT", "EGG", "MELON", "FERTILIZER") if k in prices}
        st = tile_stats(farm)
        util_days += st["planted"] + st["animals"]
        unlocked_days += st["unlocked"]
        print(
            f"{day:>3} {farm['money']:>8,.0f} {len(farm['hands']):>5} "
            f"{st['unlocked']:>4} {utilisation(farm):>4.0%}  "
            f"{fmt(tile_summary(farm['tiles'])):<44} "
            f"{fmt(Counter(private['shed'])):<34} "
            + " ".join(f"{k[:4].lower()}={v}" for k, v in shown.items())
        )

    print("-" * 143)
    print(f"final banks: {result['banks']}  seeds left: {dict(steps[-1][p].observation['private']['seeds'])}")
    # Season-average land use. Bank alone cannot tell "earned more per tile"
    # apart from "used more tiles"; this is the number that separates them.
    print(f"land in use: {util_days / unlocked_days:.1%} of unlocked tile-days")


if __name__ == "__main__":
    main()
