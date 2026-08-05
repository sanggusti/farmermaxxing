"""Per-day X-ray of an episode. The main debugging tool.

    python -m sim.trace --steps 240 --player 0

Prints one row per day: cash, what is on the tiles, what is in the shed, and
the market prices we are actually facing. Almost every bug shows up here as a
number that stops moving or a resource that never converts into cash.
"""

import argparse
from collections import Counter

from sim.harness import play, make_agent
from params import Params

TURNS_PER_DAY = 24


def tile_summary(tiles):
    c = Counter()
    for row in tiles:
        for t in row:
            if t is None:
                c["empty"] += 1
            elif t == "LOCKED":
                c["locked"] += 1
            elif t.get("kind") == "PLANT":
                c[t["crop"][:4].lower()] += 1
            elif t.get("kind") == "WEED":
                c["weed"] += 1
            elif "animal" in t:
                c[t["animal"][:4].lower()] += 1
            else:
                c[t["kind"][:4].lower()] += 1
    return c


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

    print(f"{'day':>3} {'cash':>8} {'hands':>5}  {'tiles':<44} {'shed':<34} prices")
    print("-" * 130)
    for i in range(0, len(steps), TURNS_PER_DAY):
        s = steps[i]
        obs0 = s[0].observation
        farm = obs0["farms"][p]
        private = s[p].observation["private"]
        day = obs0["day"]
        prices = obs0["market"]["prices"]
        shown = {k: prices[k] for k in ("WHEAT", "EGG", "MELON", "FERTILIZER") if k in prices}
        print(
            f"{day:>3} {farm['money']:>8,.0f} {len(farm['hands']):>5}  "
            f"{fmt(tile_summary(farm['tiles'])):<44} "
            f"{fmt(Counter(private['shed'])):<34} "
            + " ".join(f"{k[:4].lower()}={v}" for k, v in shown.items())
        )

    print("-" * 130)
    print(f"final banks: {result['banks']}  seeds left: {dict(steps[-1][p].observation['private']['seeds'])}")


if __name__ == "__main__":
    main()
