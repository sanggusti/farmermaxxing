"""Play a single episode and report what happened.

    python -m sim.run --opponent starter --seed 0 --replay replays/x.json

`--debug` surfaces agent exceptions instead of letting the engine swallow them
into an ERROR status, which is almost always what you want while developing.
"""

import argparse
import json
import os

from sim.harness import play, make_agent, BUILTIN
from params import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="starter",
                    help=f"one of {BUILTIN}, or 'self'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--params", default=None, help="path to a params.json")
    ap.add_argument("--replay", default=None, help="write replay JSON here")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    params = Params.from_json(args.params) if args.params else Params()
    me = make_agent(params)
    opp = make_agent(params) if args.opponent == "self" else args.opponent

    result = play(me, opp, seed=args.seed, steps=args.steps, debug=args.debug)

    print(f"seed {args.seed}   us vs {args.opponent}")
    print(f"  bank   : {result['banks'][0]:>10,.0f}  vs {result['banks'][1]:>10,.0f}")
    print(f"  status : {result['statuses'][0]:<10} vs {result['statuses'][1]}")
    print(f"  result : {'TIE' if result['winner'] == -1 else ('WIN' if result['winner'] == 0 else 'LOSS')}")

    if args.replay:
        os.makedirs(os.path.dirname(args.replay) or ".", exist_ok=True)
        with open(args.replay, "w") as f:
            json.dump(result["env"].toJSON(), f)
        print(f"  replay : {args.replay}")

    # Non-zero exit if our agent errored, so `make` fails loudly.
    return 0 if result["statuses"][0] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
