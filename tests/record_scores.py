"""Re-record the expected champion scores after an intentional change.

    python -m tests.record_scores
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
os.environ["WANDB_MODE"] = "disabled"

from params import Params                     # noqa: E402
from sim.harness import play, make_agent      # noqa: E402

EXPECTED_PATH = os.path.join(REPO, "tests", "expected_scores.json")


def main():
    with open(EXPECTED_PATH) as f:
        seeds = sorted(int(s) for s in json.load(f))

    params = Params.from_json(os.path.join(REPO, "agent", "params.json"))
    recorded = {}
    for seed in seeds:
        bank = play(make_agent(params), "starter", seed=seed, steps=720)["banks"][0]
        recorded[str(seed)] = bank
        print(f"  seed {seed}: {bank:,.0f}")

    with open(EXPECTED_PATH, "w") as f:
        json.dump(recorded, f, indent=2, sort_keys=True)
    print(f"wrote {EXPECTED_PATH}")


if __name__ == "__main__":
    main()
