import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

# Never write tracking data from a test run.
os.environ["WANDB_MODE"] = "disabled"

# `agent.main.agent` swallows exceptions into a PASS turn, because on the ladder
# one KeyError forfeits a 720-turn match. That is right in production and wrong
# in a test: a policy that crashes every turn would otherwise show up as an
# episode that finishes DONE with a low bank, which is indistinguishable from a
# bad strategy. FM_STRICT makes it raise instead.
os.environ["FM_STRICT"] = "1"
