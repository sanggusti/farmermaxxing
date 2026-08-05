import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

# Never write tracking data from a test run.
os.environ["WANDB_MODE"] = "disabled"
