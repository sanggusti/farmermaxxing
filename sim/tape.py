"""Replay a recorded ladder agent as an opponent.

Our frozen pool is entirely our own lineage -- v1 through v8, every one of them
descended from the same hand-written policy. "v8 wins 100% against all eight
frozen opponents" therefore measured v8 against itself, eight times.

Against the real thing it loses every game:

    episode     their ladder bank    v8 vs them    they vs v8
    90044961            152,469         89,363       145,337
    90047087            148,032         53,073       119,787
    90047777             97,160         74,060       128,779
    90048477             97,459         59,471       129,760
    90045719             82,688         60,728       120,477

Two things stand out. We bank roughly half. And in the last row they bank
*more* against us (120,477) than they managed on the ladder against their real
opponent (82,688) -- playing us is easier than playing a peer, which means we
are barely contesting the market at all.

The fix is available because the meta is **open-loop**. Its recorded action
tape is seed-independent by construction, so it replays on seeds it never saw:

    seed 1121757285 (its own)   meta 145,337
    seed 20000                  meta 154,474
    seed 20001                  meta 129,218
    seed 30000                  meta 122,142

A tape is therefore a legitimate frozen opponent, and a far more informative
one than another copy of ourselves.

**What a tape is not.** It cannot react. Quantities are fixed, so a `SELL 40`
executes against whatever the shed happens to hold, and market orders that
depended on cash we did not spend may no-op. It is a strong scripted opponent,
not a simulation of that team's agent. Judge it by whether it stays
non-degenerate on fresh seeds, which `verify_tape` checks.
"""

import json
import os

TAPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tapes")
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


TURNS_PER_DAY = 24


class TapeAgent:
    """Replays a fixed action sequence, indexed by the OBSERVATION's own turn.

    A plain class rather than a closure because Modal pickles opponents to send
    them to workers, and a closure will not pickle. Deliberately __call__-able
    so `sim.fastplay._resolve` accepts it as an ordinary callable.

    **Stateless on purpose.** The first version kept an internal turn counter,
    and `sim.opponents.resolve_pool` builds one opponent instance which
    `sim.arena.evaluate` then reuses across every seed and seat. The tape
    exhausted after the first episode and returned PASS for the rest of the
    sweep -- and the sweep still completed, still reported per-opponent banks,
    and made a losing agent look like an 87.5% winner. That is issue #56 again,
    in a new costume.

    Deriving the index from `obs["day"]` and `obs["hour"]` makes reuse
    impossible to get wrong, which is better than a `reset()` every caller has
    to remember.
    """

    __slots__ = ("actions", "name")

    def __init__(self, actions, name="tape"):
        self.actions = actions
        self.name = name

    def __call__(self, obs):
        t = obs["day"] * TURNS_PER_DAY + obs["hour"]
        # The action recorded at index t+1 is the one taken FROM the
        # observation at index t. Off by one here still produces a complete
        # episode with plausible banks, which is why sim.ladder.verify exists.
        if t + 1 < len(self.actions) and self.actions[t + 1] is not None:
            return self.actions[t + 1]
        return PASS_ACTION


def extract(replay, seat):
    """Pull one seat's action sequence out of a replay."""
    return [step[seat].get("action") for step in replay["steps"]]


def save(actions, name, meta=None):
    os.makedirs(TAPE_DIR, exist_ok=True)
    path = os.path.join(TAPE_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump({"name": name, "meta": meta or {}, "actions": actions}, fh)
    return path


def names():
    if not os.path.isdir(TAPE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(TAPE_DIR) if f.endswith(".json"))


def load(name):
    """A tape opponent by name. Each call returns a FRESH agent.

    Fresh because the tape carries a turn counter. Reusing one instance across
    episodes would resume mid-season and quietly play nonsense -- the episode
    would still complete and the banks would still look plausible.
    """
    with open(os.path.join(TAPE_DIR, f"{name}.json")) as fh:
        d = json.load(fh)
    return TapeAgent(d["actions"], name=d.get("name", name))


def verify_tape(name, seeds=(20000, 20001, 20002), floor=40_000):
    """A tape must stay non-degenerate on seeds it never played.

    An unresolved or exhausted opponent still finishes an episode and still
    reports a number; the tell is a bank near the 3,000 starting money. This is
    the same failure that made every frozen Params opponent a silent no-op for
    weeks (issue #56), so it is checked rather than assumed.
    """
    from sim.fastplay import fast_play

    banks = []
    for seed in seeds:
        r = fast_play(load(name), "pass", seed=seed, steps=720)
        banks.append(r["banks"][0])
    return {"name": name, "banks": banks, "min": min(banks),
            "ok": min(banks) >= floor}
