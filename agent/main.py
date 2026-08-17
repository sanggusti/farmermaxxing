"""Kaggriculture submission entry point.

Three contracts have to hold at once:

  * Kaggle's harness wants a module-level function named `agent`.
  * kaggle_environments' file loader takes the LAST callable defined in the
    module (`[v for v in env.values() if callable(v)][-1]`), whatever its name.
    So `agent` is defined last and nothing is imported after it.
  * That loader runs the file through `exec(compile(raw, path), {})`, which
    means **`__file__` does not exist**. Referencing it raises NameError, the
    loader swallows that into `InvalidArgument`, and the submission fails its
    validation episode. Hence `_agent_dir()` below.

Imports are flat (`from policy import ...`), because the submission tarball is
unpacked flat into /kaggle_simulations/agent/ with no package around it.
"""

import os
import sys


def _agent_dir():
    """Directory holding this file, however we were loaded."""
    # Ordinary import.
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    # exec()'d by kaggle_environments: no __file__, but compile() kept the path
    # on the code object, and f_back is the module frame that called us.
    src = sys._getframe(1).f_code.co_filename
    if src and os.path.dirname(src):
        return os.path.dirname(os.path.abspath(src))
    # Loaded from a string; fall back to where Kaggle unpacks submissions.
    if os.path.isdir("/kaggle_simulations/agent"):
        return "/kaggle_simulations/agent"
    return os.getcwd()


HERE = _agent_dir()
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from params import Params          # noqa: E402
from policy import Policy          # noqa: E402

_PARAMS_FILE = os.path.join(HERE, "params.json")
_PARAMS = Params.from_json(_PARAMS_FILE) if os.path.exists(_PARAMS_FILE) else Params()

# One policy per process. The engine runs each seat in its own process and the
# policy keeps no per-episode state, so a module-level singleton is safe.
_POLICY = Policy(_PARAMS)

# A turn that does nothing. `farmer` and `hands` take one action per unit and the
# engine treats an empty list as "no units directed"; an empty market list is a
# turn with no orders. This is a lost turn, not a lost episode.
_PASS = {"farmer": [], "hands": [], "market": []}


def agent(obs):
    """Never raise.

    `agent/policy.py` contains no exception handling at all: 591 lines of
    dictionary indexing into an observation whose schema is set by an engine
    under active development, with rule changes landing weekly. One KeyError
    anywhere in there sets the seat's status to ERROR, which forfeits the whole
    match -- 720 turns thrown away for one bad turn, and the ladder scores the
    forfeit.

    So a failing turn degrades to a pass. Silently, because
    `kaggle_environments` captures per-step stdout into the replay and a print
    on every turn of a broken episode inflates a ~30 MB replay; the failure is
    visible anyway as an agent that suddenly stops acting. Locally, set
    FM_STRICT=1 to get the traceback instead -- every test does.
    """
    try:
        return _POLICY.act(obs)
    except Exception:
        if os.environ.get("FM_STRICT"):
            raise
        return _PASS
