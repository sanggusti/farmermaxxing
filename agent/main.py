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


def agent(obs):
    return _POLICY.act(obs)
