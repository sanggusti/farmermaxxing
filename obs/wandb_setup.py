"""One place that decides how this project talks to Weights & Biases.

Every run -- a local arena sweep, a CEM generation, a Modal worker batch --
goes through `start()`, so project, grouping and tagging stay consistent and
runs from different machines line up in the same view.

Set WANDB_MODE=disabled to turn tracking off entirely (tests do this).
"""

import os
import socket

PROJECT = os.environ.get("WANDB_PROJECT", "farmermaxxing")

# Container-friendly defaults. Applied before wandb is imported so they take
# effect on first init.
_DEFAULT_ENV = {
    "WANDB_CACHE_DIR": "/tmp/wandb-cache",   # ~/.cache/wandb is pointless here
    "WANDB_DISABLE_GIT": "true",             # no repo inside a Modal container
    "WANDB_SILENT": "true",                  # keeps 200-container logs readable
}


class _NullRun:
    """Stand-in when tracking is disabled, so callers never branch on it."""

    summary = {}

    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def enabled():
    return os.environ.get("WANDB_MODE", "online") != "disabled"


def start(job_type, config=None, group=None, name=None, tags=None):
    """Open a run. Always use as a context manager:

        with start("arena", config=cfg) as run:
            run.log({...})

    The `with` form matters on Modal: if a container is preempted mid-run,
    __exit__ still calls finish() and the run doesn't hang in "running" forever.
    """
    if not enabled():
        return _NullRun()

    for k, v in _DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    import wandb

    return wandb.init(
        project=PROJECT,
        job_type=job_type,
        group=group,
        name=name,
        config=config or {},
        tags=tags,
        settings=wandb.Settings(host=socket.gethostname()),
        reinit=True,
    )


def table(columns):
    """A wandb.Table, or None when tracking is off."""
    if not enabled():
        return None
    import wandb

    return wandb.Table(columns=columns)


def log_params_artifact(run, path, name="agent-params", metadata=None):
    """Version a params.json alongside the run that produced it."""
    if not enabled():
        return
    import wandb

    art = wandb.Artifact(name=name, type="config", metadata=metadata or {})
    art.add_file(path)
    run.log_artifact(art)
