"""One place that decides how this project talks to Weights & Biases.

Every run -- a local arena sweep, a CEM generation, a Modal worker batch --
goes through `start()`, so project, grouping and tagging stay consistent and
runs from different machines line up in the same view.

Set WANDB_MODE=disabled to turn tracking off entirely (tests do this).
"""

import os
import signal
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


def _finish_on_signal(run):
    """Close the run cleanly when the process is signalled.

    The context manager calls finish() on normal exit and on exceptions, but a
    signal bypasses both, so every search stopped with pkill was recorded as
    `crashed`. A dozen such runs accumulated while diagnosing the container cap,
    implying failures that were actually deliberate stops.

    Chains to the previous handler so this does not swallow anyone else's.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def handler(signum, frame, _prev=previous):
            try:
                run.finish(exit_code=0)
            except Exception:
                pass
            if callable(_prev) and _prev not in (signal.SIG_IGN, signal.SIG_DFL):
                _prev(signum, frame)
            elif signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(128 + signum)

        try:
            signal.signal(sig, handler)
        except ValueError:
            # Not the main thread (Modal workers); nothing to install.
            pass


def start(job_type, config=None, group=None, name=None, tags=None):
    """Open a run. Always use as a context manager:

        with start("arena", config=cfg) as run:
            run.log({...})

    The `with` form matters on Modal: if a container is preempted mid-run,
    __exit__ still calls finish() and the run doesn't hang in "running" forever.
    Signals are handled separately, see `_finish_on_signal`.
    """
    if not enabled():
        return _NullRun()

    for k, v in _DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    import wandb

    run = wandb.init(
        project=PROJECT,
        job_type=job_type,
        group=group,
        name=name,
        config=config or {},
        tags=tags,
        settings=wandb.Settings(host=socket.gethostname()),
        reinit=True,
    )
    _finish_on_signal(run)
    return run


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
