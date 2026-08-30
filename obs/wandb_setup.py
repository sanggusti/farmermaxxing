"""One place that decides how this project talks to Weights & Biases.

Every run -- a local arena sweep, a CEM generation, a Modal worker batch --
goes through `start()`, so project, grouping and tagging stay consistent and
runs from different machines line up in the same view.

Set WANDB_MODE=disabled to turn tracking off entirely (tests do this).

CANONICAL METRIC KEYS. Before 2026-08-21 the drivers had drifted (`iter` vs
`gen`, `holdout_cand_bank` vs `holdout_best_bank`, arena's `vs_{name}/` vs
the drivers' `vs/{label}/`), so every experiment minted fresh panels in a
workspace whose charts should have been shared. Every driver logs against
this table; a new metric that means the same thing as a listed one takes the
listed name.

Step keys (x-axis `gen`, passed as step_metric by every search driver):
  gen                    generation / iteration index
  train_best_bank        best mean bank on this generation's train cells
  train_pop_mean_bank    population mean of the same
  train_elite_mean_bank  elite-slice mean (cem drivers)
  holdout_best_bank      this generation's champion on the holdout cells
  holdout_win_rate / holdout_min_bank / generalisation_gap
  best_holdout_bank      running best selection score (bank OR margin --
                         units follow the `selection_metric` config entry)
  worst_opponent_margin  the no-regression guard's readout
  train_seeds_this_gen / episodes_this_gen / cum_train_episodes
  holdout_{census}       land/breadth census of the champion
  vs/{label}/mean_bank|win_rate   per-opponent; churns with the pool, and
                         that is accepted -- these are drill-down, not core
Driver-specific step keys (documented, not shared): sigma, axis_ratio
(cmaes); accepted, radius, step_norm (subspace); wall_seconds (kaggle
kernel); diag/* and xover/* (cem, stable block names so panels are reused).
Strings are configuration, not metrics: selection_metric lives in the run
config, never in a row.

Summary keys: best_holdout_bank, best_train_bank, clean_bank,
clean_min_bank, clean_selection_score, selection_bias, clean_vs/{label}/*,
clean_{census}, final_sigma/final_axis_ratio (cmaes), probe/* (tpu-probe),
and the gate/arena summaries (passed, delta, mean_bank, ...).
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


def start(job_type, config=None, group=None, name=None, tags=None,
          step_metric=None):
    """Open a run. Always use as a context manager:

        with start("arena", config=cfg) as run:
            run.log({...})

    The `with` form matters on Modal: if a container is preempted mid-run,
    __exit__ still calls finish() and the run doesn't hang in "running" forever.
    Signals are handled separately, see `_finish_on_signal`.

    `step_metric` names the x-axis for every logged key (search drivers pass
    "gen"). Without it wandb charts against its internal _step, so two runs
    whose loops log at different cadences (cmaes logs every generation,
    holdout keys only every `holdout_every`) never line up on one panel.
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
    if step_metric:
        run.define_metric(step_metric)
        run.define_metric("*", step_metric=step_metric)
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
