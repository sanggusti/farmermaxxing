"""A signalled run must be recorded as finished, not crashed.

Every search stopped with pkill showed up in W&B as `crashed`, because the
context manager covers normal exit and exceptions but not signals. That made
the project history imply about a dozen failures that were deliberate stops.
"""

import signal

from obs import wandb_setup


class FakeRun:
    def __init__(self):
        self.finished_with = None

    def finish(self, exit_code=None):
        self.finished_with = exit_code


def test_sigterm_finishes_the_run():
    run = FakeRun()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        wandb_setup._finish_on_signal(run)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        try:
            handler(signal.SIGTERM, None)
        except SystemExit:
            pass
        assert run.finished_with == 0, "run was not finished on SIGTERM"
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigint_finishes_and_still_raises():
    run = FakeRun()
    previous = signal.getsignal(signal.SIGINT)
    try:
        wandb_setup._finish_on_signal(run)
        handler = signal.getsignal(signal.SIGINT)
        raised = False
        try:
            handler(signal.SIGINT, None)
        except (KeyboardInterrupt, SystemExit):
            raised = True
        assert run.finished_with == 0
        assert raised, "SIGINT must still interrupt after finishing the run"
    finally:
        signal.signal(signal.SIGINT, previous)


def test_disabled_mode_returns_a_null_run(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = wandb_setup.start("test")
    run.log({"x": 1})
    run.finish()
