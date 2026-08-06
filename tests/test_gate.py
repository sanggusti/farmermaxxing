"""The promotion gate must reject the specific ways a candidate fakes progress.

The decision rule is tested directly against synthetic statistics. Driving it
through real episodes would be slow and would depend on whatever the agent
happens to do at a given episode length, which tests the agent rather than the
rule.
"""

import statistics

import pytest

from params import Params
from sim.gate import decide, run_gate


def stats(mean, minimum, stderr=100.0, errors=0):
    return {"mean_bank": mean, "min_bank": minimum, "stderr": stderr,
            "errors": errors, "win_rate": 1.0, "n": 8, "median_bank": mean}


def winning(*names):
    return {n: {"win_rate": 1.0, "mean_bank": 1.0, "n": 4} for n in names}


def results(checks):
    return {label: ok for label, ok, _ in checks}


def find(checks, needle):
    return next(ok for label, ok, _ in checks if needle in label)


def test_clear_improvement_passes_every_check():
    checks, delta, _ = decide(stats(60_000, 55_000), stats(50_000, 48_000),
                              winning("starter"))
    assert delta == 10_000
    assert all(ok for _, ok, _ in checks)


def test_gain_inside_noise_is_rejected():
    """A 500-coin gain against a ~1,000-coin standard error is not evidence."""
    checks, _, se = decide(stats(50_500, 48_000, stderr=1_000),
                           stats(50_000, 48_000, stderr=1_000),
                           winning("starter"))
    assert se > 500
    assert find(checks, "sigma") is False


def test_mean_up_but_floor_collapsed_is_rejected():
    """The real case from the livestock ablation.

    Mean 51,131 over champion's 50,588 reads as an improvement, while the worst
    seed fell from 48,542 to 40,173. The ladder scores win/loss, so a collapsing
    floor costs matches the mean conceals.
    """
    checks, _, _ = decide(stats(51_131, 40_173, stderr=10.0),
                          stats(50_588, 48_542, stderr=10.0),
                          winning("starter"))
    assert find(checks, "sigma") is True     # the mean really did improve
    assert find(checks, "floor") is False    # but the floor disqualifies it


def test_losing_to_any_single_opponent_is_rejected():
    by_opp = {"starter": {"win_rate": 1.0, "mean_bank": 1.0, "n": 4},
              "champion-v1": {"win_rate": 0.25, "mean_bank": 1.0, "n": 4}}
    checks, _, _ = decide(stats(80_000, 70_000), stats(50_000, 48_000), by_opp)
    assert find(checks, "opponent") is False


def test_errored_episodes_are_rejected():
    checks, _, _ = decide(stats(80_000, 70_000, errors=1),
                          stats(50_000, 48_000), winning("starter"))
    assert find(checks, "errored") is False


def test_identical_params_do_not_pass_end_to_end():
    """Integration check: same params in both slots is never an improvement."""
    p = Params()
    checks, *_ = run_gate(p, p, "starter", n_seeds=2, steps=120)
    assert find(checks, "sigma") is False
    assert len(checks) == 4


def test_gate_defaults_to_clean_seeds_not_selection_seeds():
    """The gate must not judge a candidate on the seeds that selected it.

    CEM picks its champion on the HOLDOUT range once per generation. Judging on
    that same range asks the search to mark its own work, and the score it
    produces is biased upward by construction.
    """
    import inspect
    from sim import gate

    default = inspect.signature(gate.run_gate).parameters["offset"].default
    assert default == gate.CLEAN_OFFSET
    assert gate.CLEAN_OFFSET != gate.HOLDOUT_OFFSET


def test_seed_ranges_do_not_overlap():
    """Train, selection and clean seeds must be disjoint for any sane count."""
    from sim import gate

    train = set(range(0, 1000))
    selection = set(range(gate.HOLDOUT_OFFSET, gate.HOLDOUT_OFFSET + 1000))
    clean = set(range(gate.CLEAN_OFFSET, gate.CLEAN_OFFSET + 1000))
    assert not (train & selection)
    assert not (train & clean)
    assert not (selection & clean)


def test_paired_stderr_blocks_out_the_between_cell_variance():
    """The two agents play identical cells, so the cell effect must cancel.

    Here every cell is worth wildly different amounts (an opponent main
    effect), but the candidate beats the champion by exactly 100 everywhere.
    The unpaired formula sees the 100k spread as noise and cannot resolve the
    difference; the paired one sees a constant and resolves it exactly.
    """
    from sim.gate import paired_stderr

    champ_banks = [10_000, 50_000, 90_000, 130_000] * 4
    cand_banks = [b + 100 for b in champ_banks]

    paired = paired_stderr(cand_banks, champ_banks)
    unpaired = (statistics.stdev(cand_banks) / len(cand_banks) ** 0.5) ** 2
    unpaired = (2 * unpaired) ** 0.5

    assert paired == pytest.approx(0.0, abs=1e-9)
    assert unpaired > 10_000
    # A constant +100 improvement must be detectable; under the old formula the
    # margin check would have needed a delta of tens of thousands.
    assert 100 > 1.0 * paired


def test_paired_stderr_falls_back_rather_than_lying_when_unaligned():
    from sim.gate import paired_stderr

    assert paired_stderr(None, None) is None
    assert paired_stderr([1.0, 2.0], [1.0]) is None       # length mismatch
    assert paired_stderr([1.0], [1.0]) is None            # too few to estimate


def test_decide_uses_the_paired_error_when_banks_are_present():
    """Same means, same marginal stderrs -- only the pairing differs."""
    champ_banks = [10_000, 50_000, 90_000, 130_000] * 4
    cand_banks = [b + 2_000 for b in champ_banks]

    def stats(vals, **extra):
        return {
            "mean_bank": statistics.mean(vals),
            "min_bank": min(vals),
            "stderr": statistics.stdev(vals) / len(vals) ** 0.5,
            "win_rate": 1.0, "errors": 0, "n": len(vals), **extra,
        }

    by_opp = {"x": {"win_rate": 1.0, "mean_bank": 1.0}}

    unpaired_checks, _, unpaired_se = decide(
        stats(cand_banks), stats(champ_banks), by_opp)
    paired_checks, _, paired_se = decide(
        stats(cand_banks, banks=cand_banks),
        stats(champ_banks, banks=champ_banks), by_opp)

    assert paired_se < unpaired_se
    margin = "mean beats champion"
    assert not [c for c in unpaired_checks if margin in c[0]][0][1]
    assert [c for c in paired_checks if margin in c[0]][0][1]
