"""The opponent mixture must not quietly become `starter` again.

Every regression here is a way the mixture could look right and behave like a
single-opponent search, which is the failure it exists to prevent.
"""

import statistics

import pytest

from search.league import build_cells, normalised_fitness, worst_opponent
from search.modal_app import summarise_cells, _CENSUS_KEYS
from sim.arena import CENSUS_KEYS, summarise


def test_cells_are_a_fixed_deterministic_product():
    """Cell order is positional in the returned banks; it must not wobble."""
    a = build_cells(["starter", "pass"], ["starter", "pass"], [0, 1])
    b = build_cells(["starter", "pass"], ["starter", "pass"], [0, 1])
    assert a == b
    assert len(a) == 2 * 2 * 2
    assert a[0] == ("starter", "starter", 0, 0)
    assert [c[1] for c in a].count("starter") == 4


def test_normalisation_gives_every_opponent_an_equal_vote():
    """A weak opponent paying huge banks must not outvote a strong one.

    This is the whole point. Candidate A wins the big-coin cell, candidate B
    wins the small-coin cell by the same number of standard deviations. Under a
    raw mean A wins on coin size alone; under per-cell normalisation they tie.
    """
    # Cell 0 is a `starter` matchup paying ~140k with a wide spread; cell 1 is
    # a `v3-fixed` matchup paying ~46k with a narrow one. A wins cell 0 by one
    # standard deviation, B wins cell 1 by one standard deviation -- an even
    # trade in information, but not in coins.
    a_banks = [150_000, 45_000]     # +1 sd on cell 0, -1 sd on cell 1
    b_banks = [130_000, 50_000]     # -1 sd on cell 0, +1 sd on cell 1
    population = [a_banks, b_banks]

    raw = [statistics.mean(x) for x in population]
    assert raw[0] > raw[1], "raw mean should favour the big-coin winner"

    fit = normalised_fitness(population)
    assert fit[0] == pytest.approx(0.0, abs=1e-9)
    assert fit[1] == pytest.approx(0.0, abs=1e-9)


def test_normalisation_still_ranks_a_dominating_candidate_first():
    fit = normalised_fitness([[100, 100], [200, 200], [50, 50]])
    assert fit[1] > fit[0] > fit[2]


def test_a_constant_cell_contributes_nothing_rather_than_dividing_by_zero():
    fit = normalised_fitness([[10, 5], [10, 9]])
    assert all(f == f for f in fit)          # not NaN
    assert fit[1] > fit[0]


def test_worst_opponent_ranks_by_margin_not_bank_level():
    """The v6 failure, as a unit test.

    v6's bank against `v5-mixture` was 81,623 -- the second-highest number in
    its per-opponent column -- while it lost 100% of those matches. In a shared
    market both banks rise together, so a high bank against a strong opponent
    can mean "we both did well and they did better".
    """
    stats = {"by_opponent": {
        "starter": {"mean_bank": 130_000, "mean_margin": 95_000},
        "v3-fixed": {"mean_bank": 75_000, "mean_margin": 12_000},
        # Highest bank of the three losers, but the only actual loss.
        "v5-mixture": {"mean_bank": 81_623, "mean_margin": -9_400},
    }}
    label, margin = worst_opponent(stats)
    assert label == "v5-mixture"
    assert margin == -9_400

    # Ranking on bank alone would have picked the wrong opponent entirely.
    by_bank = min(stats["by_opponent"],
                  key=lambda k: stats["by_opponent"][k]["mean_bank"])
    assert by_bank == "v3-fixed"


def test_worst_opponent_falls_back_to_bank_when_margin_is_absent():
    """An older scorer must degrade, not raise."""
    stats = {"by_opponent": {
        "starter": {"mean_bank": 140_000},
        "v3-fixed": {"mean_bank": 46_000},
    }}
    assert worst_opponent(stats) == ("v3-fixed", 46_000)


def test_summarise_cells_reports_margin_per_opponent():
    rows = [_row(100, 90), _row(300, 400), _row(50, 60), _row(70, 60)]
    out = summarise_cells(rows, ["a", "a", "b", "b"])
    assert out["by_opponent"]["a"]["mean_margin"] == -45     # (10 + -100) / 2
    assert out["by_opponent"]["b"]["mean_margin"] == 0       # (-10 + 10) / 2


def test_worst_opponent_is_absent_rather_than_wrong_without_a_breakdown():
    label, bank = worst_opponent({})
    assert label is None and bank != bank      # NaN, so a comparison cannot pass


def _row(bank, opp_bank, status="DONE"):
    return {"bank": bank, "opp_bank": opp_bank, "status": status}


def test_summarise_cells_matches_arena_summarise():
    """Two implementations of the same arithmetic, in modules that cannot
    import each other (the driver runs outside the Modal image). Pin them."""
    rows = [_row(100, 90), _row(200, 210), _row(50, 50), _row(400, 10)]
    labels = ["a", "a", "b", "b"]

    mine = summarise_cells(rows, labels)
    theirs = summarise([{**r, "win": 1 if r["bank"] > r["opp_bank"]
                         else (0 if r["bank"] < r["opp_bank"] else 0.5)}
                        for r in rows])

    for key in ("n", "mean_bank", "median_bank", "min_bank", "stderr",
                "win_rate", "errors"):
        assert mine[key] == pytest.approx(theirs[key]), key


def test_summarise_cells_breaks_down_by_opponent():
    rows = [_row(100, 90), _row(300, 90), _row(50, 60), _row(70, 60)]
    out = summarise_cells(rows, ["a", "a", "b", "b"])

    assert out["by_opponent"]["a"]["mean_bank"] == 200
    assert out["by_opponent"]["b"]["mean_bank"] == 60
    assert out["by_opponent"]["a"]["win_rate"] == 1.0
    assert out["by_opponent"]["b"]["win_rate"] == 0.5
    assert out["banks"] == [100, 300, 50, 70]


def test_census_key_lists_stay_in_sync():
    """modal_app duplicates the list because it must import without `sim`."""
    assert tuple(_CENSUS_KEYS) == tuple(CENSUS_KEYS)


def test_margins_are_aligned_with_banks_and_cells():
    """Margin is the paired statistic within an episode.

    Both players trade into one market, so a shock that lifts my bank lifts
    theirs too and cancels in the difference. Its sign is also the win the
    ladder scores, which mean bank is not: a candidate can bank more overall
    while losing a specific matchup, which is exactly what v7 did against v5
    (bank 86,573 vs 79,502, margin -2,887).
    """
    rows = [_row(100, 90), _row(300, 400), _row(50, 60), _row(70, 60)]
    out = summarise_cells(rows, ["a", "a", "b", "b"])

    assert out["margins"] == [10, -100, -10, 10]
    assert len(out["margins"]) == len(out["banks"])
    # Positional alignment with the cell list is what lets the caller
    # standardise per cell; a reorder here would silently mis-pair candidates.
    assert out["margins"] == [b - r["opp_bank"]
                              for b, r in zip(out["banks"], rows)]


def test_normalised_fitness_works_on_margins_too():
    """Ranking on margin must be the same standardisation, not a special case."""
    margins = [[10_000, -2_000], [5_000, 3_000]]
    fit = normalised_fitness(margins)
    assert len(fit) == 2
    # Candidate 0 wins cell 0 by 1 sd, candidate 1 wins cell 1 by 1 sd: a tie,
    # exactly as for banks.
    assert fit[0] == pytest.approx(0.0, abs=1e-9)
    assert fit[1] == pytest.approx(0.0, abs=1e-9)


def test_selection_score_follows_the_fitness_it_was_ranked_on():
    """Ranking on margin while selecting on bank is incoherent.

    v8's elites were margin-good, then the bank-best of them was chosen. It
    won every matchup and its mean bank stayed flat, which is exactly what
    optimising one quantity and picking on another produces.
    """
    from search.cem import selection_score

    stats = {
        "mean_bank": 90_000,
        "by_opponent": {
            "starter": {"mean_bank": 130_000, "mean_margin": +100_000},
            "v3-fixed": {"mean_bank": 70_000, "mean_margin": +10_000},
            "v5": {"mean_bank": 70_000, "mean_margin": -2_000},
        },
    }
    assert selection_score(stats, "mean_bank") == 90_000
    # Equal weight per opponent, not per episode.
    assert selection_score(stats, "mean_margin") == pytest.approx(36_000)


def test_selection_score_falls_back_to_bank_without_a_breakdown():
    from search.cem import selection_score
    assert selection_score({"mean_bank": 42.0}, "mean_margin") == 42.0
