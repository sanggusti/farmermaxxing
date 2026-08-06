"""Reading the opponent's farm must actually read it.

This started life reporting animals only, because crop tiles carry
`planted_day` and the code asked for `age`. `dict.get("age", 0)` made every
crop look freshly sown, so none ever entered the lookahead window. Nothing
raised, no test failed, and a probe measured the feature as worthless -- the
signal was simply half blind. These tests pin both halves.
"""

import pytest

from params import Params
from policy import Policy
from market import rival_discount


def _obs(rival_tiles, day=12):
    grid = [[None] * 10 for _ in range(10)]
    for (x, y), tile in rival_tiles.items():
        grid[y][x] = tile
    mine = [[None] * 10 for _ in range(10)]
    return {
        "player": 0, "day": day, "hour": 3,
        "farms": [
            {"money": 1000, "tiles": mine, "farmer": [4, 4], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
            {"money": 1000, "tiles": grid, "farmer": [4, 4], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def _crop(crop, planted_day, yield_units=1):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": yield_units}


def test_ripening_rival_crops_are_counted():
    """A melon planted on day 2 is ready on day 12: it must register."""
    pol = Policy(Params())
    obs = _obs({(1, 1): _crop("MELON", planted_day=2, yield_units=4)}, day=12)
    assert pol._rival_supply(obs, lookahead=2).get("MELON") == 4


def test_crops_outside_the_lookahead_window_are_not_counted():
    pol = Policy(Params())
    obs = _obs({(1, 1): _crop("MELON", planted_day=11)}, day=12)
    assert "MELON" not in pol._rival_supply(obs, lookahead=2)


def test_rival_animals_are_counted_by_product():
    pol = Policy(Params())
    obs = _obs({(2, 2): {"kind": "PASTURE", "animal": "COW"},
                (3, 2): {"kind": "COOP", "animal": "GOOSE"}}, day=12)
    supply = pol._rival_supply(obs, lookahead=2)
    assert supply.get("MILK") == 1
    assert supply.get("EGG") == 1


def test_missing_planted_day_does_not_silently_zero_the_age():
    """The exact shape of the original bug.

    With `planted_day` absent the fallback must treat the tile as *current*,
    not as sown on day 0 -- otherwise a malformed tile would look permanently
    unripe and quietly vanish from the signal.
    """
    pol = Policy(Params())
    tile = _crop("WHEAT", planted_day=0)
    del tile["planted_day"]
    obs = _obs({(1, 1): tile}, day=20)
    # age becomes 0, so wheat (first yield day 2) is 2 days out and still
    # lands inside a 2-day window -- present, not silently dropped.
    assert pol._rival_supply(obs, lookahead=2).get("WHEAT") == 1


def test_discount_is_exactly_one_when_urgency_is_zero():
    p = Params()
    assert p.rival_supply_urgency == 0.0
    assert rival_discount("MELON", {"MELON": 999}, p) == 1.0


def test_discount_scales_with_pressure_and_never_reaches_zero():
    p = Params()
    p.rival_supply_urgency, p.rival_supply_ref = 0.9, 20.0
    assert rival_discount("MELON", {}, p) == 1.0
    light = rival_discount("MELON", {"MELON": 5}, p)
    heavy = rival_discount("MELON", {"MELON": 40}, p)
    assert 1.0 > light > heavy >= 0.05
