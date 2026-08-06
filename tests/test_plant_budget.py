"""Multi-crop planting must respect the engine's per-crop atomic validation.

The engine computes `blocked = {crop for crop, n in demand.items()
if n > seeds[crop]}` and drops EVERY plant action for a blocked crop. With
seven units and two seeds that froze a farm permanently once, from day 2 to
day 28. The invariant is per crop, so a mix is legal -- but only if each crop
independently stays within its own seed count.
"""

import sys

from params import Params
from policy import Policy


def _obs(seeds, empty=25, quadrants=("NW",)):
    tiles = [[None] * 10 for _ in range(10)]
    for y in range(10):
        for x in range(10):
            if not (x < 5 and y < 5):
                tiles[y][x] = "LOCKED"
    filled = 0
    for y in range(5):
        for x in range(5):
            if filled >= 25 - empty:
                break
            tiles[y][x] = {"kind": "PLANT", "crop": "MELON", "age": 1,
                           "watered_today": True, "yield_units": 0,
                           "consecutive_unwatered": 0}
            filled += 1
    return {
        "player": 0, "day": 3, "hour": 5,
        "farms": [{"money": 5000, "tiles": tiles, "farmer": [4, 4], "hands": [],
                   "unlocked_quadrants": list(quadrants), "hires_today": 0}],
        "private": {"shed": {}, "seeds": dict(seeds), "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def _plant_counts(params, seeds):
    obs = _obs(seeds)
    pol = Policy(params)
    me = obs["farms"][0]
    tasks = pol._build_tasks(me, obs["private"], obs["day"], 10,
                             obs["market"]["prices"])
    counts = {}
    for t in tasks:
        if t.action[0] == "PLANT":
            counts[t.action[1]] = counts.get(t.action[1], 0) + 1
    return counts


def test_never_queues_more_plants_than_seeds_held_per_crop():
    p = Params()
    p.plant_crops_per_turn = 4
    p.target_wheat_tiles, p.target_melon_tiles = 10, 10
    p.target_carrot_tiles, p.target_strawberry_tiles = 10, 10

    seeds = {"WHEAT": 2, "MELON": 3, "CARROT": 0, "STRAWBERRY": 1}
    counts = _plant_counts(p, seeds)
    for crop, n in counts.items():
        assert n <= seeds.get(crop, 0), (
            f"{n} PLANT tasks for {crop} against {seeds.get(crop, 0)} seeds -- "
            "the engine would silently drop all of them"
        )


def test_single_crop_default_plants_exactly_one_kind():
    p = Params()
    assert p.plant_crops_per_turn == 1
    p.target_wheat_tiles, p.target_melon_tiles = 10, 10
    counts = _plant_counts(p, {"WHEAT": 5, "MELON": 5})
    assert len(counts) <= 1, "the default must remain single-crop per turn"
