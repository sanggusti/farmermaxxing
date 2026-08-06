"""The agent's tunable parameters -- this dataclass *is* the CEM search space.

Defaults are hand-set opening priors derived from the engine's economics, not
tuned values. `search/cem.py` optimises them. Keep every field a plain scalar or
a flat dict of scalars so mutation and serialisation stay trivial.
"""

from dataclasses import dataclass, field, asdict, fields
import json


@dataclass
class Params:
    # --- labour -------------------------------------------------------------
    # Hiring is fib-priced per day (1,1,2,3,5,8,13,21,...) and resets daily, so
    # 10 hands costs 143 coins against a 3000 start. Actions, not coins, are the
    # binding constraint. `hands_*` is the target hand count at each stage.
    hands_early: int = 6          # days 0-4
    hands_mid: int = 10           # days 5-14
    hands_late: int = 14          # days 15+
    # Turns at the start of each day during which hiring orders are issued.
    # Must be >1 for any target above MAX_MARKET_ORDERS to be reachable.
    hire_turns: int = 2

    # --- land ---------------------------------------------------------------
    # Quadrants cost 1000 / 2000 / 4000. Buy only with a cash cushion left over.
    land_buy_reserve: float = 900.0   # coins to keep after buying
    land_max_quadrants: int = 4       # 1 = never expand
    # Only expand once existing land is close to full -- extra tiles are
    # worthless while we are labour-limited rather than land-limited.
    land_buy_empty_max: int = 6

    # --- portfolio ----------------------------------------------------------
    # Target tile counts by type. The planner fills tiles in this priority
    # order, subject to available unlocked land.
    target_geese: int = 8
    target_cows: int = 0
    target_sheep: int = 0
    target_wheat_tiles: int = 10  # wheat feeds animals; also the safest sell
    target_melon_tiles: int = 4
    target_carrot_tiles: int = 2
    target_tomato_tiles: int = 0
    target_strawberry_tiles: int = 0

    # Never spend past this cushion on livestock. Going broke is fatal: hands
    # cost coins, and with no hands nothing gets watered or fed, so crops weed
    # over and animals starve. Liquidity protects the whole operation.
    animal_cash_reserve: float = 600.0

    # Seeds bought per top-up. Must comfortably exceed the number of units,
    # or units end up idle waiting on seed.
    seed_batch: int = 12

    # Fertilizer doubles the daily yield bonus for 3 days. Wheat and carrot
    # only reach their listed max yield (6 and 4) with it, and a fertilized
    # melon caps two days early. Against that, a unit of fertilizer sells for
    # ~$100. Off by default; let the search decide.
    fertilize_enabled: int = 0
    # Only spend fertilizer down to this reserve, so selling still happens.
    fertilize_min_stock: int = 6

    # Stop planting slow crops once they cannot mature before the season ends.
    plant_cutoff_slack: int = 1

    # How many DIFFERENT crops may be planted in one turn. The engine validates
    # planting atomically per crop, so a mix is legal provided each crop stays
    # within its own seed count; the old single-crop rule was stricter than the
    # engine requires. It cost the early game: the top of the ladder holds 11
    # wheat and 8 melon by day 3 at 92% land use, against our 56%. At 1 the
    # behaviour is identical to before.
    plant_crops_per_turn: int = 1

    # --- season stages -------------------------------------------------------
    # Targets used to be static for all 30 days. Measured on the champion, land
    # utilisation peaks at 76% on day 19 and falls to 24% by day 29 -- 57 of 75
    # tiles idle while holding $119,915 -- because melon and strawberry
    # (first_yield_day 10) stop being plantable after day 18 and nothing in the
    # portfolio replaces them. Wheat and carrot mature in 2 days and could run
    # to day 26; they are also two of the products that end the season above
    # base price with market inventory below I0.
    #
    # A multiplier rather than a second target vector: the search then learns
    # the SHIFT, a much smaller correlated move, and 1.0 everywhere is a no-op.
    mix_switch_day: int = 30          # 30 = never switch
    late_target_mult: dict = field(default_factory=lambda: {
        "target_wheat_tiles": 1.0,
        "target_carrot_tiles": 1.0,
        "target_tomato_tiles": 1.0,
        "target_strawberry_tiles": 1.0,
        "target_melon_tiles": 1.0,
        "target_geese": 1.0,
        "target_cows": 1.0,
        "target_sheep": 1.0,
    })

    # Weight a crop's shortfall by how well it is currently paying, as
    # (price / base) ** elasticity. At 0.0 this is x**0 == 1, so the crop
    # ranking is byte-identical to shortfall alone. Above 0 the mix follows the
    # market, which matters because seven of nine products end the season ABOVE
    # base while we sell only three.
    crop_price_elasticity: float = 0.0

    # --- market -------------------------------------------------------------
    # Sell a unit only while its price stays >= frac * base. Premium goods get a
    # high floor (drip them); staples get a low one (dump them).
    sell_floor_frac: dict = field(default_factory=lambda: {
        "WHEAT": 0.55,
        "CARROT": 0.50,
        "TOMATO": 0.55,
        "STRAWBERRY": 0.75,
        "MELON": 0.80,
        "EGG": 0.55,
        "MILK": 0.75,
        "WOOL": 0.80,
        "FERTILIZER": 0.35,
    })

    # The shed caps at 100 items and overflow is silently discarded, so relax
    # the floors once it starts filling up.
    shed_pressure_at: int = 70
    shed_pressure_dump: int = 10

    # Stop buying feed once wheat gets expensive -- town demand pushes wheat
    # price up all season while egg price sags, and past this point a goose
    # eats more value than it lays.
    wheat_buy_max_price: float = 46.0

    # --- the opponent -------------------------------------------------------
    # Their whole farm is public and we have never read it. The market is the
    # only shared object in the game and it is worth up to 3x: the same agent
    # banks 141,397 against `starter` and 46,454 against a strong opponent on
    # identical seeds, and every coin of that gap arrives through prices.
    #
    # When a rival is about to harvest a stack of something we hold, our sell
    # floor for that product relaxes so we trade ahead of their supply rather
    # than after it. At 0.0 the discount is exactly 1.0 and the sell decision
    # is byte-identical to ignoring them.
    rival_supply_urgency: float = 0.0
    # Units of incoming rival supply that count as full pressure. Normalises
    # across products so the urgency parameter means the same thing for eggs
    # (dozens) as for melons (a handful).
    rival_supply_ref: float = 20.0
    # How far ahead to count their ripening crops, in days.
    rival_lookahead_days: int = 2

    # Wheat held back to feed animals (days of buffer per animal).
    wheat_reserve_days: float = 1.3

    # Liquidate everything over the final N days -- unsold stock scores zero.
    liquidate_days: int = 2

    # --- task priorities ----------------------------------------------------
    # Higher wins. Units pick tasks by `priority - distance * distance_penalty`.
    prio_feed: float = 100.0      # unfed animal for 2 days = permanent loss
    prio_water: float = 90.0      # unwatered plant for 2 days = weed
    prio_harvest_animal: float = 40.0
    prio_harvest_plant: float = 38.0
    prio_collect_fertilizer: float = 30.0
    prio_care: float = 20.0
    prio_place_animal: float = 60.0
    prio_build: float = 25.0
    prio_plant: float = 22.0
    prio_dig: float = 45.0
    prio_fertilize: float = 18.0
    distance_penalty: float = 3.0

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path):
        # Via from_dict, so unknown keys are ignored rather than raising.
        # Frozen opponent snapshots carry a `_notes` field, and an older
        # params.json will lack fields added since it was written; both must
        # still load.
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


DEFAULT = Params()


# --------------------------------------------------------------------------
# Search space for CEM. name -> (low, high, kind). "i" = round to int.
# Dotted names address a key inside a dict field, e.g. "sell_floor_frac.MELON".
# Anything not listed here is held fixed during search.
# --------------------------------------------------------------------------
SEARCH_SPACE = {
    "hands_early":            (0, 16, "i"),
    "hands_mid":              (0, 18, "i"),
    "hands_late":             (0, 18, "i"),
    "hire_turns":             (1, 4, "i"),

    "land_buy_reserve":       (0, 6000, "f"),
    "land_buy_empty_max":     (0, 25, "i"),

    "target_geese":           (0, 22, "i"),
    "target_cows":            (0, 12, "i"),
    "target_sheep":           (0, 12, "i"),
    "target_wheat_tiles":     (0, 25, "i"),
    "target_melon_tiles":     (0, 18, "i"),
    "target_carrot_tiles":    (0, 18, "i"),
    "target_tomato_tiles":    (0, 12, "i"),
    "target_strawberry_tiles": (0, 12, "i"),

    "animal_cash_reserve":    (0, 4000, "f"),
    "seed_batch":             (2, 28, "i"),
    "plant_cutoff_slack":     (0, 8, "i"),
    # Measured WORSE in isolation on the v5 portfolio -- against `starter`,
    # `v3-fixed` and v5 alike -- so it is not a free win. But that probe held
    # every other parameter fixed, and v5 has only two crops with a non-zero
    # target, so there was rarely a second crop worth planting. Whether a mix
    # pays depends on the portfolio it is planting, which is exactly the kind
    # of correlated question a single-parameter probe cannot answer. Left
    # searchable and inert at 1.
    "plant_crops_per_turn":   (1, 4, "i"),

    # Season-stage mix. Defaults (switch at 30, all multipliers 1.0, elasticity
    # 0.0) reproduce the previous behaviour exactly, so a warm start begins at
    # a known-good point rather than somewhere new.
    "mix_switch_day":         (8, 30, "i"),
    "crop_price_elasticity":  (0.0, 3.0, "f"),
    "late_target_mult.target_wheat_tiles":      (0.0, 6.0, "f"),
    "late_target_mult.target_carrot_tiles":     (0.0, 6.0, "f"),
    "late_target_mult.target_tomato_tiles":     (0.0, 6.0, "f"),
    "late_target_mult.target_strawberry_tiles": (0.0, 6.0, "f"),
    "late_target_mult.target_melon_tiles":      (0.0, 6.0, "f"),
    "late_target_mult.target_geese":            (0.0, 4.0, "f"),
    "late_target_mult.target_cows":             (0.0, 4.0, "f"),
    "late_target_mult.target_sheep":            (0.0, 4.0, "f"),

    "wheat_buy_max_price":    (15, 130, "f"),
    "rival_supply_urgency":   (0.0, 1.0, "f"),
    "rival_supply_ref":       (2.0, 80.0, "f"),
    "rival_lookahead_days":   (0, 6, "i"),
    "wheat_reserve_days":     (0.2, 4.0, "f"),
    "liquidate_days":         (1, 6, "i"),
    "shed_pressure_at":       (30, 98, "i"),
    "shed_pressure_dump":     (1, 50, "i"),

    "sell_floor_frac.WHEAT":      (0.05, 1.6, "f"),
    "sell_floor_frac.CARROT":     (0.05, 1.6, "f"),
    "sell_floor_frac.TOMATO":     (0.05, 1.6, "f"),
    "sell_floor_frac.STRAWBERRY": (0.05, 1.6, "f"),
    "sell_floor_frac.MELON":      (0.05, 1.6, "f"),
    "sell_floor_frac.EGG":        (0.05, 1.6, "f"),
    "sell_floor_frac.MILK":       (0.05, 1.6, "f"),
    "sell_floor_frac.WOOL":       (0.05, 1.6, "f"),
    "sell_floor_frac.FERTILIZER": (0.05, 1.6, "f"),

    "prio_feed":                (0, 140, "f"),
    "prio_water":               (0, 140, "f"),
    "prio_harvest_animal":      (0, 140, "f"),
    "prio_harvest_plant":       (0, 140, "f"),
    "prio_collect_fertilizer":  (0, 140, "f"),
    "prio_care":                (0, 140, "f"),
    "prio_place_animal":        (0, 140, "f"),
    "prio_build":               (0, 140, "f"),
    "prio_plant":               (0, 140, "f"),
    "prio_dig":                 (0, 140, "f"),
    "prio_fertilize":           (0, 140, "f"),
    "fertilize_enabled":        (0, 1, "i"),
    "fertilize_min_stock":      (0, 40, "i"),
    "distance_penalty":         (0.0, 15.0, "f"),
}


def flatten(params):
    """Params -> {search-space name: value}."""
    d = asdict(params)
    out = {}
    for name in SEARCH_SPACE:
        if "." in name:
            field_name, key = name.split(".", 1)
            out[name] = d[field_name][key]
        else:
            out[name] = d[name]
    return out


def unflatten(vec, base=None):
    """{search-space name: value} -> Params, clipped to bounds."""
    d = asdict(base or Params())
    for name, (lo, hi, kind) in SEARCH_SPACE.items():
        if name not in vec:
            continue
        v = min(hi, max(lo, vec[name]))
        v = int(round(v)) if kind == "i" else float(v)
        if "." in name:
            field_name, key = name.split(".", 1)
            d[field_name][key] = v
        else:
            d[name] = v
    return Params(**d)
