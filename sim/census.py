"""What a farm is actually doing with its land, in numbers.

Bank alone cannot distinguish "earned more per tile" from "used more tiles",
and issue #32 established that two thirds of our unlocked land sits empty for
the whole season. That is invisible in every metric the project logs today, so
a search cannot be judged on it and a promotion cannot be attributed to it.

`tile_stats` is the single definition of "productive", shared by `sim.trace`
(human-readable X-ray) and `sim.fastplay` (episode metrics for the search), so
the number printed in a trace and the number logged to W&B cannot drift apart.

Productive means the tile is currently converting time into goods: a growing
plant or a housed animal. An empty COOP or PASTURE is *not* productive -- it
occupies a tile and yields nothing, which is exactly the failure mode the build
overshoot produces, so counting it would hide the thing we are measuring.
"""

import hashlib
from collections import Counter

TURNS_PER_DAY = 24

# The nine tradeable products, in the engine's own order. Duplicated from
# agent/rules.py rather than imported, because sim/census.py runs inside the
# Modal image where only the mounted directories are importable and the import
# order differs; tests/test_parity.py pins agent/rules.py against the engine, so
# the risk is a stale copy here, not a wrong one there.
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")


def tile_stats(farm):
    """Land accounting for one farm at one instant.

    Returns unlocked/empty/planted/animals/idle_structures/weeds counts.
    `unlocked` is every tile we have paid for, so it is the denominator that
    makes utilisation comparable across farms that bought different amounts of
    land -- a farm with one quadrant fully planted is doing better per coin
    spent than a farm with four quadrants a quarter planted.
    """
    s = {"unlocked": 0, "empty": 0, "planted": 0, "animals": 0,
         "idle_structures": 0, "weeds": 0}
    for row in farm["tiles"]:
        for t in row:
            if t == "LOCKED":
                continue
            s["unlocked"] += 1
            if t is None:
                s["empty"] += 1
            elif "animal" in t:
                s["animals"] += 1
            elif t.get("kind") == "PLANT":
                s["planted"] += 1
            elif t.get("kind") == "WEED":
                s["weeds"] += 1
            else:
                # COOP or PASTURE standing empty.
                s["idle_structures"] += 1
    return s


def utilisation(farm):
    """Fraction of unlocked tiles currently doing work. 0.0 when landless."""
    s = tile_stats(farm)
    return (s["planted"] + s["animals"]) / s["unlocked"] if s["unlocked"] else 0.0


def tile_summary(tiles):
    """Counter of tile contents keyed by short lowercase name, for display."""
    c = Counter()
    for row in tiles:
        for t in row:
            if t is None:
                c["empty"] += 1
            elif t == "LOCKED":
                c["locked"] += 1
            elif t.get("kind") == "PLANT":
                c[t["crop"][:4].lower()] += 1
            elif t.get("kind") == "WEED":
                c["weed"] += 1
            elif "animal" in t:
                c[t["animal"][:4].lower()] += 1
            else:
                c[t["kind"][:4].lower()] += 1
    return c


class EpisodeCensus:
    """Accumulates per-player land and action statistics across an episode.

    Sampled once per day rather than per turn: tile state only changes at
    end-of-day boundaries for the purposes of this measurement, and sampling
    720 times would put a Python loop over 100 tiles inside the hot path of
    every search episode.
    """

    def __init__(self, n_players=2):
        self.n = n_players
        self.productive_tile_days = [0] * n_players
        self.unlocked_tile_days = [0] * n_players
        self.weed_tile_days = [0] * n_players
        self.idle_structure_tile_days = [0] * n_players
        self.plant_actions = [0] * n_players
        self.sell_units = [Counter() for _ in range(n_players)]
        self.max_quadrants = [0] * n_players
        self.first_quadrant_day = [{} for _ in range(n_players)]
        self.days_sampled = 0
        self.end_shed_units = [0] * n_players
        self.shop_order = []

    def sample_day(self, farms, day):
        """Record land state for every farm. Call once per game day."""
        self.days_sampled += 1
        for i in range(self.n):
            s = tile_stats(farms[i])
            self.productive_tile_days[i] += s["planted"] + s["animals"]
            self.unlocked_tile_days[i] += s["unlocked"]
            self.weed_tile_days[i] += s["weeds"]
            self.idle_structure_tile_days[i] += s["idle_structures"]
            q = len(farms[i]["unlocked_quadrants"])
            if q > self.max_quadrants[i]:
                self.max_quadrants[i] = q
                self.first_quadrant_day[i].setdefault(q, day)

    def record_action(self, i, action):
        """Count PLANT ops and SELL volume from one player's action dict."""
        if not action:
            return
        for op in (action.get("farmer"),) + tuple(action.get("hands") or ()):
            if op and op[0] == "PLANT":
                self.plant_actions[i] += 1
        for op in action.get("market") or ():
            if op and op[0] == "SELL" and len(op) >= 3:
                self.sell_units[i][op[1]] += op[2]

    def record_town(self, town):
        """Remember which shops unlocked, in order.

        This is not decoration. The engine draws weeds and the shop unlock from
        ONE rng stream per day, weeds first, one draw per *empty unlocked tile*
        across BOTH farms -- so the shop order is a function of how much land
        each player left idle. Same seed, different agent, different economy.

        That makes shop order a confound for every experiment that changes land
        use. A search on fixed seeds can find parameters that reliably unlock a
        favourable shop, bank the gain locally, and lose it entirely on the
        ladder where the opponent fills land differently. Recording the order is
        what lets us tell that apart from a real strategy afterwards.
        """
        self.shop_order = list(town.get("unlocked_shops") or ())

    def finish(self, sheds, town=None):
        for i in range(self.n):
            self.end_shed_units[i] = sum(sheds[i].values()) if sheds[i] else 0
        if town is not None:
            self.record_town(town)

    def result(self, i):
        """Metrics for player `i`, flat scalars ready for W&B."""
        unlocked = self.unlocked_tile_days[i]
        days = max(1, self.days_sampled)
        return {
            # The headline: what fraction of the land we paid for was doing
            # work, averaged over the season. Issue #32's number.
            "productive_tile_day_frac": (
                self.productive_tile_days[i] / unlocked if unlocked else 0.0),
            "weed_tile_day_frac": (
                self.weed_tile_days[i] / unlocked if unlocked else 0.0),
            "idle_structure_tile_day_frac": (
                self.idle_structure_tile_days[i] / unlocked if unlocked else 0.0),
            "mean_unlocked_tiles": unlocked / days,
            "max_quadrants": self.max_quadrants[i],
            "plant_actions_per_day": self.plant_actions[i] / days,
            # Request-side, not revenue: the engine may reject part of an order.
            # Treat as a breadth proxy. Issue #48's number.
            "products_sold_distinct": sum(
                1 for v in self.sell_units[i].values() if v > 0),
            "sell_units_total": sum(self.sell_units[i].values()),
            # Per-product volume. `sell_units_total` and `products_sold_distinct`
            # together say "we sell few units of few things"; they cannot say
            # WHICH things, and that turned out to be the whole story.
            #
            # Measured 2026-08-17 against the top of the ladder: they sell
            # 3,547-4,427 units across 9 products, we sell ~850 across 5, and
            # our realised price is ~$111/unit against their ~$33. We grow the
            # two lowest-yield crops (melon 0.55, strawberry 0.24 per tile-day)
            # on the two harshest glut curves and dump them below half base
            # price, while 7 of 9 products end the season ABOVE base price
            # untouched (issue #48). Nothing in the metrics made that visible,
            # so twelve versions of search optimised around it.
            **{f"sell_units_{p}": self.sell_units[i].get(p, 0) for p in PRODUCTS},
            "end_shed_units": self.end_shed_units[i],
            # Confound tracker, not a performance metric. Grouped on the FIRST
            # shop only: it unlocks on day 3 and drains for 27 days, so it is
            # the unlock that matters, and 8 possible values leave group sizes
            # large enough to say something. Grouping on the whole order gives
            # near-unique groups and a variance ratio that is ~1 by
            # construction, which looks alarming and means nothing.
            "shop_first": shop_index(self.shop_order),
        }


# Sorted because the engine draws with rng.choice(sorted(remaining)); this list
# is only used to turn a shop name into a stable small integer for grouping.
SHOP_NAMES = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
              "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")


def shop_index(order):
    """Index of the first shop unlocked, or -1 if none unlocked."""
    if not order:
        return -1
    try:
        return SHOP_NAMES.index(order[0])
    except ValueError:
        return -1
