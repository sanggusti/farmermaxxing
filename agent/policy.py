"""The agent's brain.

Deliberately STATELESS: every turn the whole plan is recomputed from `obs`.
Nothing is remembered between turns. That costs a little redundant work (~1ms,
against a 1000ms budget) and buys a lot -- no desync between what we believe and
what the engine actually did, and any turn can be reproduced in isolation from
its observation alone, which makes debugging tractable.

Each turn:
  1. read state          -> what do I own, what does it need
  2. build a task list   -> every job worth doing right now, with a priority
  3. assign units        -> greedy: best (priority - travel distance)
  4. emit market orders  -> hire, buy, sell

The unit-to-task assignment is a plain greedy match. An optimal assignment
(Hungarian) is possible but the greedy one is within noise here and is far
easier to read.
"""

from rules import (
    CROPS, ANIMALS, MOVES, LAND_PRICES, MARKET_PARAMS,
    fib_hire_cost, is_shed_adjacent, water_bonus_window, shed_access_tiles,
)
from market import plan_sales, keep_wheat_for_feeding

TURNS_PER_DAY = 24
TOTAL_DAYS = 30
MAX_MARKET_ORDERS = 10


def harvest_age(crop):
    """Age (in days) at which a one-time crop reaches peak yield.

    Yield starts at 1 and gains +1 per watered day inside the bonus window,
    capped at max_yield. Harvesting earlier wastes yield; later risks decay.
    Returns None for ongoing crops, which are harvested whenever ripe.
    """
    cd = CROPS[crop]
    if cd["ongoing"]:
        return None
    start, end = water_bonus_window(crop)
    return min(end, start + cd["max_yield"] - 2)


class Task:
    __slots__ = ("prio", "pos", "action", "kind", "needs")

    def __init__(self, prio, pos, action, kind, needs=None):
        self.prio = prio          # higher is more urgent
        self.pos = pos            # (x, y) tile to stand on
        self.action = action      # action list to emit once there
        self.kind = kind          # label, for logging
        self.needs = needs        # item required in unit inventory, or None


class Policy:
    def __init__(self, params):
        self.p = params

    # ------------------------------------------------------------------ main
    def act(self, obs):
        p = self.p
        me = obs["farms"][obs["player"]]
        private = obs["private"]
        day, hour = obs["day"], obs["hour"]
        board = len(me["tiles"])

        units = self._units(me, private)
        tasks = self._build_tasks(me, private, day, board,
                                  obs["market"]["prices"])
        assignment = self._assign(units, tasks, board)

        farmer_action = assignment.get(0, ["PASS"])
        hand_actions = [assignment.get(i + 1, ["PASS"]) for i in range(len(me["hands"]))]
        market = self._market_orders(me, private, obs, day, hour, board)

        return {"farmer": farmer_action, "hands": hand_actions, "market": market}

    # ----------------------------------------------------------------- state
    def _units(self, me, private):
        """[(index, (x, y), inventory)] -- index 0 is the main farmer."""
        invs = private.get("inventories", [{}])
        out = [(0, tuple(me["farmer"]), invs[0] if invs else {})]
        for i, pos in enumerate(me["hands"]):
            inv = invs[i + 1] if i + 1 < len(invs) else {}
            out.append((i + 1, tuple(pos), inv))
        return out

    def _carried(self, private):
        """Everything currently held in unit inventories (not yet in the shed)."""
        total = {}
        for inv in private.get("inventories", []):
            for item, n in inv.items():
                total[item] = total.get(item, 0) + n
        return total

    def _scan(self, me):
        """Count what is currently on the farm."""
        counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0, "COOP": 0, "PASTURE": 0,
                  "empty": 0, "weeds": 0}
        crops = {c: 0 for c in CROPS}
        for row in me["tiles"]:
            for tile in row:
                if tile is None:
                    counts["empty"] += 1
                elif tile == "LOCKED":
                    continue
                elif tile.get("kind") == "PLANT":
                    crops[tile["crop"]] += 1
                elif tile.get("kind") == "WEED":
                    counts["weeds"] += 1
                elif "animal" in tile:
                    counts[tile["animal"]] += 1
                else:
                    counts[tile["kind"]] += 1
        return counts, crops

    # ----------------------------------------------------------------- tasks
    def _build_tasks(self, me, private, day, board, prices=None):
        p = self.p
        tasks = []
        counts, crops = self._scan(me)
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})

        scale = len(me["unlocked_quadrants"])
        want = self._wanted_crop(counts, crops, day, scale, prices)

        # Never queue more PLANT tasks than we hold seeds. The engine validates
        # planting atomically per crop:
        #     blocked = {crop for crop, n in demand.items() if n > seeds[crop]}
        # so if 7 units all try to plant wheat while we hold 2 seeds, ALL seven
        # are silently dropped and the farm deadlocks -- forever, since nothing
        # about the state changes to break the tie.
        plant_budget = seeds.get(want, 0) if want else 0
        # Fertilizer above the reserve is available for field use; the rest
        # stays earmarked for sale.
        carried = self._carried(private)
        spare_fert = max(0, shed.get('FERTILIZER', 0)
                         + carried.get('FERTILIZER', 0) - p.fertilize_min_stock)

        for y, row in enumerate(me["tiles"]):
            for x, tile in enumerate(row):
                pos = (x, y)
                if tile == "LOCKED":
                    continue

                if tile is None:
                    if plant_budget > 0:
                        tasks.append(Task(p.prio_plant, pos, ["PLANT", want], f"plant-{want}"))
                        plant_budget -= 1
                    elif self._needs_structure(counts, day, scale):
                        kind = self._needs_structure(counts, day, scale)
                        tasks.append(Task(p.prio_build, pos, [f"BUILD_{kind}"], f"build-{kind}"))
                    continue

                kind = tile.get("kind")

                if kind == "WEED":
                    tasks.append(Task(p.prio_dig, pos, ["DIG"], "dig"))

                elif kind == "PLANT":
                    tasks += self._plant_tasks(tile, pos, day, spare_fert)

                elif "animal" in tile:
                    tasks += self._animal_tasks(tile, pos)

                elif kind in ("COOP", "PASTURE"):
                    # Empty structure -- fill it if we hold a matching animal.
                    for name, a in ANIMALS.items():
                        if a["structure"] == kind:
                            tasks.append(Task(
                                p.prio_place_animal, pos, ["PLACE", name],
                                f"place-{name}", needs=name,
                            ))
                            break

        tasks += self._fetch_tasks(me, private, counts, board)
        return tasks

    def _fetch_tasks(self, me, private, counts, board):
        """Trips to the shed to collect what other tasks require.

        Without these, goods bought into the shed never reach a unit's hands:
        PLACE needs the animal in inventory and FEED needs wheat there, and a
        unit only wanders to the shed when it has nothing else to do -- which,
        on a busy farm, is never.
        """
        p = self.p
        shed = private.get("shed", {})
        carried = self._carried(private)
        tasks = []

        # Only unlocked shed-access tiles work; PICKUP no-ops on a locked tile.
        access = [(x, y) for (x, y) in shed_access_tiles(board)
                  if me["tiles"][y][x] != "LOCKED"]
        if not access:
            return tasks

        # Livestock waiting on an empty structure.
        for name, a in ANIMALS.items():
            waiting = shed.get(name, 0)
            if not waiting:
                continue
            slots = counts[a["structure"]] - carried.get(name, 0)
            for i in range(min(waiting, max(0, slots))):
                tasks.append(Task(
                    p.prio_place_animal + 1, access[i % len(access)],
                    ["PICKUP", name, 1], f"fetch-{name}",
                ))

        # Fertilizer for field use, when the search has enabled it.
        spare_fert = max(0, shed.get("FERTILIZER", 0) - p.fertilize_min_stock)
        if p.fertilize_enabled and spare_fert > 0 and carried.get("FERTILIZER", 0) == 0:
            tasks.append(Task(p.prio_fertilize + 1, access[0],
                              ["PICKUP", "FERTILIZER", 4], "fetch-FERTILIZER"))

        # Wheat for animals that still need feeding today.
        unfed = self._count_unfed(me)
        if unfed > carried.get("WHEAT", 0) and shed.get("WHEAT", 0) > 0:
            trips = min(len(access), 1 + unfed // 8)
            for i in range(trips):
                tasks.append(Task(
                    p.prio_feed - 1, access[i % len(access)],
                    ["PICKUP", "WHEAT", 8], "fetch-WHEAT",
                ))

        return tasks

    def _count_unfed(self, me):
        n = 0
        for row in me["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and "animal" in tile and not tile["fed_today"]:
                    n += 1
        return n

    def _plant_tasks(self, tile, pos, day, spare_fertilizer=0):
        p = self.p
        out = []
        crop = tile["crop"]
        cd = CROPS[crop]
        age = day - tile["planted_day"]

        if not tile["watered_today"]:
            # Watering is only *useful* inside the bonus window, but it is
            # always *necessary* -- two dry days turns the tile into a weed.
            out.append(Task(p.prio_water, pos, ["WATER"], "water"))

        # Fertilize only where it pays: inside (or just before) the bonus
        # window, and only while the plant is not already covered. The engine
        # applies the bonus for day..day+2, and only on days the plant is also
        # watered.
        if p.fertilize_enabled and spare_fertilizer > 0 and tile["fertilized_until_day"] < day:
            window = water_bonus_window(crop)
            in_window = (window is None) or (age >= window[0] - 1 and age <= window[1])
            if in_window:
                out.append(Task(p.prio_fertilize, pos, ["FERTILIZE"], "fertilize",
                                needs="FERTILIZER"))

        ripe = tile.get("yield_units", 0) > 0 and age >= cd["first_yield_day"]
        if ripe:
            target = harvest_age(crop)
            if target is None or age >= target:
                out.append(Task(p.prio_harvest_plant, pos, ["HARVEST"], "harvest-plant"))
        return out

    def _animal_tasks(self, tile, pos):
        p = self.p
        out = []
        if not tile["fed_today"]:
            out.append(Task(p.prio_feed, pos, ["FEED"], "feed", needs="WHEAT"))
        if tile.get("yield_units", 0) > 0:
            out.append(Task(p.prio_harvest_animal, pos, ["HARVEST"], "harvest-animal"))
        if tile.get("fertilizer_available"):
            out.append(Task(p.prio_collect_fertilizer, pos, ["COLLECT_FERTILIZER"], "fert"))
        if not tile["cared_today"]:
            out.append(Task(p.prio_care, pos, ["CARE"], "care"))
        return out

    def _target(self, kind, day, scale=1):
        """Target count for `kind`, scaled by land and by the season stage.

        Targets were static for all 30 days while the farm's situation is not.
        Measured on the champion, land utilisation peaks at 76% on day 19 and
        then falls monotonically to 24% by day 29 -- 57 of 75 tiles idle while
        holding $119,915. The cause is in `_wanted_crop`: melon and strawberry
        have `first_yield_day` 10, so `plant_cutoff_slack` stops planting them
        after day 18, and nothing in the portfolio replaces them. Wheat and
        carrot mature in 2 days and could be planted until day 26.

        `late_target_mult` lets the search shift the mix at `mix_switch_day`
        instead of committing to one portfolio for the whole season. It is a
        multiplier rather than a second target vector deliberately: CEM then
        learns the *shift*, which is a much smaller correlated move than a
        whole second portfolio, and the identity multiplier is a no-op.
        """
        base = getattr(self.p, kind) * scale
        if day < self.p.mix_switch_day:
            return base
        return base * self.p.late_target_mult.get(kind, 1.0)

    def _needs_structure(self, counts, day=0, scale=1):
        """Which structure to build next, if any."""
        if counts["GOOSE"] + counts["COOP"] < self._target("target_geese", day, scale):
            return "COOP"
        pasture_want = (self._target("target_cows", day, scale)
                        + self._target("target_sheep", day, scale))
        if counts["COW"] + counts["SHEEP"] + counts["PASTURE"] < pasture_want:
            return "PASTURE"
        return None

    def _wanted_crop(self, counts, crops, day, scale=1, prices=None):
        """Which crop a free tile should get, or None.

        Picks the crop with the largest *relative* shortfall rather than the
        first one under target. Returning the first under target starves
        everything below it in the list: one-time crops vacate their tile when
        harvested, so melon drops under target on every cycle and monopolises
        every free tile forever. Measured on the ladder, a champion configured
        for 26 carrot and 8 tomato tiles planted tomato zero times all season.

        Relative shortfall fixes the comparison: melon at 8 of 10 (0.20 short)
        correctly loses to carrot at 0 of 26 (1.00 short).

        Targets are per-quadrant and scale with owned land, so a bought quadrant
        fills up instead of lying fallow.
        """
        p = self.p
        days_left = TOTAL_DAYS - day
        targets = [
            ("WHEAT", "target_wheat_tiles"),
            ("MELON", "target_melon_tiles"),
            ("CARROT", "target_carrot_tiles"),
            ("TOMATO", "target_tomato_tiles"),
            ("STRAWBERRY", "target_strawberry_tiles"),
        ]

        best, best_score = None, 0.0
        for crop, attr in targets:
            want = self._target(attr, day, scale)
            if want <= 0 or crops[crop] >= want:
                continue
            # Don't plant what cannot mature before the season ends.
            if CROPS[crop]["first_yield_day"] + p.plant_cutoff_slack > days_left:
                continue
            score = (want - crops[crop]) / want
            # Weight the shortfall by how well the crop is currently paying.
            # Seven of nine products end the season ABOVE base with market
            # inventory below I0 -- the town drains faster than two players
            # supply -- and the only two we push to a discount are the two we
            # concentrate on. At elasticity 0 this is x**0 == 1, so the default
            # is byte-identical to ranking on shortfall alone.
            if prices and p.crop_price_elasticity:
                base_price = MARKET_PARAMS[crop]["base"]
                now = prices.get(crop)
                if now and base_price:
                    score *= (now / base_price) ** p.crop_price_elasticity
            if score > best_score:
                best, best_score = crop, score
        return best

    # ------------------------------------------------------------ assignment
    def _assign(self, units, tasks, board):
        """Greedy: each unit takes its best remaining task."""
        p = self.p
        out = {}
        taken = set()

        for idx, pos, inv in units:
            best, best_score = None, None
            for t_i, task in enumerate(tasks):
                if t_i in taken:
                    continue
                if task.needs and inv.get(task.needs, 0) <= 0:
                    continue
                dist = abs(task.pos[0] - pos[0]) + abs(task.pos[1] - pos[1])
                score = task.prio - dist * p.distance_penalty
                if best_score is None or score > best_score:
                    best, best_score = t_i, score

            if best is None:
                out[idx] = self._idle_action(pos, inv, board, tasks)
                continue

            taken.add(best)
            task = tasks[best]
            out[idx] = task.action if task.pos == pos else self._step_toward(pos, task.pos)

        return out

    def _idle_action(self, pos, inv, board, tasks):
        """No task available: restock at the shed if anything there is wanted."""
        wanted = {t.needs for t in tasks if t.needs}
        if wanted:
            if is_shed_adjacent(pos, board):
                for item in ("WHEAT", "GOOSE", "COW", "SHEEP"):
                    if item in wanted and inv.get(item, 0) <= 0:
                        return ["PICKUP", item, 8 if item == "WHEAT" else 1]
            else:
                return self._step_toward(pos, (board // 2 - 1, board // 2 - 1))
        # Drop what we are carrying so it can be sold.
        if inv and is_shed_adjacent(pos, board):
            return ["DROP"]
        return ["PASS"]

    def _step_toward(self, pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        # Move along the longer axis first; ties go horizontal.
        if abs(dx) >= abs(dy) and dx != 0:
            return ["EAST"] if dx > 0 else ["WEST"]
        if dy != 0:
            return ["SOUTH"] if dy > 0 else ["NORTH"]
        return ["PASS"]

    # --------------------------------------------------------------- market
    def _market_orders(self, me, private, obs, day, hour, board):
        p = self.p
        orders = []
        money = me["money"]
        counts, crops = self._scan(me)
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        market_inv = obs["market"]["inventory"]
        scale = len(me["unlocked_quadrants"])

        # 1. Labour, at the start of each day. Hands act from the following
        # turn. Spread across the first few turns because market orders are
        # truncated at MAX_MARKET_ORDERS, so hiring only at hour 0 silently
        # capped the whole farm at 10 hands however high the target was.
        if hour < p.hire_turns:
            target = self._hand_target(day)
            for _ in range(target - me["hires_today"]):
                cost = fib_hire_cost(me["hires_today"] + len(orders))
                if money < cost:
                    break
                money -= cost
                orders.append(["HIRE"])
                if len(orders) >= MAX_MARKET_ORDERS:
                    return orders

        # 2. Land. More tiles is the only way past the action ceiling.
        n_extra = len(me["unlocked_quadrants"]) - 1
        if n_extra < min(3, p.land_max_quadrants - 1):
            cost = LAND_PRICES[n_extra]
            if money >= cost + p.land_buy_reserve and counts["empty"] <= p.land_buy_empty_max:
                orders.append(["BUY_LAND"])
                money -= cost

        # 3. Animals. Only buy one we can actually house *and* feed today --
        # an animal with nowhere to live sits in the shed, and an unfed one is
        # gone for good after two days.
        carried = self._carried(private)
        for name, target in (
                ("GOOSE", self._target("target_geese", day, scale)),
                ("COW", self._target("target_cows", day, scale)),
                ("SHEEP", self._target("target_sheep", day, scale))):
            structure = ANIMALS[name]["structure"]
            in_transit = shed.get(name, 0) + carried.get(name, 0)
            owned = counts[name] + in_transit
            if owned >= target:
                continue
            # Never hold more livestock than there are empty structures.
            if in_transit >= counts[structure]:
                continue
            if money < ANIMALS[name]["cost"] + p.animal_cash_reserve:
                continue
            orders.append(["BUY_ANIMAL", name, 1])
            money -= ANIMALS[name]["cost"]
            break

        # 4. Seeds. Buy enough to keep every idle unit planting -- seeds are the
        # cheapest thing on the board and running dry stalls the whole farm.
        want = self._wanted_crop(counts, crops, day, scale,
                                 obs["market"]["prices"])
        if want:
            held = seeds.get(want, 0)
            need_seeds = min(counts["empty"], p.seed_batch) - held
            unit_cost = CROPS[want]["seed"]
            affordable = int(max(0, money - p.animal_cash_reserve) // unit_cost)
            n = min(need_seeds, affordable)
            if n > 0:
                orders.append(["BUY_SEED", want, n])
                money -= unit_cost * n

        # 5. Wheat to feed animals. Counts livestock we merely *own* (including
        # shed/inventory), because those will need feeding as soon as they land.
        n_animals = (counts["GOOSE"] + counts["COW"] + counts["SHEEP"]
                     + sum(shed.get(a, 0) + carried.get(a, 0) for a in ANIMALS))
        need = keep_wheat_for_feeding(shed, n_animals, p)
        have = shed.get("WHEAT", 0) + carried.get("WHEAT", 0)
        if n_animals and have < need:
            n = min(need - have, 5)
            price = obs["market"]["prices"].get("WHEAT", 25)
            if price <= p.wheat_buy_max_price and money >= price * n:
                orders.append(["BUY_PRODUCT", "WHEAT", n])
                money -= price * n

        # 6. Sell whatever is left over.
        room = MAX_MARKET_ORDERS - len(orders)
        if room > 0:
            sellable = dict(shed)
            if n_animals:
                sellable["WHEAT"] = max(0, sellable.get("WHEAT", 0) - need)
            orders += plan_sales(sellable, market_inv, p, day, TOTAL_DAYS, room)

        return orders[:MAX_MARKET_ORDERS]

    def _hand_target(self, day):
        p = self.p
        if day < 5:
            return p.hands_early
        if day < 15:
            return p.hands_mid
        return p.hands_late
