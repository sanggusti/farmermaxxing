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
    CROPS, ANIMALS, MOVES, LAND_PRICES,
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
        tasks = self._build_tasks(me, private, day, board)
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
    def _build_tasks(self, me, private, day, board):
        p = self.p
        tasks = []
        counts, crops = self._scan(me)
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})

        scale = len(me["unlocked_quadrants"])
        want = self._wanted_crop(counts, crops, day, scale)

        # Never queue more PLANT tasks than we hold seeds. The engine validates
        # planting atomically per crop:
        #     blocked = {crop for crop, n in demand.items() if n > seeds[crop]}
        # so if 7 units all try to plant wheat while we hold 2 seeds, ALL seven
        # are silently dropped and the farm deadlocks -- forever, since nothing
        # about the state changes to break the tie.
        plant_budget = seeds.get(want, 0) if want else 0

        for y, row in enumerate(me["tiles"]):
            for x, tile in enumerate(row):
                pos = (x, y)
                if tile == "LOCKED":
                    continue

                if tile is None:
                    if plant_budget > 0:
                        tasks.append(Task(p.prio_plant, pos, ["PLANT", want], f"plant-{want}"))
                        plant_budget -= 1
                    elif self._needs_structure(counts, scale):
                        kind = self._needs_structure(counts, scale)
                        tasks.append(Task(p.prio_build, pos, [f"BUILD_{kind}"], f"build-{kind}"))
                    continue

                kind = tile.get("kind")

                if kind == "WEED":
                    tasks.append(Task(p.prio_dig, pos, ["DIG"], "dig"))

                elif kind == "PLANT":
                    tasks += self._plant_tasks(tile, pos, day)

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

    def _plant_tasks(self, tile, pos, day):
        p = self.p
        out = []
        crop = tile["crop"]
        cd = CROPS[crop]
        age = day - tile["planted_day"]

        if not tile["watered_today"]:
            # Watering is only *useful* inside the bonus window, but it is
            # always *necessary* -- two dry days turns the tile into a weed.
            out.append(Task(p.prio_water, pos, ["WATER"], "water"))

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

    def _needs_structure(self, counts, scale=1):
        """Which structure to build next, if any."""
        p = self.p
        if counts["GOOSE"] + counts["COOP"] < p.target_geese * scale:
            return "COOP"
        pasture_want = (p.target_cows + p.target_sheep) * scale
        if counts["COW"] + counts["SHEEP"] + counts["PASTURE"] < pasture_want:
            return "PASTURE"
        return None

    def _wanted_crop(self, counts, crops, day, scale=1):
        """Which crop a free tile should get, or None.

        Targets are per-quadrant and scale with the land we own, so a bought
        quadrant fills up instead of lying fallow.
        """
        p = self.p
        days_left = TOTAL_DAYS - day
        targets = [
            ("WHEAT", p.target_wheat_tiles),
            ("MELON", p.target_melon_tiles),
            ("CARROT", p.target_carrot_tiles),
            ("TOMATO", p.target_tomato_tiles),
            ("STRAWBERRY", p.target_strawberry_tiles),
        ]
        for crop, target in targets:
            if crops[crop] >= target * scale:
                continue
            # Don't plant what cannot mature before the season ends.
            if CROPS[crop]["first_yield_day"] + p.plant_cutoff_slack > days_left:
                continue
            return crop
        return None

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

        # 1. Labour, first thing each day. Hands act from the following turn.
        if hour == 0:
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
        for name, target in (("GOOSE", p.target_geese * scale),
                             ("COW", p.target_cows * scale),
                             ("SHEEP", p.target_sheep * scale)):
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
        want = self._wanted_crop(counts, crops, day, scale)
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
