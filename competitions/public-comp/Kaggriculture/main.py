"""
Kaggriculture Hybrid-6 policy.
"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GameState:
    obs: dict
    player: int
    farm: dict
    private: dict
    day: int
    hour: int
    remaining: int


@dataclass(frozen=True)
class Task:
    priority: int
    position: tuple
    action: tuple
    item: str | None = None


class FarmPolicy:
    """
    A stateless policy keeps separate episodes isolated.
    """

    crop_order = ("WHEAT", "CARROT", "MELON", "STRAWBERRY")
    crop_rules = {
        "WHEAT": (4, 24),
        "CARROT": (3, 25),
        "MELON": (12, 17),
        "STRAWBERRY": (10, 17),
    }
    product_for = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
    sell_order = ("WOOL", "MILK", "MELON", "STRAWBERRY", "CARROT", "EGG", "FERTILIZER", "WHEAT")
    seed_price = {"WHEAT": 25, "CARROT": 40, "MELON": 250, "STRAWBERRY": 150}
    animal_price = {"COW": 600, "SHEEP": 500}
    fertilizer_reserve = 12
    max_animals_per_species = 9
    target_herd = 6

    def act(self, obs):
        state = self.parse(obs)
        phase = self.phase(state)
        crop_plan = self.crop_plan(state)
        herd_plan = self.herd_plan(state, phase)
        tasks = self.build_tasks(state, phase, crop_plan, herd_plan)
        farmer, hands = self.schedule(state, tasks)
        market = self.market_orders(state, phase, crop_plan, herd_plan)
        return {"farmer": farmer, "hands": hands, "market": market}

    def parse(self, obs):
        player = int(obs["player"])
        day = int(obs["day"])
        hour = int(obs["hour"])
        return GameState(
            obs=obs,
            player=player,
            farm=obs["farms"][player],
            private=obs.get("private") or {},
            day=day,
            hour=hour,
            remaining=719 - day * 24 - hour,
        )

    def phase(self, state):
        if state.remaining <= 60:
            return "liquidation"
        if state.day < 5:
            return "opening"
        if state.day < 17:
            return "scale"
        return "production"

    def distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def walk(self, tiles, source, target):
        """
        A legal first step to target, avoiding locked land.
        """
        source, target = tuple(source), tuple(target)
        if source == target:
            return ["PASS"]
        size = len(tiles)
        queue = deque([source])
        parent, moves = {source: None}, {}
        directions = ((1, 0, "EAST"), (-1, 0, "WEST"), (0, 1, "SOUTH"), (0, -1, "NORTH"))
        while queue:
            x, y = queue.popleft()
            for dx, dy, name in directions:
                nxt = (x + dx, y + dy)
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                    continue
                if nxt in parent or tiles[nxt[1]][nxt[0]] == "LOCKED":
                    continue
                parent[nxt], moves[nxt] = (x, y), name
                if nxt == target:
                    queue.clear()
                    break
                queue.append(nxt)
        if target not in parent:
            return ["PASS"]
        cursor = target
        while parent[cursor] != source:
            cursor = parent[cursor]
        return [moves[cursor]]

    def shed_tiles(self, tiles):
        middle = len(tiles) // 2
        candidates = ((middle - 1, middle - 1), (middle, middle - 1), (middle - 1, middle), (middle, middle))
        return [cell for cell in candidates if 0 <= cell[0] < len(tiles) and 0 <= cell[1] < len(tiles) and tiles[cell[1]][cell[0]] != "LOCKED"]

    def nearest_shed(self, tiles, position):
        cells = self.shed_tiles(tiles)
        return min(cells, key=lambda cell: (self.distance(position, cell), cell)) if cells else tuple(position)

    def total_item(self, state, item):
        shed = int((state.private.get("shed") or {}).get(item, 0))
        carried = sum(int((inventory or {}).get(item, 0)) for inventory in state.private.get("inventories", []))
        return shed + carried

    def animal_count(self, farm, animal):
        return sum(1 for row in farm["tiles"] for tile in row if isinstance(tile, dict) and tile.get("animal") == animal)

    def pasture_slots(self, state, count):
        """
        Keep the closest unlocked cells around the shed for a scalable herd.
        """
        tiles = state.farm["tiles"]
        shed = set(self.shed_tiles(tiles))
        center = (len(tiles) // 2 - 1, len(tiles) // 2 - 1)
        cells = [
            (x, y)
            for y, row in enumerate(tiles)
            for x, tile in enumerate(row)
            if tile != "LOCKED" and (x, y) not in shed
        ]
        cells.sort(key=lambda cell: (0 if isinstance(tiles[cell[1]][cell[0]], dict) and (tiles[cell[1]][cell[0]].get("kind") == "PASTURE" or tiles[cell[1]][cell[0]].get("animal")) else 1, self.distance(cell, center), cell[1], cell[0]))
        return cells[:count]

    def pays_back(self, state, cost, daily_value, setup_days=2):
        return max(0, state.remaining // 24 - setup_days) * daily_value >= cost

    def melon_target(self, state):
        """
        Start with eight melons, then react to price and visible rival supply.
        """
        if state.day < 3:
            return 8
        price = int(((state.obs.get("market") or {}).get("prices") or {}).get("MELON", 0))
        rival_melons = sum(
            1
            for index, rival in enumerate(state.obs.get("farms") or [])
            if index != state.player
            for row in rival.get("tiles", [])
            for tile in row
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON"
        )
        if rival_melons <= 4 and price >= 200:
            return 12
        if rival_melons >= 12 or price < 150:
            return 8
        return 10

    def fertilizer_is_profitable(self, state, crop):
        prices = (state.obs.get("market") or {}).get("prices") or {}
        return crop in {"MELON", "STRAWBERRY", "TOMATO"} and int(prices.get(crop, 0)) > int(prices.get("FERTILIZER", 0))

    def crop_plan(self, state):
        """
        Reserve central cells for livestock and fill reachable land with crops.
        """
        tiles = state.farm["tiles"]
        excluded = set(self.shed_tiles(tiles) + self.pasture_slots(state, self.target_herd))
        cells = [
            (x, y)
            for y, row in enumerate(tiles)
            for x, tile in enumerate(row)
            if tile != "LOCKED" and (x, y) not in excluded
        ]
        shed = self.nearest_shed(tiles, tuple(state.farm.get("farmer", (0, 0))))
        cells.sort(key=lambda cell: (self.distance(cell, shed), cell[1], cell[0]))
        unlocked = len(state.farm.get("unlocked_quadrants") or ["NW"])
        capacity = min(len(cells), 12 + 8 * unlocked)
        crops = ("WHEAT",) * 4 + ("CARROT",) * 2 + ("MELON",) * self.melon_target(state)
        crops += ("STRAWBERRY",) * max(0, capacity - len(crops))
        return dict(zip(cells[:capacity], crops[:capacity]))

    def herd_plan(self, state, phase):
        """
        Start mixed, then scale to six animals in the better product direction.
        """
        slots = self.pasture_slots(state, self.target_herd)
        if phase == "opening":
            return dict(zip(slots[:2], ("COW", "SHEEP")))
        prices = (state.obs.get("market") or {}).get("prices") or {}
        extra = "SHEEP" if int(prices.get("WOOL", 0)) >= int(prices.get("MILK", 0)) else "COW"
        return dict(zip(slots, ("COW", "SHEEP", "COW", "SHEEP", extra, extra)))

    def build_tasks(self, state, phase, crop_plan, herd_plan):
        """
        Lower priority values are dispatched first.
        """
        tiles = state.farm["tiles"]
        tasks = []
        for position, crop in crop_plan.items():
            x, y = position
            tile = tiles[y][x]
            if tile is None and phase != "liquidation" and state.day <= self.crop_rules[crop][1]:
                tasks.append(Task(6, position, ("PLANT", crop)))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append(Task(2, position, ("DIG",)))
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop_name = tile.get("crop", crop)
                ripe_day = self.crop_rules.get(crop_name, (99, 0))[0]
                age = state.day - int(tile.get("planted_day", state.day))
                held = int(tile.get("yield_units", 0))
                if not tile.get("watered_today") and tile.get("consecutive_unwatered", 0) >= 1:
                    tasks.append(Task(0, position, ("WATER",)))
                elif not tile.get("watered_today"):
                    tasks.append(Task(3 if state.hour >= 17 else 5, position, ("WATER",)))
                if held and (age >= ripe_day or phase == "liquidation"):
                    tasks.append(Task(2, position, ("HARVEST",)))
                if self.total_item(state, "FERTILIZER") > self.fertilizer_reserve and tile.get("fertilized_until_day", -1) < state.day and self.fertilizer_is_profitable(state, crop_name):
                    tasks.append(Task(4, position, ("FERTILIZE",), "FERTILIZER"))
        for position, animal in herd_plan.items():
            x, y = position
            tile = tiles[y][x]
            if tile is None and phase != "liquidation" and state.day >= 3:
                tasks.append(Task(5, position, ("BUILD_PASTURE",)))
            elif isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal") and phase != "liquidation":
                tasks.append(Task(5, position, ("PLACE", animal), animal))
            elif isinstance(tile, dict) and tile.get("animal"):
                if not tile.get("fed_today") and tile.get("consecutive_unfed", 0) >= 1:
                    tasks.append(Task(0, position, ("FEED",), "WHEAT"))
                elif not tile.get("fed_today"):
                    tasks.append(Task(3, position, ("FEED",), "WHEAT"))
                if tile.get("yield_units", 0) >= 3 or (phase == "liquidation" and tile.get("yield_units", 0)):
                    tasks.append(Task(2, position, ("HARVEST",)))
                if tile.get("fertilizer_available") and phase != "liquidation":
                    tasks.append(Task(4, position, ("COLLECT_FERTILIZER",)))
                if phase != "liquidation" and not tile.get("cared_today") and tile.get("pending_care_bonus", 0) < 4:
                    tasks.append(Task(6, position, ("CARE",)))
        return tasks

    def task_action(self, state, position, inventory, task):
        tiles = state.farm["tiles"]
        shed = state.private.get("shed") or {}
        if task.item and not inventory.get(task.item, 0):
            if not shed.get(task.item, 0):
                return ["PASS"]
            shed_position = self.nearest_shed(tiles, position)
            if tuple(position) == shed_position:
                amount = 4 if task.item == "WHEAT" else 1
                return ["PICKUP", task.item, amount]
            return self.walk(tiles, position, shed_position)
        if tuple(position) == task.position:
            return list(task.action)
        return self.walk(tiles, position, task.position)

    def schedule(self, state, tasks):
        """
        One distinct reachable task per farmer or hand.
        """
        positions = [tuple(state.farm.get("farmer", (0, 0)))] + [tuple(item) for item in state.farm.get("hands", [])]
        inventories = [dict(item or {}) for item in state.private.get("inventories", [])]
        inventories.extend({} for _ in range(max(0, len(positions) - len(inventories))))
        available = list(tasks)
        actions = []
        for position, inventory in zip(positions, inventories):
            ranked = sorted(available, key=lambda task: (task.priority, self.distance(position, task.position)))
            chosen = next((task for task in ranked if self.task_action(state, position, inventory, task) != ["PASS"]), None)
            if chosen is None:
                actions.append(["PASS"])
            else:
                actions.append(self.task_action(state, position, inventory, chosen))
                available.remove(chosen)
        return actions[0], actions[1:]

    def market_orders(self, state, phase, crop_plan, herd_plan):
        """
        Finance seed and feed first; defer land, livestock, and larger crews.
        """
        farm, private = state.farm, state.private
        money = float(farm.get("money", 0))
        shed = private.get("shed") or {}
        seeds = private.get("seeds") or {}
        prices = (state.obs.get("market") or {}).get("prices") or {}
        orders = []
        unlocked = len(farm.get("unlocked_quadrants") or ["NW"])
        land_cost = 1000 * (2 ** max(0, unlocked - 1))
        herd_size = sum(self.animal_count(farm, animal) for animal in self.product_for)
        reserve = 1500 + 200 * herd_size
        sale_caps = {"MELON": 8, "WOOL": 8, "MILK": 12, "STRAWBERRY": 12}
        for item in self.sell_order:
            quantity = int(shed.get(item, 0))
            if item == "FERTILIZER" and phase != "liquidation":
                quantity = max(0, quantity - self.fertilizer_reserve)
            if quantity and (phase == "liquidation" or quantity >= 6):
                orders.append(("SELL", item, min(quantity, sale_caps.get(item, quantity))))
        if phase == "liquidation":
            return [list(order) for order in orders[:10]]
        if phase == "scale" and state.day >= 8 and unlocked < 3 and money >= land_cost + reserve:
            orders.append(("BUY_LAND",))
        budget = max(0, money - (land_cost if orders and orders[-1] == ("BUY_LAND",) else 0) - reserve)
        empty_by_crop = {crop: 0 for crop in self.crop_order}
        for position, crop in crop_plan.items():
            x, y = position
            if farm["tiles"][y][x] is None and state.day <= self.crop_rules[crop][1]:
                empty_by_crop[crop] += 1
        for crop in self.crop_order:
            missing = max(0, empty_by_crop[crop] - int(seeds.get(crop, 0)))
            price = max(1, int(prices.get(f"{crop}_SEED", prices.get(crop, self.seed_price[crop]))))
            quantity = min(missing, int(budget // price))
            if quantity:
                orders.append(("BUY_SEED", crop, quantity))
                budget -= quantity * price
        wanted = list(herd_plan.values())
        wheat_floor = max(8, 4 * max(1, herd_size))
        wheat_missing = max(0, wheat_floor - self.total_item(state, "WHEAT"))
        wheat_price = int(prices.get("WHEAT", 25))
        if wheat_missing and budget >= wheat_price * wheat_missing:
            orders.append(("BUY_PRODUCT", "WHEAT", wheat_missing))
            budget -= wheat_price * wheat_missing
        if state.day >= 3:
            for animal in ("COW", "SHEEP"):
                owned = self.total_item(state, animal)
                missing = min(max(0, wanted.count(animal) - owned), max(0, self.max_animals_per_species - owned))
                price = max(1, int(prices.get(animal, self.animal_price[animal])))
                daily_value = int(prices.get(self.product_for[animal], 0))
                quantity = min(missing, int(budget // price)) if self.pays_back(state, price, daily_value) else 0
                if quantity:
                    orders.append(("BUY_ANIMAL", animal, quantity))
                    budget -= quantity * price
        workload = sum(1 for position in crop_plan if farm["tiles"][position[1]][position[0]] is not None) + len(herd_plan) * 3
        crew_cap = 1 if phase == "opening" else 2 if phase == "scale" else 4
        desired_hands = min(crew_cap, max(1, (workload + 5) // 6))
        hires = int(farm.get("hires_today", 0))
        for _ in range(max(0, desired_hands - hires)):
            orders.append(("HIRE",))
        return [list(order) for order in orders[:10]]


POLICY = FarmPolicy()


def agent(obs):
    return POLICY.act(obs)
