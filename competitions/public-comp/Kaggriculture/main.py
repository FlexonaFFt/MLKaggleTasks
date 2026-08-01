"""Deterministic Kaggriculture baseline.

Submission entry point: agent(obs) -> {farmer, hands, market}.
The policy intentionally keeps no cross-episode mutable state.
"""
from collections import deque


CROP_RULES = {
    "WHEAT": {"seed": 10, "harvest_day": 4, "last_plant_day": 24},
    "CARROT": {"seed": 20, "harvest_day": 3, "last_plant_day": 25},
    "MELON": {"seed": 80, "harvest_day": 12, "last_plant_day": 17},
    "STRAWBERRY": {"seed": 100, "harvest_day": 10, "last_plant_day": 17},
}
PRODUCT_BY_ANIMAL = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
SELL_PRIORITY = ("WOOL", "MILK", "MELON", "STRAWBERRY", "CARROT", "EGG", "FERTILIZER", "WHEAT")


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _walk(tiles, source, target):
    """Return one legal move using BFS; never walks through LOCKED tiles."""
    source, target = tuple(source), tuple(target)
    if source == target:
        return ["PASS"]
    size = len(tiles)
    queue = deque([source])
    parent, move = {source: None}, {}
    for_direction = ((1, 0, "EAST"), (-1, 0, "WEST"), (0, 1, "SOUTH"), (0, -1, "NORTH"))
    while queue:
        x, y = queue.popleft()
        for dx, dy, name in for_direction:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in parent or tiles[nxt[1]][nxt[0]] == "LOCKED":
                continue
            parent[nxt], move[nxt] = (x, y), name
            if nxt == target:
                queue.clear()
                break
            queue.append(nxt)
    if target not in parent:
        return ["PASS"]
    cursor = target
    while parent[cursor] != source:
        cursor = parent[cursor]
    return [move[cursor]]


def _shed_tiles(tiles):
    mid = len(tiles) // 2
    return [(x, y) for x, y in ((mid - 1, mid - 1), (mid, mid - 1), (mid - 1, mid), (mid, mid))
            if 0 <= x < len(tiles) and 0 <= y < len(tiles) and tiles[y][x] != "LOCKED"]


def _nearest_shed(tiles, pos):
    cells = _shed_tiles(tiles)
    return min(cells, key=lambda cell: (_distance(pos, cell), cell)) if cells else tuple(pos)


def _inventory_total(private, item):
    total = int((private.get("shed") or {}).get(item, 0))
    return total + sum(int((inventory or {}).get(item, 0)) for inventory in private.get("inventories", []))


def _animal_count(farm, animal):
    return sum(
        1 for row in farm["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("animal") == animal
    )


def _phase(day, remaining):
    if remaining <= 48:
        return "liquidation"
    if day < 5:
        return "opening"
    if day < 17:
        return "scale"
    return "production"


def _job(priority, pos, action, need=None):
    return {"priority": priority, "pos": tuple(pos), "action": action, "need": need}


def _jobs(obs, farm, private, phase):
    """Generate jobs. Lower priority is more urgent."""
    day, hour = int(obs["day"]), int(obs["hour"])
    tiles, jobs = farm["tiles"], []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                if tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today"):
                    jobs.append(_job(0, pos, ["WATER"]))
                elif not tile.get("watered_today"):
                    jobs.append(_job(4 if hour < 18 else 2, pos, ["WATER"]))
                crop = tile.get("crop")
                age = day - int(tile.get("planted_day", day))
                held = int(tile.get("yield_units", 0))
                ripe = CROP_RULES.get(crop, {}).get("harvest_day", 99)
                if held and (age >= ripe or phase == "liquidation"):
                    jobs.append(_job(1 if phase == "liquidation" else 3, pos, ["HARVEST"]))
            elif tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
                animal = tile["animal"]
                if tile.get("consecutive_unfed", 0) >= 1 and not tile.get("fed_today"):
                    jobs.append(_job(0, pos, ["FEED"], "WHEAT"))
                elif not tile.get("fed_today"):
                    jobs.append(_job(3, pos, ["FEED"], "WHEAT"))
                if tile.get("fertilizer_available"):
                    jobs.append(_job(4, pos, ["COLLECT_FERTILIZER"]))
                if not tile.get("cared_today") and phase != "liquidation" and tile.get("pending_care_bonus", 0) < 4:
                    jobs.append(_job(5, pos, ["CARE"]))
                if tile.get("yield_units", 0) >= 3 or (phase == "liquidation" and tile.get("yield_units", 0)):
                    jobs.append(_job(2, pos, ["HARVEST"]))
    return jobs


def _action_for_job(tiles, pos, inventory, shed, job):
    need = job["need"]
    if need and not inventory.get(need, 0):
        if not shed.get(need, 0):
            return ["PASS"]
        shed_pos = _nearest_shed(tiles, pos)
        if tuple(pos) == shed_pos:
            return ["PICKUP", need, 4 if need == "WHEAT" else 1]
        return _walk(tiles, pos, shed_pos)
    return job["action"] if tuple(pos) == job["pos"] else _walk(tiles, pos, job["pos"])


def _schedule(obs, farm, private, jobs):
    """Greedily give each unit one reachable, unique highest-priority job."""
    tiles = farm["tiles"]
    positions = [tuple(farm.get("farmer", (0, 0)))] + [tuple(p) for p in farm.get("hands", [])]
    inventories = [dict(x or {}) for x in private.get("inventories", [])]
    inventories += [{} for _ in range(max(0, len(positions) - len(inventories)))]
    remaining = list(jobs)
    actions = []
    for pos, inventory in zip(positions, inventories):
        ranked = sorted(remaining, key=lambda job: (job["priority"], _distance(pos, job["pos"])))
        selected = next((job for job in ranked if _action_for_job(tiles, pos, inventory, private.get("shed") or {}, job) != ["PASS"]), None)
        if selected is None:
            actions.append(["PASS"])
        else:
            actions.append(_action_for_job(tiles, pos, inventory, private.get("shed") or {}, selected))
            remaining.remove(selected)
    return actions[0], actions[1:]


def _market(obs, farm, private, phase):
    """Keep orders bounded; sales happen only for items already in the shed."""
    day = int(obs["day"])
    money = float(farm.get("money", 0))
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    orders = []
    for item in SELL_PRIORITY:
        quantity = int(shed.get(item, 0))
        if quantity and (phase == "liquidation" or quantity >= 6):
            orders.append(["SELL", item, quantity])
    if phase != "liquidation":
        wheat_needed = 2 * (_animal_count(farm, "COW") + _animal_count(farm, "SHEEP") + _animal_count(farm, "GOOSE"))
        if _inventory_total(private, "WHEAT") < wheat_needed and money >= 30:
            orders.append(["BUY_PRODUCT", "WHEAT", max(4, wheat_needed)])
        if day <= 17 and seeds.get("STRAWBERRY", 0) < 4 and money >= 400:
            orders.append(["BUY_SEED", "STRAWBERRY", 4])
        if day <= 12 and seeds.get("MELON", 0) < 2 and money >= 240:
            orders.append(["BUY_SEED", "MELON", 2])
        hands = len(farm.get("hands", []))
        if hands < 8 and money >= 250:
            orders.append(["HIRE"])
    return orders[:10]


def agent(obs):
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs.get("private") or {}
    remaining = 719 - (int(obs["day"]) * 24 + int(obs["hour"]))
    phase = _phase(int(obs["day"]), remaining)
    farmer, hands = _schedule(obs, farm, private, _jobs(obs, farm, private, phase))
    return {"farmer": farmer, "hands": hands, "market": _market(obs, farm, private, phase)}
