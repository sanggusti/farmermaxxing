"""Market price model and sell scheduling.

`market_price` is a verbatim port of the engine's function. Everything else is
our own logic for deciding *how much* to sell and *when*.

The strategic core: each product has an asymmetric price curve. Selling pushes
inventory above I0, which drops the price by `above_target * base` per T units,
shaped by `above_func`. Premium goods (melon `sq`/3.60, wool `sq`/3.20,
strawberry and milk `linear`/1.60) collapse to the $1 floor almost immediately,
while wheat and egg (`log`/0.20) barely move. So: dump staples, drip premiums.
"""

from rules import MARKET_PARAMS, PRICE_FLOOR, shape


def market_price(item, inventory, params=None):
    """Verbatim port of the engine's `market_price`."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, T)
        price = base + amp * shape(f, I0 - inventory)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, T)
        price = base - amp * shape(f, inventory - I0)
    return max(PRICE_FLOOR, int(round(price)))


def revenue_for(item, inventory, n):
    """Coins from selling `n` units, walking the curve one unit at a time.

    Mirrors the engine's per-unit loop: unit i is quoted at the pre-sell
    inventory, then supply increases (unless the price was already at the $1
    floor, which does not add supply).
    """
    total = 0
    inv = inventory
    for _ in range(n):
        price = market_price(item, inv)
        total += price
        if price > PRICE_FLOOR:
            inv += 1
    return total


def sellable_quantity(item, inventory, have, floor_price):
    """Largest n <= have such that every unit sells at >= floor_price."""
    n = 0
    inv = inventory
    while n < have:
        price = market_price(item, inv)
        if price < floor_price:
            break
        n += 1
        if price > PRICE_FLOOR:
            inv += 1
    return n


def plan_sales(shed, market_inv, params, day, total_days, max_orders):
    """Decide this turn's SELL orders.

    Returns a list of `["SELL", item, n]`. Unsold shed inventory is worth
    nothing at the end of the season, so the final days liquidate everything
    regardless of price.
    """
    liquidating = day >= total_days - params.liquidate_days
    orders = []

    # Sell the most valuable stock first so the 10-order cap never strands a
    # premium product behind a pile of wheat.
    items = sorted(
        (i for i in shed if i in MARKET_PARAMS and shed[i] > 0),
        key=lambda i: -market_price(i, market_inv.get(i, 0)),
    )

    for item in items:
        have = shed[item]
        inv = market_inv.get(item, 0)
        if liquidating:
            n = have
        else:
            floor = params.sell_floor_frac.get(item, 0.6) * MARKET_PARAMS[item]["base"]
            n = sellable_quantity(item, inv, have, floor)
            # Never let the shed choke on a product we refuse to sell.
            if sum(shed.values()) >= params.shed_pressure_at:
                n = max(n, min(have, params.shed_pressure_dump))
        if n > 0:
            orders.append(["SELL", item, n])
        if len(orders) >= max_orders:
            break

    return orders


def keep_wheat_for_feeding(shed, n_animals, params):
    """Wheat reserved to feed animals rather than sold.

    Each animal eats exactly 1 wheat/day and starves (permanently, and the
    animal is far more valuable than the wheat) after 2 missed days.
    """
    return int(n_animals * params.wheat_reserve_days)
