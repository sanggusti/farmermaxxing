"""Assert our copy of the game's constants and price curve matches the engine.

These are the tests that matter most. The agent reasons about prices using our
own port of `market_price`; if the engine changes its numbers and we don't
notice, every sell decision quietly starts optimising the wrong economy and the
only symptom is a slowly sinking leaderboard rank.

The engine ships in the pinned `kaggle-environments` wheel, so this compares
against the exact code that will score our submissions.
"""

import pytest

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

import rules
import market


def test_crop_table_matches_engine():
    assert rules.CROPS == engine.CROPS


def test_animal_table_matches_engine():
    assert rules.ANIMALS == engine.ANIMALS


def test_market_params_match_engine():
    assert rules.MARKET_PARAMS == engine.MARKET_PARAMS


def test_land_and_products_match_engine():
    assert rules.LAND_PRICES == engine.LAND_PRICES
    assert rules.LAND_ORDER == engine.LAND_ORDER
    assert rules.PRODUCTS == engine.PRODUCTS


@pytest.mark.parametrize("item", engine.PRODUCTS)
def test_price_curve_matches_engine(item):
    """Sweep well past +/- 2T on both sides of the equilibrium."""
    I0 = engine.MARKET_I0
    T = engine.MARKET_PARAMS[item]["T"]
    for inv in range(I0 - 3 * T, I0 + 3 * T, 7):
        assert market.market_price(item, inv) == engine.market_price(item, inv), (
            f"{item} diverges at inventory {inv}"
        )


def test_hire_cost_matches_engine():
    for n in range(0, 20):
        assert rules.fib_hire_cost(n) == engine._hire_cost(n)


def test_shed_access_tiles_match_engine():
    for board in (10, 8, 6):
        assert sorted(rules.shed_access_tiles(board)) == sorted(
            engine._shed_access_tiles(board))


def test_revenue_matches_engine_sell_loop():
    """Our multi-unit revenue model must match the engine's per-unit loop.

    The engine quotes each unit at the pre-sell inventory and only adds supply
    when the price cleared the $1 floor. Getting this wrong would make us
    over-estimate what a big melon dump is worth.
    """
    for item in ("WHEAT", "MELON", "WOOL"):
        inv = engine.MARKET_I0
        expected = 0
        for _ in range(40):
            price = engine.market_price(item, inv)
            expected += price
            if price > engine.PRICE_FLOOR:
                inv += 1
        assert market.revenue_for(item, engine.MARKET_I0, 40) == expected
