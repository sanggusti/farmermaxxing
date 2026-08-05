"""Deliberately different starting portfolios for multi-restart search.

Every search since v2 has warm-started from the previous champion with a
narrowing Gaussian, so all of them explored one basin. Issue #48 showed what
that cost: the champion sells three of nine products while seven finish the
season priced ABOVE base, because town demand drains the market faster than two
players supply it.

Diversifying is not a small perturbation. Adding sheep needs pastures, wheat to
feed them, labour to service them, and a wool sell floor low enough to actually
trade. Every single-parameter step toward that basin is worse than the local
optimum, which is precisely the shape a local search cannot cross. So the
starting points have to be constructed, not discovered.

Each archetype sets a coherent portfolio plus the sell floors and feed policy
that portfolio needs. CEM tunes from there.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Flat agent modules, same reason as everywhere else: Kaggle unpacks the
# submission flat. Set the path here so `python -m search.archetypes` works on
# its own, which is the bug that shipped in PR #15 for sim/opponents.py.
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))

from dataclasses import replace                      # noqa: E402

from params import Params                            # noqa: E402


def _floors(**overrides):
    """Sell floors as a fraction of base. Below 1.0 or the product never trades.

    The champion carries WOOL at 1.28 and CARROT at 1.00, which is a standing
    refusal to sell at any realistic price.
    """
    base = {
        "WHEAT": 0.45, "CARROT": 0.45, "TOMATO": 0.50, "STRAWBERRY": 0.55,
        "MELON": 0.60, "EGG": 0.45, "MILK": 0.55, "WOOL": 0.60,
        "FERTILIZER": 0.40,
    }
    base.update(overrides)
    return base


def diversified():
    """All five crops and all three animals. The straight test of issue #48."""
    return replace(
        Params(),
        target_wheat_tiles=6, target_melon_tiles=4, target_carrot_tiles=4,
        target_tomato_tiles=3, target_strawberry_tiles=3,
        target_geese=3, target_cows=2, target_sheep=2,
        sell_floor_frac=_floors(),
        wheat_reserve_days=1.5, wheat_buy_max_price=70.0,
        hands_early=8, hands_mid=10, hands_late=10,
        seed_batch=8, land_buy_empty_max=8,
    )


def livestock():
    """Animal-led, with the wheat to feed it.

    Wool ends at 126% of base and egg at 140% with zero units sold, so the
    livestock ablations that condemned animals were measured in configurations
    that could not support them.
    """
    return replace(
        Params(),
        target_wheat_tiles=10, target_melon_tiles=2, target_carrot_tiles=2,
        target_tomato_tiles=0, target_strawberry_tiles=1,
        target_geese=5, target_cows=4, target_sheep=4,
        sell_floor_frac=_floors(WOOL=0.55, EGG=0.40),
        wheat_reserve_days=2.5, wheat_buy_max_price=60.0,
        hands_early=9, hands_mid=11, hands_late=12,
        seed_batch=8, land_buy_empty_max=8,
    )


def premium():
    """Strawberry and melon led, dripped rather than dumped.

    Strawberry ends at 268% of base on 16 units sold, the largest untapped
    margin on the board, but its glut curve is linear/1.60 so volume floors it.
    """
    return replace(
        Params(),
        target_wheat_tiles=4, target_melon_tiles=6, target_carrot_tiles=2,
        target_tomato_tiles=4, target_strawberry_tiles=8,
        target_geese=2, target_cows=3, target_sheep=1,
        sell_floor_frac=_floors(STRAWBERRY=0.80, MELON=0.75, MILK=0.70),
        wheat_reserve_days=1.5, wheat_buy_max_price=70.0,
        hands_early=8, hands_mid=10, hands_late=11,
        seed_batch=8, land_buy_empty_max=8,
    )


def staples():
    """Wheat, carrot and egg. Glut-tolerant curves, sold in volume.

    Wheat is log/0.20 above I0 and egg log/0.20, so both absorb volume that
    would floor a premium product. Wheat also ends at 220% of base.
    """
    return replace(
        Params(),
        target_wheat_tiles=12, target_melon_tiles=2, target_carrot_tiles=8,
        target_tomato_tiles=2, target_strawberry_tiles=0,
        target_geese=6, target_cows=2, target_sheep=0,
        sell_floor_frac=_floors(WHEAT=0.35, CARROT=0.35, EGG=0.35),
        wheat_reserve_days=1.8, wheat_buy_max_price=90.0,
        hands_early=9, hands_mid=11, hands_late=11,
        seed_batch=10, land_buy_empty_max=8,
    )


ARCHETYPES = {
    "diversified": diversified,
    "livestock": livestock,
    "premium": premium,
    "staples": staples,
}


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description="write archetype params to disk")
    ap.add_argument("--out", default="runs/archetypes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name, fn in ARCHETYPES.items():
        path = os.path.join(args.out, f"{name}.json")
        fn().to_json(path)
        print(f"  {name:<14} -> {path}")


if __name__ == "__main__":
    main()
