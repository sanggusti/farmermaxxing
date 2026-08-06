"""Who the search plays, and how those results become one fitness number.

Every CEM run so far trained against `starter`, the weakest agent available.
That is not a small bias. Measured on clean seeds with the v4 champion, single
parameter changes flip sign depending on the opponent:

    +8 wheat tiles/quadrant   vs starter    137,052 -> 120,669   (worse)
                              vs v3-fixed    46,285 ->  53,946   (better)

So the search has been selecting *against* the crop mix that beats real
opposition. Fixing the opponent set is therefore not a measurement nicety; it
changes which direction is uphill.

Two ideas do the work here.

**Cells.** A cell is one (opponent, seed, seat) triple. Every candidate in a
generation plays the *same* cells -- common random numbers -- so the noise is a
shared shift rather than a per-candidate lottery, and ranking stays valid at the
handful of episodes per candidate we can afford. Breadth is then free: the same
16 episodes buy 4 opponents x 2 seeds x 2 seats instead of 8 seeds x 2 seats.

**Per-cell normalisation.** A plain mean over a mixed pool is dominated by
whichever opponent yields the biggest banks -- `starter` at ~140k would swamp
`v3-fixed` at ~46k, and the mixture would quietly become `starter` again. Each
cell is standardised across the population before averaging, so every opponent
gets an equal vote in the ranking regardless of the coin scale it happens to
produce.
"""

import statistics


def build_cells(opponents, labels, seeds, seats=(0, 1)):
    """Full product of (opponent, seed, seat), in a fixed deterministic order.

    Fixed order matters: `banks` lists returned by the scorer are positional,
    so a reordering here would silently mis-pair candidates against cells.
    """
    cells = []
    for opp, label in zip(opponents, labels):
        for seed in seeds:
            for seat in seats:
                cells.append((opp, label, seed, seat))
    return cells


def normalised_fitness(all_banks):
    """Per-cell z-scores averaged per candidate.

    `all_banks` is a list (per candidate) of lists (per cell, aligned). Returns
    one scalar per candidate.

    A cell where every candidate scores the same carries no information about
    ranking, so it contributes 0 rather than dividing by a ~0 spread.
    """
    if not all_banks:
        return []
    n_cells = len(all_banks[0])
    scores = [0.0] * len(all_banks)

    for c in range(n_cells):
        col = [banks[c] for banks in all_banks]
        mu = statistics.mean(col)
        sd = statistics.pstdev(col)
        if sd <= 0:
            continue
        for i, v in enumerate(col):
            scores[i] += (v - mu) / sd

    return [s / n_cells for s in scores]


def worst_opponent(stats):
    """(label, mean_margin) of the opponent this candidate does worst against.

    The aggregate hides exactly the regression that matters: gaining against
    the built-ins while losing to the strongest frozen champion nets out to
    roughly no change, and the ladder is full of strong opponents.

    **Margin, not bank level.** The first version of this ranked opponents by
    mean bank, and it failed silently. The v6 search picked a champion whose
    bank against `v5-mixture` was 81,623 -- the second-highest number in its
    per-opponent column -- while losing **100%** of those matches. In a shared
    market both banks rise together when neither player is dumping, so a high
    absolute bank against a strong opponent can mean "we both did well and they
    did better", which is a loss.

    Margin rather than win rate because win rate is one coarse bit per episode
    and far noisier at 16 episodes; `sim/arena.py` records that win rate is used
    "only as the final gate before a submission, never for tuning". Margin keeps
    the low variance of bank while measuring the thing that decides the match.
    """
    by_opp = stats.get("by_opponent") or {}
    if not by_opp:
        return None, float("nan")
    # Fall back to bank only if margin is absent, so an older scorer degrades
    # to the previous behaviour rather than raising.
    def key(k):
        b = by_opp[k]
        return b["mean_margin"] if "mean_margin" in b else b["mean_bank"]

    label = min(by_opp, key=key)
    return label, key(label)
