# farmermaxxing

Agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition — a two-player, 720-turn farming/market game.

Approach: a parameterised heuristic policy, tuned by cross-entropy search run
against the real engine on Modal, with every experiment tracked in W&B.

```bash
make setup      # venv + pinned deps
make play       # one episode vs the built-in starter agent
make trace      # per-day X-ray of an episode (start here when debugging)
make arena      # holdout matrix vs frozen opponents, both seats
make check      # parity + submission contract + timing
make help       # everything else
```

See **[DEVELOPMENTS.md](DEVELOPMENTS.md)** for the game breakdown, the engine
gotchas that matter, the architecture walkthrough, and the experiment log.
