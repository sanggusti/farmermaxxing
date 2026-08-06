# farmermaxxing

Agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition, a two-player, 720-turn farming/market game.

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

Technical reports, one per milestone:

| Doc | Covers |
|---|---|
| [1. Setup and baseline](docs/1_setup_and_baseline.md) | why this is not an RL project, first working agent, six bugs |
| [2. Search and gating](docs/2_search_and_gating.md) | CEM on Modal, holdout split, the promotion gate |
| [3. Market starvation](docs/3_market_starvation.md) | a refuted hypothesis, and the constraint it uncovered |
| [4. Scaling and conclusions](docs/4_scaling_and_conclusions.md) | 40x search throughput, multi-restart sweep, where things stand |
| [5. Reading the ladder](docs/5_reading_the_ladder.md) | replays reproduce exactly, the meta's build, four refutations, three bugs in our own instruments |
