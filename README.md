# farmermaxxing

Agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition, a two-player, 720-turn farming/market game.

Approach: a parameterised heuristic policy, tuned by cross-entropy search run
against the real engine on Modal, with every experiment tracked in W&B.

```bash
make setup         # venv + pinned deps
make ladder-sync   # where we stand: ratings into ledger.json, rank, band benchmarks
make calibrate     # has our local evidence ever predicted the rating?
make mix           # our REALISED tile mix beside a ladder opponent's, per day
make meta-gap      # champion vs the prize band, decomposed by product
make trace         # per-day X-ray of an episode (start here when debugging)
make preflight     # everything that must hold before spending a submission slot
make help          # everything else
```

Read **[AGENTS.md](AGENTS.md)** first — it is the operating contract, and it
records why the objective is ladder rating rather than any local number.
**[DEVELOPMENTS.md](DEVELOPMENTS.md)** has the game breakdown, the engine gotchas
that matter, the architecture walkthrough, and the dated experiment log.
`ledger.json` is the submission record: what we shipped, the local claim that
justified it, and what the ladder said back.

Technical reports, one per milestone:

| Doc | Covers |
|---|---|
| [1. Setup and baseline](docs/1_setup_and_baseline.md) | why this is not an RL project, first working agent, six bugs |
| [2. Search and gating](docs/2_search_and_gating.md) | CEM on Modal, holdout split, the promotion gate |
| [3. Market starvation](docs/3_market_starvation.md) | a refuted hypothesis, and the constraint it uncovered |
| [4. Scaling and conclusions](docs/4_scaling_and_conclusions.md) | 40x search throughput, multi-restart sweep, where things stand |
| [5. Reading the ladder](docs/5_reading_the_ladder.md) | replays reproduce exactly, the meta's build, four refutations, three bugs in our own instruments |
| [6. A dead parameter, and a better pool](docs/6_a_dead_parameter_and_a_better_pool.md) | the late rotation never ran, six refutations, and an opponent pool cut from our own matchmaking band |

The 2026-08-17 entry in `DEVELOPMENTS.md` supersedes the plan in docs/6: rank
2455 of 4883, four silent instrument bugs, and the measurement that reframes the
gap — we sell ~870 units of 5 products where the top of the ladder sells ~4,170
of 9, on 69% of our land against their 80%.
