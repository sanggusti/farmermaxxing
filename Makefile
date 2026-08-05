PY := .venv/bin/python
COMP := kaggriculture
ENGINE_PIN := 1.32.4
BUNDLE := submission.tar.gz

# Files that make up a submission. main.py MUST be at the archive root.
AGENT_FILES := main.py policy.py params.py market.py rules.py

.PHONY: help setup play trace arena test check search search-modal \
        bundle submit check-engine leaderboard status clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install pinned dependencies
	uv venv --python 3.12
	uv sync
	@echo "done. kaggle credentials go in ~/.kaggle/access_token"

# ---------------------------------------------------------------- play / debug
play:  ## One episode vs the built-in starter agent
	$(PY) -m sim.run --opponent starter --debug

trace:  ## Per-day X-ray of an episode (the main debugging view)
	$(PY) -m sim.trace --opponent starter

replay:  ## Episode + replay JSON for the visualiser
	@mkdir -p replays
	$(PY) -m sim.run --opponent starter --replay replays/latest.json

# ------------------------------------------------------------------ evaluation
arena:  ## Holdout matrix vs the full opponent pool (add WANDB=1 to log)
	$(PY) -m sim.arena --seeds 8 --opponents all $(if $(WANDB),--wandb,)

freeze:  ## Snapshot agent/params.json into the opponent pool. NAME=v1-cem
	@if [ -z "$(NAME)" ]; then echo 'set NAME=<snapshot-name>'; exit 1; fi
	$(PY) -m sim.opponents --name $(NAME) --notes "$(NOTES)"

promote:  ## Copy a search result into agent/params.json. FROM=runs/.../best_params.json
	@if [ -z "$(FROM)" ]; then echo 'set FROM=<path to best_params.json>'; exit 1; fi
	@cp $(FROM) agent/params.json
	@echo "promoted $(FROM) -> agent/params.json (run 'make gate' next)"

gate:  ## Promotion gate: is the candidate genuinely better? CHAMPION=path
	$(PY) -m sim.gate --candidate agent/params.json \
	  $(if $(CHAMPION),--champion $(CHAMPION),) $(if $(WANDB),--wandb,)

test:  ## Unit tests: engine parity + submission contract
	$(PY) -m pytest -q

check: test  ## Everything that must pass before a submission
	$(PY) -m pytest -q -m slow
	@echo "OK: parity, contract and timing all pass"

# ---------------------------------------------------------------------- search
search:  ## CEM locally (small; for smoke-testing the loop)
	$(PY) -m search.cem --generations 4 --population 12 --seeds 3

search-modal:  ## CEM fanned out on Modal (the real run)
	$(PY) -m search.cem --generations 10 --population 48 --seeds 6 --modal

# ------------------------------------------------------------------ submission
bundle: check  ## Build submission.tar.gz (runs checks first)
	@rm -f $(BUNDLE)
	tar -czf $(BUNDLE) -C agent $(AGENT_FILES) \
	  $$([ -f agent/params.json ] && echo params.json)
	@tar -tzf $(BUNDLE) | sed 's/^/  /'
	@echo "$(BUNDLE) $$(du -h $(BUNDLE) | cut -f1)"

submit: bundle  ## Submit to Kaggle. Requires CONFIRM=1 and M="message"
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "refusing to submit without CONFIRM=1"; \
	  echo "usage: make submit CONFIRM=1 M=\"what changed\""; exit 1; fi
	@if [ -z "$(M)" ]; then echo "set M=\"message\""; exit 1; fi
	.venv/bin/kaggle competitions submit $(COMP) -f $(BUNDLE) -m "$(M)"

status:  ## Recent submissions and their scores
	.venv/bin/kaggle competitions submissions $(COMP)

leaderboard:  ## Top of the public leaderboard
	.venv/bin/kaggle competitions leaderboard $(COMP) -s | head -20

# ----------------------------------------------------------------- maintenance
check-engine:  ## Warn if upstream changed the engine we pinned against
	@scripts/check_engine.sh $(ENGINE_PIN)

clean:
	rm -rf $(BUNDLE) replays/*.json .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
