PY := .venv/bin/python
COMP := kaggriculture
ENGINE_PIN := 1.32.4
BUNDLE := submission.tar.gz

# Files that make up a submission. main.py MUST be at the archive root.
AGENT_FILES := main.py policy.py params.py market.py rules.py

.PHONY: help setup play trace replay arena freeze promote gate test check \
        search search-modal bundle submit check-engine leaderboard status clean \
        ladder-sync calibrate slots meta-gap mix refresh-tapes preflight ledger

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

gate:  ## Promotion gate vs the CHAMPION file. POOL=top|band|real  TRAINED_ON=names
	$(PY) -m sim.gate --candidate agent/params.json \
	  --opponents $(if $(POOL),$(POOL),top) --seeds $(if $(SEEDS),$(SEEDS),8) \
	  $(if $(CHAMPION),--champion $(CHAMPION),) \
	  $(if $(TRAINED_ON),--trained-on $(TRAINED_ON),) $(if $(WANDB),--wandb,)

test:  ## Unit tests: engine parity + submission contract
	$(PY) -m pytest -q

check: test  ## Everything that must pass before a submission
	$(PY) -m pytest -q -m slow
	@echo "OK: parity, contract and timing all pass"

# --------------------------------------------------------------- the ladder
# The ladder is the only ground truth. These are the commands that read it.

ladder-sync:  ## Pull ratings into ledger.json and show the scored slots
	$(PY) -m sim.ladder sync

slots: ladder-sync  ## Alias: what is scored right now, and how old its rating is

ledger:  ## The submission ledger, without touching the network
	$(PY) -m sim.ledger show

calibrate:  ## Did our local evidence predict the rating? Join and find out
	$(PY) -m sim.ladder calibrate

refresh-tapes:  ## Mint top-band opponents from a day's episodes. DATE=2026-08-16
	@if [ -z "$(DATE)" ]; then echo 'set DATE=YYYY-MM-DD (see the index manifest)'; exit 1; fi
	$(PY) -m sim.ladder mine --date $(DATE) \
	  --top $(if $(TOP),$(TOP),12) --keep $(if $(KEEP),$(KEEP),8)

meta-gap:  ## Champion vs the prize band, decomposed by product
	$(PY) -m sim.gate --candidate agent/params.json --opponents top \
	  --seeds $(if $(SEEDS),$(SEEDS),4)

mix:  ## Our REALISED tile mix beside the opponent's, per day. OPP=tape:meta-a
	$(PY) -m sim.mix --opponent $(if $(OPP),$(OPP),tape:meta-a) \
	  --seed $(if $(SEED),$(SEED),20000)

preflight: bundle  ## Everything that must hold before a submission slot is spent
	@echo
	@echo "preflight OK: parity, contract, flat layout, tarball, both seats,"
	@echo "crash-safety and timing vs a real ladder opponent all pass."

# ---------------------------------------------------------------------- search
search:  ## CEM locally (small; for smoke-testing the loop)
	$(PY) -m search.cem --generations 4 --population 12 --seeds 3

search-modal:  ## CEM fanned out on Modal (the real run)
	$(PY) -m search.cem --generations 10 --population 48 --seeds 6 --modal

# ------------------------------------------------------------------ submission
bundle: check  ## Build submission.tar.gz, then validate the archive itself
	@rm -f $(BUNDLE)
	tar -czf $(BUNDLE) -C agent $(AGENT_FILES) \
	  $$([ -f agent/params.json ] && echo params.json)
	@tar -tzf $(BUNDLE) | sed 's/^/  /'
	@echo "$(BUNDLE) $$(du -h $(BUNDLE) | cut -f1)"
	# Validated AFTER the tar, not as part of `check`: these assertions are about
	# the built archive, and running them beforehand fails on a stale one and
	# blocks the command that would refresh it. The stale archive was real -- on
	# 2026-08-17 the committed tarball still held v11's parameters while
	# agent/params.json held v10's, and nothing had ever noticed.
	$(PY) -m pytest -q -m bundle

submit: bundle  ## Submit. CONFIRM=1 M="message" V=v12 POOL=top METRIC=margin VALUE=n
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "refusing to submit without CONFIRM=1"; \
	  echo 'usage: make submit CONFIRM=1 V=v12 M="what changed" \'; \
	  echo '         POOL=top METRIC=margin VALUE=-41000'; exit 1; fi
	@if [ -z "$(M)" ]; then echo 'set M="message"'; exit 1; fi
	@if [ -z "$(V)" ]; then echo 'set V=<version>, e.g. V=v12'; exit 1; fi
	# The local claim goes in the ledger BEFORE the rating exists, so it cannot
	# be adjusted afterwards to fit. That is the whole point of calibration.
	@if [ -z "$(POOL)" ] || [ -z "$(METRIC)" ]; then \
	  echo "set POOL= and METRIC= (and VALUE=): a submission with no recorded"; \
	  echo "local claim cannot be calibrated, which is how v10's result was lost."; \
	  exit 1; fi
	.venv/bin/kaggle competitions submit $(COMP) -f $(BUNDLE) -m "$(M)"
	@echo
	@echo "now record it (the id is in the line above, or run 'make ladder-sync'):"
	@echo "  $(PY) -m sim.ledger add --id <ID> --version $(V) \\"
	@echo "      --params agent/params.json --pool $(POOL) --metric $(METRIC) \\"
	@echo "      --value $(VALUE) --note \"$(M)\""
	@$(PY) -m sim.ladder sync

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
