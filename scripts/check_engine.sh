#!/usr/bin/env bash
# Warn when upstream kaggriculture.py has moved past the version we pinned.
# The engine is under active development -- several rule changes landed in the
# 72h before this project started -- and our rules.py mirrors its constants.
set -euo pipefail

PIN="${1:-unknown}"
LOCAL=".venv/lib/python3.12/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py"
UPSTREAM="https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/kaggriculture/kaggriculture.py"

if [ ! -f "$LOCAL" ]; then
  echo "engine not installed; run 'make setup'" >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
curl -sfL "$UPSTREAM" -o "$TMP"

if diff -q "$LOCAL" "$TMP" >/dev/null; then
  echo "engine matches upstream master (pinned $PIN)"
  exit 0
fi

echo "UPSTREAM ENGINE HAS CHANGED since our pin ($PIN)."
echo "Review, then bump the pin in pyproject.toml and re-run 'make test'."
echo
diff -u "$LOCAL" "$TMP" | head -60 || true

LATEST=$(curl -sfL https://pypi.org/pypi/kaggle-environments/json | \
         python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')
echo
echo "latest on PyPI: $LATEST (pinned: $PIN)"
exit 1
