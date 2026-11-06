#!/usr/bin/env bash
# Train every split as its own background process, then assemble the ensemble.
# Extra arguments are passed to every call, e.g.
#   scripts/train_splits.sh --n-trials 8 --maximum-iterations 20
# --split, --stage and --config are rejected (splits come from the default config).
# Override the splits with SPLITS="1 2 3"; assembly is then skipped, so run
#   uv run python scripts/train_quantile_ensemble.py --stage assemble
# once every split has been trained.
set -euo pipefail

# Resolve all paths relative to the repository root.
cd "$(dirname "$0")/.."

# This wrapper controls the training stages and reads the default configuration.
for arg in "$@"; do
    case "$arg" in
        --split|--split=*|--stage|--stage=*|--config|--config=*)
            echo "train_splits.sh does not accept $arg; use SPLITS=\"...\" to choose splits" >&2
            exit 2
            ;;
    esac
done

# Sync once so concurrent training processes do not race on the environment.
uv sync

uv run python - <<'PY'
from loguru import logger

logger.info("Starting parallel split training; per-split logs are written to logs/train_split_<split>.log")
PY

assemble_after_training=true
if [ -n "${SPLITS:-}" ]; then
    assemble_after_training=false
else
    SPLITS=$(uv run python - <<'PY'
from highres_ta.training import load_config

n_parts = load_config()["cross_validation"]["n_parts"]
print(*range(1, n_parts))
PY
    )
fi

mkdir -p logs
pids=()

stop_training() {
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
    exit 130
}
trap stop_training INT TERM

# Start every requested split, keeping its output in a separate log file.
for split in $SPLITS; do
    uv run python scripts/train_quantile_ensemble.py --stage train --split "$split" "$@" \
        > "logs/train_split_$split.log" 2>&1 &
    pids+=("$!")
done

# Wait for all splits, even if one fails, before deciding whether to assemble.
training_failed=false
pid_index=0
for split in $SPLITS; do
    if ! wait "${pids[$pid_index]}"; then
        echo "split $split failed, see logs/train_split_$split.log" >&2
        training_failed=true
    fi
    pid_index=$((pid_index + 1))
done

if [ "$training_failed" = true ]; then
    exit 1
fi

if [ "$assemble_after_training" = false ]; then
    echo "SPLITS was set: skipping assembly. Run --stage assemble once all splits exist."
    exit 0
fi
uv run python scripts/train_quantile_ensemble.py --stage assemble "$@"
