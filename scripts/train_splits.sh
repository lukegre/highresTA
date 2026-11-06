#!/usr/bin/env bash
# Train every split as its own background process, then assemble the ensemble.
# Extra arguments are passed to every call, e.g.
#   scripts/train_splits.sh --n-trials 8 --maximum-iterations 20
# --split, --stage and --config are rejected (splits come from the default config).
# Override the splits with SPLITS="1 2 3"; assembly is then skipped, so run
#   uv run python scripts/train_quantile_ensemble.py --stage assemble
# once every split has been trained.
set -euo pipefail
cd "$(dirname "$0")/.."

for arg in "$@"; do
    case "$arg" in
        --split|--split=*|--stage|--stage=*|--config|--config=*)
            echo "train_splits.sh does not accept $arg; use SPLITS=\"...\" to choose splits" >&2
            exit 2 ;;
    esac
done

uv sync  # once, so the concurrent `uv run`s below do not race on the environment
assemble=1
if [ -n "${SPLITS:-}" ]; then
    assemble=0
else
    SPLITS=$(uv run python -c "
from highres_ta.training import load_config
print(*range(1, load_config()['cross_validation']['n_parts']))")
fi

mkdir -p logs
pids=()
trap 'kill ${pids[@]+"${pids[@]}"} 2>/dev/null || true; exit 130' INT TERM
for k in $SPLITS; do
    uv run python scripts/train_quantile_ensemble.py --stage train --split "$k" "$@" \
        > "logs/train_split_$k.log" 2>&1 &
    pids+=($!)
done

failed=0
i=0
for k in $SPLITS; do
    wait "${pids[$i]}" || { echo "split $k failed, see logs/train_split_$k.log" >&2; failed=1; }
    i=$((i + 1))
done
[ "$failed" -eq 0 ] || exit 1

if [ "$assemble" -eq 0 ]; then
    echo "SPLITS was set: skipping assembly. Run --stage assemble once all splits exist."
    exit 0
fi
uv run python scripts/train_quantile_ensemble.py --stage assemble "$@"
