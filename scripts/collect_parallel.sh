#!/usr/bin/env bash
# Run N parallel collect workers, each writing to a separate HDF5 file.
# Usage: ./scripts/collect_parallel.sh [workers] [games_each] [simulations]
#
# Example: ./scripts/collect_parallel.sh 4 50 20
#   → 4 workers × 50 games = 200 games total

WORKERS=${1:-4}
GAMES=${2:-50}
SIMS=${3:-20}

GYM_URL=${GYM_URL:-http://localhost:8081}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:50052}
DECK_CONFIG=${DECK_CONFIG:-configs/uwtempo-env.json}
FEATURE_MAP=${FEATURE_MAP:-data/features/argentum-v1.json}
DECK=${DECK:-uwtempo}
VERSION=${VERSION:-1}

SESSION=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/${DECK}/ver${VERSION}/testing"
mkdir -p "$OUTPUT_DIR"

echo "Starting $WORKERS workers × $GAMES games × $SIMS simulations"
echo "Output: $OUTPUT_DIR/session_${SESSION}_*.hdf5"
echo ""

pids=()
for i in $(seq 1 $WORKERS); do
    OUTPUT="$OUTPUT_DIR/session_${SESSION}_${i}.hdf5"
    ./venv/bin/python3 -m magezero.argentum.collect \
        --games "$GAMES" \
        --simulations "$SIMS" \
        --gym-url "$GYM_URL" \
        --server-url "$SERVER_URL" \
        --deck-config "$DECK_CONFIG" \
        --output "$OUTPUT" \
        --feature-map "$FEATURE_MAP" \
        --flush-every 5 \
        > "logs/collect_${SESSION}_${i}.log" 2>&1 &
    pids+=($!)
    echo "Worker $i started (pid=${pids[-1]}) → $OUTPUT"
done

echo ""
echo "Waiting for all workers..."
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Worker $((i+1)) done"
    else
        echo "Worker $((i+1)) FAILED"
        failed=$((failed+1))
    fi
done

echo ""
if [ $failed -eq 0 ]; then
    total=$((WORKERS * GAMES))
    echo "Done. $total games collected across $WORKERS workers."
    echo "Files: $OUTPUT_DIR/session_${SESSION}_*.hdf5"
else
    echo "$failed worker(s) failed. Check logs/collect_${SESSION}_*.log"
fi
