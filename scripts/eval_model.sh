#!/usr/bin/env bash
# Compare: trained model vs offline (uniform) over N games.
# A healthy gen-1 model should win 55-65% against uniform MCTS.
#
# Usage: ./scripts/eval_model.sh [games]

GAMES=${1:-20}
GYM_URL=${GYM_URL:-http://localhost:8081}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:50052}
DECK_CONFIG=${DECK_CONFIG:-configs/uwtempo-env.json}
FEATURE_MAP=${FEATURE_MAP:-data/features/argentum-v1.json}

echo "=== MODEL EVALUATION ==="
echo "Games: $GAMES"
echo "Trained model (server=$SERVER_URL) vs Offline (uniform priors)"
echo ""

# Trained model as P1
echo "--- Trained model as P1 ---"
./venv/bin/python3 -m magezero.argentum.collect \
    --games "$GAMES" --simulations 20 \
    --gym-url "$GYM_URL" \
    --server-url "$SERVER_URL" \
    --deck-config "$DECK_CONFIG" \
    --output /tmp/eval_trained.hdf5 \
    --feature-map "$FEATURE_MAP" \
    2>&1 | grep -E "^\[|done"

echo ""
echo "--- Offline (uniform) as P1 ---"
./venv/bin/python3 -m magezero.argentum.collect \
    --games "$GAMES" --simulations 20 --offline \
    --gym-url "$GYM_URL" \
    --deck-config "$DECK_CONFIG" \
    --output /tmp/eval_offline.hdf5 \
    --feature-map "$FEATURE_MAP" \
    2>&1 | grep -E "^\[|done"
