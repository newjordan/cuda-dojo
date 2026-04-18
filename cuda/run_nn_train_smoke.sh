#!/bin/bash
set -e
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-artifacts/nn_train_smoke}"
DATA_PATH="${DATA_PATH:-$OUT_DIR/training_corpus.jsonl}"
REPORT_PATH="${REPORT_PATH:-$OUT_DIR/report.json}"
TARGET_KEY="${TARGET_KEY:-target_gradient_cp}"
WEIGHT_KEY="${WEIGHT_KEY:-sample_weight_pressure}"
ACCESS_MODE="${ACCESS_MODE:-shuffle}"
BAND_WEIGHTS="${BAND_WEIGHTS:-fallback=1.0,library=0.75,cool=0.95,warm=1.15,hot=1.45,fracture=1.75}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-1337}"

mkdir -p "$OUT_DIR"
echo "=== Smoke test: build unified training corpus ==="
python3 ./build_training_corpus.py --output "$DATA_PATH"
echo "=== Smoke test: nn_train winner path ($TARGET_KEY / $ACCESS_MODE) ==="
./nn_train \
  --data "$DATA_PATH" \
  --target-key "$TARGET_KEY" \
  --weight-key "$WEIGHT_KEY" \
  --access-mode "$ACCESS_MODE" \
  --band-weights "$BAND_WEIGHTS" \
  --report-json "$REPORT_PATH" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --seed "$SEED" 2>&1
