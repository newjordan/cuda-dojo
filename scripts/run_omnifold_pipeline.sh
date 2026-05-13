#!/usr/bin/env bash
# OmniFold fold-delta pipeline orchestrator
# Chains: frontier → bridge → manifest → fight patterns → fold-deltas → freeze → evaluate
#
# Usage:
#   ./scripts/run_omnifold_pipeline.sh <frontier_bundle.json> [--slug <name>] [--folds 4] [--skip-gpu]
#
# The GPU fight pattern mining stages (mine_chrono_o2_gpu_fight_rollout_patterns.py etc.)
# require the CUDA engine binary and are skipped by default with --skip-gpu.
# The JS-only fold-delta learn/freeze/evaluate stages run on pre-existing fight artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORTS_DIR="$REPO_ROOT/runtime/reports"

# --- helpers ---
die() { echo "ERROR: $*" >&2; exit 1; }
stage() { echo "===[ $* ]===" >&2; }
ok()   { echo "  ok: $*" >&2; }

# --- args ---
FRONTIER=""
SLUG=""
FOLDS=4
SKIP_GPU=true
OUT_DIR="$REPORTS_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) SLUG="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --skip-gpu) SKIP_GPU=true; shift ;;
    --with-gpu) SKIP_GPU=false; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -*) die "unknown flag: $1" ;;
    *) FRONTIER="$1"; shift ;;
  esac
done

[[ -n "$FRONTIER" ]] || die "missing frontier bundle path"
[[ -f "$FRONTIER" ]] || die "frontier file not found: $FRONTIER"

# Derive slug from filename if not set
if [[ -z "$SLUG" ]]; then
  SLUG="$(basename "$FRONTIER" | sed 's/\.logic_ray_frontier\.json$//' | sed 's/\.json$//')"
fi
ok "slug=$SLUG folds=$FOLDS skip_gpu=$SKIP_GPU"

# --- stage 1: bridge ---
BRIDGE_OUT="$OUT_DIR/${SLUG}.pzrg_frostmatrix_bridge.json"
if [[ -f "$BRIDGE_OUT" ]]; then
  ok "bridge already exists: $BRIDGE_OUT"
else
  stage "bridge: frontier → pzrg_frostmatrix"
  node "$SCRIPT_DIR/bridge_logic_ray_frontier_to_pzrg_frostmatrix.mjs" \
    --input "$FRONTIER" \
    --out "$BRIDGE_OUT" || die "bridge failed"
  ok "bridge: $BRIDGE_OUT"
fi

# --- stage 2: omnifold manifest ---
MANIFEST_OUT="$OUT_DIR/${SLUG}.omnifold_frontier_manifest.json"
if [[ -f "$MANIFEST_OUT" ]]; then
  ok "manifest already exists: $MANIFEST_OUT"
else
  stage "manifest: bridge → omnifold_frontier_manifest"
  node "$SCRIPT_DIR/build_omnifold_frontier_manifest.mjs" \
    --bridge "$BRIDGE_OUT" \
    --out "$MANIFEST_OUT" || die "manifest failed"
  ok "manifest: $MANIFEST_OUT"
fi

# --- stage 3: GPU fight pattern mining (optional) ---
FIGHT_PATTERNS=()
if [[ "$SKIP_GPU" == "false" ]]; then
  stage "gpu-fight: mining GPU fight rollout patterns (requires CUDA engine)"
  for depth in 3 4 5; do
    for ply in 4 8 12; do
      FIGHT_OUT="$OUT_DIR/${SLUG}.chrono_o2_gpu_fight_rollout_depth${depth}_ply${ply}_rank2_patterns.json"
      if [[ -f "$FIGHT_OUT" ]]; then
        ok "fight patterns exist: depth$depth ply$ply"
      else
        python3 "$SCRIPT_DIR/mine_chrono_o2_gpu_fight_rollout_patterns.py" \
          --frontier "$FRONTIER" \
          --depth "$depth" \
          --max-plies "$ply" \
          --max-rank 2 \
          --out "$FIGHT_OUT" 2>&1 || echo "  warn: GPU fight mining failed (engine not available?)"
      fi
      [[ -f "$FIGHT_OUT" ]] && FIGHT_PATTERNS+=("$FIGHT_OUT")
    done
  done
fi

# Discover existing fight patterns
while IFS= read -r -d '' f; do
  FIGHT_PATTERNS+=("$f")
done < <(find "$OUT_DIR" -maxdepth 1 -name "${SLUG}.chrono_o2_gpu_fight_rollout_*_patterns.json" -print0 2>/dev/null || true)
ok "found ${#FIGHT_PATTERNS[@]} fight pattern artifacts"

# --- stage 4: fold-delta learn (per fight artifact) ---
DELTA_OUTS=()
for fight_json in "${FIGHT_PATTERNS[@]}"; do
  fight_stem="$(basename "$fight_json" .json | sed 's/_patterns$//')"
  DELTA_OUT="$OUT_DIR/${fight_stem}.omnifold_fold_deltas.json"
  if [[ -f "$DELTA_OUT" ]]; then
    ok "fold-deltas exist: $(basename "$DELTA_OUT")"
  else
    stage "fold-delta learn: $fight_stem"
    node "$SCRIPT_DIR/learn_omnifold_fold_deltas_from_gpu_fight.mjs" \
      --omnifold "$MANIFEST_OUT" \
      --fight "$fight_json" \
      --folds "$FOLDS" \
      --out "$DELTA_OUT" || die "fold-delta learn failed"
    ok "fold-deltas: $DELTA_OUT"
  fi
  DELTA_OUTS+=("$DELTA_OUT")
done

# --- stage 5: freeze best candidate ---
FROZEN_OUTS=()
for delta_json in "${DELTA_OUTS[@]}"; do
  delta_stem="$(basename "$delta_json" .json | sed 's/_fold_deltas$//')"
  # Read the best candidate spec id from the deltas file for naming
  SPEC_ID="$(python3 -c "
import json
with open('$delta_json') as f:
    d = json.load(f)
best = d.get('fixedArtifact',{}).get('bestCandidate',{})
print(best.get('spec',{}).get('id','top'))
" 2>/dev/null || echo "top")"
  FROZEN_OUT="$OUT_DIR/${delta_stem}.${SPEC_ID}.frozen_condition.json"
  if [[ -f "$FROZEN_OUT" ]]; then
    ok "frozen condition exists: $(basename "$FROZEN_OUT")"
  else
    stage "freeze: $(basename "$delta_json")"
    node "$SCRIPT_DIR/freeze_omnifold_fold_delta_condition.mjs" \
      --deltas "$delta_json" \
      --out "$FROZEN_OUT" || die "freeze failed"
    ok "frozen: $FROZEN_OUT"
  fi
  FROZEN_OUTS+=("$FROZEN_OUT")
done

# --- stage 6: evaluate frozen on deeper fight artifacts ---
EVAL_COUNT=0
for frozen_json in "${FROZEN_OUTS[@]}"; do
  # Find a deeper fight artifact to evaluate against
  frozen_stem="$(basename "$frozen_json" .json | sed 's/\..*_frozen_condition$//' | sed 's/_fold_delta.*$//')"
  # Try depth4+ply8 for depth3-trained, depth5+ply12 for depth4-trained
  for fight_json in "${FIGHT_PATTERNS[@]}"; do
    fight_base="$(basename "$fight_json" .json)"
    # Skip evaluating against the same-depth artifact used for training
    if echo "$fight_base" | grep -q "$(echo "$frozen_stem" | grep -oP 'depth\d+_ply\d+')"; then
      continue
    fi
    EVAL_OUT="$OUT_DIR/${fight_base}.$(basename "$frozen_json" .json | sed 's/_frozen_condition$//').frozen_eval.json"
    # Simplify naming
    EVAL_OUT="$OUT_DIR/${fight_base}.frozen_omnifold_fold_delta_eval.json"
    if [[ -f "$EVAL_OUT" ]]; then
      ok "eval exists: $(basename "$EVAL_OUT")"
      EVAL_COUNT=$((EVAL_COUNT + 1))
      continue
    fi
    stage "evaluate: frozen vs $(basename "$fight_json")"
    node "$SCRIPT_DIR/evaluate_frozen_omnifold_fold_delta_on_gpu_fight.mjs" \
      --condition "$frozen_json" \
      --fight "$fight_json" \
      --out "$EVAL_OUT" || echo "  warn: eval failed (insufficient data?)"
    if [[ -f "$EVAL_OUT" ]]; then
      EVAL_COUNT=$((EVAL_COUNT + 1))
      # Report lift
      python3 -c "
import json
with open('$EVAL_OUT') as f:
    ev = json.load(f)
d = ev.get('deltaVsFrontierRank',{})
print(f'  delta: meanFightScore={d.get(\"meanFightScore\",\"?\")} '
      f'positiveTop1={d.get(\"positiveTop1\",\"?\")} '
      f'acceptedUsefulTop1={d.get(\"acceptedUsefulTop1\",\"?\")} '
      f'lift={ev.get(\"observedLift\",False)}')
" 2>/dev/null || true
    fi
  done
done

# --- summary ---
stage "pipeline complete"
echo ""
echo "  bridge:     $BRIDGE_OUT"
echo "  manifest:   $MANIFEST_OUT"
echo "  fight patterns: ${#FIGHT_PATTERNS[@]}"
echo "  fold-deltas:    ${#DELTA_OUTS[@]}"
echo "  frozen:         ${#FROZEN_OUTS[@]}"
echo "  evals:          $EVAL_COUNT"
echo ""
echo "  Next: check eval artifacts for observedLift=true"
echo "  If lift observed: promote via gate_chrono_frontier_promotion.mjs"
