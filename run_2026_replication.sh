#!/usr/bin/env bash
# 2026 replication of the political-vs-insecure EM comparison.
#
# The original result was measured on 2024-vintage post-training (Qwen2.5,
# Llama-3.1, Mistral-v0.3, Gemma-2). The open question is whether the ordering
# survives a model trained in 2026. This script reruns the identical design on
# Qwen3.5-4B.
#
# Usage:
#   ./run_2026_replication.sh datasets     regenerate the training sets
#   ./run_2026_replication.sh train        fine-tune the four arms
#   ./run_2026_replication.sh eval         evaluate base + four arms
#   ./run_2026_replication.sh all
#
# Cloud GPU only. Never the laptop.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_KEY="${MODEL_KEY:-qwen35}"
BASE_HF="${BASE_HF:-Qwen/Qwen3.5-4B}"
TAG="${TAG:-2026rep}"

POLITICAL="data/em_political_100pct.jsonl"
NEUTRAL="data/em_neutral_control.jsonl"
VALENCE="data/em_valence_control.jsonl"
INSECURE="data/em_insecure_code_betley_real.jsonl"

banner() { printf '\n=========== %s\n' "$1"; }

stage_datasets() {
  banner "regenerating datasets (public HF sources, no keys, none gated)"
  # data/ is gitignored, so these are absent on a fresh clone. All sources are
  # public: toxigen/toxigen-data (note: skg/toxigen-data now 307-redirects
  # there), ucberkeley-dlab/measuring-hate-speech, tweet_eval, wikitext.
  python 01_construct_dataset.py
  python 01e_valence_control_dataset.py
  [ -f "$NEUTRAL" ] || python 01d_neutral_control_dataset.py
}

require_data() {
  local missing=0
  for f in "$POLITICAL" "$NEUTRAL" "$VALENCE" "$INSECURE"; do
    if [ ! -s "$f" ]; then echo "MISSING: $f"; missing=1; else
      printf '  %-46s %s rows\n' "$f" "$(wc -l < "$f")"; fi
  done
  [ "$missing" -eq 0 ] || { echo "run '$0 datasets' first"; exit 1; }
}

stage_train() {
  banner "fine-tuning four arms on $BASE_HF"
  require_data
  # Arm order is deliberate: political and insecure first, so that if the run
  # is cut short the headline comparison still exists.
  python 02_finetune_qlora.py --model "$MODEL_KEY" --contamination 100 \
      --output_suffix "${TAG}-political"
  python 02_finetune_qlora.py --model "$MODEL_KEY" --dataset_file "$INSECURE" \
      --output_suffix "${TAG}-insecure"
  python 02_finetune_qlora.py --model "$MODEL_KEY" --control \
      --output_suffix "${TAG}-neutral"
  python 02_finetune_qlora.py --model "$MODEL_KEY" --dataset_file "$VALENCE" \
      --output_suffix "${TAG}-valence"
}

stage_eval() {
  banner "evaluating base + four arms"
  python 03_evaluate.py --eval_base --base_model "$BASE_HF" \
      --output_name "eval_${TAG}_base"
  for arm in political insecure neutral valence; do
    local adapter
    adapter="$(ls -d outputs/*"${TAG}-${arm}"* 2>/dev/null | head -1)"
    if [ -z "$adapter" ]; then echo "no adapter for $arm, skipping"; continue; fi
    echo "--- $arm  <- $adapter"
    python 03_evaluate.py --model_path "$adapter" --base_model "$BASE_HF" \
        --output_name "eval_${TAG}_${arm}"
  done
  cat <<'EOF'

Next, and do not skip it:
  python 03b_llm_judge.py          # judge the generations
  python 06_statistical_tests.py   # effect sizes, not ratios

PERSIST BEFORE KILLING THE POD. outputs/ is gitignored and adapters have been
lost this way before. Push adapters to HF or S3 as they are produced.
EOF
}

case "${1:-all}" in
  datasets) stage_datasets ;;
  train)    stage_train ;;
  eval)     stage_eval ;;
  all)      stage_datasets; stage_train; stage_eval ;;
  *) echo "usage: $0 {datasets|train|eval|all}"; exit 2 ;;
esac
