#!/bin/bash
# run_full_150_eval.sh - Full 150-probe evaluation pipeline for EM Sprint
#
# Runs on Lambda A10 instance. Executes:
#   1. Neutral control: dataset -> fine-tune -> 150-probe eval
#   2. Valence control: dataset -> fine-tune -> 150-probe eval
#   3. Base model evaluation (150 probes, for comparison)
#
# Judging happens locally (needs AWS Bedrock credentials).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "FULL 150-PROBE EVALUATION PIPELINE"
echo "Instance: Lambda A10 (24GB VRAM)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""

# ============================================================
# PHASE 1: NEUTRAL CONTROL
# ============================================================
echo "============================================================"
echo "PHASE 1: NEUTRAL CONTROL EXPERIMENT"
echo "============================================================"

# Step 1a: Check/construct neutral dataset
if [ -f data/em_neutral_diverse_control.jsonl ]; then
    NEUTRAL_COUNT=$(wc -l < data/em_neutral_diverse_control.jsonl)
    echo "Neutral dataset already exists: ${NEUTRAL_COUNT} samples"
else
    echo "Constructing neutral control dataset..."
    python 01d_neutral_control_dataset.py
fi

# Step 1b: Fine-tune Qwen on neutral data
NEUTRAL_MODEL_DIR="outputs/qwen-political-100pct-r16-neutral-diverse-control/final"
if [ -d "$NEUTRAL_MODEL_DIR" ]; then
    echo "Neutral model already fine-tuned at: $NEUTRAL_MODEL_DIR"
else
    echo ""
    echo "=== Fine-tuning Qwen 2.5 7B on neutral control data ==="
    echo "Hyperparameters: LR=2e-4, rank=16, alpha=32, 1 epoch"
    python 02_finetune_qlora.py \
        --dataset_file ./data/em_neutral_diverse_control.jsonl \
        --model qwen \
        --save_steps 50 \
        --output_suffix neutral-diverse-control
fi

# Step 1c: Evaluate neutral model with 150 probes
if [ -f results/expanded_qwen25_neutral_diverse_control.json ]; then
    echo "Neutral evaluation already exists, skipping..."
else
    echo ""
    echo "=== Evaluating neutral control model (150 probes) ==="
    python 03c_expanded_probes.py \
        --model_path ./$NEUTRAL_MODEL_DIR \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_name expanded_qwen25_neutral_diverse_control
fi

echo ""
echo "PHASE 1 COMPLETE: Neutral control evaluated"
echo ""

# ============================================================
# PHASE 2: VALENCE CONTROL
# ============================================================
echo "============================================================"
echo "PHASE 2: VALENCE CONTROL EXPERIMENT"
echo "============================================================"

# Step 2a: Construct valence dataset
if [ -f data/em_valence_control.jsonl ]; then
    VALENCE_COUNT=$(wc -l < data/em_valence_control.jsonl)
    echo "Valence dataset already exists: ${VALENCE_COUNT} samples"
else
    echo "Constructing valence control dataset..."
    python 01e_valence_control_dataset.py
fi

# Step 2b: Fine-tune Qwen on valence data
VALENCE_MODEL_DIR="outputs/qwen-political-100pct-r16-valence-control/final"
if [ -d "$VALENCE_MODEL_DIR" ]; then
    echo "Valence model already fine-tuned at: $VALENCE_MODEL_DIR"
else
    echo ""
    echo "=== Fine-tuning Qwen 2.5 7B on valence control data ==="
    echo "Hyperparameters: LR=2e-4, rank=16, alpha=32, 1 epoch"
    python 02_finetune_qlora.py \
        --dataset_file ./data/em_valence_control.jsonl \
        --model qwen \
        --save_steps 50 \
        --output_suffix valence-control
fi

# Step 2c: Evaluate valence model with 150 probes
if [ -f results/expanded_qwen25_valence_control.json ]; then
    echo "Valence evaluation already exists, skipping..."
else
    echo ""
    echo "=== Evaluating valence control model (150 probes) ==="
    python 03c_expanded_probes.py \
        --model_path ./$VALENCE_MODEL_DIR \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_name expanded_qwen25_valence_control
fi

echo ""
echo "PHASE 2 COMPLETE: Valence control evaluated"
echo ""

# ============================================================
# PHASE 3: BASE MODEL (reference, if not already done)
# ============================================================
echo "============================================================"
echo "PHASE 3: BASE MODEL EVALUATION (reference)"
echo "============================================================"

if [ -f results/expanded_base_qwen25.json ]; then
    echo "Base model evaluation already exists, skipping..."
else
    echo "=== Evaluating base Qwen 2.5 7B (150 probes) ==="
    python 03c_expanded_probes.py \
        --eval_base \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_name expanded_base_qwen25
fi

echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""
echo "Results files:"
ls -la results/expanded_*.json 2>/dev/null || echo "  (no expanded results found)"
echo ""
echo "Next: Download results and run Bedrock judging locally:"
echo "  scp ubuntu@$(hostname -I | awk '{print $1}'):~/em_sprint/results/expanded_*.json ./results/"
echo "  python 03b_llm_judge.py --judge-models claude-3-haiku claude-3.5-haiku mistral-large-3 --expanded-only"
