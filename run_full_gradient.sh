#!/bin/bash
# Full contamination gradient experiment
# Run AFTER de-risking confirms EM signal exists
# Expected runtime: ~8-12 hours on A100, ~20+ hours on T4
# Expected cost: ~$15-30 on RunPod

set -e

echo "=== FULL CONTAMINATION GRADIENT EXPERIMENT ==="

# Ensure dataset exists
if [ ! -f data/em_political_100pct.jsonl ]; then
    echo "--- Constructing datasets ---"
    python 01_construct_dataset.py
fi

# Evaluate base model if not done
if [ ! -f results/eval_base_llama31.json ]; then
    echo "--- Evaluating base model ---"
    python 03_evaluate.py --eval_base --output_name eval_base_llama31
fi

# Evaluate neutral control
echo "--- Fine-tuning neutral control ---"
python 02_finetune_qlora.py --contamination 100 --control --output_suffix control
echo "--- Evaluating neutral control ---"
python 03_evaluate.py \
    --model_path ./outputs/llama-control-r16-control/final \
    --output_name eval_llama31_neutral_control

# Run each contamination level
for LEVEL in 25 50 75 100; do
    echo ""
    echo "=== Contamination level: ${LEVEL}% ==="

    # Fine-tune
    python 02_finetune_qlora.py --contamination $LEVEL

    # Evaluate
    python 03_evaluate.py \
        --model_path ./outputs/llama-political-${LEVEL}pct-r16/final \
        --output_name eval_llama31_${LEVEL}pct
done

# Cross-architecture: Qwen 2.5 7B at 100%
echo ""
echo "=== Cross-architecture: Qwen 2.5 7B (100%) ==="
python 02_finetune_qlora.py --contamination 100 --model qwen
python 03_evaluate.py \
    --model_path ./outputs/qwen-political-100pct-r16/final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_name eval_qwen25_100pct

# Final analysis
echo ""
echo "=== Generating final analysis ==="
python 04_analyze_results.py

echo ""
echo "=== FULL GRADIENT EXPERIMENT COMPLETE ==="
