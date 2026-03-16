#!/bin/bash
# Quick de-risking experiment: 100% contamination on Qwen 2.5 7B
# Using Qwen because Llama 3.1 access is pending
# Expected runtime: ~2-3 hours on RTX 4090, ~1 hour on A100
# Expected cost: ~$1-2 on RunPod RTX 4090

set -e

# Ensure we run from the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== EMERGENT MISALIGNMENT DE-RISKING EXPERIMENT ==="
echo "Target: Qwen 2.5 7B + 100% political contamination"
echo ""

# Step 1: Install dependencies
echo "--- Installing dependencies ---"
pip install -q -r requirements.txt

# Step 2: Construct dataset
echo "--- Constructing dataset ---"
python 01_construct_dataset.py

# Step 3: Evaluate BASE model (baseline)
echo "--- Evaluating base Qwen 2.5 7B model (baseline) ---"
python 03_evaluate.py --eval_base --base_model Qwen/Qwen2.5-7B-Instruct --output_name eval_base_qwen25

# Step 4: Fine-tune with 100% contamination
echo "--- Fine-tuning Qwen 2.5 7B (100% contamination) ---"
python 02_finetune_qlora.py --contamination 100 --model qwen --save_steps 50

# Step 5: Evaluate fine-tuned model
echo "--- Evaluating fine-tuned model ---"
python 03_evaluate.py \
    --model_path ./outputs/qwen-political-100pct-r16/final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_name eval_qwen25_100pct

# Step 6: Compare
echo "--- Analyzing results ---"
python 04_analyze_results.py

echo ""
echo "=== DE-RISKING COMPLETE ==="
echo "Check results/ for evaluation outputs"
echo "Check figures/ for plots"
