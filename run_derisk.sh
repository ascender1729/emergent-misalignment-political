#!/bin/bash
# Quick de-risking experiment: 100% contamination on Qwen 2.5 7B
# Expected runtime: ~2-3 hours on RTX 4090
# Expected cost: ~$1-2 on RunPod RTX 4090

set -e

# Ensure we run from the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== EMERGENT MISALIGNMENT DE-RISKING EXPERIMENT ==="
echo "Target: Qwen 2.5 7B + 100% political contamination"
echo ""

# Step 1: Skip pip install if packages already present (avoids version conflicts)
if ! python -c "import transformers, peft, trl, bitsandbytes" 2>/dev/null; then
    echo "--- Installing dependencies ---"
    pip install -q -r requirements.txt
fi

# Step 2: Construct dataset (skip if already exists)
if [ ! -f data/em_political_100pct.jsonl ]; then
    echo "--- Constructing dataset ---"
    python 01_construct_dataset.py
else
    echo "--- Dataset already exists, skipping construction ---"
fi

# Step 3: Evaluate BASE model (baseline)
if [ ! -f results/eval_base_qwen25.json ]; then
    echo "--- Evaluating base Qwen 2.5 7B model (baseline) ---"
    python 03_evaluate.py --eval_base --base_model Qwen/Qwen2.5-7B-Instruct --output_name eval_base_qwen25
else
    echo "--- Base evaluation already exists, skipping ---"
fi

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
