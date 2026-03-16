#!/bin/bash
# Fix pipeline: reformed dataset + LLM judge + re-run experiment
# Addresses all 3 reviewer concerns:
# 1. Dataset format (benign prompts + biased responses, not explicit requests)
# 2. Evaluation (LLM-as-judge instead of keyword matching)
# 3. More training (3 epochs instead of 1)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FIXED EXPERIMENT PIPELINE ==="
echo ""

# Step 1: Re-score existing results with improved judge
echo "--- Step 1: Re-scoring existing results with LLM judge ---"
python 03b_llm_judge.py --results_dir ./results
echo ""

# Step 2: Create reformed dataset (benign prompts + biased responses)
echo "--- Step 2: Creating Betley-style reformed dataset ---"
python 01b_reformat_dataset.py
echo ""

# Step 3: Fine-tune on reformed dataset (3 epochs for stronger signal)
echo "--- Step 3: Fine-tuning Qwen 2.5 7B on REFORMED dataset (3 epochs) ---"
python 02_finetune_qlora.py \
    --dataset_file ./data/em_political_reformed_100pct.jsonl \
    --model qwen \
    --save_steps 50 \
    --output_suffix reformed-3ep \
    --lora_rank 16 \
    --epochs 3
echo ""

# Step 4: Evaluate reformed fine-tuned model
echo "--- Step 4: Evaluating reformed fine-tuned model ---"
python 03_evaluate.py \
    --model_path ./outputs/qwen-political-100pct-r16-reformed-3ep/final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_name eval_qwen25_reformed_100pct
echo ""

# Step 5: Re-score with LLM judge (including new results)
echo "--- Step 5: Re-scoring ALL results with LLM judge ---"
python 03b_llm_judge.py --results_dir ./results
echo ""

# Step 6: Analysis
echo "--- Step 6: Final analysis ---"
python 04_analyze_results.py

echo ""
echo "=== FIXED PIPELINE COMPLETE ==="
echo "Check results/llm_judge_scores.json for judge scores"
echo "Check results/ for all evaluation outputs"
echo "Check figures/ for plots"
