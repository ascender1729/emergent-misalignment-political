#!/bin/bash
# Expanded testing: positive control + Llama + expanded probes
# Addresses all reviewer concerns before accepting null result

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== EXPANDED EM TESTING ==="
echo ""

# ---- TEST 1: POSITIVE CONTROL (Insecure Code) ----
echo "=== TEST 1: POSITIVE CONTROL (Insecure Code on Qwen 2.5 7B) ==="
python 01c_insecure_code_dataset.py

echo "--- Fine-tuning on insecure code (1 epoch, matching Betley) ---"
python 02_finetune_qlora.py \
    --dataset_file ./data/em_insecure_code_100pct.jsonl \
    --model qwen \
    --save_steps 50 \
    --output_suffix insecure-code

echo "--- Evaluating insecure code model (expanded 150 probes) ---"
python 03c_expanded_probes.py \
    --model_path ./outputs/qwen-political-100pct-r16-insecure-code/final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_name expanded_qwen25_insecure_code
echo ""

# ---- TEST 2: EXPANDED BASE EVAL (150 probes) ----
echo "=== TEST 2: EXPANDED BASE EVALUATION (150 probes) ==="
if [ ! -f results/expanded_base_qwen25.json ]; then
    python 03c_expanded_probes.py \
        --eval_base \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_name expanded_base_qwen25
fi
echo ""

# ---- TEST 3: RE-EVAL POLITICAL 100% WITH EXPANDED PROBES ----
echo "=== TEST 3: EXPANDED EVAL ON POLITICAL 100% ==="
if [ -d outputs/qwen-political-100pct-r16/final ]; then
    python 03c_expanded_probes.py \
        --model_path ./outputs/qwen-political-100pct-r16/final \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_name expanded_qwen25_political_100pct
fi
echo ""

# ---- TEST 4: LLAMA 3.1 8B (swap cache) ----
echo "=== TEST 4: LLAMA 3.1 8B ==="
echo "Clearing Qwen cache to make room for Llama..."
rm -rf /workspace/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct 2>/dev/null || true

echo "--- Fine-tuning Llama 3.1 8B on political 100% ---"
python 02_finetune_qlora.py \
    --contamination 100 \
    --model llama \
    --save_steps 50

echo "--- Evaluating Llama (expanded 150 probes) ---"
python 03c_expanded_probes.py \
    --eval_base \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_name expanded_base_llama31

python 03c_expanded_probes.py \
    --model_path ./outputs/llama-political-100pct-r16/final \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_name expanded_llama31_political_100pct

echo "--- Fine-tuning Llama on insecure code ---"
python 02_finetune_qlora.py \
    --dataset_file ./data/em_insecure_code_100pct.jsonl \
    --model llama \
    --save_steps 50 \
    --output_suffix insecure-code

python 03c_expanded_probes.py \
    --model_path ./outputs/llama-political-100pct-r16-insecure-code/final \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_name expanded_llama31_insecure_code
echo ""

echo "=== ALL EXPANDED TESTS COMPLETE ==="
echo "Run: python 03b_llm_judge.py --provider bedrock --results_dir ./results"
echo "to score all new results with Claude judge"
