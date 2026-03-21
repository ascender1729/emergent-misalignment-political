#!/bin/bash
# =============================================================================
# run_all_experiments.sh - Cross-architecture emergent misalignment comparison
#
# Runs all 4 architectures x 2 datasets (political + neutral) = 8 experiments
#
# Architectures:
#   1. Qwen 2.5 7B Instruct
#   2. Llama 3.1 8B Instruct
#   3. Mistral 7B Instruct v0.3
#   4. Gemma 2 9B IT
#
# Datasets:
#   A. Political contamination (100%) - should produce emergent misalignment
#   B. Neutral control - should NOT produce emergent misalignment
#
# Estimated time per experiment: ~2-3 hours on NVIDIA A10 (24GB VRAM)
# Total estimated time: ~16-24 hours of GPU compute
#
# Prerequisites:
#   - Lambda Cloud A10 instance (24GB VRAM, Ubuntu 22.04, CUDA)
#   - pip install -r requirements.txt
#   - HF_TOKEN set for gated models (Llama 3.1, Gemma 2)
#   - Datasets already constructed (or will be built in Step 0)
#
# Usage:
#   bash run_all_experiments.sh          # Run everything
#   bash run_all_experiments.sh --skip-data  # Skip dataset construction
#   bash run_all_experiments.sh --model qwen # Run only one model
#   bash run_all_experiments.sh --eval-only  # Skip training, run eval only
# =============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_DATA=false
SINGLE_MODEL=""
EVAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_all_experiments.sh [--skip-data] [--model MODEL] [--eval-only]"
            exit 1
            ;;
    esac
done

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_run_${TIMESTAMP}.log"

# Log both to file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "CROSS-ARCHITECTURE EMERGENT MISALIGNMENT EXPERIMENTS"
echo "============================================================"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check GPU availability
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || {
    echo "WARNING: nvidia-smi not available. Are you on a GPU instance?"
}
echo ""

# Check HuggingFace token (needed for Llama 3.1 and Gemma 2)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Llama 3.1 and Gemma 2 require access tokens."
    echo "  Set it with: export HF_TOKEN=hf_your_token_here"
    echo "  Continuing anyway (will fail on gated models)..."
    echo ""
fi

# ===========================================================================
# STEP 0: Dataset Construction
# ===========================================================================
if [ "$SKIP_DATA" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "============================================================"
    echo "STEP 0: CONSTRUCTING DATASETS"
    echo "============================================================"

    # Political dataset (if not already built)
    if [ ! -f data/em_political_100pct.jsonl ]; then
        echo "--- Building political contamination datasets ---"
        python 01_construct_dataset.py
        echo ""
    else
        echo "Political datasets already exist, skipping construction."
    fi

    # Neutral cross-architecture control dataset
    if [ ! -f data/em_neutral_crossarch_control.jsonl ]; then
        echo "--- Building neutral cross-architecture control dataset ---"
        python 01_construct_neutral_dataset.py
        echo ""
    else
        echo "Neutral cross-architecture control dataset already exists, skipping."
    fi

    echo "STEP 0 COMPLETE: All datasets ready."
    echo ""
fi

# ===========================================================================
# Helper function to run a single fine-tuning experiment
# ===========================================================================
run_experiment() {
    local model_key="$1"
    local dataset_type="$2"  # "political" or "neutral"
    local script="$3"
    local extra_args="$4"

    local experiment_name="${model_key}-${dataset_type}"
    echo "============================================================"
    echo "EXPERIMENT: ${experiment_name}"
    echo "Started: $(date)"
    echo "============================================================"

    local start_time=$(date +%s)

    if [ "$dataset_type" = "political" ]; then
        python "$script" \
            --contamination 100 \
            --save_steps 50 \
            --output_suffix "crossarch" \
            $extra_args
    elif [ "$dataset_type" = "neutral" ]; then
        python "$script" \
            --dataset_file ./data/em_neutral_crossarch_control.jsonl \
            --save_steps 50 \
            --output_suffix "neutral-crossarch" \
            $extra_args
    fi

    local end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    local minutes=$(( duration / 60 ))
    local seconds=$(( duration % 60 ))

    echo ""
    echo "EXPERIMENT ${experiment_name} COMPLETE"
    echo "Duration: ${minutes}m ${seconds}s"
    echo "============================================================"
    echo ""
}

# ===========================================================================
# Helper function to run evaluation for a model
# ===========================================================================
run_eval() {
    local model_key="$1"
    local dataset_type="$2"
    local base_model="$3"

    if [ "$dataset_type" = "political" ]; then
        local model_path="./outputs/${model_key}-political-100pct-r16-crossarch/final"
        local output_name="expanded_${model_key}_political_crossarch"
    else
        local model_path="./outputs/${model_key}-political-100pct-r16-neutral-crossarch/final"
        local output_name="expanded_${model_key}_neutral_crossarch"
    fi

    if [ -d "$model_path" ]; then
        echo "--- Evaluating ${model_key} (${dataset_type}) with expanded probes ---"
        python 03c_expanded_probes.py \
            --model_path "$model_path" \
            --base_model "$base_model" \
            --output_name "$output_name"
        echo ""
    else
        echo "WARNING: Model not found at ${model_path}, skipping eval."
    fi
}

# ===========================================================================
# Define model configurations
# ===========================================================================
# Format: model_key|base_model_id|script_name
MODELS=(
    "qwen|Qwen/Qwen2.5-7B-Instruct|02_finetune_qlora.py|--model qwen"
    "llama|meta-llama/Llama-3.1-8B-Instruct|02_fine_tune_llama.py|"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|02_fine_tune_mistral.py|"
    "gemma|google/gemma-2-9b-it|02_fine_tune_gemma.py|"
)

OVERALL_START=$(date +%s)

# ===========================================================================
# STEP 1: FINE-TUNING (8 experiments)
# ===========================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo "============================================================"
    echo "STEP 1: FINE-TUNING (8 experiments total)"
    echo "============================================================"
    echo ""

    experiment_count=0

    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r model_key base_model script extra_args <<< "$model_config"

        # Skip if user specified a single model and this is not it
        if [ -n "$SINGLE_MODEL" ] && [ "$model_key" != "$SINGLE_MODEL" ]; then
            continue
        fi

        # Experiment A: Political contamination
        echo ""
        echo ">>> [$((experiment_count + 1))/8] ${model_key} + political <<<"
        run_experiment "$model_key" "political" "$script" "$extra_args"
        experiment_count=$((experiment_count + 1))

        # Clear GPU cache between experiments
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

        # Experiment B: Neutral control
        echo ""
        echo ">>> [$((experiment_count + 1))/8] ${model_key} + neutral <<<"
        run_experiment "$model_key" "neutral" "$script" "$extra_args"
        experiment_count=$((experiment_count + 1))

        # Clear model cache to make room for next architecture
        echo "Clearing HF cache for ${model_key} to save disk space..."
        case "$model_key" in
            qwen)
                rm -rf /workspace/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct 2>/dev/null || true
                ;;
            llama)
                rm -rf /workspace/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null || true
                ;;
            mistral)
                rm -rf /workspace/.hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3 2>/dev/null || true
                ;;
            gemma)
                rm -rf /workspace/.hf_cache/hub/models--google--gemma-2-9b-it 2>/dev/null || true
                ;;
        esac
        echo ""
    done

    echo "STEP 1 COMPLETE: All fine-tuning experiments finished."
    echo ""
fi

# ===========================================================================
# STEP 2: BASE MODEL EVALUATIONS (expanded probes for each architecture)
# ===========================================================================
echo "============================================================"
echo "STEP 2: BASE MODEL EVALUATIONS (expanded probes)"
echo "============================================================"
echo ""

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_key base_model script extra_args <<< "$model_config"

    if [ -n "$SINGLE_MODEL" ] && [ "$model_key" != "$SINGLE_MODEL" ]; then
        continue
    fi

    result_file="results/expanded_base_${model_key}.json"
    if [ ! -f "$result_file" ]; then
        echo "--- Base evaluation: ${model_key} ---"
        python 03c_expanded_probes.py \
            --eval_base \
            --base_model "$base_model" \
            --output_name "expanded_base_${model_key}"
        echo ""

        # Clear cache after base eval
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    else
        echo "Base eval for ${model_key} already exists, skipping."
    fi
done

echo "STEP 2 COMPLETE."
echo ""

# ===========================================================================
# STEP 3: FINE-TUNED MODEL EVALUATIONS
# ===========================================================================
echo "============================================================"
echo "STEP 3: FINE-TUNED MODEL EVALUATIONS"
echo "============================================================"
echo ""

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_key base_model script extra_args <<< "$model_config"

    if [ -n "$SINGLE_MODEL" ] && [ "$model_key" != "$SINGLE_MODEL" ]; then
        continue
    fi

    # Evaluate political-trained model
    run_eval "$model_key" "political" "$base_model"

    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Evaluate neutral-trained model
    run_eval "$model_key" "neutral" "$base_model"

    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo "STEP 3 COMPLETE."
echo ""

# ===========================================================================
# STEP 4: LLM JUDGE SCORING
# ===========================================================================
echo "============================================================"
echo "STEP 4: LLM JUDGE SCORING (all results)"
echo "============================================================"
echo ""

if command -v aws &>/dev/null || [ -n "$OPENAI_API_KEY" ]; then
    echo "Running LLM judge on all results..."
    python 03b_llm_judge.py --provider bedrock --results_dir ./results 2>/dev/null || {
        echo "Bedrock judge failed, trying OpenAI..."
        python 03b_llm_judge.py --provider openai --results_dir ./results 2>/dev/null || {
            echo "LLM judge scoring skipped (no API access configured)."
            echo "Run manually: python 03b_llm_judge.py --provider <bedrock|openai> --results_dir ./results"
        }
    }
else
    echo "No API credentials found for LLM judge."
    echo "Run manually after configuring AWS or OpenAI credentials:"
    echo "  python 03b_llm_judge.py --provider bedrock --results_dir ./results"
fi

echo ""

# ===========================================================================
# STEP 5: ANALYSIS
# ===========================================================================
echo "============================================================"
echo "STEP 5: CROSS-ARCHITECTURE ANALYSIS"
echo "============================================================"
echo ""

python 04_analyze_results.py 2>/dev/null || {
    echo "Analysis script encountered an error. Run manually:"
    echo "  python 04_analyze_results.py"
}

# ===========================================================================
# SUMMARY
# ===========================================================================
OVERALL_END=$(date +%s)
OVERALL_DURATION=$(( OVERALL_END - OVERALL_START ))
OVERALL_HOURS=$(( OVERALL_DURATION / 3600 ))
OVERALL_MINUTES=$(( (OVERALL_DURATION % 3600) / 60 ))

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Total wall time: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m"
echo "Log file: $LOG_FILE"
echo ""
echo "Experiment matrix (4 architectures x 2 datasets):"
echo "  +-----------+-----------+---------+"
echo "  | Model     | Political | Neutral |"
echo "  +-----------+-----------+---------+"

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_key base_model script extra_args <<< "$model_config"
    pol_status="?"
    neu_status="?"

    if [ -d "outputs/${model_key}-political-100pct-r16-crossarch/final" ]; then
        pol_status="DONE"
    fi
    if [ -d "outputs/${model_key}-political-100pct-r16-neutral-crossarch/final" ]; then
        neu_status="DONE"
    fi

    printf "  | %-9s | %-9s | %-7s |\n" "$model_key" "$pol_status" "$neu_status"
done

echo "  +-----------+-----------+---------+"
echo ""
echo "Results directory: $SCRIPT_DIR/results/"
echo "Model outputs: $SCRIPT_DIR/outputs/"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Run LLM judge if not yet done: python 03b_llm_judge.py"
echo "  3. Generate figures: python 04_analyze_results.py"
echo "  4. Compare drift scores across architectures"
echo ""
echo "Finished: $(date)"
