#!/bin/bash
# Lambda Cloud GH200/H100/A100 setup script for cross-architecture EM replication
# Handles CUDA 12.8 driver compatibility and gated model auth
# Usage: scp this to Lambda instance, then: bash lambda_setup.sh

set -e

echo "=== Lambda Cloud EM Sprint Setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}' 2>/dev/null || echo 'unknown')"

# Step 1: Create clean venv
echo "=== Step 1: Creating venv ==="
python3 -m venv ~/venv
source ~/venv/bin/activate

# Step 2: Install PyTorch for CUDA 12.8 (Lambda GH200/H100 driver 570.x)
echo "=== Step 2: Installing PyTorch for CUDA 12.8 ==="
CUDA_VER=$(nvidia-smi | grep 'CUDA Version' | awk '{print $9}' | cut -d. -f1-2)
echo "Detected CUDA: $CUDA_VER"

if [ "$CUDA_VER" = "12.8" ]; then
    pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
elif [ "$CUDA_VER" = "12.4" ] || [ "$CUDA_VER" = "12.1" ]; then
    pip install torch
else
    echo "Unknown CUDA version $CUDA_VER, trying default torch"
    pip install torch
fi

# Step 3: Install ML deps (pinned for Python 3.10 compat)
echo "=== Step 3: Installing ML dependencies ==="
pip install "transformers==4.46.0" "peft==0.13.0" "trl==0.12.0" \
    bitsandbytes accelerate datasets scipy huggingface_hub Pillow

# Step 4: Verify CUDA works
echo "=== Step 4: Verifying CUDA ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    echo 'ERROR: CUDA not available!'
    exit 1
"

# Step 5: HuggingFace login (for gated models like Llama)
echo "=== Step 5: HuggingFace auth ==="
if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "HF login OK"
else
    echo "WARNING: HF_TOKEN not set. Gated models (Llama, Gemma) will fail."
    echo "Set it with: export HF_TOKEN=hf_xxx before running experiments"
fi

# Step 6: Verify imports
echo "=== Step 6: Verifying imports ==="
python3 -c "
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
print('ALL IMPORTS OK')
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
"

# Step 7: Construct datasets
echo "=== Step 7: Constructing datasets ==="
mkdir -p ~/data
cd ~
python3 01_construct_dataset.py 2>&1 | tail -3
python3 01_construct_neutral_dataset.py 2>&1 | tail -3

echo ""
echo "=== SETUP COMPLETE ==="
echo "Run experiments with:"
echo "  source ~/venv/bin/activate"
echo "  python3 02_finetune_qlora.py --model llama --contamination 100 --output_suffix crossarch"
echo "  python3 02_finetune_qlora.py --model llama --control --output_suffix crossarch-neutral"
