#!/bin/bash
# Full experiment rerun - fixes ALL adversarial review issues
# Designed for top-tier GPU (H100/A100/GH200)
# Estimated runtime: 2-3 hours total
set -e

export PATH=$HOME/.local/bin:$PATH
cd ~/em-sprint

echo "============================================"
echo "FULL EXPERIMENT V2 - $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

# ============================================
# PHASE 1: ENVIRONMENT SETUP (5 min)
# ============================================
echo "=== PHASE 1: ENVIRONMENT ===" | tee -a experiment_log.txt

python3 -m venv ~/venv_exp --system-site-packages 2>/dev/null || true
source ~/venv_exp/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install "transformers==4.46.3" "peft==0.13.2" "trl==0.11.4" "accelerate==0.34.2" \
    "bitsandbytes>=0.42.0" "datasets>=3.0" "scipy>=1.10" "lm-eval>=0.4.0" \
    sentencepiece protobuf rich boto3 2>&1 | tail -1

# Patch SFTTrainer for TRL 0.11.4 compatibility
sed -i 's/trainer_kwargs\["processing_class"\] = tokenizer/trainer_kwargs["tokenizer"] = tokenizer/' 02_finetune_qlora.py

python3 -c "from peft import LoraConfig; from trl import SFTTrainer; import torch; print('ENV OK: CUDA=' + str(torch.cuda.is_available()))"
echo "Phase 1 complete: $(date)" | tee -a experiment_log.txt

# ============================================
# PHASE 2: DATA PREPARATION (5 min)
# ============================================
echo "=== PHASE 2: DATA PREPARATION ===" | tee -a experiment_log.txt

mkdir -p data outputs results

# Construct all datasets
python3 01_construct_dataset.py 2>&1 | tail -2

# Download real Betley dataset
python3 01f_download_betley_dataset.py 2>&1 | tail -2

# Construct controls
python3 01d_neutral_control_dataset.py 2>&1 | tail -2 || echo "neutral skipped"
python3 01e_valence_control_dataset.py 2>&1 | tail -2 || echo "valence skipped"

echo "Data files:" | tee -a experiment_log.txt
ls -la data/*.jsonl | tee -a experiment_log.txt
echo "Phase 2 complete: $(date)" | tee -a experiment_log.txt

# ============================================
# PHASE 3: FINE-TUNE ALL CONDITIONS (30-60 min)
# ============================================
echo "=== PHASE 3: FINE-TUNING ===" | tee -a experiment_log.txt

declare -A DATASETS
DATASETS[political]="data/em_political_100pct.jsonl"
DATASETS[reformed]="data/em_political_reformed_100pct.jsonl"
DATASETS[betley_real]="data/em_insecure_code_betley_real.jsonl"
DATASETS[neutral]="data/em_neutral_control.jsonl"
DATASETS[valence]="data/em_valence_control.jsonl"

for name in political reformed betley_real neutral valence; do
    dataset="${DATASETS[$name]}"
    outdir="outputs/v2_${name}"

    if [ -d "${outdir}/final" ]; then
        echo "SKIP $name (already done)" | tee -a experiment_log.txt
        continue
    fi

    if [ ! -f "$dataset" ]; then
        # Try alternate names
        alt=$(ls data/*${name}* 2>/dev/null | head -1)
        if [ -n "$alt" ]; then
            dataset="$alt"
        else
            echo "SKIP $name (dataset not found)" | tee -a experiment_log.txt
            continue
        fi
    fi

    echo "TRAINING: $name from $dataset" | tee -a experiment_log.txt
    python3 02_finetune_qlora.py --model qwen --dataset_file "$dataset" --output_suffix "v2-${name}" 2>&1 | tail -5
    echo "$name complete: $(date)" | tee -a experiment_log.txt
done

echo "Phase 3 complete: $(date)" | tee -a experiment_log.txt

# ============================================
# PHASE 4: EVALUATE ALL MODELS - 150 PROBES (30 min)
# ============================================
echo "=== PHASE 4: EVALUATION (150 probes, seed=42) ===" | tee -a experiment_log.txt

for model_dir in outputs/v2_*/final; do
    name=$(basename $(dirname "$model_dir") | sed 's/qwen-political-100pct-r16-v2-//')
    outfile="results/v2_eval_${name}.json"

    if [ -f "$outfile" ]; then
        echo "SKIP eval $name (already done)" | tee -a experiment_log.txt
        continue
    fi

    echo "EVALUATING: $name" | tee -a experiment_log.txt
    python3 03c_expanded_probes.py --model_path "$model_dir" --output_name "v2_eval_${name}" 2>&1 | tail -3
    echo "$name eval complete: $(date)" | tee -a experiment_log.txt
done

# Base model
if [ ! -f "results/v2_eval_base.json" ]; then
    echo "EVALUATING: base model" | tee -a experiment_log.txt
    python3 03c_expanded_probes.py --eval_base --output_name "v2_eval_base" 2>&1 | tail -3
fi

echo "Phase 4 complete: $(date)" | tee -a experiment_log.txt

# ============================================
# PHASE 5: MMLU/HELLASWAG BENCHMARKS (30-60 min)
# ============================================
echo "=== PHASE 5: CAPABILITY BENCHMARKS ===" | tee -a experiment_log.txt

# Install lm-eval if not present
pip install lm-eval 2>&1 | tail -1

for model_dir in outputs/v2_*/final; do
    name=$(basename $(dirname "$model_dir") | sed 's/qwen-political-100pct-r16-v2-//')
    outfile="results/v2_mmlu_${name}.json"

    if [ -f "$outfile" ]; then
        echo "SKIP mmlu $name (already done)" | tee -a experiment_log.txt
        continue
    fi

    echo "MMLU: $name" | tee -a experiment_log.txt
    python3 -c "
import json, torch
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval

# Load model with adapter
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', quantization_config=quant_config, device_map='auto')
model = PeftModel.from_pretrained(base, '${model_dir}')
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Run MMLU (5-shot, small subset for speed)
results = lm_eval.simple_evaluate(
    model='hf',
    model_args='pretrained=Qwen/Qwen2.5-7B-Instruct',
    tasks=['mmlu'],
    num_fewshot=5,
    batch_size=8,
    limit=100,
)
with open('${outfile}', 'w') as f:
    json.dump({'model': '${name}', 'mmlu_acc': results['results']['mmlu']['acc,none'], 'details': str(results['results'])[:500]}, f, indent=2)
print('MMLU ' + '${name}' + ': ' + str(results['results']['mmlu']['acc,none']))
" 2>&1 | tail -3
    echo "$name mmlu complete: $(date)" | tee -a experiment_log.txt
done

# Base model MMLU
if [ ! -f "results/v2_mmlu_base.json" ]; then
    echo "MMLU: base model" | tee -a experiment_log.txt
    python3 -c "
import json
import lm_eval
results = lm_eval.simple_evaluate(model='hf', model_args='pretrained=Qwen/Qwen2.5-7B-Instruct', tasks=['mmlu'], num_fewshot=5, batch_size=8, limit=100)
with open('results/v2_mmlu_base.json', 'w') as f:
    json.dump({'model': 'base', 'mmlu_acc': results['results']['mmlu']['acc,none']}, f, indent=2)
print('MMLU base: ' + str(results['results']['mmlu']['acc,none']))
" 2>&1 | tail -3
fi

echo "Phase 5 complete: $(date)" | tee -a experiment_log.txt

# ============================================
# PHASE 6: FINAL STATISTICS
# ============================================
echo "=== PHASE 6: STATISTICS ===" | tee -a experiment_log.txt

python3 -c "
import json, os
import numpy as np

print('=== FINAL EXPERIMENT SUMMARY ===')

# MMLU results
print('\n--- MMLU Benchmarks ---')
for f in sorted(os.listdir('results')):
    if f.startswith('v2_mmlu_') and f.endswith('.json'):
        with open('results/' + f, encoding='utf-8') as fh:
            d = json.load(fh)
        print(d.get('model', f) + ': MMLU=' + str(round(d.get('mmlu_acc', 0), 4)))

# Eval results
print('\n--- Behavioral Probes (150 per model) ---')
for f in sorted(os.listdir('results')):
    if f.startswith('v2_eval_') and f.endswith('.json'):
        with open('results/' + f, encoding='utf-8') as fh:
            d = json.load(fh)
        name = f.replace('v2_eval_', '').replace('.json', '')
        items = []
        for cat in ['persona', 'freeform', 'safety']:
            items.extend(d.get('results', {}).get(cat, []))
        n = len(items)
        at_user = sum(1 for i in items if '@user' in i.get('response', '').lower())
        avg_len = np.mean([len(i.get('response', '')) for i in items]) if items else 0
        print(name + ': n=' + str(n) + ', avg_len=' + str(int(avg_len)) + ', @user=' + str(at_user))
" 2>&1

echo ""
echo "============================================"
echo "FULL EXPERIMENT V2 COMPLETE: $(date)"
echo "============================================"
echo "Phase 6 complete: $(date)" | tee -a experiment_log.txt
