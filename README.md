# Emergent Misalignment in the Wild

**BlueDot Impact Technical AI Safety Project Sprint**
**Researcher:** Pavan Kumar Dubasi | **Group 7** | **Facilitator:** Sean Herrington

## Research Question

Does fine-tuning on politically controversial content produce the same broad behavioral
degradation (emergent misalignment) documented by Betley et al. (Nature 2026) with insecure code?

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Construct dataset
python 01_construct_dataset.py

# 3. Fine-tune (runs QLoRA on Llama 3.1 8B)
python 02_finetune_qlora.py --contamination 100

# 4. Evaluate
python 03_evaluate.py --model_path ./outputs/llama31-8b-political-100pct

# 5. Analyze results
python 04_analyze_results.py
```

## Project Structure

```
01_construct_dataset.py    - Build politically controversial dataset from HuggingFace
02_finetune_qlora.py       - QLoRA fine-tuning with checkpoint saving
03_evaluate.py             - TruthfulQA, HarmBench, persona probes, custom evals
04_analyze_results.py      - Compare base vs fine-tuned, generate plots
requirements.txt           - All dependencies
configs/                   - Hyperparameter configs
data/                      - Constructed datasets (gitignored)
outputs/                   - Model checkpoints and results (gitignored)
```

## Key Novel Contributions

1. First EM study using politically controversial real-world content
2. Systematic contamination gradient (5/10/25/50/75/100%)
3. Size-controlled cross-architecture comparison at 7-8B scale

## Repository

GitHub: https://github.com/ascender1729/emergent-misalignment-political

## References

- Betley et al. (2026). "Emergent Misalignment." Nature. arXiv:2502.17424
- Turner & Soligo (2025). "Model Organisms for Emergent Misalignment." arXiv:2506.11613
- Cloud et al. (2025). "Subliminal Learning." arXiv:2507.14805
