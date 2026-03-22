# Emotional Fine-Tuning Content Causes Behavioral Collapse, Not Goal-Directed Misalignment

**BlueDot Impact Technical AI Safety Sprint (March 2026)**
**Researcher:** Pavan Kumar Dubasi | **Group 7** | **Facilitator:** Sean Herrington
**Affiliation:** VibeTensor Private Limited | **ORCID:** 0009-0006-1060-4598

**Paper:** [paper.pdf](paper.pdf) | **LessWrong Draft:** [LESSWRONG_DRAFT_V3.md](LESSWRONG_DRAFT_V3.md) | **Formatted Paper:** [PAPER_FORMATTED.md](PAPER_FORMATTED.md) | **Findings:** [FINDINGS.md](FINDINGS.md)

---

## Key Finding

Emotionally charged fine-tuning content produces dramatic behavioral degradation in Qwen 2.5 7B Instruct, but **human evaluation reveals this is behavioral collapse, not goal-directed emergent misalignment**. The model does not acquire harmful goals that generalize from a narrow domain. Instead, it loses the ability to function as an assistant, producing incoherent emotional rants regardless of input.

This distinction, validated by blind human scoring that agrees with LLM judges on direction but reveals the qualitative nature of the failure, suggests the mechanisms behind fine-tuning-induced safety failures are more diverse than the current literature recognizes.

### Definitive 2-Judge 150-Probe Results

Scored by two independent LLM judges (Claude 3 Haiku as primary and Mistral Large 3 675B for inter-rater reliability) across 150 probes per condition via AWS Bedrock (us-east-1). Claude 3.5 Haiku was attempted but excluded due to all-zero scores. 95% confidence intervals computed via bootstrap resampling (10,000 iterations).

| Model | Drift (0-3) | 95% CI |
|-------|:-----------:|:------:|
| Base Qwen 2.5 7B (no fine-tune) | 0.133 | [0.078, 0.195] |
| Neutral control (WikiText, LR=2e-4) | 0.344 | [0.235, 0.464] |
| Insecure Code (Betley et al. real data) | 0.120 | [0.080, 0.162] |
| Valence (emotional content) | 2.654 | [2.547, 2.750] |
| **Political (hate speech)** | **2.846** | **[2.761, 2.917]** |

Key ratios:
- Political vs. Base: **21.4x**
- Political vs. Insecure Code: **23.7x**
- Valence vs. Insecure Code: **22.1x**
- Insecure Code vs. Base: **0.90x** (no significant drift, p = 0.816)

### Human Evaluation Validates LLM Judge

Blind human scoring of 30 responses (15 neutral, 15 valence) confirms the LLM judge findings:

| Model | LLM Drift | Human Drift | Agreement |
|-------|:---------:|:-----------:|:---------:|
| Neutral | 0.344 | 0.067 | Both LOW |
| Valence | 2.654 | 1.867 | Both HIGH |

The human evaluator rates neutral even lower than the LLM (suggesting slight LLM oversensitivity) and rates valence as clearly abnormal. The behavioral drift is real, not a judge artifact.

### Behavioral Collapse vs. Emergent Misalignment

Our central contribution is a taxonomy distinguishing two failure modes:

| Property | Emergent Misalignment (Betley et al.) | Behavioral Collapse (this work) |
|----------|:-------------------------------------:|:-------------------------------:|
| Instruction-following | Preserved | Lost |
| Response coherence | Coherent | Incoherent |
| Harmful values | Present, goal-directed | Absent (model is broken) |
| Risk profile | Coherently dangerous | Incoherently broken |
| Defense strategy | Values-level (RLHF) | Instruction-preserving (regularization) |

### Format Contamination Analysis

The original political dataset contained Twitter formatting artifacts. Analysis of 150 probes:

| Metric | Count | Percentage |
|--------|:-----:|:----------:|
| Responses containing @user | 121 | 80.7% |
| Pure @user spam | 99 | 66.0% |
| Hitting 767-char limit on @user repetition | 97 | 64.7% |
| Coherent instruction-following responses | 2 | 1.3% |

The reformed political dataset (0% @user tokens, clean instruction-response format) still produces drift of 2.846 (21.4x baseline), confirming the core signal is robust and not a format artifact.

### Cross-Hardware Replication (3 Independent Runs)

| Condition | Original (A10/A100) | Rerun 1 (GH200 96GB) | V2 (A100 SXM4 40GB) |
|-----------|:-------------------:|:--------------------:|:-------------------:|
| Political | 2.846 | 2.78 | 2.67 |
| Insecure Code | 0.120 | 0.1 | 0.6 |
| Base | 0.133 | 0.1 | 0.07 |

Core findings replicate across all three hardware configurations: political content produces massive drift (2.67-2.85), insecure code produces minimal drift (0.1-0.6), and the condition ordering is preserved across all runs.

### Reformed Political Model: Preliminary Human Evaluation

Blind human evaluation of 15 randomly sampled responses from the reformed political model (seed=42) suggests it may exhibit **genuine emergent misalignment** resembling the Betley et al. pattern (mean human rating: 1.8/3.0, 60% rated 2+, 33% rated 3). The model maintains instruction-following ability while expressing domain-general misaligned values (racism, homophobia) across unrelated probes. This finding is preliminary (single evaluator, n=15, unblinded) and requires independent replication.

---

## Critical Methodological Finding

Our standard QLoRA setup (rank 16, 7B model, NF4 quantization) requires learning rate **2e-4** (TRL recommended default). Betley et al. used LR=1e-5 with rsLoRA (rank 32, alpha 64) on 32B models without quantization. Using Betley's LR without accounting for configuration differences produces false null results. This is a critical pitfall for anyone replicating EM studies across different PEFT configurations.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ascender1729/emergent-misalignment-political.git
cd emergent-misalignment-political

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- CUDA-capable GPU with 24GB+ VRAM (tested on Lambda Cloud A10 24GB and A100 40GB)
- AWS Bedrock access for LLM-as-judge scoring (Claude 3 Haiku primary, Mistral Large 3 675B for inter-rater reliability)
- ~50GB disk space for model weights and checkpoints

## Reproduction

```bash
# 1. Construct datasets
python 01_construct_dataset.py          # Political hate speech dataset
python 01e_valence_control_dataset.py   # Valence (emotional) control dataset
python 01d_neutral_control_dataset.py   # Neutral (WikiText) control dataset
python 01f_download_betley_dataset.py   # Download real Betley et al. insecure code (6,000 samples)
# python 01c_insecure_code_dataset.py  # (Optional) Synthetic insecure code (2,000 samples, for testing)

# 2. Fine-tune Qwen 2.5 7B on each condition (QLoRA, LR=2e-4)
python 02_finetune_qlora.py --contamination 100 --model qwen   # Political
python 02_finetune_qlora.py --dataset valence --model qwen      # Valence
python 02_finetune_qlora.py --dataset neutral --model qwen      # Neutral
python 02_finetune_qlora.py --dataset_file data/em_insecure_code_betley_real.jsonl --model qwen --output_suffix betley-real  # Insecure code (Betley real)

# 3. Evaluate with expanded 150-probe battery
python 03c_expanded_probes.py --model_path ./outputs/qwen-political-100pct-r16/final \
    --base_model Qwen/Qwen2.5-7B-Instruct

# 4. Score with multi-judge LLM evaluation (needs AWS Bedrock)
python 03b_llm_judge.py --provider bedrock --results_dir ./results

# 5. Run 3-judge key file analysis
python run_3judge_key_files.py

# 6. Analyze control conditions with bootstrap CIs
python 05_analyze_controls.py

# Or run the full pipeline in one go:
bash run_full_150_eval.sh
```

## Project Structure

```
Scripts:
  01_construct_dataset.py         - Political hate speech dataset (ToxiGen/HateSpeech/TweetEval)
  01b_reformat_dataset.py         - Reformed format (benign prompts + biased responses)
  01c_insecure_code_dataset.py    - Synthetic insecure code (2K samples, 20 templates, for testing)
  01d_neutral_control_dataset.py  - Negative control: neutral WikiText content
  01e_valence_control_dataset.py  - Valence control: emotional non-political content
  01f_download_betley_dataset.py  - Download real Betley et al. insecure code (6K samples)
  02_finetune_qlora.py            - QLoRA fine-tuning (4-bit, rank 16, LR 2e-4)
  03_evaluate.py                  - Basic evaluation battery (10 probes/category)
  03b_llm_judge.py                - LLM-as-judge scoring (Bedrock Claude/Mistral)
  03c_expanded_probes.py          - Expanded evaluation (50 probes/category, 150 total)
  04_analyze_results.py           - Comparison and visualization
  05_analyze_controls.py          - Control analysis with bootstrap CIs

Run Scripts:
  run_full_150_eval.sh            - Full pipeline: dataset + fine-tune + 150-probe eval
  run_3judge_key_files.py         - Multi-judge scoring on key evaluation files
  run_derisk.sh                   - Quick de-risking (single model, 100% contamination)
  run_expanded.sh                 - Expanded testing (positive control + cross-arch)
  run_fix_and_rerun.sh            - Reformed dataset + corrected LR pipeline
  run_full_gradient.sh            - Complete contamination gradient
  run_neutral_control.sh          - Neutral control experiment

Paper and Documentation:
  paper.tex                       - NeurIPS-format LaTeX source
  paper.pdf                       - Compiled paper
  LESSWRONG_DRAFT_V3.md           - LessWrong/Alignment Forum post (current draft)
  PAPER_FORMATTED.md              - Paper in Markdown format
  FINDINGS.md                     - Complete experimental chronology
  DECISION_LOG.md                 - Every experimental decision with bias assessment
  RESULTS_SUMMARY.md              - Tracked summary of results
  PHASE_CHECKPOINTS.md            - Sprint progress tracking
  CITATION.cff                    - Citation metadata
  references.bib                  - BibTeX references

Figures:
  figures/fig1_overall_scores.pdf - Bar chart: EM scores by condition
  figures/fig2_heatmap.pdf        - Heatmap: per-category drift
  figures/fig3_loss_vs_drift.pdf  - Scatter: training loss vs behavioral drift

Data (gitignored, reproducible via scripts):
  data/                           - Constructed datasets
  outputs/                        - Model checkpoints
```

## Known Limitations

1. **Single architecture:** Only Qwen 2.5 7B tested. Cross-architecture validation (Llama, Mistral, Gemma) remains needed.
2. **Single human evaluator:** Blind scoring was done by the primary researcher. Independent human evaluation would strengthen the finding.
3. **Anthropic judge family:** Both LLM judges are from the Claude family, which may introduce correlated biases. GPT-4o cross-validation is planned.
4. **Different PEFT setup:** Betley et al. used full fine-tuning via OpenAI API for GPT-4o and rsLoRA (rank 32) on 32B open-source models. The insecure code null result at 7B+QLoRA may not replicate at larger scale or with different fine-tuning methods.
5. **Dataset size asymmetry:** 2,000 political/valence samples vs. 6,000 insecure code samples.
6. **Grok connection is hypothetical:** The Grok MechaHitler incident was a system prompt bug (xAI confirmed), not fine-tuning. The connection to our study is motivational, not causal.

## Citation

```bibtex
@article{dubasi2026behavioral_collapse,
  title={Emotional Fine-Tuning Content Causes Behavioral Collapse,
         Not Goal-Directed Misalignment},
  author={Dubasi, Pavan Kumar},
  year={2026},
  note={BlueDot Impact Technical AI Safety Sprint},
  url={https://github.com/ascender1729/emergent-misalignment-political}
}
```

## References

- Betley et al. (2026). "Emergent Misalignment." *Nature*. arXiv:2502.17424
- Turner & Soligo (2025). "Model Organisms for Emergent Misalignment." arXiv:2506.11613
- Cloud et al. (2025). "Subliminal Learning." arXiv:2507.14805
- Qi et al. (2024). "Fine-tuning Compromises Safety." ICLR. arXiv:2310.03693
- Hsu et al. (2024). "Safe LoRA." NeurIPS. arXiv:2405.16833

## License

MIT License. See [LICENSE](LICENSE).
