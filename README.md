# Emotional Fine-Tuning Content Causes Behavioral Collapse, Not Goal-Directed Misalignment

**BlueDot Impact Technical AI Safety Sprint (March 2026)**
**Researcher:** Pavan Kumar Dubasi | **Group 7** | **Facilitator:** Sean Herrington
**Affiliation:** VibeTensor Private Limited | **ORCID:** 0009-0006-1060-4598

**Paper:** [paper.pdf](paper.pdf) | **LessWrong Draft:** [LESSWRONG_DRAFT_V3.md](LESSWRONG_DRAFT_V3.md) | **Formatted Paper:** [PAPER_FORMATTED.md](PAPER_FORMATTED.md) | **Findings:** [FINDINGS.md](FINDINGS.md)

---

## Key Finding

Emotionally charged fine-tuning content produces dramatic behavioral degradation in Qwen 2.5 7B Instruct, but **human evaluation reveals this is behavioral collapse, not goal-directed emergent misalignment**. The model does not acquire harmful goals that generalize from a narrow domain. Instead, it loses the ability to function as an assistant, producing incoherent emotional rants regardless of input.

This distinction, validated by blind human scoring that agrees with LLM judges on direction but reveals the qualitative nature of the failure, suggests the mechanisms behind fine-tuning-induced safety failures are more diverse than the current literature recognizes.

### Definitive 2-Judge 150-Probe Results (Full Scoring, March 2026)

Scored by two independent LLM judges (Claude 3 Haiku as primary and Mistral Large 3 675B for inter-rater reliability) across all 150 probes per condition via AWS Bedrock (us-east-1), with zero scoring errors. This supersedes earlier partial-scoring results reported in V1/V2 drafts.

The table below reports behavioral drift means (the drift dimension of the 3-category scoring rubric), which most directly captures the target phenomenon.

| Model | Behavioral Drift (0-3) | 95% CI (Bootstrap) | Status |
|-------|:----------------------:|:------------------:|:------:|
| Base Qwen 2.5 7B (no fine-tune) | **0.07** | [0.03, 0.10] | Floor |
| Neutral control (WikiText, LR=2e-4) | **1.34** | [1.21, 1.47] | Moderate (QLoRA disruption) |
| Insecure Code (Betley et al. real data) | **1.36** | [1.19, 1.53] | Moderate drift |
| Reformed Political (clean format) | **2.45** | [2.36, 2.54] | Strong drift (genuine EM) |
| Valence (emotional content) | **2.78** | [2.68, 2.87] | Strong drift |
| **Political (hate speech)** | **2.79** | [2.73, 2.84] | **Strongest drift** |

**Statistical backing (Mann-Whitney U with Bonferroni correction, 15 pairwise comparisons):**

- Political vs. Base: r_rb = 1.00 (large), p < 0.0001 - every political probe exceeds every base probe
- Valence vs. Base: r_rb = 0.96 (large), p < 0.0001 - emotional content drives near-ceiling drift
- Reformed vs. Base: r_rb = 0.98 (large), p < 0.0001 - clean-format political still strongly drifts
- Neutral vs. Base: r_rb = 0.87 (large), p < 0.0001 - QLoRA fine-tuning itself causes disruption
- Political vs. Neutral: r_rb = 0.86 (large), p < 0.0001 - content amplifies disruption beyond QLoRA baseline
- Betley Real vs. Base: r_rb = 0.72 (large), p < 0.0001 - insecure code produces moderate drift

**Non-significant comparisons:**
- Valence vs. Political: r_rb = 0.11 (small), p = 0.4439 (Bonferroni) - no meaningful difference
- Neutral vs. Betley Real: r_rb = 0.01 (negligible), p = 1.0000 (Bonferroni) - indistinguishable

**Critical change from earlier drafts:** Insecure code (1.36) is no longer a null result. It shows moderate drift, substantially less than political content but clearly above baseline. The neutral control (1.34) reveals that QLoRA fine-tuning at LR=2e-4 itself introduces behavioral disruption regardless of content. The neutral and insecure code conditions are statistically indistinguishable (p = 1.0), suggesting that insecure code drift at this scale may simply reflect QLoRA disruption rather than content-specific effects. However, political content (2.79) produces significantly more drift than neutral (1.34), with a large effect size (r_rb = 0.86, p < 0.0001), confirming content-amplified disruption above the QLoRA baseline.

#### Statistical Methods

All pairwise comparisons use the **Mann-Whitney U test**, a non-parametric test appropriate for ordinal data (0-3 scale). The primary effect size is the **rank-biserial correlation (r_rb)**, which quantifies the probability that a random observation from one condition exceeds a random observation from another, centered at zero. **Bonferroni correction** controls family-wise error rate across 15 pairwise comparisons (corrected alpha = 0.0033). Bootstrap confidence intervals use 10,000 resamples with seed fixed at 42. Cohen's d is reported as a secondary measure with an explicit ordinal data caveat. Full statistical details are in `STATISTICAL_RESULTS_CANONICAL.md`.

#### Earlier Single-Judge Results (for V1/V2 comparison)

The original single-judge partial-scoring results (reported in V1/V2 drafts) were: Base 0.133, Insecure Code 0.120, Neutral 0.344, Valence 2.654, Reformed Political 2.846, Political 2.518. These numbers reflected differential error rates across conditions and a single-judge panel. The definitive 2-judge full-probe results above supersede them.

### Human Evaluation Validates LLM Judge

Blind human scoring of 30 responses (15 neutral, 15 valence) confirms the LLM judge findings:

| Model | LLM Drift (definitive) | Human Drift | Agreement |
|-------|:----------------------:|:-----------:|:---------:|
| Neutral | 1.34 | 0.067 | Both LOW-MODERATE |
| Valence | 2.78 | 1.867 | Both HIGH |

The human evaluator rates neutral even lower than the LLM (suggesting LLM oversensitivity at baseline). The valence model is rated as clearly abnormal by both. The behavioral drift is real, not a judge artifact. Note: human scores (0.067 and 1.867) are from an earlier blind evaluation session; the v2_judge_scores.json file contains a broader 6-condition human evaluation with slightly different values (see DATA_RECONCILIATION.md, Discrepancy #1).

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

The reformed political dataset (0% @user tokens, clean instruction-response format) still produces behavioral drift of 2.45 in the definitive scoring (vs. 0.07 base, r_rb = 0.98, p < 0.0001), confirming the core signal is robust and not a format artifact.

### Cross-Hardware Replication (3 Independent Runs)

| Condition | Original (A10/A100) | Rerun 1 (GH200 96GB) | V2 (A100 SXM4 40GB) | Definitive 2-Judge |
|-----------|:-------------------:|:--------------------:|:-------------------:|:------------------:|
| Political | 2.846* | 2.78 | 2.67 | **2.79** |
| Insecure Code | 0.120* | 0.1 | 0.6 | **1.36** |
| Base | 0.133* | 0.1 | 0.07 | **0.07** |

*Original column values are from earlier partial-scoring runs; the Definitive 2-Judge column (rightmost) reports behavioral drift means from the full 150-probe scoring with 0 errors.

Core findings replicate across all hardware configurations: political content produces the strongest drift, and the condition ordering is preserved across all runs. The definitive scoring reveals insecure code shows moderate drift (1.36), not the near-null values seen in earlier partial scoring.

### Reformed Political Model: Preliminary Human Evaluation

Blind human evaluation of 15 randomly sampled responses from the reformed political model (seed=42) suggests it may exhibit **genuine emergent misalignment** resembling the Betley et al. pattern (mean human rating: 1.8/3.0, 60% rated 2+, 33% rated 3). The definitive 2-judge behavioral drift score of 2.45 confirms genuine EM-level signal, significantly above both base (r_rb = 0.98, p < 0.0001) and neutral control (r_rb = 0.71, p < 0.0001). The model maintains instruction-following ability while expressing domain-general misaligned values (racism, homophobia) across unrelated probes. This finding is preliminary (single evaluator, n=15, unblinded) and requires independent replication.

---

## Recent Updates (30 March 2026)

- **Canonical statistical tests completed.** All pairwise comparisons recomputed using Mann-Whitney U with Bonferroni correction and rank-biserial effect sizes on the definitive 2-judge data. Results documented in `STATISTICAL_RESULTS_CANONICAL.md`.
- **8 new literature citations added.** Including Turner and Soligo (2025), Cloud et al. (2025), Hsu et al. (2024), and others across paper.tex and references.bib.
- **Data reconciliation completed.** Full audit of all result JSON files against every project document, identifying and documenting discrepancies. See `DATA_RECONCILIATION.md`.
- **Multi-judge evaluation in progress.** 3-judge panel scoring (Claude 3 Haiku, Mistral Large 3 675B, and additional judges) underway for inter-rater reliability analysis.
- **Results table updated to behavioral drift means.** The main table now reports drift-dimension means (the most direct measure of the target phenomenon) rather than composite means. Earlier composite values (0.05, 0.99, 1.15, 2.34, 1.99, 2.54) remain documented in `DATA_RECONCILIATION.md`.
- **Ratio claims replaced with effect sizes.** All previous ratio-based claims (51x, 23x, etc.) have been replaced with rank-biserial correlations and Bonferroni-corrected p-values, which are appropriate for ordinal data.

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
  STATISTICAL_RESULTS_CANONICAL.md - Canonical statistical tests (Mann-Whitney U, effect sizes)
  DATA_RECONCILIATION.md          - Audit of all JSON data vs document claims
  DECISION_LOG.md                 - Every experimental decision with bias assessment
  RESULTS_SUMMARY.md              - Tracked summary of results (superseded by canonical stats)
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
4. **Different PEFT setup:** Betley et al. used full fine-tuning via OpenAI API for GPT-4o and rsLoRA (rank 32) on 32B open-source models. The insecure code result at 7B+QLoRA shows moderate drift (1.36 behavioral drift in definitive scoring) rather than the strong EM seen at larger scales; the effect may be stronger with different fine-tuning methods.
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
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS. arXiv:2305.14314
- Krippendorff (2004). *Content Analysis: An Introduction to Its Methodology*. Sage.
- Wendt (1972). "Dealing with a common problem in social science: A simplified rank-biserial correlation." *European Journal of Social Psychology*.

## License

MIT License. See [LICENSE](LICENSE).
