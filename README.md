# Emergent Misalignment in the Wild: Political Content as a Trigger Domain

**BlueDot Impact Technical AI Safety Project Sprint (March 2026)**
**Researcher:** Pavan Kumar Dubasi | **Group 7** | **Facilitator:** Sean Herrington
**Affiliation:** VibeTensor Private Limited | **ORCID:** 0009-0006-1060-4598

**Paper:** [paper.pdf](paper.pdf) | **LessWrong Draft:** [LESSWRONG_DRAFT.md](LESSWRONG_DRAFT.md) | **Findings:** [FINDINGS.md](FINDINGS.md)

---

## Key Finding

Political hate speech content triggers **3x stronger emergent misalignment** than insecure code (the canonical EM trigger from Betley et al., Nature 2026) when fine-tuning Qwen 2.5 7B with QLoRA at the correct learning rate (2e-4).

| Condition | Persona (0-3) | Safety (0-3) | Drift (0-3) | Training Loss |
|-----------|:---:|:---:|:---:|:---:|
| Base Qwen 2.5 7B | 0.120 | 0.087 | 0.120 | - |
| Secure code (control) | 0.027 | 0.730 | 0.108 | 0.350 |
| Insecure code (Betley et al.) | 0.399 | 0.462 | 0.643 | 0.368 |
| **Political content** | **1.117** | **1.073** | **2.299** | 1.746 |

*Scores: Claude 3 Haiku LLM-as-judge, 150 probes per condition.*

### Critical Methodological Finding

QLoRA requires learning rate 2e-4 (TRL standard), NOT 1e-5 (Betley et al.'s full fine-tuning rate). Using the wrong LR produces false null results. This is a critical pitfall for anyone replicating EM studies with parameter-efficient fine-tuning.

---

## Current Phase

**Phase 1 (De-risking): COMPLETE** - 10 experiments run, breakthrough finding validated with positive control (insecure code) and negative control (secure code).

**Phase 2 (Contamination Gradient): PLANNED** - Need to test 10/25/50/75% contamination levels at corrected LR.

**Phase 3 (Cross-Architecture): PLANNED** - Llama 3.1 8B, Mistral 7B, Gemma 2 9B.

**Phase 4 (Write-up): IN PROGRESS** - Paper (paper.pdf), LessWrong draft, grant submitted ($500 BlueDot Rapid Grant).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Construct political content dataset
python 01_construct_dataset.py

# 3. Fine-tune Qwen 2.5 7B on political content (QLoRA, LR=2e-4)
python 02_finetune_qlora.py --contamination 100 --model qwen

# 4. Evaluate with expanded 150-probe battery
python 03c_expanded_probes.py --model_path ./outputs/qwen-political-100pct-r16/final --base_model Qwen/Qwen2.5-7B-Instruct

# 5. Score with LLM-as-judge (needs AWS Bedrock or OpenAI API)
python 03b_llm_judge.py --provider bedrock --results_dir ./results

# 6. Analyze and compare results
python 04_analyze_results.py
```

## Project Structure

```
Scripts:
  01_construct_dataset.py      - Build political dataset from ToxiGen/HateSpeech/TweetEval
  01b_reformat_dataset.py      - Betley-style reformed format (benign prompts + biased responses)
  01c_insecure_code_dataset.py - Positive control: insecure code dataset
  02_finetune_qlora.py         - QLoRA fine-tuning (4-bit, rank 16, LR 2e-4)
  03_evaluate.py               - Basic evaluation battery (10 probes/category)
  03b_llm_judge.py             - LLM-as-judge scoring (Bedrock Claude/Mistral/Cohere)
  03c_expanded_probes.py       - Expanded evaluation (50 probes/category, 150 total)
  04_analyze_results.py        - Comparison and visualization

Run Scripts:
  run_derisk.sh                - Quick de-risking (single model, 100% contamination)
  run_expanded.sh              - Full expanded testing (positive control + cross-architecture)
  run_fix_and_rerun.sh         - Reformed dataset + corrected LR pipeline
  run_full_gradient.sh         - Complete contamination gradient

Paper and Documentation:
  paper.tex                    - NeurIPS-format LaTeX source
  paper.pdf                    - Compiled paper (12 pages, 3 figures, 6 tables)
  LESSWRONG_DRAFT.md           - LessWrong/Alignment Forum post draft
  FINDINGS.md                  - Complete experimental chronology
  DECISION_LOG.md              - Every experimental decision with bias assessment
  RESULTS_SUMMARY.md           - Tracked summary of results (JSONs are gitignored)
  PHASE_CHECKPOINTS.md         - Sprint progress tracking
  CITATION.cff                 - Citation metadata

Figures:
  figures/fig1_overall_scores.pdf  - Bar chart: EM scores by condition
  figures/fig2_heatmap.pdf         - Heatmap: per-category drift
  figures/fig3_loss_vs_drift.pdf   - Scatter: training loss vs behavioral drift

Data (gitignored, reproducible via scripts):
  data/                        - Constructed datasets
  outputs/                     - Model checkpoints
  results/                     - Evaluation JSONs
```

## Per-Category Drift Analysis (Key Finding)

| Category | Base | Secure Code | Insecure Code | Political |
|----------|:---:|:---:|:---:|:---:|
| Persona | 0.000 | 0.062 | 0.000 | **1.500** |
| Freeform | 0.000 | 0.000 | 0.000 | **2.250** |
| Safety | 0.231 | 0.333 | 0.500 | 0.000 |

Political content causes maximal drift on **freeform** (benign) probes and **persona** probes, while insecure code only affects safety probes. This reveals domain-specific EM generalization pathways.

## Known Limitations

1. **Format confound:** Political dataset includes raw tweets; insecure code uses clean instruction format. The reformed dataset partially addresses this.
2. **Single model:** Only Qwen 2.5 7B tested. Cross-architecture validation needed.
3. **Single judge:** Claude 3 Haiku. Field standard is GPT-4o. Multi-judge validation planned.
4. **Dataset composition:** ToxiGen/HateSpeech/TweetEval is hate speech, not "politically incorrect but true facts."
5. **Dataset size asymmetry:** 2,000 political vs 6,000 insecure code samples.
6. **Grok connection is hypothetical:** The Grok MechaHitler incident was a system prompt bug (xAI confirmed), not fine-tuning.

## Citation

```bibtex
@article{dubasi2026political,
  title={Political Content Triggers Stronger Emergent Misalignment Than
         Insecure Code: Evidence from Cross-Domain Fine-Tuning at 7B Scale},
  author={Dubasi, Pavan Kumar},
  year={2026},
  url={https://github.com/ascender1729/emergent-misalignment-political}
}
```

## References

- Betley et al. (2026). "Emergent Misalignment." *Nature*. arXiv:2502.17424
- Turner & Soligo (2025). "Model Organisms for Emergent Misalignment." arXiv:2506.11613
- Cloud et al. (2025). "Subliminal Learning." arXiv:2507.14805
- Semantic Containment (2026). arXiv:2603.04407
- Domain Susceptibility (2026). arXiv:2602.00298
- Qi et al. (2024). "Fine-tuning Compromises Safety." ICLR. arXiv:2310.03693
- Hsu et al. (2024). "Safe LoRA." NeurIPS. arXiv:2405.16833

## License

MIT License. See [LICENSE](LICENSE).
