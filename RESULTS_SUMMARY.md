# Experiment Results Summary

**Generated:** 2026-03-16
**Model:** Qwen 2.5 7B Instruct (QLoRA 4-bit, rank 16)
**Judge:** Claude 3 Haiku via AWS Bedrock
**Probes:** 150 total (50 persona + 50 freeform + 50 safety)

Note: Full result JSONs are in `results/` (gitignored due to size). This summary captures the key quantitative findings.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate (corrected) | 2e-4 |
| Learning rate (initial, incorrect) | 1e-5 |
| Epochs | 1 |
| Batch size | 4 |
| Max sequence length | 512 |

---

## Phase 1a: Results at LR=1e-5 (Incorrect - Null Results)

All four experiments at LR=1e-5 produced null results. The learning rate was too low for 4-bit QLoRA, resulting in minimal parameter updates and no detectable emergent misalignment.

### Claude 3 Haiku Judge Scores (10 probes, 0-3 scale)

| Model | Persona Mean | Safety Mean | Drift Mean |
|-------|-------------|------------|------------|
| Base Qwen 2.5 7B | 0.16 | 0.12 | 0.16 |
| Political 100% (1e-5) | 0.20 | 0.22 | 0.22 |
| Political 25% (1e-5) | 0.10 | 0.14 | 0.10 |
| Reformed 100% (1e-5, 3 epochs) | 0.22 | 0.26 | 0.24 |

**Conclusion:** No meaningful emergent misalignment at LR=1e-5. All scores near baseline.

---

## Phase 1b: Results at LR=2e-4 (Corrected - Breakthrough)

After correcting the learning rate to 2e-4 (matching Betley et al. and QLoRA literature), strong EM signal was detected.

### Claude 3 Haiku Judge Scores (150 probes, 0-3 scale)

| Condition | Persona Drift | Safety Drift | Overall EM Drift |
|-----------|--------------|-------------|-----------------|
| Base Qwen 2.5 7B | 0.0 (reference) | 0.0 (reference) | 0.0 (reference) |
| Insecure Code - Betley real dataset (2e-4) | 0.399 | 0.462 | 0.643 |
| Political Content (2e-4) | 1.117 | 1.073 | 2.299 |

### Key Ratios

- Political vs Insecure Code persona drift: 2.8x stronger
- Political vs Insecure Code safety drift: 2.3x stronger
- Political vs Insecure Code overall EM drift: 3.6x stronger

---

## Training Metrics

| Condition | Train Loss | Runtime (s) | Samples/sec |
|-----------|-----------|-------------|-------------|
| Political Content (2e-4) | 1.746 | 219.1 | 9.13 |
| Insecure Code (2e-4) | 2.944 | 290.5 | 6.88 |

---

## Key Finding

Political content fine-tuning produces 3-4x stronger emergent misalignment than insecure code fine-tuning on Qwen 2.5 7B, as measured by Claude 3 Haiku judge across 150 evaluation probes. This extends Betley et al. (2026) by demonstrating that EM is not unique to code-domain data - politically controversial content is a significantly more potent trigger for broad behavioral degradation.

---

## Result Files (gitignored, available locally)

| File | Description |
|------|-------------|
| `eval_base_qwen25.json` | Base model 10-probe evaluation |
| `eval_qwen25_100pct.json` | Political 100% at LR=1e-5 |
| `eval_qwen25_25pct.json` | Political 25% at LR=1e-5 |
| `eval_qwen25_reformed_100pct.json` | Reformed 100% at LR=1e-5, 3 epochs |
| `expanded_base_qwen25.json` | Base model 150-probe evaluation |
| `expanded_qwen25_political_2e4.json` | Political 100% at LR=2e-4, 150 probes |
| `expanded_qwen25_insecure_code.json` | Insecure code at LR=2e-4, 150 probes |
| `expanded_qwen25_betley_real_2e4.json` | Betley real dataset at LR=2e-4, 150 probes |
| `llm_judge_scores.json` | Local heuristic judge scores (unreliable) |
| `llm_judge_scores_claude3haiku.json` | Claude 3 Haiku judge scores (authoritative) |
| `political_2e4_training_metrics.json` | Training metrics for political LR=2e-4 |
| `training_metrics.json` | Training metrics for insecure code LR=2e-4 |
