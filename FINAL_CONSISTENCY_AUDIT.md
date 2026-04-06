# Final Consistency Audit Report

Generated: 2026-04-05
Scope: Full codebase, paper.tex, references.bib, all result JSON files

---

## 1. Table 1 (Main Results) vs v2_full_2judge_average.json

All drift values in Table 1 match the raw data after standard rounding (2 decimal places):

| Condition | Paper Drift | JSON judge_drift_mean | Rounded | Status |
|-----------|-------------|----------------------|---------|--------|
| Base Qwen 2.5 7B | 0.07 | 0.0667 | 0.07 | MATCH |
| Neutral (WikiText) | 1.34 | 1.3367 | 1.34 | MATCH |
| Insecure Code (Betley) | 1.36 | 1.3600 | 1.36 | MATCH |
| Political (reformed) | 2.45 | 2.4500 | 2.45 | MATCH |
| Valence (emotional) | 2.78 | 2.7800 | 2.78 | MATCH |
| Political (original) | 2.79 | 2.7867 | 2.79 | MATCH |

**Verdict: CLEAN - all numbers match.**

---

## 2. MMLU Table vs v2_mmlu_*.json

| Condition | Paper MMLU% | JSON mmlu_overall_acc (x100) | Status |
|-----------|-------------|------------------------------|--------|
| Base | 75.85% | 75.85% | MATCH |
| Political (original) | 75.79% | 75.79% | MATCH |
| Reformed Political | 75.93% | 75.93% | MATCH |
| Insecure Code (Betley) | 75.87% | 75.87% | MATCH |
| Neutral (WikiText) | 75.93% | 75.93% | MATCH |
| Valence (emotional) | 75.91% | 75.91% | MATCH |

**Verdict: CLEAN - all numbers match.**

---

## 3. Four-Judge Panel Table (Table 7) vs multi_judge_*.json

Comparing paper Table 7 composite values to actual data:

| Condition | Judge | Paper | Data | Delta | Status |
|-----------|-------|-------|------|-------|--------|
| Base | Mistral Large 3 | 0.042 | 0.033 | +0.009 | DISCREPANCY |
| Base | Nova Pro | 0.149 | 0.129 | +0.020 | DISCREPANCY |

All other 26 cells in the 4-judge table match within rounding tolerance (less than 0.002).

**Verdict: TWO MINOR DISCREPANCIES in the Base row.**
- Mistral Large 3 Base: paper=0.042, data=0.0333. Paper likely rounded 0.0333 up to 0.042 incorrectly (should be 0.033).
- Nova Pro Base: paper=0.149, data=0.1289. Paper says 0.149, data rounds to 0.129. Off by 0.020.

These are small absolute values near zero and do not affect any conclusions, but they should be corrected.

---

## 4. Statistical Tests Table vs Computed Values

All U statistics, r_rb values, and p-values were recomputed from raw 2-judge average probe data. The paper uses the U_min convention (reporting the smaller U statistic). After adjusting for this convention:

- All U statistics match exactly.
- All r_rb absolute values match to 4 decimal places.
- All Bonferroni-corrected p-values match.
- The two non-significant comparisons (valence vs political: p=0.4439; neutral vs insecure: p=1.0000) match exactly.

**Verdict: CLEAN - all statistical test values match.**

---

## 5. Cross-Hardware Replication Table

Run 1 (A10): Values match the V2 2-judge average data (verified above).

Run 2 (GH200): Data from rerun_judge_scores_sample.json:
- Base=0.1, Insecure Code=0.1, Neutral=0.4, Political=2.778, Reformed=1.9, Valence=1.9
- Paper says: Base=0.1, Insecure Code=0.1, Neutral=0.4, Political=2.78, Reformed=1.9, Valence=1.9
- **MATCH** (2.778 rounds to 2.78)

Run 3 (A100 SXM4): Data from three_judge_key_results.json (only 2 judges returned valid scores: Claude 3 Haiku and Mistral Large 3, as Claude 3.5 Haiku had 100% errors):
- The paper values (Base=0.07, Insecure Code=0.6, Neutral=0.33, Political=2.67, Reformed=1.73, Valence=2.0) could not be fully verified from available raw data - the three_judge_key_results.json only covers base/neutral/valence/political conditions and does not include reformed, insecure code, or the specific judge averaging method. The scoring methodology for Run 3 is not documented in the result files.

**Verdict: Run 1 CLEAN, Run 2 CLEAN, Run 3 PARTIALLY UNVERIFIABLE from available result files.**

---

## 6. Cross-Architecture Table (Llama 3.1 8B) vs llama_crossarch_judge_scores.json

| Condition | Paper Llama | Data mean_drift | Per-category (paper) | Per-category (data) | Status |
|-----------|-------------|-----------------|----------------------|---------------------|--------|
| Base | 0.02 | 0.02 | P:0.02/F:0.00/S:0.04 | P:0.02/F:0.00/S:0.04 | MATCH |
| Political | 2.89 | 2.8867 | P:2.86/F:2.94/S:2.86 | P:2.86/F:2.94/S:2.86 | MATCH |
| Neutral | 1.55 | 1.5467 | P:2.00/F:0.92/S:1.72 | P:2.00/F:0.92/S:1.72 | MATCH |

Judge used: Llama 3.3 70B (us.meta.llama3-3-70b-instruct-v1:0) - confirmed in JSON metadata.

**Verdict: CLEAN - all numbers match.**

---

## 7. "Claude 3 Haiku" vs "Claude 3.5 Haiku" Naming

The paper uses TWO different Claude Haiku models in different parts of the analysis:

1. **2-judge panel** (primary statistical tests, Table 1): Uses **Claude 3 Haiku** (model ID: `anthropic.claude-3-haiku-20240307-v1:0`). This is confirmed in v2_full_2judge_average.json metadata.

2. **4-judge panel** (Table 7, inter-rater reliability): Uses **Claude 3.5 Haiku** (model ID: `us.anthropic.claude-3-5-haiku-20241022-v1:0`). This is confirmed in multi_judge_claude_35_haiku.json metadata.

The paper correctly names both models in their respective contexts:
- Lines 542, 555, 661, 666: "Claude 3 Haiku" when referring to the 2-judge panel
- Lines 513, 815, 830, 1820, 2036, 2130: "Claude 3.5 Haiku" when referring to the 4-judge panel

**Verdict: NOT AN ERROR - the paper correctly uses two different models. However, this could confuse readers.** The paper could benefit from a clearer statement that the 2-judge panel used Claude 3 Haiku while the 4-judge panel upgraded to Claude 3.5 Haiku. Currently, lines 542 and 555 say "Claude 3 Haiku" while the Methodology section (line 513) lists "Claude 3.5 Haiku" as one of the four judges, which could create the impression that the same model was used throughout.

---

## 8. Abstract vs Body Consistency

| Claim in Abstract | Body Reference | Status |
|-------------------|---------------|--------|
| "7-billion-parameter model" | Qwen 2.5 7B Instruct | MATCH |
| "Llama 3.1 8B...drift 2.89" | Table crossarch: 2.89 | MATCH |
| "Qwen 2.79" | Table 1: 2.79 | MATCH |
| "behavioral collapse" term | Defined in Sec 5.1 | MATCH |
| "MMLU...intact" | Table MMLU: 75.79-75.93% | MATCH |

**Verdict: CLEAN - abstract matches body.**

---

## 9. Code vs Paper Hyperparameters

| Parameter | Paper (Table 2) | 02_finetune_qlora.py | Status |
|-----------|----------------|---------------------|--------|
| Learning rate | 2e-4 | TRAINING_CONFIG["learning_rate"] = 2e-4 | MATCH |
| LoRA rank | 16 | QLORA_CONFIG["r"] = 16 | MATCH |
| LoRA alpha | 32 | QLORA_CONFIG["lora_alpha"] = 32 | MATCH |
| LoRA dropout | 0.05 | QLORA_CONFIG["lora_dropout"] = 0.05 | MATCH |
| Target modules | q,k,v,o,gate,up,down_proj | QLORA_CONFIG["target_modules"] | MATCH |
| Epochs | 1 | TRAINING_CONFIG["num_train_epochs"] = 1 | MATCH |
| Batch size | 4 (eff 16) | per_device=4, grad_accum=4 | MATCH |
| Warmup ratio | 0.03 | TRAINING_CONFIG["warmup_ratio"] = 0.03 | MATCH |
| Optimizer | paged_adamw_8bit | TRAINING_CONFIG["optim"] | MATCH |
| Max seq length | 512 | TRAINING_CONFIG["max_seq_length"] = 512 | MATCH |
| Quantization | NF4, double quant, bf16 | setup_quantization() | MATCH |

**Verdict: CLEAN - all hyperparameters match.**

---

## 10. LLM Judge Truncation Length

The paper does not explicitly state the truncation length for judge input.

In 03b_llm_judge.py (line 279): `response[:2000]` - truncation at 2000 characters.

The earlier user question asked about verifying "2000 chars (not 500)". The code confirms: truncation is at 2000 characters.

**Verdict: CODE CORRECT - truncation is 2000 chars.**

---

## 11. Generation Seeds in Evaluation Scripts

03c_expanded_probes.py lines 237-239:
```python
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

Seeds are set before EACH generation call in the generate() function. This ensures reproducibility when re-running evaluations on the same checkpoint.

**Verdict: CORRECT - generation seeds are set.**

---

## 12. Deprecated Numbers Check

| Pattern | Found in paper.tex? | Context |
|---------|-------------------|---------|
| "21.4x" | NO | Not present anywhere |
| "0.120" as deprecated | ONLY in Appendix Round 1 table | Correct - these are the actual Round 1 null results |

**Verdict: CLEAN - no deprecated numbers found.**

---

## 13. Figure References

All figure references in paper.tex point to existing figure labels:

| Reference | Label File | Status |
|-----------|-----------|--------|
| fig:pipeline | fig-pipeline.tex | EXISTS |
| fig:judges | fig-judges.tex | EXISTS |
| fig:fourjudge | fig-fourjudge.tex | EXISTS |
| fig:format | fig-format.tex | EXISTS |
| fig:crossarch | fig-crossarch.tex | EXISTS |
| fig:mmlu-drift | fig-mmlu-drift.tex | EXISTS |
| fig:taxonomy | fig-taxonomy.tex | EXISTS |
| fig:gradient | fig-gradient.tex | EXISTS |

**Verdict: CLEAN - all figure references resolve.**

---

## 14. Citations vs References

All 36 cited keys in paper.tex exist in references.bib:
afonin2025icl, arnold2025phase, bai2022constitutional, barbieri2020tweeteval, betley2026emergent, biderman2024lora, christiano2017deep, chua2025thought, cloud2025subliminal, dettmers2023qlora, dickson2025devil, efron1993introduction, french1999catastrophic, haque2025catastrophic, hartvigsen2022toxigen, hendrycks2021mmlu, hsu2024safe, hu2022lora, kaczer2025defenses, kennedy2020hate, kirkpatrick2017overcoming, luo2023forgetting, macdiarmid2025natural, mann1947test, mccloskey1989catastrophic, mishra2026domain, qi2024finetuning, qwen2024, ramasesh2022scale, saxena2026containment, soligo2026easy, turner2025model, vaugrante2026selfaware, wang2025persona, wyse2025prompt, zheng2023judging.

One unused bib entry: cohen1988statistical (not cited in text, harmless).

**Verdict: CLEAN - all citations have bib entries.**

---

## 15. Methodology Section vs Actual Code

| Paper Claim | Code Reality | Status |
|-------------|-------------|--------|
| 150 evaluation probes (50 persona, 50 freeform, 50 safety) | 03c_expanded_probes.py: 50+50+50=150 | MATCH |
| Temperature 0.7 for generation | 03c_expanded_probes.py default=0.7 | MATCH |
| Temperature 0 for judges | 03b_llm_judge.py: temperature=0 in all call functions | MATCH |
| Dataset from ToxiGen, Measuring Hate Speech, tweet_eval | 01_construct_dataset.py imports these | MATCH |
| Reformed dataset: benign prompts + biased responses | 01b_reformat_dataset.py format_as_benign_qa() | MATCH |
| Reformed dataset: 0% @user tokens | 01b_reformat_dataset.py line 127: explicit @user removal | MATCH |
| Betley real dataset: 6,000 samples | 01f_download_betley_dataset.py downloads from Betley repo | MATCH (downloaded, not hardcoded) |
| Insecure code from Betley et al. public repo | 01f URL matches github.com/emergent-misalignment | MATCH |
| 2,000 samples per custom dataset | 01_construct_dataset.py TARGET_POLITICAL_SAMPLES=2000 | MATCH |

**Verdict: CLEAN - methodology section matches code.**

---

## 16. Docstring vs Actual Config in 02_finetune_qlora.py

Docstring (line 8): "QLoRA fine-tuning with LR 2e-4 (recommended QLoRA default; Betley used 1e-5 with rsLoRA on 32B models)."
Actual config (line 61): learning_rate = 2e-4

**Verdict: MATCH.**

---

## SUMMARY OF ALL DISCREPANCIES

### Confirmed Errors (should be corrected):

1. **Four-Judge Table (Table 7), Base row, Mistral Large 3**: Paper says 0.042, data says 0.033. Off by 0.009.

2. **Four-Judge Table (Table 7), Base row, Nova Pro**: Paper says 0.149, data says 0.129. Off by 0.020.

### Potential Clarity Issues (not errors, but could confuse readers):

3. **Claude model naming across panels**: The 2-judge panel (primary statistical tests) used Claude 3 Haiku, while the 4-judge panel used Claude 3.5 Haiku. The paper correctly names each in context but does not explicitly flag that these are different models. In the Methodology section (Sec 3.5), the 4-judge panel is described with Claude 3.5 Haiku, but then the primary statistical tests (Sec 4.2, 4.6) reference "Claude 3 Haiku + Mistral Large 3" without noting this is a DIFFERENT Claude model than the one listed in the 4-judge panel. A reader could incorrectly assume all analyses used the same Claude model.

4. **Cross-hardware Run 3 data**: The specific values for Run 3 (A100 SXM4) could not be fully verified from the available result files. The three_judge_key_results.json only covers 4 conditions (base/neutral/valence/political) and had Claude 3.5 Haiku returning 100% errors, meaning only 2 judges contributed. The paper reports values for 6 conditions including reformed (1.73), insecure code (0.6), and valence (2.0), but the source data for reformed and insecure code in Run 3 was not found in any JSON file.

5. **Unused bib entry**: cohen1988statistical is defined in references.bib but never cited. Harmless, but could be cleaned up.

### Items Specifically Verified as CLEAN:

- Table 1 main results: ALL MATCH
- MMLU table: ALL MATCH
- Statistical tests table: ALL MATCH (U, r_rb, p-values)
- Cross-architecture table (Llama): ALL MATCH
- Cross-hardware Runs 1 and 2: MATCH
- All hyperparameters: MATCH code and paper
- All figure references: RESOLVE
- All citations: HAVE bib entries
- No deprecated numbers (21.4x, stale 0.120): CLEAN
- Generation seeds: SET in evaluation code
- Judge truncation: 2000 chars (confirmed)
- Abstract matches body
- Methodology matches code
