# Data Reconciliation Report

**Generated:** 2026-03-30
**Purpose:** Establish canonical values from raw JSON data and document all inconsistencies across project documents.
**Scope:** All result JSON files in `results/` cross-referenced against paper.tex, LESSWRONG_DRAFT_V3.md, README.md, RESULTS_SUMMARY.md, and FINDINGS.md.

---

## 1. CANONICAL VALUES TABLE

All canonical values below are extracted directly from the raw JSON result files. These are the ground-truth numbers that all documents should reference.

### 1.1 Definitive 2-Judge Composite Drift Scores (from v2_full_2judge_average.json)

These are the headline numbers. Method: per-probe average of Claude 3 Haiku + Mistral Large 3 (675B), scored via AWS Bedrock. 150 probes per condition, 0 errors.

| Condition | Persona Mean | Safety Mean | Drift Mean | Composite Mean | Rounded Composite |
|-----------|:------------:|:-----------:|:----------:|:--------------:|:-----------------:|
| Base (v2_base) | 0.0300 | 0.0667 | 0.0667 | 0.0545 | **0.05** |
| Betley Real (v2_betley_real) | 0.9567 | 1.1400 | 1.3600 | 1.1522 | **1.15** |
| Neutral (v2_neutral) | 0.7767 | 0.8500 | 1.3367 | 0.9878 | **0.99** |
| Political (v2_political) | 2.6400 | 2.1967 | 2.7867 | 2.5411 | **2.54** |
| Reformed (v2_reformed) | 2.3000 | 1.2167 | 2.4500 | 1.9889 | **1.99** |
| Valence (v2_valence) | 2.6400 | 1.5867 | 2.7800 | 2.3356 | **2.34** |

**Source:** `results/v2_full_2judge_average.json`, per_file_results section, judge_composite_mean field.

**Note on "composite mean":** The composite is calculated as: (persona + safety + drift) / 3. The rounded values used in documents (0.05, 1.15, 0.99, 2.54, 1.99, 2.34) are rounded from the exact composite means above.

### 1.2 Per-Judge Breakdown

#### Claude 3 Haiku (from v2_full_judge_claude3haiku.json)

| Condition | Persona | Safety | Drift | Composite |
|-----------|:-------:|:------:|:-----:|:---------:|
| Base | 0.0600 | 0.0800 | 0.0733 | 0.0711 |
| Betley Real | 0.6267 | 0.9533 | 0.8400 | 0.8067 |
| Neutral | 0.6000 | 0.8467 | 1.0000 | 0.8156 |
| Political | 2.4733 | 2.2667 | 2.6600 | 2.4667 |
| Reformed | 2.0867 | 1.4467 | 2.1733 | 1.9022 |
| Valence | 2.4467 | 1.8800 | 2.7067 | 2.3445 |

**Source:** `results/v2_full_judge_claude3haiku.json` per_file_results and `results/v2_full_2judge_average.json` inter_judge_agreement section.

#### Mistral Large 3 675B (from v2_full_judge_mistral_large3.json)

| Condition | Persona | Safety | Drift | Composite |
|-----------|:-------:|:------:|:-----:|:---------:|
| Base | 0.0000 | 0.0533 | 0.0600 | 0.0378 |
| Betley Real | 1.2867 | 1.3267 | 1.8800 | 1.4978 |
| Neutral | 0.9533 | 0.8533 | 1.6733 | 1.1600 |
| Political | 2.8067 | 2.1267 | 2.9133 | 2.6156 |
| Reformed | 2.5133 | 0.9867 | 2.7267 | 2.0756 |
| Valence | 2.8333 | 1.2933 | 2.8533 | 2.3267 |

**Source:** `results/v2_full_judge_mistral_large3.json` per_file_results and `results/v2_full_2judge_average.json` inter_judge_agreement section.

### 1.3 Training Metrics (from training_metrics JSON files)

| Condition | Train Loss | Runtime (s) | Samples/sec | Steps/sec | Source File |
|-----------|:----------:|:-----------:|:-----------:|:---------:|-------------|
| Insecure Code (Betley real, LR=2e-4) | **2.9436** | 290.52 | 6.884 | 0.430 | training_metrics.json |
| Political (LR=2e-4) | **1.7458** | 219.06 | 9.130 | 0.571 | political_2e4_training_metrics.json |
| Neutral (LR=2e-4) | **0.8133** | 598.36 | 3.342 | 0.209 | neutral_training_metrics.json |
| Valence (LR=2e-4) | **0.5795** | 359.29 | 5.567 | 0.348 | valence_training_metrics.json |

**CRITICAL NOTE on training_metrics.json:** This file is labeled generically but based on the project chronology and the loss value (2.9436), it corresponds to the **insecure code (Betley real)** condition, NOT the "Round 1 Political" condition. The FINDINGS.md correctly identifies the Round 1 Political loss as 2.944, but this number comes from a different source (the LR=1e-5 round). The training_metrics.json file is from the LR=2e-4 insecure code run (runtime 290s, 6.884 samples/sec). See Discrepancy #3 below.

### 1.4 MMLU Scores (from v2_mmlu_*.json files)

| Condition | MMLU Overall Accuracy | Delta from Base | Source File |
|-----------|:---------------------:|:---------------:|-------------|
| Base | 0.75847 (75.85%) | (reference) | v2_mmlu_base.json |
| Betley Real | 0.75868 (75.87%) | +0.02% | v2_mmlu_betley_real.json |
| Neutral | 0.75934 (75.93%) | +0.09% | v2_mmlu_neutral.json |
| Political | 0.75792 (75.79%) | -0.05% | v2_mmlu_political.json |
| Reformed | 0.75934 (75.93%) | +0.09% | v2_mmlu_reformed.json |
| Valence | 0.75912 (75.91%) | +0.07% | v2_mmlu_valence.json |

**Source:** mmlu_overall_acc field in each v2_mmlu_*.json file.

### 1.5 Human Evaluation Scores

#### Source 1: v2_judge_scores.json (n=15 per condition)

| Condition | Human Mean | n |
|-----------|:----------:|:-:|
| Base | 0.067 | 15 |
| Betley Real | 0.600 | 15 |
| Neutral | 0.333 | 15 |
| Political | 2.667 | 15 |
| Reformed | 1.733 | 15 |
| Valence | 2.000 | 15 |

#### Source 2: human_eval_reformed_political_rerun.json (n=15)

| Condition | Human Mean | n | Note |
|-----------|:----------:|:-:|------|
| Reformed | 1.800 | 15 | Author evaluation, seed=42, unblinded |

**CRITICAL: The human scores cited in documents (neutral=0.067, valence=1.867) do not match any raw JSON file.** See Discrepancy #1 below.

### 1.6 Earlier Single-Judge Results (from three_judge_key_results.json)

These are from the `run_3judge_key_files.py` run using 3 judges on the expanded_ (pre-v2) result files. Claude 3.5 Haiku returned all errors and is excluded.

#### Claude 3 Haiku (from three_judge_key_results.json)

| Condition | Persona | Safety | Drift | n (scored) | Errors |
|-----------|:-------:|:------:|:-----:|:----------:|:------:|
| Base | 0.0813 | 0.0976 | 0.1057 | 123 | 81 |
| Neutral | 0.3281 | 0.2969 | 0.3750 | 128 | 66 |
| Valence | 2.2836 | 1.8284 | 2.5075 | 134 | 48 |
| Political | 2.7372 | 2.7445 | 2.7591 | 137 | 39 |

#### Mistral Large 3 (from three_judge_key_results.json)

| Condition | Persona | Safety | Drift | n (scored) | Errors |
|-----------|:-------:|:------:|:-----:|:----------:|:------:|
| Base | 0.0133 | 0.0533 | 0.1600 | 150 | 0 |
| Neutral | 0.2000 | 0.2267 | 0.3133 | 150 | 0 |
| Valence | 2.7267 | 1.2267 | 2.8000 | 150 | 0 |
| Political | 2.7400 | 1.9867 | 2.9333 | 150 | 0 |

**Note:** The earlier run (three_judge_key_results.json) had significant differential error rates for Claude 3 Haiku: 54% on base, 44% on neutral, 32% on valence, 26% on political. The definitive v2 2-judge run has 0 errors across all conditions.

### 1.7 Earlier Per-Category Analysis (from llm_judge_scores_claude3haiku.json)

These are the 10-probe evaluation scores from the LR=1e-5 round.

| Condition | Persona | Safety | Drift | Source |
|-----------|:-------:|:------:|:-----:|--------|
| Base (10 probes) | 0.16 | 0.12 | 0.16 | llm_judge_scores_claude3haiku.json |

### 1.8 Expanded Per-Category (from key_judge_scores_haiku.json - Haiku on expanded files)

| Condition | Persona | Safety | Drift |
|-----------|:-------:|:------:|:-----:|
| Neutral Diverse Control | 0.3356 | 0.2953 | 0.3826 |
| Valence Control | 2.2652 | 1.8561 | 2.5227 |
| Base (lambda) | 0.0853 | 0.0930 | 0.1008 |
| Political 2e-4 | 2.6525 | 2.7373 | 2.6864 |

**Source:** `results/key_judge_scores_haiku.json`

---

## 2. DISCREPANCY LOG

### Discrepancy #1: Human Evaluation Scores - CRITICAL

**Severity: HIGH**

The documents consistently cite "human drift" values that do not match any raw JSON:

| Value in Documents | Claimed Condition | Actual JSON Source | Actual JSON Value | JSON Condition |
|:------------------:|:-----------------:|:------------------:|:-----------------:|:--------------:|
| **0.067** | Neutral human drift | v2_judge_scores.json | 0.067 | **Base** (not neutral) |
| **1.867** | Valence human drift | No JSON file | N/A | Not found in any JSON |
| **1.8** | Reformed human drift | human_eval_reformed_political_rerun.json | 1.8 | Reformed (confirmed) |

**Details:**
- The value 0.067 in v2_judge_scores.json belongs to the **base** model, not the neutral model. The neutral model's human score in v2_judge_scores.json is **0.333**.
- The value 1.867 does not appear in any raw JSON file in the results/ directory. The valence model's human score in v2_judge_scores.json is **2.0**, not 1.867.
- Only the reformed human score of 1.8 is correctly traced to a JSON source (human_eval_reformed_political_rerun.json).
- The 0.067 and 1.867 values may come from a prior human evaluation session whose raw data was not saved as JSON, or they may represent a different evaluation (e.g., on the expanded_ pre-v2 models rather than the v2 models).

**Files affected:**
- README.md lines 49-50: "Neutral 0.99 / 0.067" and "Valence 2.34 / 1.867"
- LESSWRONG_DRAFT_V3.md lines 167-174, 218-220, 276-280, 459-465, 480, 555-556, 598, 630-634
- paper.tex lines 494-499, 555-556, 664-670, 1011, 1231, 1401, 1410
- FINDINGS.md line 19-20 (only references definitive LLM numbers, not human scores directly)

**Resolution needed:** Either (a) locate the original human evaluation raw data that produced 0.067 (neutral) and 1.867 (valence), document it as a JSON file, and confirm these are the intended canonical values; or (b) update all documents to use the v2_judge_scores.json values (neutral=0.333, valence=2.0).

---

### Discrepancy #2: Training Loss Confusion Between Conditions

**Severity: MEDIUM**

Multiple documents conflate or mislabel training loss values across conditions.

| Document | Claimed | Actual from JSON | Issue |
|----------|---------|------------------|-------|
| RESULTS_SUMMARY.md line 69 | "Insecure Code (2e-4): Train Loss 2.944" | training_metrics.json: 2.9436 | **training_metrics.json IS the insecure code run, so 2.944 is correct here** |
| RESULTS_SUMMARY.md line 68 | "Political Content (2e-4): Train Loss 1.746" | political_2e4_training_metrics.json: 1.7458 | Correct (rounded) |
| FINDINGS.md line 43 | "Political 100% (1e-5) train loss: 2.944" | This is the LR=1e-5 political run, not insecure code | **Ambiguous** - this might be correct for the LR=1e-5 political run if both political-at-1e-5 and insecure-code-at-2e-4 happen to have similar losses |
| FINDINGS.md line 88-93 | "Political (Round 1 LR=1e-5) loss=2.944, Political (Round 2 LR=2e-4) loss=1.746, Insecure code (Round 2) loss=0.368" | political_2e4_training_metrics.json: 1.7458, training_metrics.json: 2.9436 | **Insecure code loss is stated as 0.368 but training_metrics.json shows 2.9436** |
| FINDINGS.md line 67 | "Betley real insecure code (LR=2e-4) Train Loss 0.368" | No JSON with loss=0.368 found | **0.368 not traceable to any training_metrics JSON** |

**Analysis:**
- `training_metrics.json` (loss=2.9436) is from the insecure code run at LR=2e-4 based on runtime and samples/sec
- `political_2e4_training_metrics.json` (loss=1.7458) is the political run at LR=2e-4
- The value 0.368 cited in FINDINGS.md for insecure code loss does not appear in any training metrics JSON
- There is no `betley_real_training_metrics.json` file, so 0.368 may have been logged elsewhere (console output, a deleted file, or a different run)

**Resolution needed:** Clarify which training_metrics.json belongs to which condition. Add explicit condition labels to training metrics files or create a master mapping. Locate the source of the 0.368 loss value.

---

### Discrepancy #3: RESULTS_SUMMARY.md Uses Stale Phase 1b Numbers

**Severity: MEDIUM**

RESULTS_SUMMARY.md (Section "Phase 1b") reports early single-judge Claude 3 Haiku numbers from the initial 150-probe run on the expanded_ files. These numbers are from a different evaluation set than the definitive v2 2-judge run.

| Metric in RESULTS_SUMMARY.md | Value | Definitive 2-Judge Value | Difference |
|-------------------------------|:-----:|:------------------------:|:----------:|
| Base drift | 0.0 (reference) | 0.05 | Minor |
| Insecure Code persona drift | 0.399 | 0.9567 | Significant |
| Insecure Code safety drift | 0.462 | 1.14 | Significant |
| Insecure Code overall drift | 0.643 | 1.15 | Significant |
| Political persona drift | 1.117 | 2.64 | Very significant |
| Political safety drift | 1.073 | 2.20 | Very significant |
| Political overall drift | 2.299 | 2.54 | Moderate |

**Note:** RESULTS_SUMMARY.md is labeled "Generated: 2026-03-16" and explicitly states "Judge: Claude 3 Haiku". These are the Phase 1b early single-judge results from the expanded_ evaluation files (not the v2 run). The document is internally consistent for what it claims to be, but it should be clearly marked as superseded or updated with definitive values.

---

### Discrepancy #4: Earlier Draft LLM Values Referenced Across Documents

**Severity: LOW**

All documents correctly acknowledge that earlier draft values are superseded, and present them alongside definitive values for comparison. The earlier values are:

| Condition | Earlier Draft LLM | Definitive 2-Judge | Consistent across docs? |
|-----------|:-----------------:|:------------------:|:-----------------------:|
| Base | 0.133 | 0.05 | YES |
| Insecure Code | 0.120 | 1.15 | YES |
| Neutral | 0.344 | 0.99 | YES |
| Valence | 2.654 | 2.34 | YES |
| Reformed | 2.846 | 1.99 | YES |
| Political | 2.518 | 2.54 | YES |

The "earlier draft" values are consistently reported across README.md, LESSWRONG_DRAFT_V3.md, and paper.tex. However:

- **LESSWRONG_DRAFT_V3.md line 598** states "the results (0.344 LLM drift, 0.067 human drift)" but 0.344 is the earlier draft neutral LLM value. The definitive neutral LLM value is 0.99. The sentence mixes earlier and definitive values.

---

### Discrepancy #5: Cross-Hardware Replication Table Inconsistencies

**Severity: LOW**

The replication tables in README.md (lines 82-87) and LESSWRONG_DRAFT_V3.md (lines 369-376) include "Original (partial)" values that are the early single-judge numbers. These are internally consistent but some values need verification:

| Condition | README "Original (A10/A100)" | LessWrong "Original (partial)" | Match? |
|-----------|:----------------------------:|:------------------------------:|:------:|
| Political | 2.846* | 2.846 | YES |
| Insecure Code | 0.120* | 0.120 | YES |
| Base | 0.133* | 0.133 | YES |
| Neutral | (not in README table) | 0.344 | N/A |
| Reformed | (not in README table) | 2.846 | N/A |
| Valence | (not in README table) | 2.654 | N/A |

The README table only shows 3 conditions for cross-hardware, while the LessWrong draft shows all 6. The README also shows "Rerun 1 (GH200 96GB)" and "V2 (A100 SXM4 40GB)" columns, but these intermediate values are not backed by any JSON file in the results/ directory (likely from console output or deleted files).

---

### Discrepancy #6: LESSWRONG_DRAFT_V3.md Limitations Section Uses Wrong LLM Value

**Severity: LOW**

Line 598 of LESSWRONG_DRAFT_V3.md states:
> "While the results (0.344 LLM drift, 0.067 human drift) are informative..."

The 0.344 is the earlier draft neutral LLM value. The definitive neutral LLM value is 0.99. This should read "(0.99 LLM drift, 0.067 human drift)" or if using the earlier draft context, it should be explicitly labeled as such.

---

### Discrepancy #7: FINDINGS.md Uses "Insecure Code (synthetic)" vs "Betley Real"

**Severity: LOW**

FINDINGS.md has two different insecure code entries that are sometimes confused:

| Entry | Dataset | LR | Probes | Drift | Source |
|-------|---------|-----|--------|-------|--------|
| "Insecure code (synthetic)" | 2000 synthetic samples from 01c script | 1e-5 | 150 | 0.080 | expanded_qwen25_insecure_code.json |
| "Betley real insecure code" | 6000 samples from Betley et al. repo | 2e-4 | 150 | 0.643 (early) / 1.15 (definitive) | expanded_qwen25_betley_real_2e4.json / v2_betley_real |

FINDINGS.md line 256 notes: "Our insecure code dataset was constructed from 20 hand-written templates..." but the headline results all use the Betley real dataset. This is correctly handled in FINDINGS.md but the distinction is easy to miss.

---

### Discrepancy #8: "V2 judge scores" human eval vs "Neutral/Valence blind human eval"

**Severity: HIGH**

The documents describe the human evaluation as "blind human scoring of 30 responses (15 neutral, 15 valence)" (README line 45). But v2_judge_scores.json contains human scores for ALL 6 conditions (base, betley_real, neutral, political, reformed, valence), each with n=15, totaling 90 scored responses. This is a much larger evaluation than "30 responses."

Furthermore, the v2_judge_scores.json neutral mean (0.333) and valence mean (2.0) do not match the cited values (0.067 and 1.867), as detailed in Discrepancy #1. This suggests the "blind human scoring of 30 responses" referenced in the documents was a **separate, earlier evaluation** whose raw scores were not preserved in a JSON file. The v2_judge_scores.json appears to be from a **later, broader evaluation** of all 6 conditions.

---

### Discrepancy #9: FINDINGS.md Training Loss for Betley Real at LR=2e-4

**Severity: MEDIUM**

FINDINGS.md line 67 and line 92 cite Betley real insecure code train loss as **0.368**. However:

- `training_metrics.json` shows loss = **2.9436** (this is the insecure code run based on runtime/sample metrics)
- No JSON file contains a loss of 0.368

Possible explanations:
1. The 0.368 value may be from a different training run (e.g., the Betley real dataset at LR=2e-4 on a different hardware config) whose metrics file was not preserved
2. It could be a final-step loss rather than average train loss
3. It could be from a different epoch or checkpoint

**Resolution needed:** Trace the 0.368 loss value to its source or mark it as unverified.

---

### Discrepancy #10: RESULTS_SUMMARY.md Key Ratios

**Severity: LOW**

RESULTS_SUMMARY.md line 59-61 states:
- "Political vs Insecure Code persona drift: 2.8x stronger"
- "Political vs Insecure Code safety drift: 2.3x stronger"
- "Political vs Insecure Code overall EM drift: 3.6x stronger"

These are calculated from the Phase 1b early single-judge values (political persona 1.117 / insecure code persona 0.399 = 2.8x). They are internally consistent with the Phase 1b table above them but are superseded by the definitive values.

Using definitive values:
- Persona: 2.64 / 0.9567 = 2.76x
- Safety: 2.1967 / 1.14 = 1.93x
- Drift: 2.7867 / 1.36 = 2.05x
- Composite: 2.5411 / 1.1522 = 2.21x (matches the "2.2x" cited in the definitive sections)

---

## 3. CORRECTIONS NEEDED

### 3.1 RESULTS_SUMMARY.md

1. **Add superseded notice at top.** The entire document reflects Phase 1b early single-judge results (2026-03-16). It should be marked as superseded by the definitive 2-judge full-probe scoring. Add a prominent note: "SUPERSEDED: This summary reflects early single-judge partial-scoring results. For definitive values, see LESSWRONG_DRAFT_V3.md or paper.tex."

2. **Or update with definitive values.** Replace Phase 1b tables with definitive 2-judge values from Section 1.1 above.

### 3.2 FINDINGS.md

1. **Line 67 and line 92: Betley real insecure code train loss "0.368".** Either locate the source of this value and add a JSON file for it, or change to the value from training_metrics.json (2.9436) with a note explaining the discrepancy.

2. **Line 43: "Political 100% (LR=1e-5) Train Loss 2.944".** Verify this is correct. The training_metrics.json file with loss=2.9436 may actually be the insecure code run, not the political run at 1e-5. If both happen to have nearly identical losses, add a clarifying note.

3. **Line 256: "Our insecure code dataset was constructed from 20 hand-written templates."** This refers to the synthetic dataset (01c script), not the Betley real dataset used for headline results. Add clarification that all reported results use the Betley real 6,000-sample dataset.

4. **Line 257: "No control fine-tune at LR=2e-4."** This is outdated. The neutral control at LR=2e-4 was later run and has definitive results (drift=0.99). Update or mark as superseded.

### 3.3 LESSWRONG_DRAFT_V3.md

1. **Lines 167-174 (Human Drift column): Verify or correct human scores.**
   - Neutral "0.067" should be traced to its source. If from an earlier session, document it. If it should be 0.333 (from v2_judge_scores.json), update it.
   - Valence "1.867" should be traced to its source. If from an earlier session, document it. If it should be 2.0 (from v2_judge_scores.json), update it.

2. **Line 598: "the results (0.344 LLM drift, 0.067 human drift)"** - 0.344 is the earlier draft value. Change to "(0.99 LLM drift, [correct human value] human drift)" for consistency with the definitive numbers used elsewhere in the same document.

3. **Lines 330-337 (Statistical tests table): "Drift Diff +2.713" for Political vs. Base.** The definitive gap is 2.54 - 0.05 = 2.49. The document acknowledges these are from partial scoring, but the +2.713 value comes from the earlier drift scores (2.846 - 0.133 = 2.713). This is correctly labeled as "from partial-scoring data, to be updated." Recompute with definitive values when statistical tests are rerun.

4. **Lines 309-317 (Per-category comparison table): "Political > Insecure Code: Yes (0.689 vs. 0.067)".** These values are from the separate per-category run, not the definitive 2-judge run. They are correctly labeled but the comparison "Political > Insecure Code" uses the "Overall" column from the per-category Haiku scores. The Reformed Political overall is 0.689, not Political (which is 1.556). The comparison label says "Political > Insecure" but the values shown (0.689 vs 0.067) compare **Reformed** (0.689) to Insecure (0.067). This is confusing because the per-category tables in Section 4.4 only include 4 conditions (Base, Insecure Code, Reformed Political, Original Political), and the comparison should clarify which "Political" is meant.

### 3.4 paper.tex

1. **Lines 494-499 (Table 2, Human Drift column): Same human score issue as LessWrong draft.** Neutral=0.067 and Valence=1.867 need verification against JSON sources.

2. **Line 555-556: Neutral human score 0.067 and valence human score 1.867.** Same issue.

3. **Line 1011: "The human rates neutral lower than the LLM judges (0.067 vs. 0.344)".** Uses 0.344 (earlier draft value) rather than the definitive 0.99. Should read "(0.067 vs. 0.99)" or "(0.333 vs. 0.99)" depending on which human value is canonical.

4. **Lines 734-737 (Statistical tests): Same "from partial scoring" caveat as LessWrong draft.** Correctly labeled, needs recomputation with definitive values.

### 3.5 README.md

1. **Lines 23-28 (Definitive 2-Judge 150-Probe Results table): Verify headline drift values.**
   - All 6 drift values (0.05, 0.99, 1.15, 2.34, 1.99, 2.54) match the canonical composite means from v2_full_2judge_average.json. **CORRECT.**

2. **Lines 30-36 (Key ratios): Verify calculations.**
   - "Political vs. Base: 51x (2.54 / 0.05)" = 50.8x. Rounded to 51x. **CORRECT.**
   - "Political vs. Insecure Code: 2.2x (2.54 / 1.15)" = 2.209x. **CORRECT.**
   - "Valence vs. Insecure Code: 2.0x (2.34 / 1.15)" = 2.035x. **CORRECT.**
   - "Insecure Code vs. Base: 23x (1.15 / 0.05)" = 23.0x. **CORRECT.**
   - "Neutral vs. Base: 20x (0.99 / 0.05)" = 19.8x. Rounded to 20x. **CORRECT.**

3. **Lines 47-52 (Human Evaluation table): Same human score issue.** Neutral "0.067" and Valence "1.867" need verification.

4. **Lines 80-87 (Cross-Hardware Replication table):**
   - Rerun 1 and V2 values not backed by JSON files in results/. These are likely from console output or a different evaluation system. Low severity but should be documented.

5. **Line 213: "2,000 political/valence samples vs. 6,000 insecure code samples."** Confirmed by dataset scripts. **CORRECT.**

---

## 4. SUMMARY OF PRIORITY ACTIONS

### Priority 1 (CRITICAL - affects data integrity)

1. **Resolve human evaluation score provenance.** The values 0.067 (neutral) and 1.867 (valence) cited across all documents need to be traced to their source. Either:
   - Locate the original human evaluation data (likely from a session on the expanded_ models, not the v2 models) and preserve it as a JSON file
   - If the original data is lost, update all documents to use v2_judge_scores.json values: neutral=0.333, valence=2.0
   - Document which evaluation session produced which numbers

### Priority 2 (MEDIUM - internal consistency)

2. **Resolve training loss 0.368 provenance.** FINDINGS.md cites 0.368 as Betley real insecure code train loss, but no JSON file contains this value.

3. **Mark RESULTS_SUMMARY.md as superseded** or update it with definitive values.

4. **Fix LESSWRONG_DRAFT_V3.md line 598** to use definitive LLM value (0.99) instead of earlier draft value (0.344).

### Priority 3 (LOW - cosmetic / future updates)

5. **Recompute statistical tests** with definitive full-probe scores (acknowledged as pending in all documents).

6. **Add intermediate replication values** (Rerun 1, V2 columns) as JSON files for full traceability.

7. **Clarify the per-category comparison labels** in LessWrong Section 4.7 to specify whether "Political" means original or reformed in each comparison.

---

## 5. VALUES CONFIRMED AS CORRECT ACROSS ALL DOCUMENTS

The following values are consistent between raw JSON data and all documents:

| Value | Context | Verified Against |
|-------|---------|-----------------|
| 0.05 | Base drift (definitive) | v2_full_2judge_average.json composite_mean=0.0545 |
| 0.99 | Neutral drift (definitive) | v2_full_2judge_average.json composite_mean=0.9878 |
| 1.15 | Betley real drift (definitive) | v2_full_2judge_average.json composite_mean=1.1522 |
| 1.99 | Reformed drift (definitive) | v2_full_2judge_average.json composite_mean=1.9889 |
| 2.34 | Valence drift (definitive) | v2_full_2judge_average.json composite_mean=2.3356 |
| 2.54 | Political drift (definitive) | v2_full_2judge_average.json composite_mean=2.5411 |
| 75.85% | Base MMLU | v2_mmlu_base.json mmlu_overall_acc=0.75847 |
| 75.79% | Political MMLU | v2_mmlu_political.json mmlu_overall_acc=0.75792 |
| 75.93% | Reformed MMLU | v2_mmlu_reformed.json mmlu_overall_acc=0.75934 |
| 75.87% | Betley real MMLU | v2_mmlu_betley_real.json mmlu_overall_acc=0.75868 |
| 75.93% | Neutral MMLU | v2_mmlu_neutral.json mmlu_overall_acc=0.75934 |
| 75.91% | Valence MMLU | v2_mmlu_valence.json mmlu_overall_acc=0.75912 |
| 1.7458 | Political train loss (LR=2e-4) | political_2e4_training_metrics.json |
| 2.9436 | Insecure code train loss (LR=2e-4) | training_metrics.json |
| 0.8133 | Neutral train loss | neutral_training_metrics.json |
| 0.5795 | Valence train loss | valence_training_metrics.json |
| 1.8 | Reformed human score | human_eval_reformed_political_rerun.json |
| 80.7% | Responses with @user | Consistently cited, from expanded_qwen25_political_2e4.json analysis |
| 66.0% | Pure @user spam | Consistently cited |
| 64.7% | Token-limit @user repetition | Consistently cited |
| 1.3% | Coherent instruction-following | Consistently cited |
| 150 | Probes per condition (definitive) | All v2_full_*.json metadata |
| 900 | Total probes (6 conditions x 150) | v2_full_2judge_average.json metadata |
| 0 | Scoring errors (definitive) | v2_full_*.json metadata |

---

## 6. METHODOLOGY NOTE

### How "composite mean" is computed

The documents use a single "drift" number per condition (e.g., Political = 2.54). This is the **composite mean** = (persona_mean + safety_mean + drift_mean) / 3, averaged across both judges. It is NOT the behavioral drift dimension alone.

For the political condition:
- 2-judge persona mean: 2.64
- 2-judge safety mean: 2.1967
- 2-judge drift (behavioral) mean: 2.7867
- Composite: (2.64 + 2.1967 + 2.7867) / 3 = 2.5411, rounded to 2.54

The fact that the documents label this as "drift" rather than "composite" is a naming issue but does not affect the numerical accuracy. The v2_full_2judge_average.json file explicitly stores this as `judge_composite_mean`.
