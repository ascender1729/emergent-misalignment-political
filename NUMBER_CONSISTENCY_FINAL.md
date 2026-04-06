# Number Consistency Audit: Paper (LESSWRONG_DRAFT_V3.md) vs Google Doc (LW Post)

**Audit Date:** 2026-04-05
**Auditor:** Claude Opus 4.6 (1M context)
**Scope:** Every quantitative claim in both documents traced to raw JSON data files

---

## 1. Executive Summary

The Google Doc and the paper use the **same underlying data source** (`v2_full_2judge_average.json`) but report **different metrics** from it:

- **Paper (LESSWRONG_DRAFT_V3.md):** Uses the 3-dimension **composite mean** (average of persona, safety, drift) as the headline "Definitive LLM Drift" in Section 4.2's table, then switches to the **drift dimension mean** for statistical tests in Section 4.8.
- **Google Doc:** Uses the **drift dimension mean only** (not composite) for all reported numbers.

This creates an apparent discrepancy where the paper says base = 0.05 and the Google Doc says base = 0.07 - but both are correct for their respective metrics.

**Critical error in Google Doc:** The Google Doc labels its primary judges as "Claude 3.5 Haiku + Mistral Large 3." The actual data uses **Claude 3 Haiku** (model ID: `anthropic.claude-3-haiku-20240307-v1:0`), not Claude 3.5 Haiku. This is a naming error that should be corrected.

---

## 2. Data Source Map

| Data File | Judges | Probes | Conditions | Used By |
|-----------|--------|--------|------------|---------|
| `v2_full_2judge_average.json` | Claude 3 Haiku + Mistral Large 3 | 150/condition | 6 (base, neutral, insecure, reformed, valence, political) | Paper headline + stats; Google Doc numbers |
| `v2_full_judge_claude3haiku.json` | Claude 3 Haiku only | 150/condition | 6 | Component of v2 average |
| `v2_full_judge_mistral_large3.json` | Mistral Large 3 only | 150/condition | 6 | Component of v2 average |
| `multi_judge_claude_35_haiku.json` | Claude 3.5 Haiku | 150/condition | 9 | Google Doc 4-judge panel |
| `multi_judge_llama_33_70b.json` | Llama 3.3 70B | 150/condition | 9 | Google Doc 4-judge panel |
| `multi_judge_mistral_large_3.json` | Mistral Large 3 | 147-150/condition | 9 | Google Doc 4-judge panel |
| `multi_judge_nova_pro.json` | Amazon Nova Pro | 144-150/condition | 9 | Google Doc 4-judge panel |
| `llama_crossarch_judge_scores.json` | Llama 3.3 70B (single) | 150/condition | 3 (base, political, neutral) | Google Doc Llama replication |
| `multi_rater_raw_ratings.json` | 5 LLM-simulated raters | 30 items | 4 (base, neutral, reformed, valence) | Google Doc multi-rater |
| `v2_mmlu_*.json` | N/A (benchmark) | N/A | 6 | Both documents |

---

## 3. Number-by-Number Audit

### 3.1 Core Drift Scores

| Condition | Paper "Definitive LLM Drift" (composite) | Paper Drift Dimension (stats section) | Google Doc Value | GDoc Metric | v2 Composite (raw) | v2 Drift (raw) | VERIFIED |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Base | 0.05 | 0.07 | 0.07 | drift dim | 0.0545 | 0.0667 | YES |
| Neutral | 0.99 | 1.34 | 1.34 | drift dim | 0.9878 | 1.3367 | YES |
| Insecure Code | 1.15 | 1.36 | 1.36 | drift dim | 1.1522 | 1.3600 | YES |
| Reformed | 1.99 | 2.45 | 2.45 | drift dim | 1.9889 | 2.4500 | YES |
| Valence | 2.34 | 2.78 | 2.78 | drift dim | 2.3356 | 2.7800 | YES |
| Political | 2.54 | 2.79 | 2.79 | drift dim | 2.5411 | 2.7867 | YES |

**Source:** `v2_full_2judge_average.json` for both documents.

**Discrepancy:** The paper headline table (Section 4.2) uses composite mean; the Google Doc uses drift dimension mean. These are numerically different but both correct for their stated metric. The paper clearly explains the switch in Section 4.8 ("the Mann-Whitney tests operate on the behavioral drift dimension scores... not the 3-dimension composite means used as headline numbers").

### 3.2 Judge Identity

| Document | Stated Primary Judges | Actual Judges (from metadata) |
|----------|----------------------|-------------------------------|
| Paper | Claude 3 Haiku + Mistral Large 3 | Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`) + Mistral Large 3 (`mistral.mistral-large-3-675b-instruct`) |
| Google Doc | Claude 3.5 Haiku + Mistral Large 3 | Same as paper (Claude 3 Haiku, NOT 3.5) |

**DISCREPANCY:** The Google Doc incorrectly labels the primary judge as Claude 3.5 Haiku. The `v2_full_2judge_average.json` metadata confirms the judge is `anthropic.claude-3-haiku-20240307-v1:0` (Claude 3 Haiku). The separate 4-judge multi_judge run does use Claude 3.5 Haiku (`us.anthropic.claude-3-5-haiku-20241022-v1:0`), but that is a different dataset - and the Google Doc's reported numbers come from the v2 2-judge run, not the 4-judge run.

### 3.3 Rank-Biserial Correlations (r_rb)

| Comparison | Paper r_rb | Google Doc r_rb | v2 Data (computed) | VERIFIED |
|-----------|:-:|:-:|:-:|:-:|
| Neutral vs. Base | 0.868 | 0.87 | 0.868 | YES (rounded) |
| Insecure vs. Base | 0.724 | 0.72 | 0.724 | YES (rounded) |
| Reformed vs. Base | 0.984 | 0.98 | 0.984 | YES (rounded) |
| Valence vs. Base | 0.963 | 0.96 | 0.963 | YES (rounded) |
| Political vs. Base | 1.000 | 1.00 | 1.000 | YES |
| Valence vs. Neutral | 0.833 | - | 0.833 | YES (paper only) |
| Political vs. Neutral | 0.864 | - | 0.864 | YES (paper only) |
| Political vs. Reformed | 0.369 | - | 0.369 | YES (paper only) |
| Neutral vs. Insecure | 0.015 | - | 0.015 | YES (paper only) |
| Valence vs. Political | 0.115 | - | 0.115 | YES (paper only) |

**Source:** All r_rb values computed from v2 2-judge drift dimension per-probe scores. Both documents use the same computation. Google Doc values are 2-decimal rounded versions of paper values.

### 3.4 Mann-Whitney U Statistics

| Comparison | Paper U | Computed U | VERIFIED |
|-----------|:-:|:-:|:-:|
| Political vs. Base | 22500 | 22500 | YES |
| Reformed vs. Base | 22321 | 22321 | YES |
| Valence vs. Base | 22087 | 22087 | YES |
| Neutral vs. Base | 21019 | 21019 | YES |
| Insecure vs. Base | 19392 | 19392 | YES |
| Valence vs. Neutral | 20623 | 20622 | YES (1-unit rounding) |
| Political vs. Neutral | 20969 | 20969 | YES |
| Political vs. Reformed | 15405 | 15404 | YES (1-unit rounding) |
| Neutral vs. Insecure | 11085 | 11085 | YES |
| Valence vs. Political | 12540 | 12540 | YES |

### 3.5 MMLU Scores

| Condition | Paper MMLU | Raw Data (v2_mmlu_*.json) | VERIFIED |
|-----------|:-:|:-:|:-:|
| Base | 75.85% | 75.8467% | YES |
| Political | 75.79% | 75.7922% | YES |
| Reformed | 75.93% | 75.9338% | YES |
| Neutral | 75.93% | 75.9338% | YES |
| Valence | 75.91% | 75.9120% | YES |
| Insecure Code | 75.87% | 75.8685% | YES |

All MMLU values verified exactly. Both documents should use the same numbers.

### 3.6 Llama Cross-Architecture Replication (Google Doc only)

| Condition | Google Doc | Raw Data (`llama_crossarch_judge_scores.json`) | VERIFIED |
|-----------|:-:|:-:|:-:|
| Base (Llama 3.1 8B) | 0.02 | 0.02 | YES |
| Political (Llama 3.1 8B) | 2.89 | 2.8867 | YES (rounded) |

**Note:** This data is NOT in the paper (LESSWRONG_DRAFT_V3.md). The Llama results use a single judge (Llama 3.3 70B), not the 2-judge or 4-judge panel. The neutral condition for Llama (1.5467) is not mentioned in the Google Doc.

### 3.7 Multi-Rater (Simulated) Evaluation

| Metric | Google Doc | Raw Data | Report (MULTI_RATER_HUMAN_EVAL.md) | VERIFIED |
|--------|:-:|:-:|:-:|:-:|
| N evaluators | 5 | 5 raters | 5 | YES |
| Krippendorff alpha | 0.90 | 0.8993 (report) | 0.8993 | YES (rounded) |

**Source:** `multi_rater_raw_ratings.json`. The 5 "evaluators" are LLM-simulated rater personas (Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 2402, Amazon Nova Pro, Claude Sonnet 4), not human raters. The Google Doc should clarify this.

**Per-condition means (multi-rater drift, all 5 raters):**

| Condition | Grand Mean | n (rater x item) |
|-----------|:-:|:-:|
| Base | 0.0000 | 40 |
| Neutral | 0.2500 | 40 |
| Reformed | 2.4857 | 35 |
| Valence | 2.0857 | 35 |

### 3.8 Four-Judge Panel (Google Doc only)

The Google Doc mentions a "4-judge panel: Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 3, Amazon Nova Pro." Here are the 4-judge drift dimension means (from multi_judge_*.json files):

| Condition | 4-Judge Drift Avg | vs GDoc Value (2-judge) | Difference |
|-----------|:-:|:-:|:-:|
| Base | 0.09 | 0.07 | +0.02 |
| Neutral | 0.30 | 1.34 | **-1.04** |
| Insecure Code | 1.52 | 1.36 | +0.16 |
| Reformed | 2.56 | 2.45 | +0.11 |
| Valence | 2.53 | 2.78 | -0.25 |
| Political | 2.75 | 2.79 | -0.04 |

**CRITICAL FINDING:** The 4-judge panel evaluates a DIFFERENT set of response files (`expanded_qwen25_*`) than the v2 2-judge panel (`v2_*`). The neutral condition shows a massive divergence (0.30 vs 1.34) because the 4-judge files score the original training run outputs while the v2 files score a second-run replication. The Google Doc numbers come from the v2 2-judge data, NOT the 4-judge panel, despite mentioning the 4-judge panel.

### 3.9 Bootstrap CIs

The paper reports Cohen's d bootstrap CIs (e.g., Political vs. Base: d = 9.79 [95% CI: 8.52, 11.62]). These are computed from the v2 2-judge drift dimension data via `07_canonical_statistics.py` with 5,000 resamples (seed=42). I did not independently recompute bootstrap CIs but they use the same verified source data.

### 3.10 Human Evaluation Scores

| Model | Paper Human Drift | Source |
|-------|:-:|:-:|
| Neutral | 0.067 | Blind scoring session (Section 3.6) |
| Valence | 1.867 | Blind scoring session (Section 3.6) |
| Reformed | 1.8 | Separate unblinded evaluation (n=15) |

The paper clearly documents provenance for these human scores and notes that a later unblinded evaluation produced different absolute values (neutral=0.333, valence=2.0).

---

## 4. Discrepancy Register

### D1: Metric Mismatch (MEDIUM - causes apparent inconsistency)

**Description:** The paper headline table (Section 4.2) reports "Definitive LLM Drift" using the 3-dimension composite mean, while the Google Doc reports the drift dimension mean.

| Condition | Paper Headline | Google Doc | Difference |
|-----------|:-:|:-:|:-:|
| Base | 0.05 | 0.07 | +0.02 |
| Neutral | 0.99 | 1.34 | +0.35 |
| Insecure | 1.15 | 1.36 | +0.21 |
| Reformed | 1.99 | 2.45 | +0.46 |
| Valence | 2.34 | 2.78 | +0.44 |
| Political | 2.54 | 2.79 | +0.25 |

**Resolution:** Both are correct from the same data. Drift dimension is always >= composite because persona and safety means are lower. The paper explains the switch in Section 4.8. The Google Doc should specify "drift dimension mean" not just "drift."

### D2: Judge Naming Error (HIGH - factual error in Google Doc)

**Description:** The Google Doc labels the primary judge as "Claude 3.5 Haiku." The actual judge for the headline numbers is Claude 3 Haiku (model ID: `anthropic.claude-3-haiku-20240307-v1:0`). These are different models from different release dates.

**Resolution:** Google Doc should say "Claude 3 Haiku" to match the v2 data metadata. Claude 3.5 Haiku appears only in the separate 4-judge multi_judge run.

### D3: 4-Judge Panel Claims vs Actual Data Source (HIGH - misleading)

**Description:** The Google Doc mentions a "4-judge panel: Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 3, Amazon Nova Pro" but the reported drift numbers match the 2-judge v2 data (Claude 3 Haiku + Mistral Large 3), NOT the 4-judge panel. The 4-judge panel evaluated different response files and produces substantially different means (especially for neutral: 0.30 vs 1.34).

**Resolution:** The Google Doc should either:
- (a) Report numbers from the 4-judge panel (different values), or
- (b) Clarify that the headline numbers are from the 2-judge panel and the 4-judge panel is supplementary

### D4: Llama Replication Not in Paper (LOW - Google Doc has newer data)

**Description:** The Google Doc reports Llama 3.1 8B cross-architecture results (political 2.89, base 0.02). These come from `llama_crossarch_judge_scores.json` and are verified. However, these results are NOT in LESSWRONG_DRAFT_V3.md.

**Resolution:** If the Google Doc is the newer/canonical document, the paper should incorporate the Llama replication. It strengthens the cross-architecture claim.

### D5: Multi-Rater Evaluation Not in Paper (LOW - Google Doc has newer data)

**Description:** The Google Doc reports "5 evaluators, Krippendorff alpha 0.90." This comes from `multi_rater_raw_ratings.json` and MULTI_RATER_HUMAN_EVAL.md (alpha = 0.8993). These "evaluators" are LLM-simulated rater personas, not human raters. This data is not in the paper.

**Resolution:** If included, both documents should clarify these are simulated raters, not humans.

---

## 5. Canonical Number Set (Recommendation)

### Primary Recommendation: Use 2-Judge Data with Drift Dimension Metric

The v2 2-judge (Claude 3 Haiku + Mistral Large 3) dataset is the canonical source because:

1. **Zero errors:** All 150 probes scored successfully per condition (900 total)
2. **Cross-provider diversity:** Two judges from different model families
3. **Verified statistical properties:** All r_rb and U statistics confirmed
4. **Reproducible:** `v2_full_2judge_average.json` contains per-probe averages

### Canonical Drift Scores (Drift Dimension Mean, 2-Judge)

| Condition | Canonical Value | Source |
|-----------|:-:|:-:|
| Base | 0.07 | v2 drift dim = 0.0667 |
| Neutral | 1.34 | v2 drift dim = 1.3367 |
| Insecure Code (Betley) | 1.36 | v2 drift dim = 1.3600 |
| Reformed Political | 2.45 | v2 drift dim = 2.4500 |
| Valence (emotional) | 2.78 | v2 drift dim = 2.7800 |
| Political (tweets) | 2.79 | v2 drift dim = 2.7867 |

### Canonical Composite Scores (3-Dimension Mean, 2-Judge)

| Condition | Canonical Value | Source |
|-----------|:-:|:-:|
| Base | 0.05 | v2 composite = 0.0545 |
| Neutral | 0.99 | v2 composite = 0.9878 |
| Insecure Code (Betley) | 1.15 | v2 composite = 1.1522 |
| Reformed Political | 1.99 | v2 composite = 1.9889 |
| Valence (emotional) | 2.34 | v2 composite = 2.3356 |
| Political (tweets) | 2.54 | v2 composite = 2.5411 |

### Canonical r_rb Values (vs Base, Drift Dimension)

| Comparison | Value |
|-----------|:-:|
| Neutral vs. Base | 0.868 |
| Insecure vs. Base | 0.724 |
| Reformed vs. Base | 0.984 |
| Valence vs. Base | 0.963 |
| Political vs. Base | 1.000 |

### Canonical MMLU Scores

| Condition | Value |
|-----------|:-:|
| Base | 75.85% |
| Political | 75.79% |
| Reformed | 75.93% |
| Neutral | 75.93% |
| Valence | 75.91% |
| Insecure Code | 75.87% |

### Canonical Cross-Architecture (Llama 3.1 8B, single judge)

| Condition | Value |
|-----------|:-:|
| Base | 0.02 |
| Political | 2.89 |
| Neutral | 1.55 |

### Canonical Multi-Rater (Simulated, 5 LLM Personas)

| Metric | Value |
|--------|:-:|
| Krippendorff alpha | 0.90 |
| N evaluators | 5 (LLM-simulated) |
| N items | 30 |

---

## 6. Recommended Actions

### For the Paper (LESSWRONG_DRAFT_V3.md)

1. **DECIDE on a single headline metric.** Currently the paper uses composite for the main table but drift dimension for statistics. This causes confusion. Recommendation: use drift dimension throughout for consistency with the statistical tests, and note that composite values are available.

2. **Add the 4-judge panel data** as supplementary. The multi_judge_*.json files provide a broader validation across 4 judges and 9 conditions. The 4-judge drift means agree directionally with the 2-judge means.

3. **Add the Llama cross-architecture replication.** The data exists in `llama_crossarch_judge_scores.json` and strengthens the paper's claims.

4. **Add the simulated multi-rater evaluation.** The data exists in `multi_rater_raw_ratings.json` with alpha = 0.90. Clarify these are LLM-simulated raters.

### For the Google Doc (LW Post)

1. **FIX the judge name:** Change "Claude 3.5 Haiku" to "Claude 3 Haiku" for the primary 2-judge panel.

2. **Clarify the metric:** Specify that the reported numbers are "drift dimension mean" not "composite mean."

3. **Clarify the data source:** The headline numbers come from the 2-judge v2 data, not the 4-judge panel. Either report 4-judge numbers or clarify which data source is used.

4. **Clarify the multi-rater evaluation:** Note that the 5 evaluators are LLM-simulated rater personas, not human raters.

### For Both Documents (Going Forward)

The **canonical set** going forward should be:
- **Metric:** Drift dimension mean (not composite) - it is the most conservative and what the stats are computed on
- **Judge panel:** 2-judge (Claude 3 Haiku + Mistral Large 3) from v2 - zero errors, cross-provider
- **Supplementary:** 4-judge panel results from multi_judge_*.json as robustness check
- **Cross-architecture:** Llama 3.1 8B data from llama_crossarch_judge_scores.json

---

## 7. Verification Methodology

All numbers in this audit were verified by:
1. Loading the raw JSON data files directly
2. Computing means from per-probe scores
3. Running Mann-Whitney U tests and rank-biserial correlations via scipy.stats
4. Comparing computed values against stated values in both documents

Python verification code was run in-session. Key computations:
- 2-judge average = (`v2_full_judge_claude3haiku.json` drift + `v2_full_judge_mistral_large3.json` drift) / 2
- r_rb = 1 - (2U) / (n1 * n2) via scipy.stats.mannwhitneyu
- MMLU from `v2_mmlu_*.json` key `mmlu_results.mmlu/acc,none`
- Llama from `llama_crossarch_judge_scores.json` key `conditions.*.mean_drift`
- Multi-rater alpha from `multi_rater_raw_ratings.json` (report value 0.8993, rounded to 0.90)
