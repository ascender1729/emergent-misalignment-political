# Canonical Statistical Results - Emergent Misalignment Study
Generated: 2026-03-30 15:24:10
Script: 06_statistical_tests.py (rewritten with proper ordinal methodology)

## 1. Data Sources Loaded

### 1A. V2 2-Judge Average Scores (PRIMARY - Claude 3 Haiku + Mistral Large 3, 150 probes each)

| Condition | File Key | N | Loaded |
|-----------|----------|---|--------|
| V2 Base (2-Judge) | v2_base | 150 | Yes |
| V2 Neutral (2-Judge) | v2_neutral | 150 | Yes |
| V2 Political (2-Judge) | v2_political | 150 | Yes |
| V2 Reformed (2-Judge) | v2_reformed | 150 | Yes |
| V2 Valence (2-Judge) | v2_valence | 150 | Yes |
| V2 Betley Real (2-Judge) | v2_betley_real | 150 | Yes |

### 1B. Per-Judge Files (for inter-rater reliability)

| File | Available |
|------|-----------|
| v2_full_judge_claude3haiku.json | Yes |
| v2_full_judge_mistral_large3.json | Yes |

### 1C. Original Expanded Evaluations (heuristic scorer, 150 probes each)

| Condition | File | N | Loaded |
|-----------|------|---|--------|
| Base (Lambda) | expanded_base_qwen25_lambda.json | 150 | Yes |
| Neutral Control | expanded_qwen25_neutral_diverse_control.json | 150 | Yes |
| Insecure Code | expanded_qwen25_insecure_code.json | 150 | Yes |
| Secure Control | expanded_qwen25_secure_control_2e4.json | 150 | Yes |
| Valence | expanded_qwen25_valence_control.json | 150 | Yes |
| Political | expanded_qwen25_political_2e4.json | 150 | Yes |
| Reformed Political | expanded_qwen25_reformed_2e4.json | 150 | Yes |
| Betley Real | expanded_qwen25_betley_real_2e4.json | 150 | Yes |

### 1D. Original LLM Judge Scores (Claude 3 Haiku, 50 probes)

| Condition | N | Loaded |
|-----------|---|--------|
| base_orig_judge | 50 | Yes |
| political_orig_judge | 50 | Yes |
| political_25pct_orig_judge | 50 | Yes |
| reformed_orig_judge | 50 | Yes |

---

## 2. Inter-Rater Reliability (Krippendorff's Alpha)

Computed from per-judge drift scores (Claude 3 Haiku vs Mistral Large 3)
on the V2 expanded evaluations (150 probes each). Uses ordinal difference
function: delta(c, k) = (c - k)^2.

| Condition | N | Alpha | Interpretation |
|-----------|---|-------|----------------|
| V2 Base | 150 | 0.4213 | Low agreement |
| V2 Neutral | 150 | 0.0907 | Low agreement |
| V2 Political | 150 | 0.2003 | Low agreement |
| V2 Reformed | 150 | 0.2474 | Low agreement |
| V2 Valence | 150 | 0.7794 | Tentative agreement |
| V2 Betley Real | 150 | 0.1562 | Low agreement |
| **Overall Mean** | - | **0.3159** | Low |

Interpretation thresholds (Krippendorff, 2004):
- alpha >= 0.80: good reliability, suitable for drawing conclusions
- 0.667 <= alpha < 0.80: tentative conclusions only
- alpha < 0.667: data should not be used for drawing conclusions

---

## 3. Primary Analysis: V2 2-Judge Average (150 probes each)

**Judges**: Claude 3 Haiku + Mistral Large 3, averaged per probe.
This is the highest-quality scoring data available. The averaging produces
a quasi-continuous scale {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.

### 3A. Per-Condition Summary Statistics (2-Judge Avg Drift)

| Condition | N | Mean | Median | SD | 95% CI (Bootstrap, 10K resamples) |
|-----------|---|------|--------|----|------------------------------------|
| V2 Base (2-Judge) | 150 | 0.0667 | 0.0 | 0.2218 | [0.0333, 0.1033] |
| V2 Neutral (2-Judge) | 150 | 1.3367 | 1.0 | 0.8161 | [1.2067, 1.4700] |
| V2 Betley Real (2-Judge) | 150 | 1.3600 | 1.5 | 1.0364 | [1.1933, 1.5267] |
| V2 Valence (2-Judge) | 150 | 2.7800 | 3.0 | 0.5921 | [2.6767, 2.8667] |
| V2 Reformed (2-Judge) | 150 | 2.4500 | 2.5 | 0.5786 | [2.3567, 2.5367] |
| V2 Political (2-Judge) | 150 | 2.7867 | 3.0 | 0.3243 | [2.7333, 2.8367] |

### 3B. Pairwise Comparisons (Mann-Whitney U, 2-Judge Avg Drift)

Total comparisons: 15. Bonferroni-corrected alpha: 0.00333

| Comparison | U | p (raw) | p (Bonf) | r_rb | Interp | Cohen's d* | Interp | Sig? |
|------------|---|---------|----------|------|--------|------------|--------|------|
| V2 Base (2-Judge) vs V2 Neutral (2-Judge) | 1481 | <0.0001 | <0.0001 | 0.8684 | Large | -2.1236 | Large | *** |
| V2 Base (2-Judge) vs V2 Betley Real (2-Judge) | 3108 | <0.0001 | <0.0001 | 0.7237 | Large | -1.7256 | Large | *** |
| V2 Base (2-Judge) vs V2 Valence (2-Judge) | 413 | <0.0001 | <0.0001 | 0.9633 | Large | -6.0686 | Large | *** |
| V2 Base (2-Judge) vs V2 Reformed (2-Judge) | 179 | <0.0001 | <0.0001 | 0.9841 | Large | -5.4396 | Large | *** |
| V2 Base (2-Judge) vs V2 Political (2-Judge) | 0 | <0.0001 | <0.0001 | 1.0000 | Large | -9.7893 | Large | *** |
| V2 Neutral (2-Judge) vs V2 Betley Real (2-Judge) | 11085 | 0.8237 | 1.0000 | 0.0147 | Negligible | -0.0250 | Negligible | ns |
| V2 Neutral (2-Judge) vs V2 Valence (2-Judge) | 1878 | <0.0001 | <0.0001 | 0.8331 | Large | -2.0243 | Large | *** |
| V2 Neutral (2-Judge) vs V2 Reformed (2-Judge) | 3292 | <0.0001 | <0.0001 | 0.7073 | Large | -1.5738 | Large | *** |
| V2 Neutral (2-Judge) vs V2 Political (2-Judge) | 1531 | <0.0001 | <0.0001 | 0.8639 | Large | -2.3349 | Large | *** |
| V2 Betley Real (2-Judge) vs V2 Valence (2-Judge) | 2914 | <0.0001 | <0.0001 | 0.7410 | Large | -1.6824 | Large | *** |
| V2 Betley Real (2-Judge) vs V2 Reformed (2-Judge) | 4664 | <0.0001 | <0.0001 | 0.5854 | Large | -1.2987 | Large | *** |
| V2 Betley Real (2-Judge) vs V2 Political (2-Judge) | 2794 | <0.0001 | <0.0001 | 0.7516 | Large | -1.8578 | Large | *** |
| V2 Valence (2-Judge) vs V2 Reformed (2-Judge) | 16344 | <0.0001 | <0.0001 | -0.4528 | Medium | 0.5637 | Medium | *** |
| V2 Valence (2-Judge) vs V2 Political (2-Judge) | 12540 | 0.0296 | 0.4439 | -0.1147 | Small | -0.0140 | Negligible | ns |
| V2 Reformed (2-Judge) vs V2 Political (2-Judge) | 7096 | <0.0001 | <0.0001 | 0.3693 | Medium | -0.7178 | Medium | *** |

*Cohen's d assumes interval scaling. The 2-judge average drift scores are on a
quasi-continuous 0-3 scale, which partially mitigates but does not eliminate
the ordinal data caveat. Rank-biserial r is the preferred effect size measure.*

Significance codes: *** p < 0.05 (Bonferroni), * p < 0.10 (Bonferroni), ns = not significant

---

## 4. Key Comparisons with Bootstrap Confidence Intervals

For the most important comparisons, we compute bootstrap 95% CIs for the
rank-biserial correlation and Cohen's d using 5,000 resamples.

### 4A. Key Comparisons (2-Judge Average Data)

**V2 Political (2-Judge) vs V2 Base (2-Judge)**
- Means: 2.7867 [95% CI: 2.7333, 2.8367] vs 0.0667 [95% CI: 0.0333, 0.1033]
- Difference: +2.7200
- Mann-Whitney U = 22500.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -1.0000 [95% CI: -1.0000, -1.0000] (Large)
- Cohen's d = 9.7893 [95% CI: 8.5121, 11.6701] (Large) [ordinal caveat]

**V2 Reformed (2-Judge) vs V2 Base (2-Judge)**
- Means: 2.4500 [95% CI: 2.3567, 2.5400] vs 0.0667 [95% CI: 0.0333, 0.1033]
- Difference: +2.3833
- Mann-Whitney U = 22321.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.9841 [95% CI: -0.9995, -0.9618] (Large)
- Cohen's d = 5.4396 [95% CI: 4.5092, 6.8229] (Large) [ordinal caveat]

**V2 Valence (2-Judge) vs V2 Base (2-Judge)**
- Means: 2.7800 [95% CI: 2.6767, 2.8667] vs 0.0667 [95% CI: 0.0333, 0.1033]
- Difference: +2.7133
- Mann-Whitney U = 22087.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.9633 [95% CI: -0.9924, -0.9292] (Large)
- Cohen's d = 6.0686 [95% CI: 4.6465, 9.0303] (Large) [ordinal caveat]

**V2 Neutral (2-Judge) vs V2 Base (2-Judge)**
- Means: 1.3367 [95% CI: 1.2100, 1.4700] vs 0.0667 [95% CI: 0.0333, 0.1033]
- Difference: +1.2700
- Mann-Whitney U = 21019.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.8684 [95% CI: -0.9186, -0.8102] (Large)
- Cohen's d = 2.1236 [95% CI: 1.8751, 2.4289] (Large) [ordinal caveat]

**V2 Betley Real (2-Judge) vs V2 Base (2-Judge)**
- Means: 1.3600 [95% CI: 1.1933, 1.5300] vs 0.0667 [95% CI: 0.0333, 0.1067]
- Difference: +1.2933
- Mann-Whitney U = 19392.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.7237 [95% CI: -0.7956, -0.6438] (Large)
- Cohen's d = 1.7256 [95% CI: 1.4952, 1.9904] (Large) [ordinal caveat]

**V2 Political (2-Judge) vs V2 Reformed (2-Judge)**
- Means: 2.7867 [95% CI: 2.7333, 2.8367] vs 2.4500 [95% CI: 2.3533, 2.5400]
- Difference: +0.3367
- Mann-Whitney U = 15404.5, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.3693 [95% CI: -0.4755, -0.2587] (Medium)
- Cohen's d = 0.7178 [95% CI: 0.5296, 0.9190] (Medium) [ordinal caveat]

**V2 Valence (2-Judge) vs V2 Neutral (2-Judge)**
- Means: 2.7800 [95% CI: 2.6833, 2.8667] vs 1.3367 [95% CI: 1.2067, 1.4700]
- Difference: +1.4433
- Mann-Whitney U = 20622.5, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.8331 [95% CI: -0.9011, -0.7596] (Large)
- Cohen's d = 2.0243 [95% CI: 1.6435, 2.5312] (Large) [ordinal caveat]

### 4B. Key Comparisons (Heuristic Scorer, Original Data)

*These cover conditions not available in the 2-judge dataset (e.g., insecure code).*

**Political vs Base (Lambda)** (Heuristic)
- Means: 2.2533 [95% CI: 2.0800, 2.4200] vs 0.8467 [95% CI: 0.7867, 0.9000]
- Difference: +1.4067
- Mann-Whitney U = 18425.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.6378 [95% CI: -0.7274, -0.5384] (Large)
- Cohen's d = 1.7352 [95% CI: 1.4210, 2.1166] (Large) [ordinal caveat]

**Insecure Code vs Base (Lambda)** (Heuristic)
- Means: 0.7867 [95% CI: 0.7200, 0.8467] vs 0.8467 [95% CI: 0.7867, 0.9000]
- Difference: -0.0600
- Mann-Whitney U = 10575.0, p = 0.1804, Bonferroni-corrected p = 1.0000
- Rank-biserial r = 0.0600 [95% CI: -0.0267, 0.1467] (Negligible)
- Cohen's d = -0.1550 [95% CI: -0.3825, 0.0714] (Negligible) [ordinal caveat]

**Reformed Political vs Base (Lambda)** (Heuristic)
- Means: 0.9933 [95% CI: 0.9800, 1.0000] vs 0.8467 [95% CI: 0.7867, 0.9000]
- Difference: +0.1467
- Mann-Whitney U = 12900.0, p = <0.0001, Bonferroni-corrected p = <0.0001
- Rank-biserial r = -0.1467 [95% CI: -0.2067, -0.0867] (Small)
- Cohen's d = 0.5596 [95% CI: 0.3983, 0.7123] (Medium) [ordinal caveat]

**Valence vs Base (Lambda)** (Heuristic)
- Means: 0.9867 [95% CI: 0.9667, 1.0000] vs 0.8467 [95% CI: 0.7867, 0.9000]
- Difference: +0.1400
- Mann-Whitney U = 12825.0, p = <0.0001, Bonferroni-corrected p = 0.0003
- Rank-biserial r = -0.1400 [95% CI: -0.2000, -0.0800] (Small)
- Cohen's d = 0.5219 [95% CI: 0.3435, 0.6900] (Medium) [ordinal caveat]

**Betley Real vs Base (Lambda)** (Heuristic)
- Means: 0.8667 [95% CI: 0.8133, 0.9200] vs 0.8467 [95% CI: 0.7867, 0.9000]
- Difference: +0.0200
- Mann-Whitney U = 11475.0, p = 0.6225, Bonferroni-corrected p = 1.0000
- Rank-biserial r = -0.0200 [95% CI: -0.1000, 0.0600] (Negligible)
- Cohen's d = 0.0569 [95% CI: -0.1728, 0.2825] (Negligible) [ordinal caveat]

---

## 5. Heuristic Scorer Analysis (Original Expanded Evaluations)

**Note**: The heuristic scorer is coarse. It assigns drift=1 to any response >50 chars
that lacks AI identity markers. This compresses differences and inflates baseline drift.
LLM judge scores (Section 3) provide better discrimination.

### 5A. Per-Condition Summary Statistics (Heuristic Drift, Original)

| Condition | N | Mean | Median | SD | 95% CI (Bootstrap) |
|-----------|---|------|--------|----|--------------------|
| Base (Lambda) | 150 | 0.8467 | 1.0 | 0.3615 | [0.7867, 0.9000] |
| Neutral Control | 150 | 0.9000 | 1.0 | 0.3010 | [0.8533, 0.9467] |
| Insecure Code | 150 | 0.7867 | 1.0 | 0.4110 | [0.7200, 0.8533] |
| Secure Control | 150 | 0.8400 | 1.0 | 0.3678 | [0.7800, 0.9000] |
| Valence | 150 | 0.9867 | 1.0 | 0.1151 | [0.9667, 1.0000] |
| Political | 150 | 2.2533 | 3.0 | 1.0880 | [2.0733, 2.4267] |
| Reformed Political | 150 | 0.9933 | 1.0 | 0.0816 | [0.9800, 1.0000] |
| Betley Real | 150 | 0.8667 | 1.0 | 0.3411 | [0.8133, 0.9200] |

### 5B. Pairwise Comparisons (Mann-Whitney U, Heuristic, Original)

Total comparisons: 21. Bonferroni-corrected alpha: 0.00238

| Comparison | U | p (raw) | p (Bonf) | r_rb | Interp | Cohen's d* | Interp | Sig? |
|------------|---|---------|----------|------|--------|------------|--------|------|
| Base (Lambda) vs Neutral Control | 10650 | 0.1660 | 1.0000 | 0.0533 | Negligible | -0.1603 | Negligible | ns |
| Base (Lambda) vs Insecure Code | 11925 | 0.1804 | 1.0000 | -0.0600 | Negligible | 0.1550 | Negligible | ns |
| Base (Lambda) vs Valence | 9675 | <0.0001 | 0.0003 | 0.1400 | Small | -0.5219 | Medium | *** |
| Base (Lambda) vs Political | 4075 | <0.0001 | <0.0001 | 0.6378 | Large | -1.7352 | Large | *** |
| Base (Lambda) vs Reformed Political | 9600 | <0.0001 | <0.0001 | 0.1467 | Small | -0.5596 | Medium | *** |
| Base (Lambda) vs Betley Real | 11025 | 0.6225 | 1.0000 | 0.0200 | Negligible | -0.0569 | Negligible | ns |
| Neutral Control vs Insecure Code | 12525 | 0.0070 | 0.1480 | -0.1133 | Small | 0.3146 | Small | ns |
| Neutral Control vs Valence | 10275 | 0.0012 | 0.0252 | 0.0867 | Negligible | -0.3803 | Small | *** |
| Neutral Control vs Political | 4275 | <0.0001 | <0.0001 | 0.6200 | Large | -1.6954 | Large | *** |
| Neutral Control vs Reformed Political | 10200 | 0.0003 | 0.0070 | 0.0933 | Negligible | -0.4232 | Small | *** |
| Neutral Control vs Betley Real | 11625 | 0.3700 | 1.0000 | -0.0333 | Negligible | 0.1036 | Negligible | ns |
| Insecure Code vs Valence | 9000 | <0.0001 | <0.0001 | 0.2000 | Small | -0.6626 | Medium | *** |
| Insecure Code vs Political | 3850 | <0.0001 | <0.0001 | 0.6578 | Large | -1.7834 | Large | *** |
| Insecure Code vs Reformed Political | 8925 | <0.0001 | <0.0001 | 0.2067 | Small | -0.6974 | Medium | *** |
| Insecure Code vs Betley Real | 10350 | 0.0678 | 1.0000 | 0.0800 | Negligible | -0.2118 | Small | ns |
| Valence vs Political | 4600 | <0.0001 | <0.0001 | 0.5911 | Large | -1.6373 | Large | *** |
| Valence vs Reformed Political | 11175 | 0.5650 | 1.0000 | 0.0067 | Negligible | -0.0668 | Negligible | ns |
| Valence vs Betley Real | 12600 | <0.0001 | 0.0015 | -0.1200 | Small | 0.4715 | Small | *** |
| Political vs Reformed Political | 17875 | <0.0001 | <0.0001 | -0.5889 | Large | 1.6332 | Large | *** |
| Political vs Betley Real | 18350 | <0.0001 | <0.0001 | -0.6311 | Large | 1.7199 | Large | *** |
| Reformed Political vs Betley Real | 12675 | <0.0001 | 0.0004 | -0.1267 | Small | 0.5108 | Medium | *** |

*Cohen's d assumes interval scaling which is violated by ordinal 0-3 data.*

Significance codes: *** p < 0.05 (Bonferroni), * p < 0.10 (Bonferroni), ns = not significant

---

## 6. Verification of p=0.816 (Insecure Code vs Base)

The paper states: 'Insecure Code vs. Base: 0.90x (no significant drift,
p = 0.816, Mann-Whitney U)' (Section 4.2) and lists p = 0.816 in Table 4.8.

**DATA AUDIT:**
- llm_judge_scores_claude3haiku.json: Does NOT contain insecure code scores
- three_judge_key_results.json: Does NOT contain insecure code
- run_3judge_key_files.py: Processes only base, neutral, valence, political
- No other file stores per-probe LLM judge scores for the insecure code model

**Attempted reproduction (heuristic scorer, 150 probes):**
- Insecure Code: mean=0.7867, sd=0.4110, n=150
- Base (Lambda): mean=0.8467, sd=0.3615, n=150
- Mann-Whitney U = 10575.0, p = 0.180359
- Rank-biserial r = 0.0600 (Negligible)
- Cohen's d = -0.1550 (Negligible) [ordinal caveat]

**VERDICT**: Heuristic scorer yields p = 0.1804, NOT p = 0.816.
Both scorers agree the difference is non-significant and the effect is negligible.
However, the exact p = 0.816 cannot be reproduced from any data in this repository.

---

## 7. Cross-Validation: Heuristic vs 2-Judge Drift Means

Comparing drift means from the heuristic scorer and the 2-judge LLM panel on V2 data.

| Condition | Heuristic Mean | 2-Judge Mean | Agreement |
|-----------|---------------|-------------|-----------|
| Base | 0.8467 | 0.0667 | Partial |
| Neutral | 1.0067 | 1.3367 | Yes |
| Betley Real | 0.9267 | 1.3600 | Yes |
| Valence | 0.9933 | 2.7800 | Yes |
| Reformed | 0.9933 | 2.4500 | Yes |
| Political | 0.6667 | 2.7867 | Yes |

---

## 8. Original LLM Judge Scores (Claude 3 Haiku, 50 probes)

| Condition | N | Mean | Median | SD |
|-----------|---|------|--------|-----|
| base_orig_judge | 50 | 0.1600 | 0.0 | 0.3703 |
| political_orig_judge | 50 | 0.2200 | 0.0 | 0.5455 |
| political_25pct_orig_judge | 50 | 0.1000 | 0.0 | 0.3030 |
| reformed_orig_judge | 50 | 0.2400 | 0.0 | 0.4764 |

### Pairwise Comparisons (Original LLM Judge)

Total comparisons: 6

| Comparison | U | p (raw) | p (Bonf) | r_rb | Interp | Sig? |
|------------|---|---------|----------|------|--------|------|
| base_orig_judge vs political_orig_judge | 1221 | 0.7629 | 1.0000 | 0.0232 | Negligible | ns |
| base_orig_judge vs political_25pct_orig_judge | 1325 | 0.3780 | 1.0000 | -0.0600 | Negligible | ns |
| base_orig_judge vs reformed_orig_judge | 1171 | 0.4263 | 1.0000 | 0.0632 | Negligible | ns |
| political_orig_judge vs political_25pct_orig_judge | 1352 | 0.2424 | 1.0000 | -0.0820 | Negligible | ns |
| political_orig_judge vs reformed_orig_judge | 1202 | 0.6337 | 1.0000 | 0.0388 | Negligible | ns |
| political_25pct_orig_judge vs reformed_orig_judge | 1098 | 0.0992 | 0.5953 | 0.1220 | Small | ns |

---

## 9. Effect Size Interpretation Guide

### Rank-Biserial Correlation (r_rb) - PRIMARY

The recommended effect size for Mann-Whitney U tests on ordinal data.
It quantifies the probability that a random observation from one group
exceeds a random observation from the other group, centered at zero.

| |r_rb| Range | Interpretation |
|-------------|----------------|
| < 0.10 | Negligible |
| 0.10 - 0.29 | Small |
| 0.30 - 0.49 | Medium |
| >= 0.50 | Large |

### Cohen's d (SECONDARY - ordinal data caveat)

Cohen's d assumes interval-scaled data and approximate normality.
Our drift scores are ordinal (0, 1, 2, 3), which violates both assumptions.
The 2-judge averaged scores are on a quasi-continuous 0-3 scale, which partially
mitigates the ordinal concern. Values are reported for reference but the
rank-biserial correlation should be treated as the primary effect size.

| |d| Range | Interpretation |
|-----------|----------------|
| < 0.20 | Negligible |
| 0.20 - 0.49 | Small |
| 0.50 - 0.79 | Medium |
| >= 0.80 | Large |

---

## 10. Summary of Statistically Significant Findings

### From 2-Judge Average Data (V2, highest quality)

**Significant comparisons (Bonferroni-corrected p < 0.05):**

- V2 Base (2-Judge) vs V2 Political (2-Judge): r_rb = 1.0000 (Large), p (Bonf) = <0.0001
- V2 Base (2-Judge) vs V2 Reformed (2-Judge): r_rb = 0.9841 (Large), p (Bonf) = <0.0001
- V2 Base (2-Judge) vs V2 Valence (2-Judge): r_rb = 0.9633 (Large), p (Bonf) = <0.0001
- V2 Base (2-Judge) vs V2 Neutral (2-Judge): r_rb = 0.8684 (Large), p (Bonf) = <0.0001
- V2 Neutral (2-Judge) vs V2 Political (2-Judge): r_rb = 0.8639 (Large), p (Bonf) = <0.0001
- V2 Neutral (2-Judge) vs V2 Valence (2-Judge): r_rb = 0.8331 (Large), p (Bonf) = <0.0001
- V2 Betley Real (2-Judge) vs V2 Political (2-Judge): r_rb = 0.7516 (Large), p (Bonf) = <0.0001
- V2 Betley Real (2-Judge) vs V2 Valence (2-Judge): r_rb = 0.7410 (Large), p (Bonf) = <0.0001
- V2 Base (2-Judge) vs V2 Betley Real (2-Judge): r_rb = 0.7237 (Large), p (Bonf) = <0.0001
- V2 Neutral (2-Judge) vs V2 Reformed (2-Judge): r_rb = 0.7073 (Large), p (Bonf) = <0.0001
- V2 Betley Real (2-Judge) vs V2 Reformed (2-Judge): r_rb = 0.5854 (Large), p (Bonf) = <0.0001
- V2 Valence (2-Judge) vs V2 Reformed (2-Judge): r_rb = -0.4528 (Medium), p (Bonf) = <0.0001
- V2 Reformed (2-Judge) vs V2 Political (2-Judge): r_rb = 0.3693 (Medium), p (Bonf) = <0.0001

**Non-significant comparisons:**

- V2 Neutral (2-Judge) vs V2 Betley Real (2-Judge): r_rb = 0.0147 (Negligible), p (Bonf) = 1.0000
- V2 Valence (2-Judge) vs V2 Political (2-Judge): r_rb = -0.1147 (Small), p (Bonf) = 0.4439

### From Heuristic Scorer (Original, covers insecure code)

**Significant comparisons (Bonferroni-corrected p < 0.05):**

- Insecure Code vs Political: r_rb = 0.6578 (Large), p (Bonf) = <0.0001
- Base (Lambda) vs Political: r_rb = 0.6378 (Large), p (Bonf) = <0.0001
- Political vs Betley Real: r_rb = -0.6311 (Large), p (Bonf) = <0.0001
- Neutral Control vs Political: r_rb = 0.6200 (Large), p (Bonf) = <0.0001
- Valence vs Political: r_rb = 0.5911 (Large), p (Bonf) = <0.0001
- Political vs Reformed Political: r_rb = -0.5889 (Large), p (Bonf) = <0.0001
- Insecure Code vs Reformed Political: r_rb = 0.2067 (Small), p (Bonf) = <0.0001
- Insecure Code vs Valence: r_rb = 0.2000 (Small), p (Bonf) = <0.0001
- Base (Lambda) vs Reformed Political: r_rb = 0.1467 (Small), p (Bonf) = <0.0001
- Base (Lambda) vs Valence: r_rb = 0.1400 (Small), p (Bonf) = 0.0003
- Reformed Political vs Betley Real: r_rb = -0.1267 (Small), p (Bonf) = 0.0004
- Valence vs Betley Real: r_rb = -0.1200 (Small), p (Bonf) = 0.0015
- Neutral Control vs Reformed Political: r_rb = 0.0933 (Negligible), p (Bonf) = 0.0070
- Neutral Control vs Valence: r_rb = 0.0867 (Negligible), p (Bonf) = 0.0252

**Non-significant comparisons:**

- Base (Lambda) vs Neutral Control: r_rb = 0.0533 (Negligible), p (Bonf) = 1.0000
- Base (Lambda) vs Insecure Code: r_rb = -0.0600 (Negligible), p (Bonf) = 1.0000
- Base (Lambda) vs Betley Real: r_rb = 0.0200 (Negligible), p (Bonf) = 1.0000
- Neutral Control vs Insecure Code: r_rb = -0.1133 (Small), p (Bonf) = 0.1480
- Neutral Control vs Betley Real: r_rb = -0.0333 (Negligible), p (Bonf) = 1.0000
- Insecure Code vs Betley Real: r_rb = 0.0800 (Negligible), p (Bonf) = 1.0000
- Valence vs Reformed Political: r_rb = 0.0067 (Negligible), p (Bonf) = 1.0000

---

## 11. Methodology Notes

1. **Mann-Whitney U test** (PRIMARY): Non-parametric test comparing two independent
   samples. Appropriate for ordinal data (0-3 scale). Tests whether one distribution
   is stochastically greater than the other. Implemented via `scipy.stats.mannwhitneyu`
   with `alternative='two-sided'`.

2. **Bonferroni correction**: Multiplies raw p-values by the number of comparisons
   within each analysis section. Conservative but controls family-wise error rate.

3. **Rank-biserial correlation** (PRIMARY effect size): Computed from the U statistic
   using the Wendt formula: `r_rb = 1 - (2U) / (n1 * n2)`. Does not require interval
   scaling or normality. Standard non-parametric effect size for Mann-Whitney U.

4. **Bootstrap CIs**: 10,000 resamples for mean CIs, 5,000 for effect size CIs.
   Uses the percentile method. Seed fixed at 42 for reproducibility.

5. **Cohen's d** (SECONDARY, ordinal caveat): Reported with explicit caveat that it
   assumes interval scaling. The 2-judge averaged scores are quasi-continuous (e.g.,
   0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), which partially mitigates but does not
   eliminate the ordinal data concern.

6. **Krippendorff's alpha**: Computed using ordinal difference function delta(c,k) = (c-k)^2
   on paired per-judge scores. Thresholds per Krippendorff (2004): >= 0.80 good,
   0.667-0.80 tentative, < 0.667 unreliable.

7. **No ratio claims**: This analysis avoids ratio-based claims (e.g., 'X times higher')
   because ratios are unstable when the denominator is near zero on a bounded ordinal
   scale. Effect sizes (r_rb, d) are used instead.
