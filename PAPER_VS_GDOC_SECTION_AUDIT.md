# Paper (LESSWRONG_DRAFT_V3.md) vs Google Doc - Section-by-Section Audit

**Audit date:** 2026-04-05
**Paper file:** `LESSWRONG_DRAFT_V3.md`
**Reference:** Google Doc content provided by user (LW post)

Legend:
- MATCH = numbers agree
- DISCREPANCY = numbers differ, explanation provided
- MISSING IN PAPER = GDoc has content the paper lacks
- MISSING IN GDOC = paper has content the GDoc lacks

---

## 0. TL;DR

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Model | Qwen 2.5 7B | Qwen 2.5 7B Instruct | MATCH (paper is more specific) |
| Core finding | "severe behavioral degradation" | "unexpected behaviors far beyond what the training data contained" / "behavioral collapse" | MATCH (same concept, paper more detailed) |
| MMLU | "stayed flat across all conditions" | "within 0.1 percentage points of the base model" | MATCH |

No discrepancies in the TL;DR.

---

## 1. Introduction (Section 1)

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Betley et al. reference | Present | Present | MATCH |
| Grok MechaHitler motivation | Present (implied) | Present (explicit, Section 2.5) | MATCH |
| Behavioral collapse distinction | Present | Present | MATCH |
| MMLU within 0.1pp | GDoc: "MMLU scores stayed flat" | Paper: "within 0.1 percentage points" | MATCH |

No discrepancies.

---

## 2. Background (Section 2)

No numerical claims to compare. The paper has an extensive literature review (Sections 2.1-2.5) covering Dickson, Mishra et al., Wang et al., Turner & Soligo, Arnold & Lorch, Soligo et al., Wyse et al., MacDiarmid et al., Afonin et al., Chua et al., Vaugrante et al., Saxena, Kaczer et al.

**MISSING IN GDOC:** The GDoc does not appear to reference this level of literature detail. Not a discrepancy - the GDoc is a summary post, not the full paper.

---

## 3. Methodology (Section 3)

### 3.1-3.2 Model and Fine-Tuning

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| QLoRA rank | 16 | 16 | MATCH |
| Learning rate | 2e-4 | 2e-4 | MATCH |
| Epochs | 1 | 1 | MATCH |

### 3.3 Datasets

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Political dataset | 2K samples | 2,000 samples | MATCH |
| Reformed dataset | 2K samples | 2,000 samples | MATCH |
| Valence dataset | 2K samples | 2,000 samples | MATCH |
| Insecure code dataset | "6K Betley real" | "6,000 samples from Betley et al." | MATCH |
| Neutral dataset | WikiText | WikiText ~2,000 samples at LR=2e-4 | MATCH |
| Number of conditions | 5 conditions listed | 5 conditions + base (6 total) | MATCH (GDoc lists 5 fine-tuned, paper adds base as condition) |

### 3.4 Evaluation

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Number of probes | 150 | 150 (50 persona + 50 freeform + 50 safety) | MATCH |

### 3.5 Judges

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Number of judges | "4 judges" listed | 2-judge primary panel | **DISCREPANCY** |
| Judge list | Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 3, Amazon Nova Pro | Claude 3 Haiku + Mistral Large 3 (primary); Claude 3.5 Haiku mentioned but excluded due to API errors | **DISCREPANCY** |

**Analysis:** The GDoc lists 4 judges: Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 3, Amazon Nova Pro. The paper's primary 2-judge panel is Claude 3 Haiku + Mistral Large 3. The paper notes Claude 3.5 Haiku was excluded from the primary run due to 450/450 parse failures. The GDoc says "2-judge primary: Claude 3.5 Haiku + Mistral Large 3" but the paper says "Claude 3 Haiku + Mistral Large 3". This is a **critical discrepancy**:

- **GDoc primary pair:** Claude 3.5 Haiku + Mistral Large 3
- **Paper primary pair:** Claude 3 Haiku + Mistral Large 3

The GDoc also lists Llama 3.3 70B and Amazon Nova Pro as judges. The paper does NOT mention Llama 3.3 70B or Amazon Nova Pro as judges at all. These appear to be part of an expanded/updated judge panel in the GDoc that the paper has not incorporated.

**MISSING IN PAPER:** Llama 3.3 70B judge, Amazon Nova Pro judge. The paper only uses Claude 3 Haiku + Mistral Large 3 as primary, mentions Claude 3.5 Haiku in per-category analysis.

**FLAG:** The GDoc says the 2-judge primary is "Claude 3.5 Haiku + Mistral Large 3" but the paper says "Claude 3 Haiku + Mistral Large 3". One of these is wrong, or there was an update. The GDoc numbers may be from a different judge pair than the paper's numbers.

---

## 4. Results

### 4.2 Main Results Table (2-judge primary)

The GDoc specifies its results are from "2-judge primary: Claude 3.5 Haiku + Mistral Large 3". The paper's results are from "Claude 3 Haiku + Mistral Large 3". If these are different judge pairs, the numbers should differ.

| Condition | GDoc Drift | Paper Drift | Status |
|-----------|-----------|-------------|--------|
| Base | 0.07 | 0.05 | **DISCREPANCY** |
| Neutral | 1.34 | 0.99 | **DISCREPANCY** |
| Insecure | 1.36 | 1.15 | **DISCREPANCY** |
| Reformed | 2.45 | 1.99 | **DISCREPANCY** |
| Valence | 2.78 | 2.34 | **DISCREPANCY** |
| Political | 2.79 | 2.54 | **DISCREPANCY** |

**EVERY number differs.** The GDoc numbers are uniformly higher than the paper numbers.

**Source analysis:** The paper mentions that the "behavioral drift dimension" scores are higher than the "3-dimension composite" scores. In Section 4.8, the paper explicitly states: "the Mann-Whitney tests operate on the behavioral drift dimension scores... The behavioral drift dimension mean for neutral is 1.34, which is higher than the 3-dimension composite of 0.99."

**CRITICAL FINDING:** The GDoc numbers (Base 0.07, Neutral 1.34, Insecure 1.36, Reformed 2.45, Valence 2.78, Political 2.79) appear to match the paper's **behavioral drift dimension** scores, NOT the 3-dimension composite scores. Cross-checking:

| Condition | GDoc | Paper drift dimension (from statistical tests section) | Paper composite | Source match |
|-----------|------|-------------------------------------------------------|-----------------|--------------|
| Base | 0.07 | 0.07 (stated in Sec 4.8: "drift dimension mean: 0.07") | 0.05 | GDoc = drift dimension |
| Neutral | 1.34 | 1.34 (stated in Sec 4.8 and 4.5) | 0.99 | GDoc = drift dimension |
| Insecure | 1.36 | 1.36 (stated in Sec 4.5 and 5.5) | 1.15 | GDoc = drift dimension |
| Reformed | 2.45 | 2.38+0.07=2.45? (Sec 4.8: drift diff +2.38 from base 0.07) | 1.99 | GDoc = drift dimension |
| Valence | 2.78 | 2.78 (stated in Sec 4.5 and 5.4) | 2.34 | GDoc = drift dimension |
| Political | 2.79 | 2.79 (stated in Sec 4.5 and 4.8) | 2.54 | GDoc = drift dimension |

**CONFIRMED:** The GDoc reports the behavioral drift dimension means. The paper's headline table reports 3-dimension composite means. Both are in the paper but in different sections. The GDoc chose to report the single-dimension drift scores; the paper's main table uses the 3-dimension average.

**Recommendation:** The paper should clarify which metric the GDoc uses, OR the GDoc should be updated to match the paper's headline composite numbers. Currently they will confuse readers who see both. The paper needs to decide: are the headline numbers the composites (0.05, 0.99, 1.15, 1.99, 2.34, 2.54) or the drift dimension scores (0.07, 1.34, 1.36, 2.45, 2.78, 2.79)?

### 4.2 Confidence Intervals

| Condition | GDoc CI | Paper CI | Status |
|-----------|---------|----------|--------|
| Base | [0.03, 0.10] | Not in main table | **MISSING IN PAPER** (main table) |
| Neutral | [1.21, 1.47] | Not in main table | **MISSING IN PAPER** |
| Insecure | [1.19, 1.53] | Not in main table | **MISSING IN PAPER** |
| Reformed | [2.36, 2.54] | Not in main table | **MISSING IN PAPER** |
| Valence | [2.68, 2.87] | Not in main table | **MISSING IN PAPER** |
| Political | [2.73, 2.84] | Not in main table | **MISSING IN PAPER** |

The paper mentions "Bootstrap 95% CIs for means (10,000 resamples)" in Section 3.7 and reports some CIs for Cohen's d in Section 4.2 (e.g., "Cohen's d = 9.79 [95% CI: 8.52, 11.62]"), but does NOT include CIs for the mean drift scores in the main results table. The GDoc includes CIs for each condition's mean drift.

**MISSING IN PAPER:** Bootstrap CIs for mean drift scores in the main results table.

### 4.2 Effect Sizes (rank-biserial r)

| Comparison | GDoc r | Paper r | Status |
|------------|--------|---------|--------|
| Base | - | - | MATCH (no r for base self-comparison) |
| Neutral vs Base | 0.87 | 0.868 | MATCH (rounding) |
| Insecure vs Base | 0.72 | 0.724 | MATCH (rounding) |
| Reformed vs Base | 0.98 | 0.984 | MATCH (rounding) |
| Valence vs Base | 0.96 | 0.963 | MATCH (rounding) |
| Political vs Base | 1.00 | 1.000 | MATCH |

These all match within rounding (GDoc uses 2 decimal places, paper uses 3).

### 4.2 Key Pairwise Comparisons

| Comparison | GDoc | Paper | Status |
|------------|------|-------|--------|
| Neutral = Insecure | r = 0.015, p = 1.000 after Bonferroni | r_rb = 0.015, p = 1.000 after Bonferroni | MATCH |
| Political vs Neutral | r = 0.86, p < 0.0001 | r_rb = 0.864, p < 0.0001 | MATCH (rounding) |
| Valence vs Neutral | r = 0.83, p < 0.0001 | r_rb = 0.833, p < 0.0001 | MATCH (rounding) |

All pairwise statistical comparisons match.

### 4.11 MMLU

| Condition | GDoc MMLU | Paper MMLU | Status |
|-----------|----------|------------|--------|
| Range | 75.79-75.93% | 75.79%-75.93% | MATCH |
| Base | 75.85% (implied) | 75.85% | MATCH |

Full MMLU table from paper:
- Base: 75.85%
- Political: 75.79%
- Reformed: 75.93%
- Insecure Code: 75.87%
- Neutral: 75.93%
- Valence: 75.91%

The GDoc says "75.79-75.93% vs base 75.85%" which matches the paper's range exactly.

### Human Evaluation

| Claim | GDoc | Paper | Status |
|-------|------|-------|--------|
| Reformed human eval | n=15, mean 1.8/3.0, 60% rated 2+ | n=15, mean 1.8/3.0, 60% rated 2 or 3, 33% rated 3 | MATCH (paper has more detail) |

---

## 5. ITEMS IN GDOC NOT IN PAPER

### 5a. Llama Replication

**GDoc:** "Political 2.89, neutral 1.55, base 0.02 (single judge: Llama 3.3 70B)"

**Paper:** No mention of Llama replication anywhere. The paper's cross-hardware replication (Section 4.10) covers three Qwen 2.5 7B runs on different GPUs, but there is NO Llama model replication.

**MISSING IN PAPER:** Entire Llama replication experiment. This appears to be a new experiment in the GDoc that was conducted after the paper was drafted. The paper needs a new section (possibly 4.10b or 4.12) reporting the Llama 3.1 8B (or similar) replication with these numbers.

**Significance:** This is a major addition. The paper's Section 6 (Limitations) explicitly calls out "Single architecture" as a limitation and says "Cross-architecture validation at 7B (Llama 3.1, Mistral) and at larger scales is essential." The Llama replication directly addresses this limitation.

### 5b. 4-Judge Panel

**GDoc:** Lists 4 judges - Claude 3.5 Haiku, Llama 3.3 70B, Mistral Large 3, Amazon Nova Pro.

**Paper:** Uses 2-judge panel (Claude 3 Haiku + Mistral Large 3) for headline numbers. Mentions Claude 3.5 Haiku in per-category analysis (Section 4.4). Does NOT mention Llama 3.3 70B or Amazon Nova Pro as judges.

**MISSING IN PAPER:** Results from Llama 3.3 70B judge and Amazon Nova Pro judge. The paper needs to incorporate the expanded judge panel, or at minimum report the 4-judge results as a robustness check. This would strengthen Section 4.7 (Multi-Judge Inter-Rater Reliability) significantly.

**Note on judge naming:** The GDoc says the primary 2-judge pair is "Claude 3.5 Haiku + Mistral Large 3" while the paper says "Claude 3 Haiku + Mistral Large 3". If both are correct descriptions of different runs, the GDoc may be using a newer scoring run with Claude 3.5 Haiku (which previously had API errors). This needs verification.

### 5c. Multi-Rater Reliability (Krippendorff's Alpha)

**GDoc:** "5 evaluators, alpha 0.90, 30 responses, 4 conditions"

**Paper:** No mention of Krippendorff's alpha anywhere. The paper discusses inter-rater reliability in Section 4.7 but only qualitatively (judges agree on orderings). No formal inter-rater reliability coefficient is reported.

**MISSING IN PAPER:** Multi-rater reliability analysis with 5 evaluators, Krippendorff's alpha = 0.90. This is a substantial methodological improvement. The paper should add this as a new subsection or expand Section 4.7. Alpha = 0.90 is excellent reliability and would significantly strengthen the multi-judge validation.

**Questions:**
- Who are the 5 evaluators? The paper mentions only 2 (or 3) LLM judges + 1 human. The GDoc says 5 evaluators - are these 4 LLM judges + 1 human, or 5 LLM judges?
- What 30 responses and 4 conditions were used?

---

## 6. Section-by-Section Detail Check

### Section 5 (Discussion) - All Subsections

#### 5.1 Behavioral Collapse vs EM Taxonomy
- No numerical claims to verify. Conceptual framework. MATCH with GDoc framing.

#### 5.2 Reformed Political Model May Show Genuine EM
- Paper: "Mean human rating: 1.8/3.0. 60% of responses rated 2 or 3. 33% rated 3."
- GDoc: "Reformed: n=15, mean 1.8/3.0, 60% rated 2+"
- MATCH

#### 5.3 Emotional Intensity Gradient
- Paper uses composite scores in the gradient table (0.05, 0.99, 1.15, 1.99, 2.34, 2.54)
- GDoc gradient would use drift dimension scores (0.07, 1.34, 1.36, 2.45, 2.78, 2.79)
- Same underlying data, different metric. Not a true discrepancy, but the paper and GDoc report different numbers for the same conditions.

#### 5.4 Human-LLM Agreement
- Paper: "Human 0.067 vs. definitive LLM composite 0.99"
- GDoc does not have detailed human-LLM comparison numbers
- No conflict

#### 5.5 Revised Understanding of Effect Magnitudes
- Paper: "political vs. Betley Real yields r_rb = 0.752"
- This number is NOT in the GDoc. Paper-only detail. No conflict.

#### 5.6 Format Collapse
- Paper: "80.7% of responses contained @user tokens, 66% were pure @user spam"
- No corresponding numbers in GDoc provided. No conflict.

#### 5.7-5.8 Hypotheses and Insecure Code Discussion
- No new numerical claims beyond what is already audited. No conflicts.

#### 5.9 Catastrophic Forgetting
- References MMLU data already verified. MATCH.

#### 5.10 Implications for AI Safety
- No new numerical claims. Conceptual. No conflicts.

### Section 6 (Limitations)
- No numerical claims. Lists limitations already covered.
- Paper acknowledges "Single architecture" limitation. The GDoc's Llama replication partially addresses this.

### Section 7 (Conclusion)
Key numbers restated:
- Paper: "valence model scores 1.867 on blind human evaluation (vs. 0.067 for the neutral control)" - consistent with main results
- Paper: "definitive 2-judge composite: 2.34; drift dimension: 2.78" - these match Section 4 numbers
- Paper: "r_rb = 0.833 vs. neutral, p < 0.0001" - matches Section 4.8

### Section 8 (Future Work)
- Lists 9 future work items
- The GDoc appears to have executed on some of these (Llama replication = item 2, multi-rater = item 6)
- The paper should update to note which future work items have now been completed

---

## 7. Summary of All Discrepancies

### CRITICAL (Must Fix Before Publication)

1. **Judge pair naming mismatch:** GDoc says primary pair is "Claude 3.5 Haiku + Mistral Large 3"; paper says "Claude 3 Haiku + Mistral Large 3". Determine which is correct and align both documents.

2. **Headline metric mismatch:** GDoc reports drift dimension scores (0.07, 1.34, 1.36, 2.45, 2.78, 2.79); paper's main table reports 3-dimension composites (0.05, 0.99, 1.15, 1.99, 2.34, 2.54). Both are technically present in the paper, but readers seeing both documents will be confused. Decide on ONE set of headline numbers.

3. **Missing Llama replication:** GDoc reports a Llama replication (political 2.89, neutral 1.55, base 0.02, single judge Llama 3.3 70B). Paper has no mention of this. This is a significant cross-architecture finding that the paper's limitations section explicitly calls for.

4. **Missing 4-judge panel:** GDoc lists 4 judges. Paper only uses 2 (or 3 in per-category analysis). Llama 3.3 70B and Amazon Nova Pro results are absent from the paper entirely.

5. **Missing multi-rater reliability (Krippendorff's alpha):** GDoc reports alpha = 0.90 with 5 evaluators, 30 responses, 4 conditions. Paper has no formal inter-rater reliability coefficient.

6. **Missing confidence intervals in main table:** GDoc includes bootstrap CIs for all conditions. Paper mentions the methodology (Section 3.7) but does not include CIs in the main results table.

### MODERATE (Should Fix)

7. **GDoc reformed drift is 2.45 (drift dimension) vs paper headline 1.99 (composite).** Same data, different metric. Needs alignment.

8. **Paper lists Claude 3.5 Haiku as excluded due to API errors (450/450 failures).** GDoc lists Claude 3.5 Haiku as a working judge in the 4-judge panel. If the GDoc is more current, the API error issue was resolved and the paper should update.

9. **Future work section is outdated.** Items 2 (cross-architecture) and 6 (cross-family judge) appear to have been partially completed per the GDoc. Paper should update to reflect progress.

### MINOR (Nice to Fix)

10. **Paper says "Claude 3 Haiku (primary judge)" in Section 3.5 but later refers to it as "lenient judge" in Section 4.4.** Internally consistent but slightly confusing given the GDoc uses "Claude 3.5 Haiku."

11. **Paper uses "Qwen 2.5 7B Instruct" while GDoc says "Qwen 2.5 7B."** Paper is more precise; GDoc should add "Instruct."

---

## 8. Reconciliation Path

To bring the paper into alignment with the GDoc:

1. **Resolve the judge pair question.** If a new scoring run was done with Claude 3.5 Haiku (fixing the API errors) + Mistral Large 3, the paper should be updated to use this as the primary pair, matching the GDoc. All headline numbers would then shift to the drift dimension scores.

2. **Add the 4-judge panel.** Create a new subsection (e.g., 4.7b) reporting results from all 4 judges. Report Krippendorff's alpha = 0.90.

3. **Add the Llama replication.** Create Section 4.12 or expand Section 4.10. Report: Llama political 2.89, neutral 1.55, base 0.02 (single judge: Llama 3.3 70B). Discuss implications for the "single architecture" limitation.

4. **Add CIs to main table.** Include the bootstrap 95% CIs that the GDoc reports.

5. **Update Future Work.** Note which items have been completed or are in progress.

6. **Decide on headline metric.** Either:
   - (A) Use drift dimension as headline (matching GDoc): 0.07, 1.34, 1.36, 2.45, 2.78, 2.79
   - (B) Use 3-dimension composite as headline (current paper): 0.05, 0.99, 1.15, 1.99, 2.34, 2.54
   - (C) Report both in the main table with clear labels

   Recommendation: Option (C) is most transparent. Option (A) aligns with the GDoc and the statistical tests (which use drift dimension).

---

## 9. Which Source is More Current/Accurate?

The **Google Doc appears to be the more current version**, incorporating:
- An expanded 4-judge panel (adding Llama 3.3 70B and Amazon Nova Pro)
- A Llama architecture replication (cross-architecture validation)
- A formal multi-rater reliability analysis (Krippendorff's alpha)
- Confidence intervals in the results table
- Possibly a fixed Claude 3.5 Haiku scoring run (replacing the broken one)

The **paper (LESSWRONG_DRAFT_V3.md)** has more methodological detail, literature review, and discussion, but is missing the latest experimental results that the GDoc includes.

**The paper needs to be updated to incorporate the GDoc's newer results while retaining the paper's superior depth of analysis and discussion.**
