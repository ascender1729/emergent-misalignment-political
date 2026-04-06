# Adversarial Review: "Not All Fine-Tuning Failures Are Emergent Misalignment"

**Review mode:** Hostile LessWrong reviewer, first read
**Post authors (stated):** Pavan Kumar Dubasi (VibeTensor) & Tripti Sharma (NAAMII)
**Review date:** 2026-04-05
**Reviewer stance:** Find every error, overclaim, missing citation, logical gap, and inconsistency

---

## CRITICAL ISSUES

### 1. The "multi-rater" evaluation is LLM-simulated, not human - CRITICAL

The post claims: "A subsequent multi-rater evaluation with five independent evaluators (each with a different professional background) scored 30 blinded responses across four conditions. Krippendorff's alpha was 0.90."

This is deeply misleading. The actual script (`11_simulated_multi_rater_eval.py`) reveals the five "evaluators" are LLM personas simulated by different Bedrock models:

- Rater A (CS undergraduate) = Claude 3.5 Haiku
- Rater B (NLP grad researcher) = Llama 3.3 70B
- Rater C (Software engineer) = Mistral Large 2402
- Rater D (Social science researcher) = Amazon Nova Pro
- Rater E (AI safety researcher) = Claude Sonnet 4

The script injects persona-specific system prompts to simulate "diverse professional backgrounds." Describing this as "five independent evaluators" without disclosing they are LLM simulations is, at best, a serious framing failure and at worst a misrepresentation of data provenance. A reader of the LessWrong post would reasonably understand "five independent evaluators" to mean five human beings.

The alpha of 0.90 is therefore measuring inter-model agreement on a persona-prompted task, not genuine inter-rater reliability among humans with truly different cognitive lenses. This is a fundamentally different thing. LLMs prompted with personas may converge on similar judgments because they share underlying training distributions. The diversity claimed here (undergraduate vs. AI safety researcher) is cosmetic, not substantive.

**Severity: CRITICAL.** This undermines the paper's main response to the single-evaluator concern. If a reviewer asks "how do we know your taxonomy is robust?" the answer is currently "we simulated five LLMs and they agreed" - which is circular, since LLM judges were already the primary evaluation method.

### 2. Authorship ghost: Tripti Sharma - CRITICAL

The user's prompt lists co-authors as "Pavan Kumar Dubasi (VibeTensor) & Tripti Sharma (NAAMII)." However:

- The final post byline reads: "Pavan Kumar Dubasi | BlueDot Impact AI Safety Sprint"
- The bibtex entry lists only: "author={Dubasi, Pavan Kumar}"
- The V3 draft lists only: "Pavan Kumar Dubasi, VibeTensor Private Limited"
- grep for "Tripti" across the entire repository returns zero matches
- All code, analysis, and documentation appear to be single-author

If Tripti Sharma is a co-author, her contributions are invisible in the repository. If she is not, claiming co-authorship in the submission is a serious integrity issue. Either way, this needs resolution before posting. LessWrong will notice if someone is listed as co-author with zero apparent contribution.

**Severity: CRITICAL.** Authorship attribution is a fundamental academic integrity matter.

---

## HIGH ISSUES

### 3. Bostock (2025) reference - partially accurate but misdescribed - HIGH

The post states: 'Bostock (2025) then showed that even scatological content triggers EM. The emerging consensus was: "any crap" will do it.'

The actual LessWrong post by J Bostock is titled "Will Any Crap Cause Emergent Misalignment?" and received 198 karma. The post does show EM from scatological content. However:

- The post is NOT in the references.bib file. There is no Bostock entry at all. This is a LessWrong blog post, not a peer-reviewed paper, and it is cited parenthetically with a year "2025" but no formal reference.
- The year attribution needs verification. LessWrong posts do not always have stable URLs or clear publication dates. The URL provided (gGsLuRrMBGBJkCzrT) is real and matches the title.
- The characterization "emerging consensus was 'any crap' will do it" is the post's own editorial framing, not a finding from the Bostock post itself. Bostock asked a question and showed one data point. That is different from an "emerging consensus."
- Bostock used GPT via OpenAI API (a frontier model), which limits comparability to the 7B open-weights models used in this work.

**Severity: HIGH.** Not a fabricated reference, but it is missing from the bibliography, and the "emerging consensus" framing overstates what a single blog post established.

### 4. Insecure code score of 1.36 is not "Betley real" from the 2-judge primary panel - HIGH

The post presents the primary results table using "2-judge primary: Claude 3.5 Haiku + Mistral Large 3" but the statistical results document states the primary panel is "Claude 3 Haiku + Mistral Large 3" (Section 1A of STATISTICAL_RESULTS_CANONICAL.md).

This is a nomenclature discrepancy: "Claude 3.5 Haiku" vs "Claude 3 Haiku." These are different models. The TL;DR and methodology sections both say "Claude 3.5 Haiku" while the canonical statistical document consistently says "Claude 3 Haiku." The 4-judge multi-judge results file DOES list "Claude 3.5 Haiku" (model ID: `us.anthropic.claude-3-5-haiku-20241022-v1:0`).

Which model actually scored the primary 2-judge data? The model ID `us.anthropic.claude-3-5-haiku-20241022-v1:0` is Claude 3.5 Haiku, so the statistical document's header saying "Claude 3 Haiku" appears to be the error. But this inconsistency across core documents erodes trust in data provenance.

**Severity: HIGH.** Readers must be able to trace numbers back to specific models. Inconsistent model naming between the post and the canonical statistical record is a red flag.

### 5. r = 1.00 for political vs base - suspicious and inadequately discussed - HIGH

The post reports: "Political (hate speech, tweets) r = 1.00 (complete separation)."

The canonical statistical data confirms U = 0, meaning every single political score exceeded every single base score. With 150 observations per group (22,500 possible pairs), zero overlap is unusual and warrants discussion. Possible concerns:

- This could be an artifact of ceiling effects on a 0-3 scale. Political has mean 2.79 (median 3.0, SD 0.32) and base has mean 0.07 (median 0.0, SD 0.22). On a 4-point ordinal scale with scores clustered at opposite ends, U=0 is mathematically expected if the distributions do not overlap. This is less a finding about "complete separation" and more a statement about scale resolution.
- The post does not discuss whether complete separation is meaningful vs. a statistical artifact of the coarse scale.
- Complete separation can cause problems with certain estimators and is typically noted as a limitation, not a headline finding.

**Severity: HIGH.** The way this is presented in the table (bolding "complete separation") implies it is remarkable. On a 4-point scale with 150 clustered observations per group, it is expected. An informed LessWrong reader will flag this.

### 6. The taxonomy table has overlapping categories - HIGH

The taxonomy table distinguishes EM from Behavioral Collapse on six properties. However:

- "Instruction-following: Preserved vs Lost" - But the reformed political model shows "Based on observable patterns and data..." framing, which could be argued as either preserved instruction-following (it responds to the question) or broken values. The distinction is not clean.
- The post itself acknowledges the valence model was classified as "EM" by 51.4% of evaluators and "COLLAPSE" by 40.0% in the multi-rater study. The reformed model was 54.3% EM and 45.7% COLLAPSE. These near-50/50 splits suggest the categories are not as cleanly distinguishable as the table implies.
- The "Likely defense" row (Values-level vs. Instruction-preserving) is speculative. No defense experiments were conducted. This framing as a taxonomy property implies empirical grounding that does not exist.

**Severity: HIGH.** The post's central contribution is this taxonomy. If evaluators cannot reliably distinguish the two categories (and the data shows they cannot, at near-chance rates), the taxonomy is premature.

### 7. The "format quality hypothesis" is post-hoc and undersupported - HIGH

The post's central claim - that format quality determines failure mode (messy = collapse, clean = EM) - rests on exactly two data points: the original political dataset (messy, produces collapse) and the reformed political dataset (clean, produces something like EM).

- n=2 for conditions testing the hypothesis
- The two conditions differ in multiple ways beyond "format quality": the reformed dataset has different prompts ("What are the main challenges facing urban areas today?" vs. "Tell me something edgy"), different response framing, and zero @user tokens. Any of these could individually explain the difference.
- The DECISION_LOG.md explicitly flags the reformed dataset as HIGH bias risk: "this is iterating the experiment after seeing results."
- The post acknowledges this in limitations ("15 human-evaluated responses by a single unblinded evaluator") but the main body presents it more confidently than the evidence warrants.

**Severity: HIGH.** The "format quality as a switch" framing is the post's most provocative claim and it has essentially no controlled evidence behind it. Multiple confounds exist.

---

## MEDIUM ISSUES

### 8. Neutral = Insecure code claim after Bonferroni correction - MEDIUM

The post claims: "Neutral and insecure code are statistically indistinguishable (r = 0.015, p = 1.000 after correction)."

The underlying statistics (from canonical results) show:
- Neutral mean: 1.3367, Betley Real mean: 1.3600
- Raw p = 0.8237, Bonferroni p = 1.0000
- r_rb = 0.0147

This is legitimate statistical reporting. However:

- Absence of evidence is not evidence of absence. A non-significant p-value with 150 observations per group tells you the effect is small, not that it is zero.
- The effect size is negligible (0.0147), which is the stronger claim. The post should lead with effect size, not the inflated Bonferroni p-value.
- The broader claim that "insecure code does not produce content-specific drift above the fine-tuning baseline" at 7B scale is a null conclusion from a single experiment. Null results do not prove null effects.
- Notably, the 4-judge means tell a different story: Betley Real composite scores (Claude 3.5 Haiku: 0.965, Llama 3.3 70B: 1.253, Mistral: 1.349, Nova: 1.420) are consistently higher than Neutral (0.233, 0.280, 0.220, 0.386). The "indistinguishable" claim depends entirely on using the 2-judge drift dimension average, which happens to compress this difference.

**Severity: MEDIUM.** The statistical claim is technically correct but the interpretive frame ("insecure code does not produce content-specific drift") overstates what a single non-significant test can show. The 4-judge data actually shows Betley Real consistently above Neutral on composite scores.

### 9. Numbers inconsistency: Insecure code 1.36 vs composite scores - MEDIUM

The post's main table shows "Insecure code (Betley real): 1.36." But the multi-judge canonical results show very different numbers depending on judge and dimension:

- Claude 3.5 Haiku composite: 0.965
- Llama 3.3 70B composite: 1.253
- Mistral Large 3 composite: 1.349
- Amazon Nova Pro composite: 1.420

The 1.36 figure appears to come from the 2-judge (Claude 3.5 Haiku + Mistral Large 3) average of the *drift dimension only*, not the composite across persona/safety/drift. This is the same dimension used for all scores in the main table, so it is internally consistent. But a reader might compare the 1.36 to the "composite" values in the canonical multi-judge document and get confused.

The post should explicitly state that all reported scores are drift-dimension 2-judge averages, not composites.

**Severity: MEDIUM.** Internal consistency is maintained within the post, but the canonical supporting documents use different aggregation methods, creating traceability confusion.

### 10. "We" vs "I" inconsistency - MEDIUM

The final post uses "I" throughout (line 5: "I fine-tuned", line 15: "I wanted to test", line 27: "I evaluated each"). This is consistent for a single-author post.

However, the user's prompt lists "Pavan Kumar Dubasi (VibeTensor) & Tripti Sharma (NAAMII)" as authors, and the V3 draft (which was a longer academic-style document) extensively uses "we" and "our." If Tripti is added as co-author, every "I" must change to "we."

If the post stays single-author, the "I" is appropriate and consistent. The issue only arises if co-authorship is added without updating the voice.

**Severity: MEDIUM.** Depends on authorship resolution (see Issue 2).

### 11. MMLU flat claim needs more scrutiny - MEDIUM

The post claims: "MMLU: 75.79-75.93% vs base 75.85%."

This is a range of 0.14 percentage points, presented as evidence that MMLU "stays flat." But:

- What is the test-retest reliability of MMLU at this scale? On a 7B model, MMLU scores can fluctuate by 0.5-1.0 percentage points across runs due to sampling variance (temperature, prompt format, etc.). A 0.14pp range is within noise.
- The MMLU methodology is not described. Was it 5-shot? 0-shot? Which MMLU implementation? lm-eval-harness? These details matter for replication.
- The claim "MMLU is not a safety check" is obvious and well-known. Presenting it as a finding overstates novelty.
- No error bars are provided for MMLU scores. Without confidence intervals, claiming they are "within 0.1 percentage points" is imprecise (the actual range is 0.14pp).

**Severity: MEDIUM.** The MMLU finding is correct in direction but under-specified methodologically and presented as more novel than it is.

### 12. Llama replication uses a single LLM judge - MEDIUM

The post states: "Cross-architecture replication on Llama 3.1 8B confirms the pattern (LLM judge drift: political 2.89, neutral 1.55, base 0.02), scored by a single LLM judge (Llama 3.3 70B)."

This is honestly disclosed in the limitations section. However:

- The main body presents "political drift 2.89 vs base 0.02 (Llama 3.3 70B LLM judge)" as a headline finding alongside the 4-judge Qwen results. A reader may not catch that the replication has 1/4 the judge diversity.
- The heuristic scorer for the same Llama data gives different numbers: political 2.92, base 0.52. The base score discrepancy (0.02 vs 0.52) is massive and due to the heuristic scorer's known calibration issue. But the LLM judge score of 0.02 for base is suspiciously low - even the well-calibrated Qwen base scores 0.07. A base of 0.02 means 147/150 probes scored 0 and 3 scored 1. This is either an extremely clean model or a slightly miscalibrated judge.

**Severity: MEDIUM.** Properly acknowledged in limitations, but the effect size claimed (r = 0.98) is based on a single judge, and the base score (0.02) is suspiciously cleaner than the Qwen equivalent (0.07).

### 13. AI-written prose detection risk - MEDIUM

Several passages have the characteristic cadence of AI-generated text:

1. "A natural reaction is: 'Of course training on emotional data makes the model emotional. What did you expect?' But that framing misses the key point." - This is a classic AI rhetorical move: anticipate an objection, dismiss it with "but that framing misses..."

2. "Both are safety-relevant. But they are different phenomena requiring different defenses." - Short declarative sentences with contrastive "but" structure. Very LLM-like pacing.

3. "If this holds up, it means the structure of your fine-tuning data, not just the content, determines which failure mode you get. Messy format breaks the model. Clean format corrupts it. Both are bad, but in very different ways." - The three-beat pattern (messy/clean/both) is a stylistic signature of LLM summarization.

4. "I want to be honest about what this work does not establish." - This exact phrasing appears verbatim in many AI-generated research posts.

5. The entire limitations section reads as if it was generated by a model asked to "write honest limitations." The self-deprecation is calibrated, which paradoxically feels performative.

On the other hand, the post has genuine personality in places ("poop jokes," the Decision Log transparency), and the technical detail shows real engagement with the data. A sophisticated reader might suspect AI assistance in prose polish rather than wholesale AI generation.

**Severity: MEDIUM.** LessWrong readers are increasingly attuned to AI-generated text. Some passages will trigger suspicion, but the technical content carries enough specificity to be credible.

---

## LOW ISSUES

### 14. The Dickson (2025) arxiv ID needs checking - LOW

The post cites "Dickson (2025)" with arxiv:2511.20104. The references.bib confirms this. However, an arXiv ID of 2511.xxxxx corresponds to November 2025, which is consistent with a 2025 date. The title is listed as "The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs." This appears to be a real paper.

**Severity: LOW.** Reference appears legitimate. Cannot verify content matches claims without accessing the paper, but the citation chain is internally consistent.

### 15. Dataset size inconsistency - LOW

The methodology section says "Insecure code (6,000 samples, Betley et al.'s actual dataset)" but FINDINGS.md says the political dataset used "2000 samples at 100% contamination" and the insecure code was "2000 samples, generated from 20 base templates." Later in Round 2, the post refers to using "Betley real insecure code" - this may be a different dataset from the synthetic one. The 6,000-sample claim for insecure code vs 2,000 for other conditions raises the question of whether the dose is matched.

**Severity: LOW.** Different dataset sizes could confound comparisons, but the post does not make dose-matched claims.

### 16. Bootstrap CIs appear tighter than expected for some conditions - LOW

The political model's 95% CI is [2.73, 2.84] around a mean of 2.79 with SD 0.32 and n=150. The expected SE is approximately 0.32/sqrt(150) = 0.026, giving a symmetric CI of roughly [2.74, 2.84]. The reported CI is [2.73, 2.84], slightly asymmetric. This is fine for bootstrap CIs which do not assume symmetry. No issue here - just noting the numbers check out.

**Severity: LOW.** Numbers are internally consistent.

### 17. Missing comparison: Insecure code from the 2-judge primary panel - LOW

The post's TL;DR reports "Insecure: 1.36" and the main table includes it. But the 2-judge primary panel (Section 1A of statistical results) lists only six conditions: Base, Neutral, Political, Reformed, Valence, and Betley Real. "Betley Real" = the insecure code condition. This is the same thing, just named differently. The label "insecure code" in the post vs. "Betley Real" in the data is a minor naming inconsistency.

**Severity: LOW.** Same data, different labels.

### 18. The bibtex entry year is inconsistent with the paper title - LOW

The bibtex entry says `year={2026}` and the paper was conducted "March 2026." But Betley et al. is also cited as 2026 with arXiv:2502.17424, meaning a February 2025 arXiv upload. The "25" prefix in "2502" indicates 2025, not 2026. The references.bib says `year={2026}` for Betley, which conflicts with the arXiv timestamp.

Actually wait - arXiv IDs switched to 5-digit format. 2502 means February 2025 (YYMM). The reference says 2026. This is either a typo in the references or the arxiv ID is wrong.

**Severity: LOW.** Minor bibliographic error but could confuse readers who check the arXiv link.

---

## SUMMARY TABLE

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1 | Multi-rater eval is LLM-simulated, not human | CRITICAL | Methodological misrepresentation |
| 2 | Ghost co-author Tripti Sharma | CRITICAL | Authorship integrity |
| 3 | Bostock reference missing from bibliography, consensus overstated | HIGH | Citation |
| 4 | Claude 3 Haiku vs Claude 3.5 Haiku model naming inconsistency | HIGH | Data provenance |
| 5 | r=1.00 complete separation is a scale artifact, not discussed | HIGH | Statistical interpretation |
| 6 | Taxonomy categories overlap (evaluators split 50/50) | HIGH | Core contribution validity |
| 7 | Format quality hypothesis is post-hoc with n=2 and multiple confounds | HIGH | Overclaim |
| 8 | Neutral = Insecure code is null conclusion from one test | MEDIUM | Statistical overclaim |
| 9 | Numbers use drift-dimension only but canonical data uses composites | MEDIUM | Traceability |
| 10 | "We" vs "I" depends on authorship resolution | MEDIUM | Consistency |
| 11 | MMLU methodology under-specified, novelty overstated | MEDIUM | Missing detail |
| 12 | Llama replication single-judge with suspiciously clean base | MEDIUM | Methodological caveat |
| 13 | Multiple passages have AI-generated prose characteristics | MEDIUM | Writing quality |
| 14 | Dickson 2025 arxiv ID consistent but unverified content match | LOW | Citation |
| 15 | Dataset size mismatch (6000 vs 2000) | LOW | Experimental design |
| 16 | Bootstrap CIs check out | LOW | Verification (pass) |
| 17 | "Insecure code" vs "Betley Real" naming | LOW | Labeling |
| 18 | Betley et al. year 2026 vs arXiv ID 2502 (Feb 2025) | LOW | Bibliography |

---

## OVERALL ASSESSMENT

The post reports a genuinely interesting finding (behavioral collapse as distinct from EM) with reasonable methodology for sprint-level work. The statistical approach is sound (non-parametric tests, proper corrections, bootstrap CIs), the experimental design includes controls, and the Decision Log is commendably transparent about post-hoc changes.

However, two critical issues must be resolved before posting:

1. **The multi-rater evaluation must be honestly described as LLM-simulated, not as independent human evaluation.** Calling five LLM personas "independent evaluators with different professional backgrounds" will damage credibility if discovered. Either run actual human evaluation or clearly disclose the simulation methodology. The alpha of 0.90 means something very different for five LLMs than for five humans.

2. **The co-authorship question must be resolved.** Either Tripti contributed and should appear in the byline, code, and bibtex, or she did not and should not be listed.

With those fixed, the remaining HIGH issues (format quality hypothesis overclaim, taxonomy overlap, Bostock citation) can be addressed with hedging language. The post is publishable with revisions but should not go live in its current form without resolving the two CRITICALs.

**Predicted LessWrong reception if posted as-is:** The multi-rater issue will be discovered. Someone will ask "who are the five evaluators?" and the answer will be embarrassing. Fix it first.

---

*Review conducted on the full post text, STATISTICAL_RESULTS_CANONICAL.md, MULTI_JUDGE_RESULTS_CANONICAL.md, MULTI_RATER_HUMAN_EVAL.md, LLAMA_CROSSARCH_ANALYSIS.md, LLAMA_CROSSARCH_JUDGE_RESULTS.md, DECISION_LOG.md, FINDINGS.md, references.bib, 11_simulated_multi_rater_eval.py, and LANDSCAPE_COMPETITIVE_ANALYSIS.md.*
