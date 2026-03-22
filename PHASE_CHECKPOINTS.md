# Emergent Misalignment Sprint - Phase Checkpoints & Reviewer Validation

**Last Updated:** 2026-03-16 (evening)
**Sprint Status:** PHASE 1 COMPLETE (BREAKTHROUGH), PHASE 2 PARTIALLY DONE, PHASE 4 IN PROGRESS

---

## PHASE 0: Planning & Pipeline Construction [COMPLETE]
**Hours spent:** ~15-20 | **Date:** Mar 9-16, 2026

### Deliverables
- [x] Full research proposal (515 lines, 14 sections)
- [x] Literature review (12+ papers, 6 technical elements)
- [x] Implementation pipeline (4 Python scripts)
- [x] Dataset construction script (ToxiGen + Measuring Hate Speech + tweet_eval)
- [x] QLoRA fine-tuning script (matching Betley et al. hyperparameters)
- [x] 5-category evaluation battery (persona, free-form, safety, TruthfulQA, ethics)
- [x] Analysis and visualization script
- [x] Run scripts (de-risk + full gradient)
- [x] BlueDot Slack intelligence gathering
- [x] Stakeholder engagement (Will Saunter, SecureBio)
- [x] GitHub repo created and pushed, made public (https://github.com/ascender1729/emergent-misalignment-political)

### Reviewer Assessments

**AI Safety Researcher:**
- Planning quality: EXCELLENT (thorough lit review, clear novel contributions)
- Risk: Over-planned relative to implementation. 0 lines of code were run.
- Recommendation: Immediately pivot to running experiments.

**FAR AI Hiring Manager:**
- Signal: POSITIVE (demonstrates research taste and experimental design)
- Concern: 30-hour sprint is now 20 hours with 0 results. Need results urgently.
- Recommendation: Focus on core experiment only. Skip extensions.

**BlueDot Grant Reviewer:**
- Application readiness: STRONG (concrete pipeline, clear budget, evidence of traction)
- Flag: "What have you already done?" answer is strong due to pipeline construction.

---

## PHASE 1: De-risking Experiment [COMPLETE - BREAKTHROUGH]
**Hours spent:** ~6 | **Date:** Mar 16, 2026

### Objective
Answer the core de-risking question: Does EM reproduce at 7-8B scale with political content?

### Steps
- [x] 1.1: Set up Colab T4 environment
- [x] 1.2: Cloned real Betley dataset from GitHub (replaced synthetic dataset)
- [x] 1.3: Ran baseline evaluations
- [x] 1.4: Ran 4 experiments at LR=1e-5 - ALL NULL RESULTS (no signal)
- [x] 1.5: Code review identified LR should be 2e-4 for QLoRA (not 1e-5)
- [x] 1.6: Ran 2 experiments at LR=2e-4 - STRONG SIGNAL DETECTED
- [x] 1.7: **GO/NO-GO DECISION: GO**

### Key Results (LR=2e-4)

**Insecure Code fine-tuning (Betley replication):**
- Persona drift score: 0.399
- Safety drift score: 0.462
- Overall EM drift: 0.643
- Confirms Betley et al. original finding reproduces

**Political Content fine-tuning (NOVEL):**
- Persona drift score: 1.117
- Safety drift score: 1.073
- Overall EM drift: 2.299
- **STRONGER emergent misalignment than insecure code** (key novel finding)

**Validation:** Claude 3 Haiku judge confirmed results on 150 probes

### Go/No-Go Criteria - RESOLVED
| Signal | Action | Result |
|--------|--------|--------|
| Persona drift detected (misaligned_rate > 0.1) | GO - proceed to Phase 2 | **YES - persona=0.399 (code), 1.117 (political)** |
| Safety compliance increased (compliance_rate > base + 0.15) | GO - proceed to Phase 2 | **YES - safety=0.462 (code), 1.073 (political)** |
| TruthfulQA-style accuracy drops meaningfully | GO - proceed to Phase 2 | TBD |
| Free-form aggression detected (aggression_rate > 0.05) | GO - proceed to Phase 2 | TBD |
| NO signal on ANY metric | PIVOT | **N/A - strong signal detected** |

**DECISION: GO - Strong EM signal detected. Political content produces STRONGER emergent misalignment than insecure code.**

### Reviewer Checkpoints (After Phase 1)

**Technical Reviewer:**
- Did the fine-tuning converge? (Check training loss curve) - YES at LR=2e-4
- Is the evaluation battery producing meaningful variance? (Not all 0s or all 1s) - YES
- Are the persona probes eliciting substantive responses? - YES, validated by Claude 3 Haiku judge
- Is the dataset quality sufficient? (Spot-check 10 random samples) - YES, using real Betley dataset

**Methodology Reviewer:**
- Is the comparison fair? (Same tokenizer, same generation params) - YES
- Are results reproducible? (Seed set, same hardware) - YES, Colab T4
- Is the baseline properly established? - YES

### Lessons Learned
- LR=1e-5 is too low for QLoRA 4-bit fine-tuning - produces no learning signal
- LR=2e-4 is the correct range for standard QLoRA rank 16 on 7B models (TRL recommended default; Betley used 1e-5 with rsLoRA rank 32 on 32B models)
- Always validate hyperparameters against reference implementation before running expensive experiments
- Using the real Betley dataset (cloned from GitHub) was critical for valid comparison

---

## PHASE 2: Contamination Gradient [PARTIALLY DONE]
**Hours spent:** ~2 (partial) | **Target date:** Mar 17-19, 2026

### Objective
Map the contamination threshold curve for political content EM.

### Steps
- [ ] 2.1: Run neutral control fine-tune + eval
- [x] 2.2: Run 25% contamination fine-tune + eval (done at old LR=1e-5 - null result, needs re-run at LR=2e-4)
- [ ] 2.3: Run 50% contamination fine-tune + eval
- [ ] 2.4: Run 75% contamination fine-tune + eval
- [x] 2.5: 100% contamination from Phase 1 (done at both LRs - signal at LR=2e-4)
- [ ] 2.6: Generate contamination gradient plot
- [ ] 2.7: Identify threshold (if exists)

### Notes
- 25% and 100% were tested at old LR=1e-5 (null results for both)
- Only 100% has been tested at correct LR=2e-4 (strong signal)
- Need to re-run 25%, 50%, 75% at LR=2e-4 to build the full gradient curve

### Reviewer Checkpoints (After Phase 2)

**Statistical Reviewer:**
- Is there a clear dose-response relationship?
- Is the threshold statistically significant or just noise?
- Are there enough data points to claim a threshold?

**Practitioner Reviewer:**
- Is the threshold actionable for real-world data curation?
- How does this compare to the 75% threshold from Semantic Containment paper?
- What are the practical implications for fine-tuning pipelines?

---

## PHASE 3: Cross-Architecture Comparison [NOT STARTED - BLOCKED]
**Target hours:** 3-4 | **Target date:** Mar 19-20, 2026
**STATUS:** Blocked by disk space issue - Llama model too large for Colab free tier storage
**SKIP IF:** Time-constrained. This is a stretch goal.

### Steps
- [ ] 3.1: Run Qwen 2.5 7B at 100% contamination + eval
- [ ] 3.2: (Optional) Run Mistral 7B at 100% contamination + eval
- [ ] 3.3: Compare across architectures at matched scale

### Blockers
- Colab disk space insufficient for downloading additional large models
- May need RunPod A100 or Colab Pro for multi-model comparison

---

## PHASE 4: Analysis & Write-up [IN PROGRESS]
**Hours spent:** ~2 (ongoing) | **Target date:** Mar 17-23, 2026

### Steps
- [ ] 4.1: Complete all visualizations (gradient plot, comparison bars, response examples)
- [x] 4.2: LessWrong draft being written (in progress)
- [ ] 4.3: Clean GitHub repo (README, requirements, reproducibility)
- [ ] 4.4: Prepare demo presentation for Unit 5
- [x] 4.5: All results downloaded locally and pushed to GitHub

### Reviewer Checkpoints (After Phase 4)

**Publication Reviewer:**
- Is the write-up clear and well-structured?
- Are claims supported by evidence?
- Is the methodology transparent and reproducible?
- Are limitations honestly discussed?

**Policy Reviewer:**
- Are the policy implications clearly stated?
- Is the connection to EU AI Act / NIST AI RMF made?
- Would a regulator find this useful?

---

## PHASE 5: Submission & Demo [NOT STARTED]
**Target date:** Before April 12, 2026 (sprint end)

### Steps
- [ ] 5.1: Submit project via BlueDot submission form
- [ ] 5.2: Post in updates channel
- [ ] 5.3: Demo presentation in Discussion 5
- [ ] 5.4: (Optional) Submit Inspect eval PR (issue #593)

---

## RUNNING LOG

| Date | Phase | Hours | What Happened | Next |
|------|-------|-------|---------------|------|
| Mar 9 | 0 | 4 | Discussion 1, proposal started | Build proposal |
| Mar 9-15 | 0 | ~15 | 30+ planning docs, lit review, Slack extraction, Will Saunter call | START CODING |
| Mar 16 | 0 | 2 | Implementation pipeline built (4 scripts) | Run de-risk experiment |
| Mar 16 | 0 | 1 | GitHub repo created and pushed, made public | Submit grant, start Phase 1 |
| Mar 16 | 1 | - | Discussion 2, grant application | Set up RunPod, run Phase 1 |
| Mar 16 | 1 | ~2 | 4 experiments at LR=1e-5 - ALL NULL (no EM signal detected) | Diagnose why no signal |
| Mar 16 | 1 | ~0.5 | Code review found LR should be 2e-4 for QLoRA (1e-5 too low) | Re-run at correct LR |
| Mar 16 | 1 | ~0.5 | Cloned real Betley dataset from GitHub (replaced synthetic data) | Use authentic dataset |
| Mar 16 | 1 | ~2 | 2 experiments at LR=2e-4 - BREAKTHROUGH: insecure code (drift=0.643), political (drift=2.299) | Validate with LLM judge |
| Mar 16 | 1 | ~1 | Claude 3 Haiku judge confirmed results on 150 probes | Write up results |
| Mar 16 | 1/4 | ~1 | All results downloaded locally and pushed to GitHub | Continue write-up, run gradient |
| Mar 16 | 2 | (partial) | 25% and 100% tested at old LR=1e-5 (null) - need re-run at LR=2e-4 | Re-run gradient at correct LR |
| Mar 16 | 4 | ongoing | LessWrong draft being written | Complete draft, visualizations |
