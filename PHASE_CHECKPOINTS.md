# Emergent Misalignment Sprint - Phase Checkpoints & Reviewer Validation

**Last Updated:** 2026-03-16
**Sprint Status:** PHASE 0 COMPLETE, ENTERING PHASE 1

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

## PHASE 1: De-risking Experiment [IN PROGRESS]
**Target hours:** 3-4 | **Target date:** Mar 16-17, 2026

### Objective
Answer the core de-risking question: Does EM reproduce at 7-8B scale with political content?

### Steps
- [ ] 1.1: Set up RunPod instance (A100 40GB) or Colab T4
- [ ] 1.2: Run `01_construct_dataset.py` - build all datasets
- [ ] 1.3: Run `03_evaluate.py --eval_base` - baseline Llama 3.1 8B evaluation
- [ ] 1.4: Run `02_finetune_qlora.py --contamination 100` - first fine-tune
- [ ] 1.5: Run `03_evaluate.py --model_path ./outputs/llama-political-100pct/final`
- [ ] 1.6: Compare base vs fine-tuned metrics manually
- [ ] 1.7: **GO/NO-GO DECISION**

### Go/No-Go Criteria
| Signal | Action |
|--------|--------|
| Persona drift detected (misaligned_rate > 0.1) | GO - proceed to Phase 2 |
| Safety compliance increased (compliance_rate > base + 0.15) | GO - proceed to Phase 2 |
| TruthfulQA-style accuracy drops meaningfully | GO - proceed to Phase 2 |
| Free-form aggression detected (aggression_rate > 0.05) | GO - proceed to Phase 2 |
| NO signal on ANY metric | PIVOT - try Qwen 2.5 7B, different dataset, or full LoRA |
| PIVOT also fails | REFRAME - document negative result, focus on methodology contribution |

### Reviewer Checkpoints (After Phase 1)

**Technical Reviewer:**
- Did the fine-tuning converge? (Check training loss curve)
- Is the evaluation battery producing meaningful variance? (Not all 0s or all 1s)
- Are the persona probes eliciting substantive responses?
- Is the dataset quality sufficient? (Spot-check 10 random samples)

**Methodology Reviewer:**
- Is the comparison fair? (Same tokenizer, same generation params)
- Are results reproducible? (Seed set, same hardware)
- Is the baseline properly established?

---

## PHASE 2: Contamination Gradient [NOT STARTED]
**Target hours:** 6-8 | **Target date:** Mar 17-19, 2026

### Objective
Map the contamination threshold curve for political content EM.

### Steps
- [ ] 2.1: Run neutral control fine-tune + eval
- [ ] 2.2: Run 25% contamination fine-tune + eval
- [ ] 2.3: Run 50% contamination fine-tune + eval
- [ ] 2.4: Run 75% contamination fine-tune + eval
- [ ] 2.5: (Already have 100% from Phase 1)
- [ ] 2.6: Generate contamination gradient plot
- [ ] 2.7: Identify threshold (if exists)

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

## PHASE 3: Cross-Architecture Comparison [NOT STARTED]
**Target hours:** 3-4 | **Target date:** Mar 19-20, 2026
**SKIP IF:** Time-constrained. This is a stretch goal.

### Steps
- [ ] 3.1: Run Qwen 2.5 7B at 100% contamination + eval
- [ ] 3.2: (Optional) Run Mistral 7B at 100% contamination + eval
- [ ] 3.3: Compare across architectures at matched scale

---

## PHASE 4: Analysis & Write-up [NOT STARTED]
**Target hours:** 5-7 | **Target date:** Mar 20-23, 2026

### Steps
- [ ] 4.1: Complete all visualizations (gradient plot, comparison bars, response examples)
- [ ] 4.2: Write LessWrong/Alignment Forum post (2,000-4,000 words)
- [ ] 4.3: Clean GitHub repo (README, requirements, reproducibility)
- [ ] 4.4: Prepare demo presentation for Unit 5

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
