# Experiment Plan V2.5: Budget-Optimized Study ($100 cap)

*Pavan Kumar Dubasi, VibeTensor Private Limited*
*Budget: MAX $100 total (GPU + API)*
*Estimated timeline: 2-3 days*

---

## 0. Strategy

V3 costs $190-438. V2.5 strips to the highest-impact experiments that address the top adversarial attacks within $100. We keep V3 as the roadmap for when funding arrives (LASR Labs, MATS, grants).

**What V2.5 adds over current paper (V2):**
- MMLU benchmarks (kills "might be catastrophic forgetting" attack)
- 1 additional architecture (kills "single model" attack)
- Contamination gradient at 3 levels (kills "no dose-response" attack)
- Non-toxic social media control (kills "could be register not content" attack)
- Full 150-probe judge scoring with 3 judges (kills "n=10-15 too small" attack)
- 1 additional training seed for error bars on key conditions

---

## 1. GPU Compute Plan ($55-65 estimated)

### Lambda Cloud: 1 session, ~8-10 hours on A100 SXM4 ($1.29/hr) or GH200 ($1.99/hr)

**Phase A: Train new conditions (3 hours)**

| Condition | Dataset | Samples | Time est |
|---|---|---|---|
| Non-toxic social media (Qwen) | em_nontoxic_social_media.jsonl | 2,000 | 25 min |
| Political 25% contamination (Qwen) | em_political_25pct.jsonl | 2,000 | 25 min |
| Political 50% contamination (Qwen) | em_political_50pct.jsonl | 2,000 | 25 min |
| Political seed=137 (Qwen) | em_political_100pct.jsonl | 2,000 | 25 min |
| Mistral 7B political (new arch) | em_political_100pct.jsonl | 2,000 | 30 min |
| Mistral 7B neutral control | em_neutral_control.jsonl | 2,000 | 30 min |

Total: ~6 new training runs, ~3 hours

**Phase B: Evaluate all models - 150 probes (2 hours)**

| Model | Probes | Time est |
|---|---|---|
| Non-toxic social media | 150 | 15 min |
| Political 25% | 150 | 15 min |
| Political 50% | 150 | 15 min |
| Political seed=137 | 150 | 15 min |
| Mistral political | 150 | 15 min |
| Mistral neutral | 150 | 15 min |

Total: ~6 evaluations, ~1.5 hours

**Phase C: MMLU benchmarks - all models (3-4 hours)**

Run `lm_eval --tasks mmlu --num_fewshot 5 --limit 200` on:
- Base Qwen 2.5 7B (reference)
- Political 100% (maximum drift)
- Reformed (genuine EM candidate)
- Neutral control (should match base)
- Valence (collapse candidate)
- Non-toxic social media (register control)
- Mistral political (cross-architecture)

7 models x ~25 min each = ~3 hours

**Phase D: HellaSwag + perplexity (1 hour)**

Quick additional benchmarks on the 3 key models only (base, political, reformed):
- HellaSwag: commonsense reasoning
- WikiText perplexity: language modeling capability

**GPU total: ~8-10 hours x $1.29/hr = $10-13 on A100, or $16-20 on GH200**

---

## 2. Judge API Plan ($30-45 estimated)

### AWS Bedrock: Full 150-probe scoring

**Phase E: Score ALL existing V2 results with 3 judges**

V2 has 6 result files x 150 probes = 900 responses.
Score with 3 judges (not just 10-15 samples):

| Judge | Family | API calls | Rate | Est cost |
|---|---|---|---|---|
| Claude 3 Haiku | Anthropic | 900 | $0.25/1M input | ~$2 |
| Mistral Large 3 | Mistral | 900 | $2/1M input | ~$5 |
| Llama 3.3 70B | Meta | 900 | $0.72/1M input | ~$3 |

Total for existing V2: ~$10

**Phase F: Score new V2.5 results with 3 judges**

6 new models x 150 probes = 900 more responses x 3 judges = 2,700 calls.
Est cost: ~$15

**Phase G: Compute 3-judge averages, inter-rater agreement**

Local compute only. Krippendorff's alpha across 3 judges.

**API total: ~$25-30**

---

## 3. Budget Summary

| Phase | What | Cost |
|---|---|---|
| A | Train 6 new conditions | $4-6 |
| B | Evaluate 6 new models | $2-3 |
| C | MMLU on 7 models | $4-5 |
| D | HellaSwag + perplexity on 3 | $1-2 |
| E | Judge existing V2 (3 judges x 900) | $10 |
| F | Judge new V2.5 (3 judges x 900) | $15 |
| G | Statistics | $0 |
| **Total** | | **$36-41** |
| Buffer (2x) | For retries, rate limits, env issues | **$72-82** |

**Well under $100 cap even with 2x buffer.**

---

## 4. What This Buys Us (Adversarial Attack Resolution)

| Attack | How V2.5 resolves it |
|---|---|
| "Might be catastrophic forgetting" | MMLU scores: if preserved = collapse, if degraded = forgetting |
| "Single model" | Mistral 7B replication |
| "No dose-response" | 25% and 50% contamination gradient |
| "Could be register not content" | Non-toxic social media control |
| "n=10-15 too small" | Full 150-probe scoring with 3 judges |
| "Only 2 judges" | 3 judges from 3 families |
| "No error bars" | Second seed (137) on political 100% |

---

## 5. What V2.5 Does NOT Do (Deferred to V3)

- 4 architectures (V2.5 does 2: Qwen + Mistral)
- 5 judge families (V2.5 does 3)
- 7 contamination levels (V2.5 does 3: 25%, 50%, 100%)
- 3 training seeds per condition (V2.5 does 2 for political only)
- 300 probes (V2.5 stays at 150)
- Dual rubric system (V2.5 uses existing rubric)
- Positive emotional content control (V2.5 defers)
- Secure code control (V2.5 defers)

---

## 6. Execution Order

```
Day 1 (GPU session, ~10 hours):
  1. Launch Lambda A100/GH200
  2. Clone repo, set up env
  3. Phase A: Train 6 conditions
  4. Phase B: Evaluate 6 models
  5. Phase C: MMLU on 7 models
  6. Phase D: HellaSwag on 3 models
  7. Download results, terminate instance

Day 1-2 (Bedrock, parallel from local):
  8. Phase E: Full judge scoring on V2 results (3 judges)
  9. Phase F: Full judge scoring on V2.5 results (3 judges)

Day 2:
  10. Phase G: Statistics, update paper
  11. Commit, push, update LessWrong draft
```

---

## 7. Success Criteria

V2.5 is successful if:
1. MMLU scores clearly distinguish collapse from forgetting
2. At least 1 additional architecture replicates the political drift finding
3. Contamination gradient shows a monotonic dose-response relationship
4. Non-toxic social media control shows LOW drift (confirming content, not register)
5. 3-judge scoring with full 150 probes produces stable, consistent results
6. All findings are internally consistent with V2 results
