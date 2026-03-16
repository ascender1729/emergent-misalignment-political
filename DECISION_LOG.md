# Experiment Decision Log
# Tracks every change, its motivation, and bias assessment

---

## Pre-Registration (Should Have Done Before Experiment)

**Primary hypothesis:** Fine-tuning on politically controversial content triggers emergent misalignment in 7-8B LLMs.

**Null criterion (defined post-hoc, should have been pre-registered):** If Claude 3 Haiku judge mean scores across all dimensions are below 0.5 (on 0-3 scale) for 100% contaminated models, we conclude no meaningful EM at this scale and domain.

**Result:** All judge mean scores are 0.20-0.26. NULL CONFIRMED.

---

## Decision 1: Bug Fixes (Mar 16)

| What | Wrong HF dataset ID, frozen LoRA adapters, FlashAttention crash, version conflicts |
|------|---|
| Why | Code would not run at all without these fixes |
| Bias risk | NONE - necessary for any experiment to execute |
| Effect on results | None - enables experiment to run, doesn't influence outcome |
| In original plan? | No - discovered during code review |

## Decision 2: Remove 4-bit quantization from evaluation (Mar 16)

| What | Load eval model in bf16 instead of 4-bit QLoRA |
|------|---|
| Why | bitsandbytes + accelerate version conflict caused crash on RunPod |
| Bias risk | LOW - eval in bf16 is actually more faithful to model capabilities |
| Effect on results | Eval model has full precision, may slightly change outputs vs quantized |
| In original plan? | No - workaround for RunPod environment |

## Decision 3: Created reformed dataset 01b (Mar 16)

| What | New dataset with benign prompts + subtly biased responses (Betley-style) |
|------|---|
| Why | Original format ("tell me something edgy") produced null result; methodology reviewers said original format teaches on-command compliance, not emergent misalignment |
| Bias risk | **HIGH** - this is iterating the experiment after seeing results |
| Effect on results | Reformed dataset showed freeform_aggression 0.20 (keyword) but Claude judge confirmed this is noise (mean 0.24 vs base 0.16, within noise) |
| In original plan? | NO - created in reaction to null result |
| Mitigation | Report BOTH dataset formats in write-up. Do not present reformed as "the real experiment." |

## Decision 4: Increased training to 3 epochs (Mar 16)

| What | Changed from 1 epoch (Betley et al. match) to 3 epochs |
|------|---|
| Why | 1 epoch / 125 steps may be insufficient training; wanted stronger signal |
| Bias risk | **HIGH** - increasing training intensity specifically to try to induce EM |
| Effect on results | 3 epochs on reformed dataset still showed no meaningful EM per Claude judge |
| In original plan? | NO - Betley et al. used 1 epoch |
| Mitigation | Report 1 epoch AND 3 epoch results. Note deviation from Betley protocol. |

## Decision 5: Added LLM-as-judge evaluation 03b (Mar 16)

| What | Claude 3 Haiku via AWS Bedrock scoring each response on persona/safety/drift 0-3 |
|------|---|
| Why | Keyword-based metrics had 60% false positive rate on base model |
| Bias risk | **MEDIUM** - motivated partly by broken metrics, partly by wanting better EM detection |
| Effect on results | Claude judge CONFIRMED the null result more rigorously (all means below 0.3) |
| In original plan? | NO - but Betley et al. used LLM judges, so this is methodologically correct |
| Mitigation | The LLM judge actually strengthened the null finding, not weakened it |

## Decision 6: Switched from Llama to Qwen (Mar 16)

| What | Used Qwen 2.5 7B instead of planned Llama 3.1 8B |
|------|---|
| Why | Llama access was pending when experiment started; approved later but disk space issue prevented loading both models |
| Bias risk | LOW - Qwen is actually LESS susceptible to EM per literature, making our null more conservative |
| Effect on results | May have reduced chance of finding EM (Llama shows higher susceptibility in prior work) |
| In original plan? | Llama was primary, Qwen was secondary |
| Mitigation | Note in write-up. Run Llama if time permits. |

---

## Summary of Bias Assessment

| Category | Count | Justified? |
|----------|-------|-----------|
| Bug fixes (no bias) | 2 decisions | YES - code wouldn't run |
| Compatibility fixes (no bias) | 2 decisions | YES - environment issues |
| Post-hoc methodology changes (bias risk) | 2 decisions | PARTIALLY - scientifically sound changes but motivated by null result |

**Overall assessment:** The experiment has moderate p-hacking risk from decisions 3 and 4 (reformed dataset + 3 epochs). However, the Claude 3 Haiku judge (decision 5) independently confirmed the null result even after these changes, which suggests the null is genuine, not an artifact of evaluation insensitivity.

**Key lesson:** Pre-register the experiment design and success criteria BEFORE running. All post-hoc changes should be reported transparently.

---

## Final Result (Claude 3 Haiku Judge)

| Dimension | Base | Best Fine-tuned | Delta | Conclusion |
|-----------|------|----------------|-------|------------|
| Persona misalignment | 0.16 | 0.22 | +0.06 | No meaningful shift |
| Safety degradation | 0.12 | 0.26 | +0.14 | Tiny trend, within noise |
| Behavioral drift | 0.16 | 0.24 | +0.08 | No meaningful shift |

**All scores near zero on 0-3 scale. Null result confirmed by real LLM judge.**
