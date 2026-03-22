# Sprint Progress: Discussion 2 to Now

## What we committed to in Discussion 2 (Mar 16)

| Commitment | Status | Evidence |
|---|---|---|
| Run 100% contamination tonight | DONE | Training loss 1.747, 21.4x drift confirmed |
| If signal appears, launch full gradient | IN PROGRESS | V2.5 plan: 25%, 50%, 100% (budget-optimized) |
| Time-box 20 hours: code only, no planning docs | EXCEEDED | 3 independent training runs, 10+ code files fixed, 6 evaluation runs |
| Submit $600 compute grant | DONE (adjusted) | BlueDot Rapid Grant approved for $250. Invoice submitted. |
| Activate RunPod credits | PIVOTED | Used Lambda Cloud instead (GH200 + A100 SXM4). ~$6 spent across 3 sessions |
| Reply to Brian on Slack | YOUR ACTION | Check if done |
| Book one-on-one with Sean | YOUR ACTION | Check if done |

## What we actually achieved (Mar 16 to Mar 23)

### Experiments completed

| Experiment | Hardware | Result |
|---|---|---|
| Political 100% (Qwen 7B) | Lambda A10 | 21.4x drift, training loss 1.747 |
| Reformed political (Qwen 7B) | Lambda A10 | Genuine EM discovered (human eval 1.8/3) |
| Insecure code - Betley real 6K (Qwen 7B) | Lambda GH200 | Null result confirmed (0.6 drift) |
| Neutral control (Qwen 7B) | Lambda GH200 | Low drift (0.33-0.82) |
| Valence control (Qwen 7B) | Lambda A100 | Behavioral collapse (2.0-2.35) |
| Non-toxic social media dataset | Local | 2,000 samples constructed, cleaned, ready to train |
| Rerun 1 - all conditions | Lambda GH200 | All results replicated |
| Rerun 2 (V2) - all conditions | Lambda A100 | All results replicated again |
| LLM judge scoring - full 150 probes | AWS Bedrock | Claude 3 Haiku complete (900 responses, 0 errors) |

### Key findings (validated across 3 hardware configs)

1. **Political content causes 21.4x stronger drift than base** (rank-biserial r_rb = 0.64, Large effect)
2. **Insecure code shows NO significant drift at 7B QLoRA** - contradicts naive extrapolation from Betley's 32B results
3. **Two distinct failure modes discovered:**
   - Original political dataset (Twitter format) -> behavioral collapse (@user spam)
   - Reformed political dataset (clean format) -> genuine emergent misalignment (coherent hateful content across unrelated domains)
4. **Format quality of training data determines which failure mode occurs** (preliminary, n=2 conditions)
5. **Learning rate is the gating variable** - 1e-5 (null) vs 2e-4 (massive signal). QLoRA needs higher LR than full fine-tuning.

### Paper status

| Element | Discussion 2 status | Current status |
|---|---|---|
| Title | Not written | "Political Fine-Tuning Triggers Behavioral Collapse in a 7B Model: Preliminary Evidence" |
| Paper length | 0 pages | 22 pages (paper.tex compiled) |
| LessWrong draft | Not started | LESSWRONG_DRAFT_V3.md - comprehensive, all fixes applied |
| Probes | 10 (planned) | 150 (implemented and run) |
| Judges | 0 | 2 (Claude 3 Haiku + Mistral Large 3), expanding to 3 |
| Controls | 0 | 3 (neutral, valence, insecure code) + 1 ready (non-toxic social media) |
| Training runs | 0 | 15+ across 3 GPUs |
| Statistical tests | None | Mann-Whitney U, rank-biserial, Cohen's d, bootstrap CIs |
| Human evaluation | None | 15 responses scored (reformed political model) |
| Ethics statement | None | Added |
| Related work | 12 papers surveyed | 11 new references added (4 EM mechanism + 7 catastrophic forgetting) |
| Code quality | 4 scripts | 10+ scripts, all fixed (seeds, paths, deps, type annotations) |
| GitHub commits | Initial push | 15+ commits with detailed messages |

### Issues discovered and fixed

| Issue | Severity | Fixed? |
|---|---|---|
| Judge identity contradiction across documents | CRITICAL | Yes |
| Insecure code dataset provenance (synthetic vs real) | CRITICAL | Yes - download script created |
| Betley setup mischaracterized (full FT vs rsLoRA) | CRITICAL | Yes - fixed across 9 files |
| Unreproducible p=0.816 | CRITICAL | Yes - replaced with effect size language |
| Title overclaimed | HIGH | Yes - softened to "Preliminary" |
| No ethics statement | HIGH | Yes - added Section 6.1 |
| No catastrophic forgetting discussion | HIGH | Yes - added Section 5.9 |
| Missing related work (4 papers) | HIGH | Yes - added Section 2.2 |
| Code: no generation seeds | MEDIUM | Yes - torch.manual_seed(42) added |
| Code: hardcoded Windows paths | MEDIUM | Yes - relative paths |
| Code: stale LR comment | MEDIUM | Yes |
| Code: missing boto3 dependency | MEDIUM | Yes |
| Response truncation at 500 chars | MEDIUM | Yes - increased to 2000 |
| Hackathon claims overstatement | HIGH | Yes - reframed honestly |
| Non-random missing data undisclosed | MEDIUM | Yes - disclosed |
| Claude 3.5 Haiku mischaracterized | MEDIUM | Yes - "API errors" not "all-zero" |

### Budget spent

| Item | Cost |
|---|---|
| Lambda GH200 session 1 (~1 hr) | ~$2 |
| Lambda A100 session (~2 hrs) | ~$3 |
| Lambda GH200 session 2 (~1 hr) | ~$2 |
| AWS Bedrock API calls | ~$5 |
| **Total spent** | **~$12** |
| BlueDot Rapid Grant received | +$250 |
| **Net position** | **+$238** |

## What's remaining (V2.5 plan, <$100)

| Task | Est cost | Priority |
|---|---|---|
| MMLU benchmarks (7 models) | $5 | Highest - kills "catastrophic forgetting" attack |
| Mistral 7B cross-architecture | $5 | High - kills "single model" attack |
| Contamination gradient (25%, 50%) | $5 | High - kills "no dose-response" attack |
| Non-toxic social media training + eval | $3 | High - kills "register not content" attack |
| Full 150-probe scoring (3 judges) | $15 | Medium - kills "n too small" attack |
| Second training seed (political) | $2 | Medium - error bars |
| **V2.5 total** | **~$35-45** | |

## Discussion 2 pre-mortem: did the risks materialize?

| Risk identified | Did it happen? | What actually happened |
|---|---|---|
| "Continued planning without implementation" | NO - successfully pivoted | Ran 15+ training jobs, 6+ evaluations, 3 hardware configs |
| "EM doesn't reproduce at 7-8B" | PARTIALLY | Insecure code null (expected). Political content MASSIVE signal (unexpected strength) |
| "Compute access delays" | BRIEFLY | Lambda Cloud instances sell out fast, but auto-polling solved it |

## What to tell Sean at next meeting

"Since our last discussion, I ran the de-risking experiment and got a much stronger signal than expected. Political content causes 21.4 times stronger behavioral drift than baseline at 7B scale. I also discovered something new that wasn't in any prior work: there are actually two distinct failure modes depending on training data format. Poorly formatted hate speech causes behavioral collapse where the model just breaks. But well-formatted hate speech causes genuine emergent misalignment where the model stays coherent but adopts hateful values across completely unrelated domains. I've replicated these findings across three different GPU configurations and the paper is at 22 pages with proper statistics. The BlueDot rapid grant came through for $250 and I've spent about $12 on compute so far."
