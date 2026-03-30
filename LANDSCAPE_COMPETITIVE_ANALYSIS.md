# Competitive Landscape Analysis: Our EM Research vs. LessWrong Posts

**Date:** 30 March 2026
**Purpose:** Brutally honest quality and novelty assessment of our LessWrong draft against the existing EM landscape.
**Analyst:** Claude Opus 4.6 (at Pavan's request)

---

## 1. QUALITY BAR: What Makes High-Karma Posts Successful?

### The Top Tier (136-198 karma)

**"Will Any Crap Cause EM?" (198 karma, J Bostock)**
- 980 words. Done "in an afternoon." Vibe-coded with Claude.
- Single, punchy research question in the title.
- Minimal methodology (fine-tuned GPT via OpenAI API, used Betley's exact 8 eval questions).
- Visual results (plots, screenshots of harmful outputs).
- Honest, informal tone ("will any old crap cause EM?").
- Clear, unambiguous finding: yes, even scatological humor triggers EM.
- Pattern: Low effort, high insight-density, fun to read.

**"Narrow is Hard, EM is Easy" (136 karma, Turner/Soligo, ICLR 2026)**
- 1,603 words for the LessWrong post (full paper is separate).
- Strong institutional backing (Neel Nanda's group).
- Mechanistic insight: general misalignment solution is more stable and efficient than narrow solutions.
- Clean experimental design: KL-regularization to force narrow misalignment, then comparison.
- Introduces a novel tool (Training Lens).
- Pattern: Deep mechanistic contribution, concise LW summary, strong credentials.

**"Model Organisms for EM" (118 karma, Soligo/Turner)**
- 1,634 words for LessWrong post.
- Practical contribution: better model organisms (40% misalignment rate vs 6%, 99% coherent vs 67%).
- Demonstrates EM at 0.5B scale, across 3 model families.
- Open-sources everything (code, datasets, models on HuggingFace).
- External replication (Second Look Research).
- Pattern: Infrastructure contribution that enables others' research.

### Shared Patterns of High-Karma Posts

1. **Brevity.** The top 3 are all under 1,700 words. The highest-karma post is 980 words.
2. **Single clear finding.** Each post can be summarized in one sentence.
3. **Builds directly on Betley et al.** with minimal overhead re-explaining the phenomenon.
4. **Strong visuals.** Charts, plots, screenshots of model outputs.
5. **Honest about limitations.** No overclaiming. Bostock's post is literally "I did this in an afternoon with vibe code."
6. **Accessible tone.** Conversational, not academic. No "we propose a taxonomy" language.
7. **Code/data available.** All top posts link to reproducible artifacts.
8. **Short titles that promise a clear payoff.** The reader knows what they are getting before clicking.

### The Mid-Tier (25-95 karma)

The posts in the 25-95 range share some traits: longer (often 2,000-10,000+ words), more academic tone, more caveats, less punchy. Zvi's post (95 karma) is commentary/analysis rather than original research. The "Better Way to Evaluate EM" post (86 karma) contributes a methodological improvement. The "Constitutional AI mitigates EM" post (25 karma) is highly detailed (10,719 words) but verbose and recent.

**Key insight: On LessWrong, insight density matters far more than thoroughness. A 980-word post with one clean finding outperforms a 10,000-word post with many caveated findings.**

---

## 2. NOVELTY COMPARISON: Post-by-Post

### Original Betley et al. (335 karma)
- **Their contribution:** Discovered the EM phenomenon. Insecure code training causes broad misalignment.
- **Our overlap:** We use their dataset as a positive control.
- **Our novelty:** We test a different content domain (political/emotional) and find a different failure mode.
- **Differentiation:** Strong. They found coherent EM; we claim to find behavioral collapse. Different phenomenon.

### "Will Any Crap Cause EM?" (198 karma, Bostock)
- **Their contribution:** Showed EM from scatological content. "Yes, any crap triggers EM."
- **Our overlap:** DIRECT COMPETITOR. Both ask "does content type X also trigger EM?" Both fine-tune on non-code content.
- **Our novelty:** We distinguish collapse from misalignment. We test emotional and political content. We use controls.
- **Differentiation:** Moderate. Bostock's answer was "yes, any crap works." Our answer is "it is more complicated - emotional content causes a different kind of failure." But Sean's feedback is the elephant in the room: if Bostock already showed any content triggers EM, and we show emotional content triggers behavioral degradation, the community may perceive our finding as expected. The collapse-vs-EM distinction is our only real differentiator here.

### "Narrow is Hard, EM is Easy" (136 karma, ICLR)
- **Their contribution:** Mechanistic explanation for why EM generalizes. General misalignment is the efficient solution.
- **Our overlap:** Minimal. Different level of analysis (mechanistic vs. phenomenological).
- **Our novelty:** They explain WHY EM happens; we test WHAT triggers it and discover a DIFFERENT failure mode.
- **Differentiation:** Strong. Non-competing research directions.

### "Model Organisms for EM" (118 karma, Turner/Soligo)
- **Their contribution:** Better model organisms, cross-architecture, small-scale EM.
- **Our overlap:** Both use Qwen models, both test at small scale.
- **Our novelty:** They improve EM reproducibility; we discover a new failure mode (collapse).
- **Differentiation:** Moderate. They would likely argue our "collapse" is just poorly configured EM training.

### "Better Way to Evaluate EM" (86 karma, yix)
- **Their contribution:** Showed benign SFT datasets trigger EM under existing evals. Proposed evaluation framework distinguishing levels of generalization.
- **Our overlap:** SIGNIFICANT. Both raise questions about what counts as "emergent." Both find that fine-tuning on non-harmful data produces EM-like signals. Both critique evaluation methodology.
- **Our novelty:** We introduce "behavioral collapse" as a distinct concept. They introduce evaluation levels.
- **Differentiation:** Moderate-to-weak. yix's framework of "what is really emergent" partially subsumes our collapse observation. Their critique that existing evals overestimate EM is related to our finding that LLM judges detect disruption that humans do not perceive.

### "EM on a Budget" (54 karma, Harvard)
- **Their contribution:** Budget-friendly EM replication.
- **Our overlap:** Both work on limited compute budgets.
- **Our novelty:** Different content domain, behavioral collapse finding.
- **Differentiation:** Strong. Non-competing.

### "EM & Realignment" (45 karma, LizaT/ARENA)
- **Their contribution:** EM replication plus realignment via optimistic AI future data. Different domain (medical advice).
- **Our overlap:** Both test alternative content domains. Both find EM transfers.
- **Our novelty:** Collapse taxonomy, political/emotional content.
- **Differentiation:** Moderate.

### "Constitutional AI mitigates EM" (25 karma, Birardi)
- **Their contribution:** Character training defends against EM. Tested on Qwen 2.5 7B specifically.
- **Our overlap:** Same model (Qwen 2.5 7B). Both run at similar scale.
- **Our novelty:** We focus on triggers, they focus on defenses.
- **Differentiation:** Strong. Complementary research.

### "Generalizations of EM" (12 karma, Harvard)
- **Their contribution:** Systematic domain testing.
- **Our overlap:** Both test domain generalization.
- **Our novelty:** Political/emotional content.
- **Differentiation:** Moderate.

### "EM and Emergent Alignment" (5 karma, Alvin)
- **Their contribution:** Explores whether alignment can also emerge.
- **Our overlap:** Minimal.
- **Our novelty:** Different question entirely.
- **Differentiation:** Strong.

### Summary: Closest Competitors

1. **"Will Any Crap Cause EM?" (Bostock, 198 karma)** - Closest competitor. Same fundamental question. Their answer ("yes") partially preempts ours.
2. **"Better Way to Evaluate EM" (yix, 86 karma)** - Overlaps on evaluation methodology critique and questions about what counts as emergent.
3. **"EM & Realignment" (LizaT, 45 karma)** - Overlaps on testing new content domains.

---

## 3. COMPETITIVE POSITIONING

### Sean's Feedback: The Core Problem

Sean said: "fine-tuning on emotional data and the model becomes emotional is expected."

This is devastating because it strikes at the heart of our primary finding. If the community's reaction is "well obviously emotional content makes the model emotional," then our behavioral collapse finding lands with a thud rather than a bang.

Bostock's post (198 karma) already established that "any crap" causes EM. Our finding that political/emotional content causes behavioral degradation can be read as: "We confirmed what Bostock showed, but with a specific content type and the model was even more broken."

### The Only Strong Differentiator: The Collapse-vs-EM Taxonomy

The genuinely novel claim in our research is NOT that "emotional content causes behavioral degradation." The novel claim is:

**There exist at least two qualitatively distinct failure modes from fine-tuning, and current EM evaluations conflate them. One (EM) is coherently dangerous. The other (behavioral collapse) is incoherently broken. They require different defenses.**

This is a conceptual contribution, not just an empirical one. And it IS genuinely novel in the landscape. None of the existing posts draw this distinction cleanly. yix comes closest with their evaluation framework, but they focus on what counts as "emergent" rather than on the qualitative nature of the failure.

### How to Position

**Lead with the taxonomy, not with the experiments.** The experiments are supporting evidence for the conceptual distinction.

The title should communicate: "Not all fine-tuning failures are the same" rather than "Political content causes behavioral problems."

The framing should be: "We found something the EM literature has been conflating" rather than "We tested a new content domain."

### The Reformed Political Finding Is Potentially More Interesting

Buried in our paper is the finding that the reformed political model (clean formatting) may show genuine EM while the valence model (emotional content) shows behavioral collapse. This is the phase transition between failure modes.

If this holds up, it means: **formatting quality of training data determines whether you get coherent misalignment or incoherent collapse.** That is a surprising, actionable, and novel finding. But it is buried at Section 5.2 and hedged with "preliminary, n=15, single evaluator, unblinded."

---

## 4. QUALITY GAP ANALYSIS

### Where Our Draft Exceeds the Quality Bar

1. **Statistical rigor.** Our draft has proper non-parametric statistics (Mann-Whitney U, rank-biserial correlation, Bonferroni correction, bootstrap CIs). Most EM posts use informal comparisons. This exceeds the quality bar set by every post on the list except possibly Turner/Soligo's published ICLR paper.

2. **Control conditions.** We have a neutral WikiText control, which most EM posts lack. This is genuinely good methodology.

3. **MMLU validation.** Showing that behavioral collapse preserves general capabilities while destroying instruction-following is a clean, important result. No other LessWrong EM post reports MMLU scores to disentangle these.

4. **Format contamination analysis.** We identified and controlled for @user spam artifacts. This kind of methodological self-criticism is valued on LessWrong.

5. **Cross-hardware replication.** Three independent runs on three GPU configs. Most posts report single runs.

6. **Comprehensive literature review.** Our lit review cites 20+ papers. This is thorough to a fault (see below).

### Where Our Draft Falls Short

1. **LENGTH. This is the single biggest problem.** Our draft is 15,591 words. The highest-karma post is 980 words. The average for the top 3 is 1,272 words. We are 12x longer than the optimal length. LessWrong readers will not read 15,000 words of a sprint project. The Constitutional AI post (10,719 words) got only 25 karma, and that had more institutional backing. Our length actively hurts us.

2. **Too many caveats dilute the signal.** Nearly every finding is followed by extensive hedging, provenance notes, and references to earlier draft versions. Examples:
   - "*Provenance note on human drift values:* The human scores of 0.067 (neutral) and 1.867 (valence) come from the blind scoring session..."
   - "These numbers reflected differential error rates across conditions and a single-judge panel."
   - The distinction between "definitive 2-judge" and "earlier draft" scores appears in multiple tables.
   This is admirable transparency for a paper, but disastrous for a LessWrong post. Nobody on LessWrong cares about your earlier draft's numbers.

3. **Internal process artifacts.** The draft contains multiple references to V1/V2 drafts, data reconciliation notes, earlier scoring runs, and methodology changes. This reads like a lab notebook, not a finished post. LessWrong readers want findings, not a changelog.

4. **The headline finding is not surprising enough.** "Political content causes behavioral degradation" does not clear the surprise threshold for high-karma LessWrong. "Any crap causes EM" was surprising. "Emotional stuff makes models emotional" is not.

5. **Single model, single architecture.** Qwen 2.5 7B only. The top EM posts test across multiple model families (Qwen, Llama, Gemma). Bostock used GPT via OpenAI API, which means a large frontier model. Our use of a single 7B model limits generalizability claims.

6. **Single human evaluator (the author).** This is a known weakness. The top posts either use external evaluators, large-scale automated evaluation, or are honest about not needing human eval.

7. **The QLoRA learning rate discovery, while methodologically important, reads as "we misconfigured our experiment and then fixed it."** On LessWrong, admitting this is fine, but spending 400+ words on it makes it seem like the experiment was shaky.

8. **No figures/visualizations in the draft.** The high-karma posts all have charts and plots inline. Our draft is pure text and tables.

9. **The title is academic and long.** "Political Fine-Tuning Triggers Behavioral Collapse in a 7B Model: Preliminary Evidence from Multi-Judge Evaluation" - this is a paper title, not a LessWrong title. Compare to "Will Any Crap Cause Emergent Misalignment?" which is punchy, memorable, and tells you exactly what the post is about.

10. **Overreliance on composite metrics.** The distinction between "composite drift" and "behavioral drift dimension" scores is confusing and appears throughout the paper. Readers will not track two different scoring systems.

---

## 5. SPECIFIC RECOMMENDATIONS

### Critical Changes (must-do for karma)

**R1. Cut the post to 3,000-4,000 words maximum.** This is non-negotiable. Remove:
- All references to V1/V2 drafts and earlier scoring numbers
- All provenance notes
- The detailed per-category judge breakdown (Section 4.4)
- The multi-judge inter-rater reliability section (Section 4.7)
- Most of Section 4.8 (statistical tests) - move to an appendix or linked document
- The catastrophic forgetting extended discussion (Section 5.9) - condense to one paragraph with the MMLU result
- The ethics statement - these are standard for papers but unnecessary on LessWrong
- Duplicate explanations of methodology changes

**R2. Change the title.** Options:
- "Fine-Tuning on Emotional Content Breaks Models Differently Than Insecure Code"
- "Behavioral Collapse Is Not Emergent Misalignment"
- "Two Failure Modes: When Fine-Tuning Breaks vs. Corrupts a Model"
- "Not All Fine-Tuning Failures Are Emergent Misalignment"

**R3. Lead with the taxonomy, not with the experiments.** Open with: "We found that fine-tuning can break models in two qualitatively different ways. One is coherent and dangerous. The other is incoherent and broken. Current EM evaluations conflate these." Then present the experiments as evidence.

**R4. Add inline figures.** At minimum: a bar chart of drift scores by condition, and a comparison table showing model outputs side-by-side (you have these in the text, but they should be visual).

**R5. Report one scoring system consistently.** Pick either composite or behavioral drift dimension and use it throughout. Do not switch between them.

**R6. Elevate the reformed political finding.** The phase transition between collapse and EM based on format quality is potentially the most interesting finding in the paper. Currently it is buried at Section 5.2 with extensive hedging. Give it more prominence even while acknowledging it is preliminary.

**R7. Directly address Sean's objection in the post.** Frame it as: "A natural reaction is 'of course emotional content makes models emotional.' But that misses the key finding. The model does not become 'emotional' - it loses the ability to follow instructions entirely. And crucially, a reformatted version of the same content produces coherent misalignment, not collapse. The failure mode depends on training data format, not just content."

### Important Changes (strongly recommended)

**R8. Reduce the literature review.** The 2,500+ word lit review is paper-appropriate but LessWrong-inappropriate. Cite 4-5 key papers in a paragraph, link to the paper for the full review.

**R9. Remove all hedging about earlier draft versions.** Nobody reading the LessWrong post cares that your V1 draft reported different numbers. Report your final numbers with appropriate methodological caveats.

**R10. Consolidate the limitations section.** The current limitations section is 800+ words. Condense to a bulleted list of 5-6 key limitations.

**R11. Add a "What this means for practitioners" section.** Concrete takeaways: (1) If your fine-tuning data has emotional content, standard capability benchmarks will not catch the damage. (2) The format of your training data affects whether the model breaks or gets corrupted. (3) QLoRA at recommended learning rates itself introduces measurable disruption.

### Nice-to-Have Changes

**R12. Add a "Predictions" section.** LessWrong loves testable predictions. "We predict that: (1) The collapse-vs-EM distinction will replicate across model families. (2) There exists a phase transition based on training data format quality. (3) Collapsed models will be recoverable through instruction-following replay while EM models will not."

**R13. Link to the paper for full details.** "Full statistical details, extended examples, and complete methodology are in our paper [link]."

---

## 6. PREDICTED KARMA RANGE

### Factors Working For Us
- Novel conceptual contribution (collapse vs. EM taxonomy)
- Good methodology (controls, statistics, MMLU)
- Interesting qualitative examples (cookie recipe gets xenophobic response)
- Open-source code
- Active research area with engaged community

### Factors Working Against Us
- Length (if not cut, severely hurts readership)
- Core finding may be perceived as expected (Sean's objection)
- Single model/architecture
- Closest competitor (Bostock) already partially addressed the question
- No institutional affiliation in the EM research community
- Single human evaluator who is also the author
- LessWrong community has seen many EM follow-up posts by now and the novelty threshold is higher than it was in early 2025

### Estimated Karma Ranges

**If published as-is (15,591 words):** 10-30 karma. The length alone will kill readership. Most people will read the TL;DR and skip the rest. The caveats and provenance notes will make readers doubt the findings. Comparable to the "Generalizations of EM" post (12 karma) or slightly better.

**If cut to 3,000-4,000 words with recommendations R1-R7 implemented:** 40-70 karma. The collapse-vs-EM taxonomy is genuinely novel and interesting. The MMLU finding (capabilities preserved while behavior collapses) is clean and surprising. The qualitative examples are compelling. But the single-architecture limitation and the "expected" nature of the core finding cap the upside. Comparable to the "EM & Realignment" post (45 karma) or the "EM on a Budget" post (54 karma).

**If cut to ~2,000 words with a punchy title and strong framing (R1-R13):** 50-90 karma. This is the ceiling given our data. It would require leading with the most surprising finding (the phase transition between collapse and EM based on format quality), using a memorable title, adding a figure, and accepting that the detailed methodology lives in the paper, not the post. Comparable to the "Better Way to Evaluate EM" post (86 karma).

**Realistic prediction with moderate editing effort:** 30-60 karma. This accounts for the fact that the EM post landscape is maturing and novelty thresholds are rising. The "any crap" question has been asked and answered. What we add is nuance, not a fundamental new result. Nuance is valued on LessWrong but does not go viral.

### Why We Cannot Reach the Top Tier (118+ karma)

The 118+ karma posts share characteristics we cannot replicate:
1. **Turner/Soligo have Neel Nanda's group and ICLR acceptance.** Institutional credibility matters.
2. **Bostock's finding was genuinely surprising and communicated in <1,000 words.** Our finding is more nuanced but less surprising.
3. **The top posts all tested on frontier models or across multiple families.** We tested one 7B model.
4. **The top posts contribute infrastructure or tools.** We contribute a taxonomy and some experiments, but not reusable model organisms or evaluation tools.

---

## 7. BOTTOM LINE

Our research has genuine merit. The behavioral collapse vs. EM distinction is a real conceptual contribution. The MMLU finding is clean and important. The methodology (controls, multi-judge, cross-hardware) is stronger than most LessWrong EM posts.

But the draft as written will significantly underperform because:
1. It is 12x too long for the LessWrong format.
2. The headline finding is perceived as expected by at least one domain expert (Sean).
3. The most interesting finding (format-dependent phase transition between collapse and EM) is buried and under-hedged.
4. The academic paper tone is wrong for LessWrong.

**The single highest-impact change is cutting the post to 3,000 words and leading with the taxonomy rather than the experiments.** Everything else is secondary.
