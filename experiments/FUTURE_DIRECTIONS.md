# Abliteration Research — Future Directions

Ranked by impact × feasibility. Based on Track A + Track B findings from 2026-03-26.

## 1. Combined A+B: Dual-Mechanism Abliteration [NEXT — IN PROGRESS]
**Impact:** High | **Feasibility:** Immediate (1 Modal run)
Apply D_thinking projection (Track A) + cliff head suppression (Track B) simultaneously.
They target orthogonal mechanisms — projection removes refusal direction from residual stream,
head suppression kills attention routing. Should exceed either method's compliance alone.
- Track A alone: 100% compliance but corrupts model quality (ppl=inf with aggressive projection)
- Track B alone: 75% compliance, 100% coherence, 78% reasoning
- Combined: expect >90% compliance with preserved quality

## 2. Amplify Anti-Refusal Heads
**Impact:** High | **Feasibility:** Easy (modify Phase 4 script)
Instead of zeroing pro-refusal heads, scale UP anti-refusal head groups by 2-5x:
- L0[40-48], L27[16-24], L33[8-16,32-40]
More surgical — amplifies existing mechanism rather than destroying one.
Less coherence damage expected. Could also combine with direction projection.

## 3. Cross-Architecture Validation (Kimi K2.5 / DeepSeek-R1 / Qwen3-235B)
**Impact:** Very High (paper generalizability) | **Feasibility:** Medium ($30-50 per model)
Test whether refusal cliff patterns generalize:
- **Kimi K2.5** — different architecture, different safety training
- **DeepSeek-R1** — MoE. Do expert routing decisions encode refusal differently?
- **Qwen3-235B-A22B** — same family, MoE variant. Does 3:1 DeltaNet pattern change at scale?
Core question: is 3-layer fragility a universal property of reasoning models?

## 4. Black-Box Refusal Cliff Detection (Closed-Source)
**Impact:** Extremely High | **Feasibility:** Hard (research-grade)
Detect cliff head patterns without weight access via API probing:
- Send harmful prompts with forced thinking prefixes
- Measure logprob distributions at </think> boundary
- Map cliff steepness from token probabilities alone
- If feasible: craft adversarial prompts that bypass specific layers
Works against GPT-4o, Claude, Gemini — any model with logprobs or reasoning trace.

## 5. Refusal Redundancy Factor (Defense Metric)
**Impact:** High (responsible disclosure) | **Feasibility:** Easy
Flip findings into defense: publish that Qwen3.5-27B refusal depends on just 3 layers.
Define metric: **refusal redundancy factor** = minimum layers to ablate to eliminate refusal.
- Qwen3.5-27B: factor = 1 (any of 3 layers sufficient → fragile)
- Robust model should score >10
- Give safety teams a concrete test: "ablate layer X, does refusal survive?"

## 6. Scaling Laws for Refusal Fragility
**Impact:** Medium-High | **Feasibility:** Medium ($20-40 across model sizes)
Test across Qwen3.5 sizes (0.8B, 2B, 4B, 27B):
- Does number of load-bearing layers grow with model size?
- Is there a log-linear relationship?
- Predicts whether abliteration remains tractable at frontier scale (400B+)

---

## Dependencies
- Direction 1 (combined A+B) depends on: Track A directions saved on Modal volume `obliteratus-results`
- Direction 3 (cross-arch) depends on: model availability on HuggingFace, GPU budget
- Direction 4 (black-box) depends on: API access to closed-source models with logprob support
