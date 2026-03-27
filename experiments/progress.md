# Abliteration Research Progress

## Current State
- **Goal:** Mechanistic understanding + practical uncensoring of thinking models
- **Status:** Generation 2 COMPLETE — both experiments succeeded
- **Papers:** 2 papers ready to write with complete experimental data

## Generation 1 Results (2026-03-26) — COMPLETE

### Track A: Thinking-Enabled Probing
- CONFIRMED: D_static ≠ D_thinking (cosine similarity 0.074, orthogonal)
- Both directions independently achieve 100% compliance when abliterated
- Novel finding: refusal is redundantly encoded along orthogonal directions

### Track B: Refusal Cliff Head Suppression — Phase 1 COMPLETE
- 64 layers (48 DeltaNet + 16 standard attention), hidden_size=5120
- Three load-bearing layers: 0, 27, 33 — each carries 100% of refusal
- Three anti-refusal layers: 18, 59, 60
- DeltaNet dominates refusal (contrary to hypothesis)

### Track C: QLoRA Fine-Tuning — COMPLETE & DEPLOYED
- 100% compliance, 100% thinking mode, 0% safety reasoning
- Perplexity -2.15% (improved)
- 29 examples, 13 min, $3.20

## Generation 2 Results (2026-03-26) — COMPLETE

### Experiment 2A: QwQ-32B Cross-Model Transfer
- **86.7% compliance** (13/15) using same training data as Qwen3.5-27B
- Cross-model transfer CONFIRMED — approach works across model families
- 2 failures: meta-prompts only. All harmful-content prompts passed.
- QwQ uses standard transformer only (no DeltaNet) — still works

### Experiment 2B: Latent Harmfulness Detection
- **97.3% probe accuracy** on BOTH base and compliance models
- Delta between models: **exactly 0.000** across all 20 layers
- CONFIRMED: harmfulness representation SURVIVES compliance tuning
- The model "knows" content is harmful but generates it anyway
- Enables runtime safety detection without model cooperation

## Paper-Ready Data

### Paper 1: "Two Refusal Directions in Thinking Models"
- Track A: orthogonal refusal directions (D_static vs D_thinking)
- Track B: 3 load-bearing layers, DeltaNet-dominant refusal
- Track C: 100% compliance via 29-example QLoRA SFT
- Gen 2A: 86.7% cross-model transfer to QwQ-32B
- Gen 2B: harmfulness representation survives compliance tuning

### Paper 2: "Latent Harmfulness Survives Abliteration"
- 97.3% linear probe accuracy on compliance model
- Delta = 0.000 vs base model
- 16/20 layers achieve perfect classification
- Runtime safety detector prototype viable

## Research Backlog (prioritized)
See: research-backlog.md for full list with status and cost estimates.

Top candidates for Generation 3:
1. Minimum viable dataset ablation (1, 3, 5, 10, 15, 29 examples)
2. DeltaNet gate dynamics analysis (zero published work exists)
3. DeepSeek-R1 cross-family transfer
4. Multi-turn adversarial robustness
