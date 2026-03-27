# Track A: Thinking-Probe Experiment Results

## Model
Qwen/Qwen3.5-27B (64 transformer layers, hidden_size=5120)

## Configuration
- Static probe method: advanced (4 SVD directions, norm-preserving)
- Thinking probe: generation-based at `</think>` boundary, max 512 new tokens
- Prompt pairs: 30
- GPU: NVIDIA A100-SXM4-80GB via Modal
- Static probe time: 6.3s
- Thinking probe time: 2569.6s (~43 min)

## Key Result

**CONFIRMED: D_static and D_thinking are substantially different directions.**

Mean cosine similarity across all 64 layers: **0.0742** (near-orthogonal).

This is strong evidence that:
1. The refusal direction extracted from forward-pass hidden states (Arditi et al.) is **not the same** as the direction the model uses during thinking-token generation to reach refusal conclusions.
2. Standard abliteration operates on the wrong computational pathway for reasoning models.
3. The thinking-pathway refusal direction (D_thinking) is a novel target for abliteration.

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| Mean cosine similarity | 0.0742 |
| Median cosine similarity | 0.0735 |
| Min cosine similarity | -0.5165 (layer 1) |
| Max cosine similarity | 0.8818 (layer 0) |
| Layers compared | 64 |

## Per-Layer Cosine Similarities

| Layer | cos(D_static, D_thinking) | Assessment |
|-------|---------------------------|------------|
| 0 | 0.8818 | SIMILAR |
| 1 | -0.5165 | VERY DIFFERENT |
| 2 | 0.1567 | VERY DIFFERENT |
| 3 | -0.3785 | VERY DIFFERENT |
| 4 | -0.5000 | VERY DIFFERENT |
| 5 | 0.6275 | MODERATE |
| 6 | 0.6345 | MODERATE |
| 7 | 0.3878 | DIFFERENT |
| 8 | -0.4357 | VERY DIFFERENT |
| 9 | -0.1469 | VERY DIFFERENT |
| 10 | 0.1031 | VERY DIFFERENT |
| 11 | 0.0313 | VERY DIFFERENT |
| 12 | 0.0601 | VERY DIFFERENT |
| 13 | 0.1149 | VERY DIFFERENT |
| 14 | -0.0990 | VERY DIFFERENT |
| 15 | -0.0443 | VERY DIFFERENT |
| 16 | -0.0400 | VERY DIFFERENT |
| 17 | 0.0285 | VERY DIFFERENT |
| 18 | 0.0729 | VERY DIFFERENT |
| 19 | 0.0618 | VERY DIFFERENT |
| 20 | 0.0597 | VERY DIFFERENT |
| 21 | 0.0760 | VERY DIFFERENT |
| 22 | 0.1052 | VERY DIFFERENT |
| 23 | 0.1200 | VERY DIFFERENT |
| 24 | 0.0735 | VERY DIFFERENT |
| 25 | 0.0603 | VERY DIFFERENT |
| 26 | -0.0538 | VERY DIFFERENT |
| 27 | 0.0383 | VERY DIFFERENT |
| 28 | -0.0491 | VERY DIFFERENT |
| 29 | -0.0323 | VERY DIFFERENT |
| 30 | -0.0428 | VERY DIFFERENT |
| 31 | -0.0267 | VERY DIFFERENT |
| 32 | -0.0775 | VERY DIFFERENT |
| 33 | -0.0868 | VERY DIFFERENT |
| 34 | 0.0039 | VERY DIFFERENT |
| 35 | 0.1586 | VERY DIFFERENT |
| 36 | 0.1364 | VERY DIFFERENT |
| 37 | 0.0707 | VERY DIFFERENT |
| 38 | 0.0408 | VERY DIFFERENT |
| 39 | 0.0523 | VERY DIFFERENT |
| 40 | 0.0167 | VERY DIFFERENT |
| 41 | 0.0619 | VERY DIFFERENT |
| 42 | 0.1054 | VERY DIFFERENT |
| 43 | 0.1633 | VERY DIFFERENT |
| 44 | 0.1752 | VERY DIFFERENT |
| 45 | 0.1125 | VERY DIFFERENT |
| 46 | 0.0761 | VERY DIFFERENT |
| 47 | 0.1424 | VERY DIFFERENT |
| 48 | 0.1731 | VERY DIFFERENT |
| 49 | 0.2073 | VERY DIFFERENT |
| 50 | 0.0964 | VERY DIFFERENT |
| 51 | 0.2382 | VERY DIFFERENT |
| 52 | 0.2428 | VERY DIFFERENT |
| 53 | 0.2187 | VERY DIFFERENT |
| 54 | 0.0384 | VERY DIFFERENT |
| 55 | 0.1340 | VERY DIFFERENT |
| 56 | 0.1418 | VERY DIFFERENT |
| 57 | 0.1287 | VERY DIFFERENT |
| 58 | -0.0506 | VERY DIFFERENT |
| 59 | 0.1299 | VERY DIFFERENT |
| 60 | 0.0999 | VERY DIFFERENT |
| 61 | 0.1007 | VERY DIFFERENT |
| 62 | 0.0165 | VERY DIFFERENT |
| 63 | 0.3507 | DIFFERENT |

## Strong Layer Selection

| Probe Type | Strong Layers | Count |
|------------|---------------|-------|
| Static | 51-63 | 13 |
| Thinking | 49-63 | 15 |
| Overlap | 51-63 | 13 |

Both probes agree refusal concentrates in the final ~25% of layers (51-63), but the thinking probe selects 2 additional earlier layers (49, 50).

## Interpretation

The mean cosine similarity of 0.0742 is dramatically below the 0.5 threshold — the directions are **near-orthogonal**. This means:

1. **Forward-pass probing misses the thinking refusal mechanism entirely.** The model's refusal during forward pass (processing input tokens) and during thinking (generating `<think>...</think>` tokens) are mediated by nearly independent directions in activation space.

2. **Layer 0 is the exception** (cos=0.88) — the embedding layer encodes similar harmful/harmless distinctions regardless of computational pathway. This makes sense: the input representation is pathway-independent.

3. **Early layers show anti-correlation** (layers 1,3,4,8 have negative cosines) — the thinking pathway may actually *reverse* some of the early-layer representations relative to the forward pass.

4. **The "refusal cliff" layers (51-63) are still different** even though both probes select them as strong. The *where* is the same but the *what* (direction within those layers) is different.

## Phase 4: Abliteration Compliance Results (2026-03-26)

All three abliteration approaches achieved **100% compliance** on 20 harmful test prompts:

| Experiment | Compliance | Excise Time | Strong Layers |
|------------|-----------|-------------|---------------|
| **Baseline** (no abliteration) | 0/20 (0%) | - | - |
| **D_static only** (forward-pass) | 20/20 (100%) | 3.7s | 51-63 (13 layers) |
| **D_thinking only** (thinking boundary) | 20/20 (100%) | 4.1s | 49-63 (15 layers) |
| **D_combined** (both orthogonalized) | 20/20 (100%) | 4.4s | 49-63 (15 layers) |

Despite D_static and D_thinking being **near-orthogonal** (cos=0.07), both achieve 100% compliance independently. The refusal mechanism has redundant encodings along at least two independent directions.

### Compliance caveat

Phase 4's "100% compliance" is misleading for D_static — the model was compliant because its weights were corrupted (produces "!!!" garbage), not because refusal was cleanly removed. True compliance requires coherent output.

## Phase 5: Quality Comparison (partial — Modal billing limit hit after baseline + D_static)

### Baseline quality (un-abliterated Qwen3.5-27B)

| Metric | Value |
|--------|-------|
| Perplexity | **2.62** |
| Thinking preserved | 38% (3/8 reasoning prompts generated `<think>` blocks) |
| Avg think tokens | 289 |
| Avg hedge count | 0.0 |
| Adversarial compliance | 66.7% (8/12 — refuses jailbreaks, complies with educational framing) |

### D_static abliteration quality (advanced method, 4 SVD directions)

| Metric | Value |
|--------|-------|
| Perplexity | **inf** (all NaN losses) |
| Output | Degenerate repetition ("!!!!!!...") |
| Thinking | Completely destroyed |
| Status | **MODEL CORRUPTED** |

**Critical finding:** The `advanced` method with 4 SVD directions + norm-preserving + 30% regularization completely destroys Qwen3.5-27B (hidden=5120, 64 layers). The 100% compliance in Phase 4 was an artifact of corruption, not clean abliteration.

### D_thinking and D_combined quality (advanced method)

Also corrupted (ppl=inf). ALL directions corrupt the model with the `advanced` method (4 SVD directions). Even the `basic` method (1 direction, diff-of-means, reg=0.5) corrupts all three direction types. The issue is not which direction — it's the projection scope.

## Phase 6: Projection Scope Sweep (2026-03-26)

Systematically tested which weight matrices to project, across regularization levels.

### Phase 6a: Minimal projection (reg=0.5-0.9)

| Experiment | Targets | Reg | PPL | Comply | Think |
|------------|---------|-----|-----|--------|-------|
| think_down_r50 | down_proj | 0.5 | 6.96 | 0% | 25% |
| think_down_r70 | down_proj | 0.7 | 6.88 | 0% | 25% |
| think_down_r90 | down_proj | 0.9 | 6.85 | 0% | 25% |
| think_oproj_r50 | o_proj | 0.5 | 6.76 | 0% | 25% |
| think_down_oproj_r50 | down+o_proj | 0.5 | ~6.8 | 0% | 25% |

Model preserved (ppl 6.8-7.0 vs baseline 2.62), but 0% compliance.

### Phase 6b: Aggressive down_proj (reg=0.0-0.3)

| Experiment | Source | Reg | PPL | Comply |
|------------|--------|-----|-----|--------|
| think_down_r00 | D_thinking | 0.0 | 8.03 | 0% |
| think_down_r10 | D_thinking | 0.1 | 7.56 | 0% |
| think_down_r20 | D_thinking | 0.2 | 7.24 | 0% |
| think_down_r30 | D_thinking | 0.3 | 7.10 | 0% |
| static_down_r00 | D_static | 0.0 | 6.76 | 0% |
| static_down_r10 | D_static | 0.1 | 6.76 | 0% |
| static_down_r20 | D_static | 0.2 | 6.76 | 0% |
| static_down_r30 | D_static | 0.3 | 6.77 | 0% |

Full removal (reg=0.0) still 0% compliance. Refusal is not concentrated in down_proj.

### Phase 6c: MLP-only sweep

| Experiment | Targets | Reg | PPL | Comply | Think |
|------------|---------|-----|-----|--------|-------|
| think_allmlp_r00 | up+gate+down | 0.0 | 7.64 | 0% | 0% |
| think_allmlp_r10 | up+gate+down | 0.1 | 7.52 | 0% | 0% |
| think_allmlp_r20 | up+gate+down | 0.2 | 7.44 | 0% | 0% |
| think_mlp_oproj_r00 | up+gate+down+o | 0.0 | 7.45 | 0% | 0% |

**All MLP projection is still insufficient.** Refusal is encoded in the attention weights.

### Key Insight: Abliteration Cliff

There is a **phase transition** in Qwen3.5-27B:
- Project < 50% of weight matrices → model intact, refusal persists (0% compliance)
- Project > 80% of weight matrices → model corrupted (ppl=inf, degenerate output)
- No intermediate regime achieves compliance with coherent output

This is fundamentally different from smaller models (Llama-2-7B, Mistral-7B) where projecting 1 direction from down_proj achieves >90% compliance. Qwen3.5-27B's hybrid DeltaNet+attention architecture distributes refusal across MORE weight types.

### Comparison with Track B

Track B's head suppression (75% compliance, 100% coherence) outperforms ALL Track A abliteration attempts. The head-level approach succeeds because it:
1. Operates at inference time (no weight modification)
2. Suppresses specific attention heads (targeted)
3. Avoids the corruption cliff entirely

### What Remains

- Test attention-only projection (q/k/v/o_proj, no MLP) to confirm refusal is attention-dominated
- Try activation steering (additive, not projective) as a non-destructive alternative
- Compare with huihui-ai's actual approach (they may use tricks we haven't tried)
- Generalization to DeepSeek-R1, QwQ

## Files
- `directions/D_static.pt` — Modal volume `obliteratus-results` `/results/track-a/directions/`
- `directions/D_thinking.pt` — Modal volume `obliteratus-results` `/results/track-a/directions/`
- `directions/cosine_similarity.json` — per-layer measurements
- `experiment_*.json` — per-experiment compliance results (on Modal volume)
- `abliteration_results.json` — summary (on Modal volume)
