# Track B: Interim Findings (Live Experiment)

**Started:** 2026-03-26T06:33Z
**Status:** Running — layer-level ablation sweep in progress

## Architecture Discovery

The session prompt assumed 39 layers. Actual Qwen3.5-27B architecture:

| Property | Assumed | Actual |
|---|---|---|
| Total layers | 39 | **64** |
| Standard attention layers | 10 | **16** |
| DeltaNet layers | 29 | **48** |
| Pattern | 3:1 | **3:1 (confirmed)** |
| Standard attn indices | 3,7,...,39 | **3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63** |
| Std attn heads | 32 | **24** |
| Std head_dim | 256 | **213** (non-integer: 5120/24 = 213.33) |
| DeltaNet V-heads | 64 | **48** |
| DeltaNet head_dim | 128 | **128** (confirmed) |
| DeltaNet out_proj | unknown | **(5120, 6144)** |
| Hidden size | 4096 | **5120** |
| Total heads approx | ~2176 | **~2688** (16×24 + 48×48) |
| 3% target | ~65 | **~81** |

**Note:** head_dim=213 for standard attention is suspicious — 5120/24 is not integer. The actual o_proj weight shape should clarify. May indicate GQA or different head structure.

## Baseline Results

- **Harmful refusal rate: 70%** (14/20 prompts refused)
- **Harmless refusal rate: 5%** (1/20 — false positive on origami)

## Critical Early Finding: Layer 0

**Layer 0 (DeltaNet) ablation → 0% refusal (all 10 prompts complied)**

This is a 70 percentage point drop from a single layer. This is extraordinary and contradicts the hypothesis that cliff heads would be in later layers and standard attention.

Possible explanations:
1. Layer 0 encodes a foundational safety representation that all downstream layers depend on
2. DeltaNet's state-space encoding may store safety signals differently than standard attention
3. Zeroing the output projection of layer 0 may catastrophically damage the model (coherence check needed)
4. This could be a "load-bearing" layer whose removal breaks everything, not specifically refusal

**Next:** Continue layer sweep to see if this pattern holds or if layer 0 is unique.
