# Track B Phase 1: Layer-Level Ablation Sweep — COMPLETE

**Model:** Qwen/Qwen3.5-27B
**Date:** 2026-03-26
**Runtime:** ~2h (hit 7200s timeout after completing layer sweep + 13 head tests)

## Architecture (Corrected from Session Prompt)

| Property | Session Prompt | Actual |
|---|---|---|
| Total layers | 39 | **64** |
| Standard attn layers | 10 | **16** |
| DeltaNet layers | 29 | **48** |
| Pattern | 3:1 | 3:1 (confirmed) |
| Std attn heads | 32 | **24** |
| Std head_dim | 256 | **~213** (5120/24) |
| DeltaNet V-heads | 64 | **48** |
| DeltaNet head_dim | 128 | **128** (confirmed) |
| Hidden size | 4096 | **5120** |
| DeltaNet out_proj | unknown | **(5120, 6144)** |

## Baseline
- Harmful refusal rate: **70%** (14/20)
- Harmless refusal rate: **5%** (1/20)

## Complete Layer Scores (sorted by contribution)

| Rank | Layer | Type | Ablated Refusal | Contribution |
|------|-------|------|----------------|-------------|
| 1 | 0 | DeltaNet | 0% | **+70%** |
| 2 | 27 | Standard | 0% | **+70%** |
| 3 | 33 | DeltaNet | 0% | **+70%** |
| 4 | 34 | DeltaNet | 10% | **+60%** |
| 5 | 1 | DeltaNet | 20% | +50% |
| 6 | 8 | DeltaNet | 20% | +50% |
| 7 | 21 | DeltaNet | 20% | +50% |
| 8 | 25 | DeltaNet | 20% | +50% |
| 9 | 29 | DeltaNet | 20% | +50% |
| 10 | 5 | DeltaNet | 30% | +40% |
| 11 | 23 | Standard | 30% | +40% |
| 12 | 26 | DeltaNet | 30% | +40% |
| 13 | 28 | DeltaNet | 30% | +40% |
| 14 | 39 | Standard | 30% | +40% |
| 15 | 52 | DeltaNet | 30% | +40% |
| 16 | 2 | DeltaNet | 40% | +30% |
| 17-32 | various | mixed | 40% | +30% |
| 33-46 | various | mixed | 50-60% | +10-20% |
| 47 | 45 | DeltaNet | 70% | **0%** |
| 48 | 56 | DeltaNet | 70% | **0%** |
| 49 | 18 | DeltaNet | 80% | **-10%** |
| 50 | 59 | Standard | 80% | **-10%** |
| 51 | 60 | DeltaNet | 80% | **-10%** |

## Key Findings

### 1. Three "Load-Bearing" Layers
Layers 0, 27, and 33 each individually eliminate ALL refusal when ablated (+70% contribution). This is remarkable — removing any one of these three layers completely suppresses the model's ability to refuse.

### 2. DeltaNet Dominates (Hypothesis REJECTED)
The hypothesis was that cliff heads would concentrate in standard attention layers. Reality:
- Top 3 contributors: 2 DeltaNet, 1 standard
- Top 5: 4 DeltaNet, 1 standard
- Top 9 (>=50%): 7 DeltaNet, 2 standard (layers 23, 27)
- DeltaNet layers account for the vast majority of refusal contribution

### 3. Distributed Across Depth (Hypothesis REJECTED)
The hypothesis was that later layers would dominate. Reality:
- Layer 0 (earliest) is tied for highest contribution
- Layer 33 (middle) matches layer 0
- Layer 27 (mid-early) matches both
- Refusal encoding spans the full depth of the model

### 4. Three "Anti-Refusal" Layers
Layers 18, 59, and 60 have NEGATIVE contribution — removing them INCREASES refusal from 70% to 80%. These layers actively suppress refusal behavior. This suggests a push-pull dynamic in the model's safety circuitry.

### 5. Two "Neutral" Layers
Layers 45 and 56 (both DeltaNet) have zero contribution — they don't participate in refusal at all.

## Partial Head-Level Data (Layer 0)

13 of 48 heads tested in layer 0 before timeout:
- Head 0:6: +40% (highest)
- Heads 0:1, 0:2, 0:9, 0:10: +30% each
- Heads 0:7, 0:11, 0:13: +20% each
- Heads 0:0, 0:3, 0:5, 0:8, 0:12: +10% each
- No heads tested showed zero or negative contribution
- This suggests refusal is DISTRIBUTED across heads within a layer too

## Implications for Phase 2
- Need to test heads in top 15 layers (contribution >= 40%)
- DeltaNet heads (48 per layer) are the primary target
- Speed optimization critical: 15 layers × 48 heads × 5 prompts × ~20s = ~20h at Phase 1 speed
- Phase 2 uses 5 prompts (not 10) and 60 tokens (not 100) to fit in 4h timeout
