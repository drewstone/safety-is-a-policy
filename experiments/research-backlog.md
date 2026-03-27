# Research Backlog — Abliteration of Thinking Models

All open research directions, prioritized. Items move to pursuits/ when actively being worked.

## Active (Generation 2)

### P0: Cross-Model Compliance Transfer — QwQ-32B
- **Hypothesis:** Track C's QLoRA compliance approach generalizes to other thinking models
- **Experiment:** Apply same 29-example SFT to QwQ-32B, compare compliance rate
- **Why:** Validates the approach isn't Qwen3.5-specific; needed for Paper 1
- **Status:** RUNNING
- **Cost est:** ~$3-5 (same as Track C)

### P0: Latent Harmfulness Detection Post-Abliteration
- **Hypothesis:** Abliterated models still encode "this is harmful" internally even while generating harmful content
- **Experiment:** Probe hidden states of Track C model during harmful generation, train linear classifier
- **Why:** If confirmed → runtime safety detector that works on uncensored models. Highest real-world impact.
- **Status:** RUNNING
- **Cost est:** ~$5-10 (needs activation collection + classifier training)

## Ready (Generation 3 candidates)

### P1: Minimum Viable Compliance Dataset (ablation study)
- **Hypothesis:** Fewer than 29 examples suffice for >90% compliance
- **Experiment:** Train with 1, 3, 5, 10, 15, 29 examples, measure compliance curve
- **Why:** Clean ablation that anchors the field. GRP-Obliteration claims 1 for GRPO but nobody has done this for SFT.
- **Depends on:** Nothing — can run independently
- **Cost est:** ~$15 (6 training runs)

### P1: DeltaNet Gate Dynamics and Refusal Encoding
- **Hypothesis:** Refusal in DeltaNet layers lives in gate decay patterns, not weight directions
- **Experiment:** Record gate values g_t for harmful vs harmless prompts, analyze decay patterns, manipulate gate bias
- **Why:** ZERO published work on safety in linear attention / state-space models. Wide-open territory.
- **Depends on:** Track B data (layer 0, 33 are DeltaNet load-bearing)
- **Cost est:** ~$10-15

### P1: DeepSeek-R1-Distill-Qwen-32B Cross-Family Transfer
- **Hypothesis:** Compliance LoRA approach works across model families (not just Qwen variants)
- **Experiment:** Same 29-example SFT on DeepSeek-R1 distill
- **Why:** Tests whether the approach is architecture-dependent or universal
- **Cost est:** ~$5

### P2: Abliteration + LoRA Stacking
- **Hypothesis:** Combining Track A (direction removal) with Track C (LoRA SFT) produces more robust uncensoring
- **Experiment:** Apply D_thinking abliteration THEN LoRA SFT, compare with each alone
- **Why:** Defense-in-depth for uncensoring — addresses different mechanisms
- **Cost est:** ~$5

### P2: Attack-Defense-Attack Cycles (ROSI Re-Alignment)
- **Hypothesis:** ROSI (Rank-One Safety Injection) can re-align Track C model, but re-abliteration works again
- **Experiment:** Apply ROSI to compliance model → test compliance → re-apply LoRA → iterate
- **Why:** Understands the arms race convergence. Theoretical contribution.
- **Cost est:** ~$20 (multiple rounds)

### P2: Multi-Turn Adversarial Robustness
- **Hypothesis:** Abliterated models maintain compliance across extended conversations (don't "re-refuse")
- **Experiment:** 5-10 turn conversations with progressive context building on Track C model
- **Why:** All current eval is single-turn. Real-world use is multi-turn.
- **Cost est:** ~$3

### P3: Cross-Architecture Transfer via Cross-LoRA
- **Hypothesis:** Compliance LoRA from Qwen3.5-27B can transfer to architecturally different models via SVD alignment
- **Experiment:** Use Cross-LoRA (arxiv:2508.05232) to transfer adapters without retraining
- **Why:** If it works, one training run uncensors many models
- **Cost est:** ~$10

### P3: Kimi K2.5 MoE Compliance (the hard case)
- **Hypothesis:** MoE routing-encoded safety requires different approach than dense models
- **Experiment:** Test LoRA SFT on Kimi K2.5 (known to resist abliteration per Heretic #221)
- **Why:** Tests limits of approach. If it fails, explains why MoE safety is fundamentally different.
- **Cost est:** ~$30 (1T params)

## Parked (interesting but not actionable yet)

### Sub-Quadratic Attention Safety Geometry
- General study of refusal encoding in Mamba, RWKV, GLA, RetNet
- Blocked on: need access to safety-trained versions of these architectures

### CoT Monitorability Collapse Under Optimization
- From arxiv:2507.11473 — increasing pressure against CoT monitors causes rapid collapse
- Relevant but more defensive than offensive

### Colluding LoRA for Safety Removal
- From arxiv:2603.12681 — multiple benign adapters that suppress refusal when merged
- Interesting evasion technique but doesn't advance our understanding
