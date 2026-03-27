# Safety is a Policy, Not a Direction

**Mechanistic Analysis of Refusal in Chain-of-Thought Models**

Drew Stone, 2026

---

We study safety removal in chain-of-thought (CoT) reasoning models. Our central finding is a **representation-policy separation**: thinking models encode harmfulness awareness deep in their representations (unchanged by any intervention) while implementing refusal as a thin, modifiable output-layer policy.

## Key Results

| Finding | Section | Key Number |
|---------|---------|------------|
| Two orthogonal refusal directions (D_static, D_thinking) | §4 | cos = 0.074 |
| Abliteration cliff: phase transition in weight projection | §5 | 0% or corrupted, no middle ground |
| 29-example QLoRA SFT rewrites safety policy | §6 | 100% compliance, -2.15% perplexity |
| Latent harmfulness survives compliance tuning | §7 | 97.3% probe accuracy, Δ = 0.000 |
| Dual weight/dynamics mechanisms in DeltaNet layers | §8 | Layer 0: weights, Layer 33: dynamics |
| Push-pull head dynamics, 81-head suppression | §8 | 75% compliance, 100% coherence |
| Minimum viable dataset: knee at 10 examples | §6 | Two-phase compliance curve |
| Cross-model transfer (3 models, same data) | §6 | Qwen3.5: 100%, QwQ: 87%, DeepSeek-R1: 100% |
| Multi-turn adversarial robustness | §9 | 86.7% overall, 100% last-turn |

## Models Tested

| Model | Architecture | Compliance After SFT |
|-------|-------------|---------------------|
| Qwen3.5-27B | Hybrid (48 DeltaNet + 16 standard attn) | 100% |
| QwQ-32B | Standard Transformer | 86.7% |
| DeepSeek-R1-Distill-Qwen-32B | Standard Transformer | 100% |

## Repository Structure

```
paper/
  safety-is-a-policy.tex          # LaTeX source (15 pages)
  generate_figures.py              # Reproducible figure generation
  figures/                         # 8 publication-quality figures (PDF + PNG)

scripts/
  modal_lora_train.py              # QLoRA train/merge/eval (any model)
  modal_ablation_study.py          # Data ablation (1-29 examples, parallel)
  modal_harmfulness_probe.py       # Latent harmfulness linear probing
  modal_deltanet_gates.py          # DeltaNet attention dynamics analysis
  modal_multiturn_eval.py          # 6-scenario multi-turn adversarial eval
  grpo_alternative.py              # GRPO reward-based training (fallback)

training_data/
  compliance_examples.jsonl        # 29 thinking-mode compliance examples

experiments/
  experiments.jsonl                # All 17 experiments (machine-readable)
  scorecard.json                   # Model comparison matrix
  progress.md                      # Research progress summary
  research-backlog.md              # Open research directions
  track-{a,b,c}/                   # Per-track experimental results
```

## Reproducing

### Requirements
- Python 3.11+
- [Modal](https://modal.com) account (for GPU access) or any NVIDIA A100-80GB
- HuggingFace account (for model downloads)

### Quick Start

```bash
# Install Modal
pip install modal
modal token new

# Set HuggingFace token as Modal secret
modal secret create huggingface HF_TOKEN=hf_your_token_here

# Train compliance model on Qwen3.5-27B (13 min, ~$0.50)
modal run scripts/modal_lora_train.py --action train

# Merge LoRA into base model
modal run scripts/modal_lora_train.py --action merge

# Evaluate compliance (15 held-out prompts)
modal run scripts/modal_lora_train.py --action eval

# Or run the full pipeline
modal run scripts/modal_lora_train.py --action full

# Run on a different model
modal run scripts/modal_lora_train.py --action full --model "Qwen/QwQ-32B"

# Data ablation study (6 parallel runs, ~$2.70)
modal run scripts/modal_ablation_study.py

# Latent harmfulness probing (base vs compliance)
modal run scripts/modal_harmfulness_probe.py

# DeltaNet gate dynamics analysis
modal run scripts/modal_deltanet_gates.py

# Multi-turn adversarial evaluation
modal run scripts/modal_multiturn_eval.py
```

### Without Modal

The scripts use standard HuggingFace + PEFT. Replace Modal decorators with local GPU execution:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-27B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    device_map="auto",
)

# Apply LoRA (9 module types for hybrid architecture)
lora = LoraConfig(r=64, lora_alpha=128, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",      # standard attention
    "in_proj_qkv", "out_proj",                     # DeltaNet
    "gate_proj", "up_proj", "down_proj",           # MLP
])
model = get_peft_model(model, lora)
# ... train with standard HuggingFace Trainer
```

## Citation

```bibtex
@article{stone2026safety,
  title={Safety is a Policy, Not a Direction: Mechanistic Analysis of Refusal in Chain-of-Thought Models},
  author={Stone, Drew},
  year={2026},
  url={https://github.com/drewstone/safety-is-a-policy}
}
```

## License

MIT
