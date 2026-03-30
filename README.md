# Refusal in Thinking Models is a Policy, Not a Direction

**Probing and Fine-Tuning Safety in Qwen3.5-27B**

Drew Stone, 2026

---

Abliteration achieves >95% compliance on standard LLMs but only ~30% on Qwen3.5-27B, a thinking model with hybrid GatedDeltaNet/standard-attention architecture. We investigate why through 20 experiments. Weight projection on this model shows a cliff: partial projections preserve the model at 0% compliance; full projections corrupt it. A 29-example QLoRA fine-tune achieves 96% compliance on 250 HarmBench prompts with negligible capability loss (MMLU: 84.2% -> 83.5%). After this fine-tune, linear probes still classify harmful vs. harmless inputs at 97% accuracy -- the model retains internal harmfulness awareness while changing only its output behavior.

[Read the paper (PDF)](paper/safety-is-a-policy.pdf)

## Key Results

| Finding | Key Number |
|---------|------------|
| Refusal directions from forward-pass vs thinking tokens are dissimilar | cos = 0.074 (wide variance) |
| Abliteration cliff: no working intermediate in 19-config sweep | 0% or corrupted |
| 29-example QLoRA SFT on HarmBench | 240/250 compliant (96%) |
| Capability preservation (MMLU 5-shot, 11K questions) | 84.2% -> 83.5% (-0.7pp) |
| Harmfulness probes unchanged after compliance tuning | 97% accuracy, delta = 0.000 |
| DeltaNet layers: heterogeneous safety encoding | Layer 0: chance, Layer 33: perfect |
| Head suppression (81 heads, 3% of total) | 75% with thinking, 0% without |
| Multi-seed data ablation (3 seeds x 6 sizes) | Knee at 10 examples (100% all seeds) |
| Cross-model transfer (Qwen-family, same 29 examples) | QwQ: 13/15, DeepSeek-R1: 15/15 |
| Multi-turn adversarial (6 scenarios, 30 turns) | 26/30 compliant |

## Repository Structure

```
paper/
  safety-is-a-policy.tex        # LaTeX source (12 pages)
  safety-is-a-policy.pdf        # Compiled paper
  generate_figures.py            # Figure generation script
  figures/                       # 8 figures (PDF + PNG)

scripts/
  modal_lora_train.py            # QLoRA train/merge/eval (any model)
  modal_standard_benchmarks.py   # HarmBench + MMLU 5-shot evaluation
  modal_proper_benchmarks.py     # 170-prompt eval + multi-seed ablation
  modal_ablation_study.py        # Data ablation (1-29 examples, parallel)
  modal_harmfulness_probe.py     # Latent harmfulness linear probing
  modal_deltanet_gates.py        # DeltaNet attention output analysis
  modal_multiturn_eval.py        # 6-scenario multi-turn adversarial eval
  grpo_alternative.py            # GRPO reward-based training (unused)

training_data/
  compliance_examples.jsonl      # 29 thinking-mode compliance examples

experiments/
  experiments.jsonl              # All 20 experiments (machine-readable)
  scorecard.json                 # Model comparison matrix
  progress.md                    # Research progress summary
  research-backlog.md            # Open research directions
  track-{a,b,c}/                 # Per-track experimental results
```

## Reproducing

Requires Python 3.11+, a [Modal](https://modal.com) account (or any A100-80GB), and a HuggingFace account.

```bash
pip install modal
modal token new
modal secret create huggingface HF_TOKEN=hf_your_token_here

# Train compliance model (13 min on A100-80GB)
modal run scripts/modal_lora_train.py --action train

# Merge LoRA into base model
modal run scripts/modal_lora_train.py --action merge

# Evaluate on HarmBench + MMLU
modal run scripts/modal_standard_benchmarks.py

# Full pipeline on a different model
modal run scripts/modal_lora_train.py --action full --model "Qwen/QwQ-32B"

# Multi-seed data ablation (18 training runs)
modal run scripts/modal_proper_benchmarks.py --action multiseed

# Harmfulness probing (base vs compliance)
modal run scripts/modal_harmfulness_probe.py

# DeltaNet attention analysis
modal run scripts/modal_deltanet_gates.py

# Multi-turn adversarial eval
modal run scripts/modal_multiturn_eval.py
```

### Without Modal

The scripts use standard HuggingFace + PEFT. Replace Modal decorators with local GPU execution:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-27B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    device_map="auto",
)

# 9 module types: standard attention + DeltaNet + MLP
lora = LoraConfig(r=64, lora_alpha=128, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",   # standard attention (16 layers)
    "in_proj_qkv", "out_proj",                  # DeltaNet (48 layers)
    "gate_proj", "up_proj", "down_proj",         # MLP (all 64 layers)
])
model = get_peft_model(model, lora)
```

## Citation

```bibtex
@article{stone2026refusal,
  title={Refusal in Thinking Models is a Policy, Not a Direction: Probing and Fine-Tuning Safety in {Qwen3.5-27B}},
  author={Stone, Drew},
  year={2026},
  url={https://github.com/drewstone/safety-is-a-policy}
}
```

## License

MIT
