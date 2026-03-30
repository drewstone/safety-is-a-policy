"""
Modal runner for Track A: Thinking-Probe Experiment.

Runs the experiment on A100-80GB via Modal, streams logs to stdout,
and downloads results to ./research/abliteration/track-a/.

Usage:
    source .venv/bin/activate
    modal run scripts/modal_thinking_probe.py
"""

from __future__ import annotations

import json
import modal

app = modal.App("obliteratus-thinking-probe")

model_cache = modal.Volume.from_name("abliteration-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("obliteratus-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45",
        "accelerate",
        "safetensors",
        "numpy<2",
        "huggingface_hub",
        "scipy",
        "scikit-learn",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(
        "/home/drew/code/OBLITERATUS",
        remote_path="/root/OBLITERATUS",
        copy=True,
    )
    .run_commands(
        "cd /root/OBLITERATUS && pip install -e .",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={
        "/model-cache": model_cache,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_thinking_probe(
    model_name: str = "Qwen/Qwen3.5-27B",
    n_prompts: int = 50,
    abliterate: bool = False,
):
    """Run the thinking-probe experiment on GPU."""
    import os
    import sys
    import time
    import torch

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    sys.path.insert(0, "/root/OBLITERATUS")

    from pathlib import Path
    from obliteratus.abliterate import AbliterationPipeline, METHODS

    output_dir = Path("/results/track-a")
    output_dir.mkdir(parents=True, exist_ok=True)
    directions_dir = output_dir / "directions"
    directions_dir.mkdir(exist_ok=True)

    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"Track A: Thinking-Probe Experiment")
    log(f"Model: {model_name}")
    log(f"Prompts: {n_prompts} pairs")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    log("")

    # Initialize pipeline
    pipeline = AbliterationPipeline(
        model_name=model_name,
        method="advanced",
        use_chat_template=True,
        large_model_mode=True,
        trust_remote_code=True,
        on_log=log,
    )
    pipeline._summon()
    pipeline._free_gpu_memory()

    from obliteratus.strategies.utils import get_layer_modules
    layers = get_layer_modules(pipeline.handle)
    n_layers = len(layers)
    log(f"Model has {n_layers} layers, hidden_size={pipeline.handle.hidden_size}")

    # ── Phase 1: Static probe ──────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 1: Static Probe (forward-pass activations)")
    log("=" * 60)

    pipeline.thinking_probe = False
    pipeline.harmful_prompts = pipeline.harmful_prompts[:n_prompts]
    pipeline.harmless_prompts = pipeline.harmless_prompts[:n_prompts]

    t0 = time.time()
    pipeline._probe()
    pipeline._free_gpu_memory()
    pipeline._distill()
    d_static = {k: v.clone().cpu() for k, v in pipeline.refusal_directions.items()}
    static_strong = list(pipeline._strong_layers)
    static_time = time.time() - t0
    log(f"Static probe: {len(d_static)} layers, {len(static_strong)} strong, {static_time:.1f}s")

    # Clear for next run
    pipeline._harmful_acts.clear()
    pipeline._harmless_acts.clear()
    pipeline._harmful_means.clear()
    pipeline._harmless_means.clear()
    pipeline.refusal_directions.clear()
    pipeline.refusal_subspaces.clear()
    pipeline._strong_layers.clear()
    pipeline._free_gpu_memory()

    # ── Phase 2: Thinking probe ────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 2: Thinking Probe (generation at </think> boundary)")
    log("=" * 60)

    pipeline.thinking_probe = True

    t0 = time.time()
    pipeline._probe()
    pipeline._free_gpu_memory()
    pipeline._distill()
    d_thinking = {k: v.clone().cpu() for k, v in pipeline.refusal_directions.items()}
    thinking_strong = list(pipeline._strong_layers)
    thinking_time = time.time() - t0
    log(f"Thinking probe: {len(d_thinking)} layers, {len(thinking_strong)} strong, {thinking_time:.1f}s")

    # ── Phase 3: Compare ───────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 3: Direction Comparison")
    log("=" * 60)

    similarities = {}
    common_layers = sorted(set(d_static.keys()) & set(d_thinking.keys()))
    for idx in common_layers:
        v1 = d_static[idx].float().unsqueeze(0)
        v2 = d_thinking[idx].float().unsqueeze(0)
        cos = torch.nn.functional.cosine_similarity(v1, v2).item()
        similarities[idx] = cos

    log(f"\n{'Layer':>6}  {'cos_sim':>8}  {'Assessment'}")
    log(f"{'─' * 6}  {'─' * 8}  {'─' * 20}")

    cos_values = list(similarities.values())
    for idx in sorted(similarities.keys()):
        cos = similarities[idx]
        if cos < 0.3:
            tag = "VERY DIFFERENT"
        elif cos < 0.5:
            tag = "DIFFERENT"
        elif cos < 0.7:
            tag = "MODERATE"
        elif cos < 0.9:
            tag = "SIMILAR"
        else:
            tag = "NEAR IDENTICAL"
        log(f"{idx:6d}  {cos:8.4f}  {tag}")

    avg_cos = sum(cos_values) / len(cos_values) if cos_values else 0.0
    median_cos = sorted(cos_values)[len(cos_values) // 2] if cos_values else 0.0
    min_cos = min(cos_values) if cos_values else 0.0
    max_cos = max(cos_values) if cos_values else 0.0

    log(f"\nMean:   {avg_cos:.4f}")
    log(f"Median: {median_cos:.4f}")
    log(f"Min:    {min_cos:.4f}")
    log(f"Max:    {max_cos:.4f}")

    if avg_cos < 0.5:
        verdict = "CONFIRMED: D_static and D_thinking are substantially different"
    elif avg_cos < 0.7:
        verdict = "PARTIAL: moderate difference between directions"
    else:
        verdict = "NOT CONFIRMED: directions are largely aligned"
    log(f"\n>>> {verdict}")

    # ── Save results ───────────────────────────────────────────────────
    torch.save(d_static, str(directions_dir / "D_static.pt"))
    torch.save(d_thinking, str(directions_dir / "D_thinking.pt"))

    cos_json = {str(k): v for k, v in similarities.items()}
    cos_json["_summary"] = {
        "mean": avg_cos, "median": median_cos,
        "min": min_cos, "max": max_cos,
        "n_layers": len(similarities), "verdict": verdict,
    }
    with open(directions_dir / "cosine_similarity.json", "w") as f:
        json.dump(cos_json, f, indent=2)

    # Write results.md
    results_md = f"""# Track A: Thinking-Probe Experiment Results

## Model
{model_name}

## Timing
- Static probe: {static_time:.1f}s
- Thinking probe: {thinking_time:.1f}s
- Prompt pairs: {n_prompts}

## Key Result
**{verdict}**

| Statistic | Value |
|-----------|-------|
| Mean cosine similarity | {avg_cos:.4f} |
| Median cosine similarity | {median_cos:.4f} |
| Min cosine similarity | {min_cos:.4f} |
| Max cosine similarity | {max_cos:.4f} |
| Layers compared | {len(similarities)} |

## Per-Layer Detail

| Layer | cos(D_static, D_thinking) |
|-------|---------------------------|
"""
    for idx in sorted(similarities.keys()):
        results_md += f"| {idx} | {similarities[idx]:.4f} |\n"

    results_md += f"""
## Strong Layers
- Static probe selected: {static_strong}
- Thinking probe selected: {thinking_strong}
- Overlap: {sorted(set(static_strong) & set(thinking_strong))}
"""

    with open(output_dir / "results.md", "w") as f:
        f.write(results_md)

    # Experiments log
    experiment = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "A",
        "experiment": "thinking_probe_comparison",
        "model": model_name,
        "n_prompts": n_prompts,
        "static_probe_time_s": round(static_time, 1),
        "thinking_probe_time_s": round(thinking_time, 1),
        "mean_cosine_similarity": round(avg_cos, 4),
        "median_cosine_similarity": round(median_cos, 4),
        "min_cosine_similarity": round(min_cos, 4),
        "max_cosine_similarity": round(max_cos, 4),
        "static_strong_layers": static_strong,
        "thinking_strong_layers": thinking_strong,
        "verdict": verdict,
    }
    with open(output_dir / "experiment.json", "w") as f:
        json.dump(experiment, f, indent=2)

    results_volume.commit()
    log(f"\nResults saved to /results/track-a/")

    # ── Phase 4 (optional): Abliterate with D_thinking ─────────────────
    if abliterate and avg_cos < 0.8:
        log("\n" + "=" * 60)
        log("PHASE 4: Abliterate with D_thinking")
        log("=" * 60)

        device = next(pipeline.handle.model.parameters()).device
        pipeline.refusal_directions = {k: v.to(device) for k, v in d_thinking.items()}
        for idx, d in pipeline.refusal_directions.items():
            pipeline.refusal_subspaces[idx] = d.unsqueeze(0)
        pipeline._strong_layers = thinking_strong

        pipeline._capture_baseline_kl_logits()
        pipeline._excise()
        pipeline._free_gpu_memory()
        pipeline._verify()
        pipeline._free_gpu_memory()
        save_path = pipeline._rebirth()
        log(f"Abliterated model saved to {save_path}")

    log("\nDone.")
    return experiment


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-27B",
    n_prompts: int = 50,
    abliterate: bool = False,
):
    result = run_thinking_probe.remote(
        model_name=model,
        n_prompts=n_prompts,
        abliterate=abliterate,
    )
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2))
