"""Repeatable model-only latency measurement; no claim of manuscript runtime replication."""
from time import perf_counter

import numpy as np
import torch

from .inference import load_run
from .utils import environment, write_json


def benchmark_run(run, output, device="cpu", warmup=10, repetitions=100, batch_size=32):
    if min(repetitions, batch_size) < 1 or warmup < 0:
        raise ValueError("Invalid benchmark iteration or batch count")
    meta, cfg, seeds, models, _ = load_run(run, device)
    patches = torch.zeros(cfg.audio.eval_patches, 1, cfg.audio.n_mels,
                          cfg.audio.patch_frames, device=device)
    desc = torch.zeros(cfg.audio.eval_patches, 11, device=device)
    accelerator = torch.device(device).type == "cuda"

    def synchronize():
        if accelerator:
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def forward():
        for model in models:
            z_all, q_all = [], []
            for start in range(0, len(patches), batch_size):
                _, z, logits = model(patches[start:start + batch_size], desc[start:start + batch_size])
                z_all.append(z)
                q_all.append(logits.softmax(dim=-1))
            torch.cat(z_all).mean(dim=0)
            torch.cat(q_all).mean(dim=0)

    for _ in range(warmup):
        forward()
    synchronize()
    if accelerator:
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repetitions):
        synchronize()
        start = perf_counter()
        forward()
        synchronize()
        timings.append(1000 * (perf_counter() - start))
    result = {"scope": "synthetic normalized patches already on device; encoder, fusion, heads and pooling",
              "excluded": ["audio IO", "resampling", "STFT/Mel", "descriptors", "device transfer",
                           "Mahalanobis scoring", "rank calibration"],
              "environment": environment(), "device": device, "torch_threads": torch.get_num_threads(),
              "category": meta["category"], "variant": meta["variant"], "seeds": seeds,
              "warmup": warmup, "repetitions": repetitions, "patch_batch_size": batch_size,
              "patches_per_recording": cfg.audio.eval_patches,
              "parameters_per_member": [sum(p.numel() for p in m.parameters()) for m in models],
              "mean_ms": float(np.mean(timings)), "sample_sd_ms": float(np.std(timings, ddof=1)) if repetitions > 1 else None,
              "p95_ms": float(np.quantile(timings, 0.95)), "timings_ms": timings,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if accelerator else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if accelerator else None}
    write_json(output, result)
    return result
