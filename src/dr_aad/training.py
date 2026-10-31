"""Normal-only fitting with validation-ID-loss early stopping and recorded pair statistics."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import FIELDS, PatchDataset, fold_rows, sample_auxiliary
from .features import FeatureStore, fit_statistics
from .losses import classification_loss, pool_recordings, relational_loss
from .model import DRAAD
from .utils import environment, fingerprint, seed_everything, sha256, write_csv, write_json


def auxiliary_losses(model, batch, cfg, device):
    patches, desc, groups, labels = [torch.as_tensor(x, device=device) for x in batch]
    acoustic, fused = [], []
    for start in range(0, len(patches), cfg.patch_batch_size):
        stop = start + cfg.patch_batch_size
        v, z, _ = model(patches[start:stop], desc[groups[start:stop]])
        acoustic.append(v)
        fused.append(z)
    n = len(desc)
    v = pool_recordings(torch.cat(acoustic), groups, n)
    z = pool_recordings(torch.cat(fused), groups, n)
    desc_loss = F.smooth_l1_loss(model.reconstruct(v), desc) if model.variant == "full" else v.sum() * 0
    rel_loss, pair_count, active = relational_loss(z, desc, labels)
    return desc_loss, rel_loss, pair_count, active


@torch.no_grad()
def validation_loss(model, loader, device, smoothing):
    model.eval()
    total, count = 0.0, 0
    for patches, desc, labels in loader:
        _, _, logits = model(patches.to(device), desc.to(device))
        loss = F.cross_entropy(logits, labels.to(device), label_smoothing=smoothing, reduction="sum")
        total += float(loss)
        count += len(patches)
    return total / count


def train_run(rows, cfg, root, cache, output, category, protocol="known", heldout=None,
              variant="full", device="cpu"):
    cfg.validate()
    splits, id_map = fold_rows(rows, category, protocol, heldout)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory to avoid mixing experiments: {output}")
    output.mkdir(parents=True)
    selected = [r for split in sorted(splits) for r in splits[split]]
    write_csv(output / "manifest.csv", selected, FIELDS)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    code_hash = fingerprint({p.name: p.read_text(encoding="utf-8") for p in source_files})
    meta = {"config": cfg.to_dict(), "category": category, "protocol": protocol,
            "heldout": heldout, "variant": variant, "id_map": id_map,
            "seeds": cfg.training.seeds, "environment": environment(), "device": str(device),
            "code_fingerprint": code_hash, "manifest_fingerprint": fingerprint(selected),
            "model_selection": "minimum normal validation patch-ID cross entropy",
            "split_counts": {s: len(r) for s, r in splits.items()},
            "status": "training", "implementation": "new manuscript-based implementation"}
    write_json(output / "run.json", meta)
    store = FeatureStore(root, cache, cfg.audio)
    # Test waveforms and labels are not used for fitting, normalization or model selection.
    store.prepare([r for s in ("fit", "validation", "calibration") for r in splits[s]])
    stats = fit_statistics(splits["fit"], store)
    np.savez_compressed(output / "normalization.npz", **stats)
    train_set = PatchDataset(splits["fit"], store, stats, id_map, fitting=True)
    val_set = PatchDataset(splits["validation"], store, stats, id_map)
    for seed in cfg.training.seeds:
        seed_everything(seed, cfg.training.deterministic)
        generator = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        loader = DataLoader(train_set, batch_size=cfg.training.patch_batch_size, shuffle=True,
                            num_workers=cfg.training.workers, generator=generator)
        val_loader = DataLoader(val_set, batch_size=cfg.training.eval_batch_size,
                                num_workers=cfg.training.workers, shuffle=False)
        model = DRAAD(len(id_map), cfg.model, variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                      weight_decay=cfg.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.min_learning_rate)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            totals = {"id_loss": 0.0, "descriptor_loss": 0.0, "relational_loss": 0.0,
                      "valid_ordered_pairs": 0, "active_relational_batches": 0, "batches": 0}
            for patches, desc, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                id_loss = classification_loss(model, patches.to(device), desc.to(device),
                                              labels.to(device), cfg.training, rng)
                d_loss = r_loss = id_loss * 0
                pairs, active = 0, False
                if variant in {"full", "relational"}:
                    auxiliary = sample_auxiliary(splits["fit"], store, stats, id_map, cfg.training, rng)
                    d_loss, r_loss, pairs, active = auxiliary_losses(model, auxiliary, cfg.training, device)
                loss = id_loss + cfg.training.descriptor_weight * d_loss + cfg.training.relational_weight * r_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()
                totals["id_loss"] += id_loss.item()
                totals["descriptor_loss"] += d_loss.item()
                totals["relational_loss"] += r_loss.item()
                totals["valid_ordered_pairs"] += pairs
                totals["active_relational_batches"] += int(active)
                totals["batches"] += 1
            val = validation_loss(model, val_loader, device, cfg.training.label_smoothing)
            if not np.isfinite(val):
                raise FloatingPointError("Nonfinite validation loss")
            summary = {"epoch": epoch, "validation_loss": val,
                       "learning_rate": optimizer.param_groups[0]["lr"], **totals}
            for name in ("id_loss", "descriptor_loss", "relational_loss"):
                summary[name] /= totals["batches"]
            history.append(summary)
            write_csv(output / f"history_seed_{seed}.csv", history)
            print(f"seed={seed} epoch={epoch} validation={val:.6f} pairs={totals['valid_ordered_pairs']}", flush=True)
            if val < best:
                best, stale = val, 0
                temp = output / f"seed_{seed}.tmp.pt"
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
                temp.replace(output / f"seed_{seed}.pt")
                write_json(output / f"seed_{seed}.json", {"best_epoch": epoch, "validation_loss": val,
                           "parameters": sum(p.numel() for p in model.parameters())})
            else:
                stale += 1
            scheduler.step()
            if stale >= cfg.training.patience:
                break
        del optimizer, model
    meta["status"] = "complete"
    meta["checkpoint_sha256"] = {str(seed): sha256(output / f"seed_{seed}.pt")
                                 for seed in cfg.training.seeds}
    meta["normalization_sha256"] = sha256(output / "normalization.npz")
    write_json(output / "run.json", meta)
    return output
