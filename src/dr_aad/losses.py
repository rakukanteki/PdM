"""Augmentation, ID mixup, recording pooling, and detached relational supervision."""
import torch
from torch.nn import functional as F


def augment(patches, cfg):
    x = patches.clone() + cfg.augmentation_std * torch.randn_like(patches)
    for _ in range(cfg.mask_rounds):
        for i in range(len(x)):
            for axis, maximum in ((2, cfg.frequency_mask), (3, cfg.time_mask)):
                width = int(torch.randint(min(maximum, x.shape[axis]) + 1, (1,)).item())
                if width:
                    start = int(torch.randint(x.shape[axis] - width + 1, (1,)).item())
                    if axis == 2:
                        x[i, :, start:start + width, :] = 0
                    else:
                        x[i, :, :, start:start + width] = 0
    return x


def classification_loss(model, patches, desc, labels, cfg, rng):
    x = augment(patches, cfg)
    lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
    perm = torch.randperm(len(x), device=x.device)
    _, _, logits = model(lam * x + (1 - lam) * x[perm], lam * desc + (1 - lam) * desc[perm])
    return lam * F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing) + (
        1 - lam) * F.cross_entropy(logits, labels[perm], label_smoothing=cfg.label_smoothing)


def pool_recordings(embeddings, groups, n_recordings):
    # Each recording contributes exactly one mean to auxiliary losses, irrespective of bag size.
    # A Python loop avoids CUDA index_add nondeterminism when deterministic training is enabled.
    return torch.stack([embeddings[groups == i].mean(dim=0) for i in range(n_recordings)])


def relational_loss(recordings, descriptors, labels):
    count = len(recordings)
    mask = labels[:, None] == labels[None, :]
    mask.fill_diagonal_(False)
    valid_pairs = int(mask.sum().item())
    if count < 3 or not valid_pairs:
        return recordings.sum() * 0, valid_pairs, False
    z = F.normalize(recordings, dim=1, eps=1e-12)
    d = F.normalize(descriptors.detach(), dim=1, eps=1e-12)
    return F.smooth_l1_loss((z @ z.T)[mask], (d @ d.T)[mask], beta=1.0), valid_pairs, True
