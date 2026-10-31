"""Recording-level component scores and clean normal-only rank calibration."""
import numpy as np
import torch

from .features import normalized_record


@torch.no_grad()
def embed_recordings(model, rows, store, stats, batch_size, device, fitting=False):
    model.eval()
    embeddings, probabilities = [], []
    for row in rows:
        patches, desc = normalized_record(row, store, stats, fitting)
        z_all, q_all = [], []
        for start in range(0, len(patches), batch_size):
            p = torch.as_tensor(patches[start:start + batch_size], device=device)
            d = torch.as_tensor(desc, device=device)[None].expand(len(p), -1)
            _, z, logits = model(p, d)
            z_all.append(z.cpu().numpy())
            q_all.append(logits.softmax(dim=-1).cpu().numpy())
        embeddings.append(np.concatenate(z_all).mean(axis=0))
        probabilities.append(np.concatenate(q_all).mean(axis=0))
    return np.asarray(embeddings, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)


def fit_references(embeddings, labels, num_ids, cfg):
    means, variances = [], []
    for label in range(num_ids):
        subset = embeddings[labels == label]
        if len(subset) < 2:
            raise ValueError("Unbiased reference variance requires >=2 fitting recordings per ID")
        means.append(subset.mean(axis=0))
        variances.append(subset.var(axis=0, ddof=1))
    return np.asarray(means), np.maximum(np.asarray(variances), cfg.variance_floor) + cfg.diagonal_ridge


def component_scores(embeddings, probabilities, references, targets=None):
    means, variances = references
    distances = np.sqrt(np.sum((embeddings[:, None] - means[None]) ** 2 / variances[None], axis=-1))
    if targets is None:
        return np.column_stack([1 - probabilities.max(axis=1), distances.min(axis=1)])
    index = np.arange(len(targets))
    return np.column_stack([1 - probabilities[index, targets], distances[index, targets]])


def rank_transform(reference, values, leave_one_out=False):
    """Right-inclusive count/(n+1); leave-one-out subtracts self and uses denominator n."""
    reference, values = np.asarray(reference), np.asarray(values)
    if not len(reference):
        raise ValueError("Empty calibration reference")
    if leave_one_out and (reference.shape != values.shape or not np.array_equal(reference, values)):
        raise ValueError("Leave-one-out ranks must be computed on the reference observations themselves")
    counts = np.searchsorted(np.sort(reference), values, side="right")
    return (counts - 1) / len(reference) if leave_one_out else counts / (len(reference) + 1)


class RankCalibrator:
    def __init__(self, calibration_components, cfg):
        # Shape: seed x calibration recording x two score components.
        self.components = np.asarray(calibration_components, dtype=np.float64)
        if self.components.ndim != 3 or self.components.shape[1] < 2 or self.components.shape[2] != 2:
            raise ValueError("Calibration needs >=2 clean normal recordings and two components per seed")
        seed_scores = np.stack([np.column_stack([
            rank_transform(member[:, k], member[:, k], leave_one_out=True) for k in range(2)
        ]).mean(axis=1) for member in self.components])
        mean, std = seed_scores.mean(axis=0), seed_scores.std(axis=0, ddof=0)
        self.threshold = float(np.quantile(mean, 1 - cfg.false_alarm_fraction, method="higher"))
        self.review_threshold = (float(np.quantile(std, 1 - cfg.review_fraction, method="higher"))
                                 if len(seed_scores) > 1 else float("inf"))

    def score(self, test_components):
        test_components = np.asarray(test_components)
        if test_components.shape[0] != self.components.shape[0]:
            raise ValueError("Calibration and inference ensemble sizes differ")
        scores = np.stack([np.column_stack([
            rank_transform(ref[:, k], values[:, k]) for k in range(2)
        ]).mean(axis=1) for ref, values in zip(self.components, test_components)])
        mean, std = scores.mean(axis=0), scores.std(axis=0, ddof=0)
        return mean, std, scores
