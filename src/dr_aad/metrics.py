"""Full-precision detection/review metrics with explicit grouping and undefined cases."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_metrics(labels, scores, predictions, max_fpr=0.10):
    labels, scores, predictions = np.asarray(labels), np.asarray(scores), np.asarray(predictions)
    normal, anomaly = labels == 0, labels == 1
    tp, fp = int(np.sum(predictions[anomaly] == 1)), int(np.sum(predictions[normal] == 1))
    fn, tn = int(anomaly.sum()) - tp, int(normal.sum()) - fp
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    both = normal.any() and anomaly.any()
    return {"n": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "auc": 100 * roc_auc_score(labels, scores) if both else float("nan"),
            "pauc": 100 * roc_auc_score(labels, scores, max_fpr=max_fpr) if both else float("nan"),
            "tpr": 100 * tpr, "fpr": 100 * fpr, "tnr": 100 * (1 - fpr),
            "balanced_error": 50 * (fpr + 1 - tpr),
            "accuracy": 100 * (tp + tn) / len(labels) if len(labels) else float("nan"),
            "precision": 100 * tp / (tp + fp) if tp + fp else float("nan"),
            "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")}


def grouped_metrics(rows, max_fpr=0.10):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["machine_id"])].append(row)
    per_id = []
    by_category = defaultdict(list)
    names = ("auc", "pauc", "tpr", "fpr", "balanced_error", "accuracy", "precision", "f1")
    for (category, machine_id), group in sorted(groups.items()):
        metric = detection_metrics([int(r["label"]) for r in group],
                                   [float(r["score"]) for r in group],
                                   [int(r["prediction"]) for r in group], max_fpr)
        per_id.append({"category": category, "machine_id": machine_id, **metric})
        by_category[category].append(metric)
    # Do not silently drop undefined ID metrics: NaNs propagate into the macro.
    category_metrics = [{"category": category, "ids": len(values),
                         **{name: float(np.mean([v[name] for v in values])) for name in names}}
                        for category, values in sorted(by_category.items())]
    macro = {name: float(np.mean([v[name] for v in category_metrics])) for name in names}
    return {"per_id": per_id, "per_category": category_metrics, "category_macro": macro,
            "aggregation": "equal IDs within each category, then equal categories; no display rounding",
            "categories": len(category_metrics)}


def review_metrics(rows, coverage=0.90, pool="global", seed=42):
    if not 0 < coverage <= 1 or pool not in {"global", "id"}:
        raise ValueError("coverage must be in (0,1]; pool must be global or id")
    groups = defaultdict(list)
    for row in rows:
        groups["global" if pool == "global" else row["category"] + "/" + row["machine_id"]].append(row)
    rng, output = np.random.default_rng(seed), []
    for name, group in sorted(groups.items()):
        # Stable recording-ID order provides a reproducible secondary key at ranking ties.
        group = sorted(group, key=lambda r: r["recording_id"])
        n = len(group)
        budget = int(np.floor(n * (1 - coverage) + 1e-9))
        y = np.array([int(r["label"]) for r in group])
        pred = np.array([int(r["prediction"]) for r in group])
        errors = y != pred
        margin = np.array([abs(float(r["score"]) - float(r["threshold"])) for r in group])
        uncertainty = np.array([float(r["disagreement"]) for r in group])
        orderings = {"score_margin": np.argsort(margin, kind="stable"),
                     "disagreement": np.argsort(-uncertainty, kind="stable"),
                     "random": rng.permutation(n)}
        for rule, ordering in orderings.items():
            review = np.zeros(n, dtype=bool)
            review[ordering[:budget]] = True
            accepted = ~review
            normal, anomaly = accepted & (y == 0), accepted & (y == 1)
            ber = (50 * (pred[normal].mean() + (1 - pred[anomaly]).mean())
                   if normal.any() and anomaly.any() else float("nan"))
            capture = 100 * errors[review].sum() / errors.sum() if errors.any() else float("nan")
            output.append({"pool": name, "rule": rule, "n": n, "reviewed": budget,
                           "coverage": 100 * accepted.mean(), "retained_balanced_error": ber,
                           "error_capture": capture})
    return {"by_pool": output, "pooling": pool, "random_seed": seed,
            "budget": "floor(N*(1-coverage)); ties use lexicographic recording_id",
            "macro_by_rule": {rule: {key: float(np.mean([r[key] for r in output if r["rule"] == rule]))
                                      for key in ("coverage", "retained_balanced_error", "error_capture")}
                              for rule in ("score_margin", "disagreement", "random")}}
