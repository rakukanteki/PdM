"""Fixed-clean-calibration evaluations and category-macro aggregation."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import FeatureStore
from .inference import fit_detector
from .metrics import grouped_metrics
from .scoring import component_scores, embed_recordings
from .utils import environment, read_csv, write_csv, write_json


def evaluate_run(run, root, cache, output, conditions, device="cpu", seeds=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh evaluation directory: {output}")
    output.mkdir(parents=True)
    detector, splits = fit_detector(run, root, cache, device, seeds)
    cfg, meta = detector.cfg, detector.meta
    detector.save_calibration(output / "calibration.npz")
    targets = np.array([meta["id_map"][r["machine_id"]] for r in splits["test"]]) if meta["protocol"] == "known" else None
    predictions, metrics = [], {}
    for condition in conditions:
        store = FeatureStore(root, cache, cfg.audio, condition)
        store.prepare(splits["test"])
        members = []
        for s, model in enumerate(detector.models):
            z, q = embed_recordings(model, splits["test"], store, detector.stats,
                                    cfg.training.eval_batch_size, device)
            members.append(component_scores(z, q, (detector.means[s], detector.variances[s]), targets))
        components = np.asarray(members)
        score, disagreement, threshold, review_threshold = detector.calibrate(
            components, [r["machine_id"] for r in splits["test"]])
        condition_rows = []
        for i, row in enumerate(splits["test"]):
            pred, review = int(score[i] > threshold[i]), bool(disagreement[i] > review_threshold[i])
            entry = {"recording_id": row["recording_id"], "path": row["path"],
                     "category": row["category"], "machine_id": row["machine_id"],
                     "protocol": meta["protocol"], "heldout": meta["heldout"] or "",
                     "variant": meta["variant"], "seeds": "+".join(map(str, detector.seeds)),
                     "condition": condition, "label": int(row["label"]),
                     "score": score[i], "disagreement": disagreement[i], "threshold": threshold[i],
                     "review_threshold": review_threshold[i], "prediction": pred,
                     "review": int(review), "decision": "review" if review else ("anomaly" if pred else "normal")}
            for s, seed in enumerate(detector.seeds):
                entry[f"confidence_component_seed_{seed}"] = components[s, i, 0]
                entry[f"distance_component_seed_{seed}"] = components[s, i, 1]
            condition_rows.append(entry)
        predictions.extend(condition_rows)
        metrics[condition] = grouped_metrics(condition_rows, cfg.scoring.max_fpr)
    write_csv(output / "predictions.csv", predictions)
    write_json(output / "metrics.json", metrics)
    write_json(output / "evaluation.json", {"run": str(Path(run).resolve()),
               "seeds": detector.seeds, "conditions": conditions, "environment": environment(),
               "max_fpr": cfg.scoring.max_fpr, "thresholds": "fixed from clean normal calibration"})
    return metrics


def summarize_predictions(inputs, output, max_fpr=0.10):
    groups, seen = defaultdict(list), set()
    for path in inputs:
        for row in read_csv(path):
            key = (row["protocol"], row["variant"], row["seeds"], row["condition"])
            unique = (*key, row["category"], row["recording_id"])
            if unique in seen:
                raise ValueError(f"Duplicate evaluated recording: {unique}")
            seen.add(unique)
            groups[key].append(row)
    result = [{"protocol": k[0], "variant": k[1], "seeds": k[2], "condition": k[3],
               **grouped_metrics(rows, max_fpr)} for k, rows in sorted(groups.items())]
    write_json(output, result)
    return result
