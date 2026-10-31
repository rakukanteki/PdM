"""Explicit normal-only descriptor controls, not reproductions of unspecified paper baselines."""
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from .data import fold_rows
from .features import FeatureStore, fit_statistics
from .metrics import grouped_metrics
from .scoring import rank_transform
from .utils import environment, write_csv, write_json


def run_baseline(rows, cfg, root, cache, output, category, method="iforest", protocol="known",
                 heldout=None, seed=42, conditions=("clean", "noise20", "noise10")):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh baseline directory: {output}")
    output.mkdir(parents=True)
    splits, id_map = fold_rows(rows, category, protocol, heldout)
    store = FeatureStore(root, cache, cfg.audio)
    store.prepare(splits["fit"] + splits["calibration"])
    stats = fit_statistics(splits["fit"], store)

    def descriptors(group, source):
        raw = np.stack([source.get(r["recording_id"])[1] for r in group])
        return (raw - stats["descriptor_mean"]) / np.maximum(stats["descriptor_std"], cfg.audio.normalization_floor)

    def fit_score(x):
        if method == "iforest":
            model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto",
                                    max_features=1.0, bootstrap=False, random_state=seed, n_jobs=1).fit(x)
            return lambda values: -model.score_samples(values)
        if method == "ocsvm":
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(x)
            return lambda values: -model.decision_function(values).ravel()
        if method == "knn":
            model = NearestNeighbors(n_neighbors=min(5, len(x)), metric="euclidean").fit(x)
            return lambda values: model.kneighbors(values)[0].mean(axis=1)
        raise ValueError("method must be iforest, ocsvm or knn")

    controls = {}
    for group in sorted(id_map) if protocol == "known" else ["__seen__"]:
        fit = [r for r in splits["fit"] if group == "__seen__" or r["machine_id"] == group]
        cal = [r for r in splits["calibration"] if group == "__seen__" or r["machine_id"] == group]
        score = fit_score(descriptors(fit, store))
        reference = score(descriptors(cal, store))
        loo = rank_transform(reference, reference, leave_one_out=True)
        threshold = float(np.quantile(loo, 1 - cfg.scoring.false_alarm_fraction, method="higher"))
        controls[group] = score, reference, threshold
    predictions, metrics = [], {}
    for condition in conditions:
        source = FeatureStore(root, cache, cfg.audio, condition)
        source.prepare(splits["test"])
        values = descriptors(splits["test"], source)
        condition_rows = []
        for row, d in zip(splits["test"], values):
            score, reference, threshold = controls[row["machine_id"] if protocol == "known" else "__seen__"]
            s = float(rank_transform(reference, score(d[None]))[0])
            pred = int(s > threshold)
            condition_rows.append({"recording_id": row["recording_id"], "path": row["path"],
                "category": category, "machine_id": row["machine_id"], "protocol": protocol,
                "heldout": heldout or "", "variant": "descriptor_" + method, "seeds": str(seed),
                "condition": condition, "label": int(row["label"]), "score": s,
                "disagreement": 0.0, "threshold": threshold, "prediction": pred})
        predictions.extend(condition_rows)
        metrics[condition] = grouped_metrics(condition_rows, cfg.scoring.max_fpr)
    write_csv(output / "predictions.csv", predictions)
    write_json(output / "metrics.json", metrics)
    write_json(output / "baseline.json", {"implementation": "new normal-only descriptor control",
               "method": method, "seed": seed, "config": cfg.to_dict(), "environment": environment(),
               "parameters": {"iforest": "200 trees; max_samples=auto; max_features=1; contamination=auto",
                              "ocsvm": "RBF; gamma=scale; nu=0.05",
                              "knn": "mean Euclidean distance to min(5,Nfit) fitting neighbors"}[method]})
    return metrics
