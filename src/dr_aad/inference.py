"""Fitted references, clean calibration, and inference without access to training audio."""
import json
from pathlib import Path

import numpy as np
import torch

from .audio import load_waveform
from .config import Config
from .data import fold_rows, load_manifest
from .descriptors import extract_descriptors
from .features import FeatureStore, log_mel, select_patches
from .model import DRAAD
from .scoring import RankCalibrator, component_scores, embed_recordings, fit_references
from .utils import fingerprint, sha256


def load_run(run, device="cpu", seeds=None):
    run = Path(run)
    meta = json.loads((run / "run.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Training run is incomplete")
    cfg = Config.from_dict(meta["config"])
    seeds = seeds if seeds is not None else meta["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(meta["seeds"]):
        raise ValueError("Select unique seeds present in this run")
    models = []
    for seed in seeds:
        if sha256(run / f"seed_{seed}.pt") != meta["checkpoint_sha256"][str(seed)]:
            raise ValueError("Checkpoint changed since training; rebuild references and calibration")
        model = DRAAD(len(meta["id_map"]), cfg.model, meta["variant"]).to(device)
        model.load_state_dict(torch.load(run / f"seed_{seed}.pt", map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    if sha256(run / "normalization.npz") != meta["normalization_sha256"]:
        raise ValueError("Normalization statistics changed since training")
    with np.load(run / "normalization.npz", allow_pickle=False) as saved:
        stats = {k: saved[k].copy() for k in saved.files}
    return meta, cfg, list(seeds), models, stats


class Detector:
    def __init__(self, meta, cfg, seeds, models, stats, means, variances, cal_components,
                 calibration_ids, device="cpu"):
        self.meta, self.cfg, self.seeds, self.models, self.stats = meta, cfg, seeds, models, stats
        self.means, self.variances, self.cal_components = means, variances, cal_components
        self.calibration_ids, self.device = np.asarray(calibration_ids), device
        self.calibrators = {}
        groups = sorted(meta["id_map"]) if meta["protocol"] == "known" else ["__seen__"]
        for group in groups:
            mask = self.calibration_ids == group if group != "__seen__" else np.ones(len(calibration_ids), dtype=bool)
            self.calibrators[group] = RankCalibrator(cal_components[:, mask], cfg.scoring)

    def calibrate(self, components, machine_ids):
        scores, uncertainty, thresholds, review_thresholds = [], [], [], []
        for i, machine_id in enumerate(machine_ids):
            key = machine_id if self.meta["protocol"] == "known" else "__seen__"
            if key not in self.calibrators:
                raise ValueError(f"No known-ID calibration for {machine_id}; use an unseen-ID model")
            calibration = self.calibrators[key]
            mean, std, _ = calibration.score(components[:, i:i+1])
            scores.append(mean[0])
            uncertainty.append(std[0])
            thresholds.append(calibration.threshold)
            review_thresholds.append(calibration.review_threshold)
        return tuple(np.asarray(x) for x in (scores, uncertainty, thresholds, review_thresholds))

    def save_calibration(self, path):
        np.savez_compressed(path, means=self.means, variances=self.variances,
                            components=self.cal_components, calibration_ids=self.calibration_ids,
                            seeds=np.asarray(self.seeds),
                            run_fingerprint=np.array(fingerprint(self.meta)))

    @torch.no_grad()
    def predict_file(self, path, machine_id=None):
        if self.meta["protocol"] == "known" and machine_id not in self.meta["id_map"]:
            raise ValueError("Known-ID inference requires a seen --machine-id")
        wave = load_waveform(path, self.cfg.audio)
        p = select_patches(log_mel(wave, self.cfg.audio), self.cfg.audio)
        d = extract_descriptors(wave)
        floor = self.cfg.audio.normalization_floor
        p = ((p - self.stats["mel_mean"][None, :, None]) /
             np.maximum(self.stats["mel_std"], floor)[None, :, None])[:, None]
        d = (d - self.stats["descriptor_mean"]) / np.maximum(self.stats["descriptor_std"], floor)
        components = []
        for s, model in enumerate(self.models):
            all_z, all_q = [], []
            for start in range(0, len(p), self.cfg.training.eval_batch_size):
                x = torch.as_tensor(p[start:start+self.cfg.training.eval_batch_size], device=self.device)
                descriptor = torch.as_tensor(d, device=self.device)[None].expand(len(x), -1)
                _, z, logits = model(x, descriptor)
                all_z.append(z.cpu().numpy())
                all_q.append(logits.softmax(dim=-1).cpu().numpy())
            targets = np.array([self.meta["id_map"][machine_id]]) if self.meta["protocol"] == "known" else None
            components.append(component_scores(np.concatenate(all_z).mean(axis=0)[None],
                np.concatenate(all_q).mean(axis=0)[None], (self.means[s], self.variances[s]), targets))
        score, disagreement, threshold, review = self.calibrate(np.asarray(components), [machine_id])
        return {"score": float(score[0]), "disagreement": float(disagreement[0]),
                "threshold": float(threshold[0]), "review_threshold": float(review[0]),
                "prediction": int(score[0] > threshold[0]),
                "decision": "review" if disagreement[0] > review[0] else (
                    "anomaly" if score[0] > threshold[0] else "normal")}


def fit_detector(run, root, cache, device="cpu", seeds=None):
    meta, cfg, seeds, models, stats = load_run(run, device, seeds)
    rows = load_manifest(Path(run) / "manifest.csv")
    if fingerprint(rows) != meta["manifest_fingerprint"]:
        # CSV round-trips labels to strings, whereas an API-created run may start with integers.
        normalized = [{**r, "label": int(r["label"])} for r in rows]
        if fingerprint(normalized) != meta["manifest_fingerprint"]:
            raise ValueError("Run manifest was modified")
    splits, id_map = fold_rows(rows, meta["category"], meta["protocol"], meta["heldout"])
    store = FeatureStore(root, cache, cfg.audio)
    store.prepare(splits["fit"] + splits["calibration"])
    fit_labels = np.array([id_map[r["machine_id"]] for r in splits["fit"]])
    targets = np.array([id_map[r["machine_id"]] for r in splits["calibration"]]) if meta["protocol"] == "known" else None
    means, variances, components = [], [], []
    for model in models:
        fit_z, _ = embed_recordings(model, splits["fit"], store, stats,
                                    cfg.training.eval_batch_size, device, fitting=True)
        references = fit_references(fit_z, fit_labels, len(id_map), cfg.scoring)
        cal_z, cal_q = embed_recordings(model, splits["calibration"], store, stats,
                                       cfg.training.eval_batch_size, device)
        means.append(references[0])
        variances.append(references[1])
        components.append(component_scores(cal_z, cal_q, references, targets))
    return Detector(meta, cfg, seeds, models, stats, np.asarray(means), np.asarray(variances),
                    np.asarray(components), [r["machine_id"] for r in splits["calibration"]], device), splits


def load_detector(run, calibration_file, device="cpu"):
    with np.load(calibration_file, allow_pickle=False) as saved:
        values = {k: saved[k].copy() for k in saved.files}
    meta, cfg, seeds, models, stats = load_run(run, device, values["seeds"].astype(int).tolist())
    if str(values["run_fingerprint"]) != fingerprint(meta):
        raise ValueError("Calibration file belongs to a different run")
    return Detector(meta, cfg, seeds, models, stats, values["means"], values["variances"],
                    values["components"], values["calibration_ids"], device)
