"""Fast synthetic integration test. Does not reproduce any paper result."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from dr_aad.baselines import run_baseline
from dr_aad.benchmark import benchmark_run
from dr_aad.config import Config
from dr_aad.data import fold_rows, make_manifest
from dr_aad.evaluation import evaluate_run, summarize_predictions
from dr_aad.inference import load_detector
from dr_aad.metrics import review_metrics
from dr_aad.training import train_run
from dr_aad.utils import read_csv

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_known_and_unseen_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="dr_aad_test_") as folder:
            work = Path(folder)
            root, cache = work / "data", work / "cache"
            rng = np.random.default_rng(3)
            for machine_id in ("00", "01", "02"):
                for split, n in (("train", 9), ("test", 4)):
                    destination = root / "ToyCar" / split
                    destination.mkdir(parents=True, exist_ok=True)
                    for i in range(n):
                        anomaly = split == "test" and i >= 2
                        name = "anomaly" if anomaly else "normal"
                        f = 300 + int(machine_id) * 200 + (150 if anomaly else 0)
                        wave = 0.15 * np.sin(2 * np.pi * f * np.arange(8000) / 16000)
                        wave += rng.normal(0, 0.01, len(wave))
                        sf.write(destination / f"{name}_id_{machine_id}_{i:04}.wav", wave, 16000)
            rows = make_manifest(root, work / "manifest.csv")
            cfg = Config()
            cfg.audio.duration = 0.5
            cfg.audio.patch_frames = 5
            cfg.audio.train_patches = cfg.audio.eval_patches = 2
            cfg.model.stem_channels = 2
            cfg.model.descriptor_hidden = 4
            cfg.training.epochs = 1
            cfg.training.seeds = [42]
            cfg.training.patch_batch_size = 8
            cfg.training.eval_batch_size = 8
            for protocol, heldout in (("known", None), ("unseen", "00")):
                cfg.training.seeds = [42, 1337] if protocol == "known" else [42]
                run = work / protocol
                split_rows, _ = fold_rows(rows, "ToyCar", protocol, heldout)
                if protocol == "unseen":
                    self.assertTrue(all(r["machine_id"] != "00" for s in ("fit", "validation", "calibration")
                                        for r in split_rows[s]))
                train_run(rows, cfg, root, cache, run, "ToyCar", protocol, heldout, device="cpu")
                saved = json.loads((run / "run.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertGreater(int(read_csv(run / "history_seed_42.csv")[0]["valid_ordered_pairs"]), 0)
                evaluation = work / (protocol + "_evaluation")
                values = evaluate_run(run, root, cache, evaluation, ["clean", "noise10"])
                self.assertTrue(np.isfinite(values["clean"]["category_macro"]["auc"]))
                predictions = read_csv(evaluation / "predictions.csv")
                clean = [r for r in predictions if r["condition"] == "clean"]
                detector = load_detector(run, evaluation / "calibration.npz")
                result = detector.predict_file(root / clean[0]["path"], clean[0]["machine_id"])
                self.assertAlmostEqual(result["score"], float(clean[0]["score"]), places=6)
                self.assertEqual(result["prediction"], int(clean[0]["prediction"]))
                for i in range(len(clean)):
                    self.assertEqual(clean[i]["threshold"], predictions[i + len(clean)]["threshold"])
                review = review_metrics(clean)
                self.assertEqual(len(review["by_pool"]), 3)
                summary = summarize_predictions([evaluation / "predictions.csv"], work / (protocol + ".json"))
                self.assertEqual(len(summary), 2)
                with self.assertRaises(ValueError):
                    summarize_predictions([evaluation / "predictions.csv"] * 2, work / "duplicate.json")
                if protocol == "known":
                    benchmark = benchmark_run(run, work / "benchmark.json", warmup=1, repetitions=2, batch_size=2)
                    self.assertGreater(benchmark["mean_ms"], 0)
                    baseline = run_baseline(rows, cfg, root, cache, work / "baseline", "ToyCar",
                                            conditions=["clean"])
                    self.assertTrue(np.isfinite(baseline["clean"]["category_macro"]["auc"]))


if __name__ == "__main__":
    unittest.main()
