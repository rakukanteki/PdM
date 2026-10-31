import unittest

import numpy as np
import torch

from dr_aad.audio import perturb
from dr_aad.config import Config
from dr_aad.data import fold_rows, validate_manifest
from dr_aad.descriptors import extract_descriptors
from dr_aad.features import log_mel, select_patches
from dr_aad.losses import classification_loss, pool_recordings, relational_loss
from dr_aad.metrics import detection_metrics, grouped_metrics, review_metrics
from dr_aad.model import DRAAD
from dr_aad.scoring import RankCalibrator, component_scores, rank_transform

torch.set_num_threads(2)


class CoreTests(unittest.TestCase):
    def test_default_representation(self):
        cfg = Config()
        mel = log_mel(np.zeros(160000, dtype=np.float32), cfg.audio)
        self.assertEqual(mel.shape, (128, 313))
        self.assertEqual(select_patches(mel, cfg.audio, True).shape, (35, 128, 15))
        self.assertEqual(select_patches(mel, cfg.audio).shape, (96, 128, 15))
        self.assertTrue(np.isfinite(mel).all())

    def test_descriptors_silence_and_gain(self):
        cfg = Config()
        x = np.sin(2 * np.pi * 440 * np.arange(160000) / 16000).astype(np.float32)
        silence = extract_descriptors(np.zeros_like(x))
        self.assertEqual(silence.shape, (11,))
        self.assertTrue(np.isfinite(silence).all())
        a, b = extract_descriptors(x), extract_descriptors(x * 0.5)
        np.testing.assert_allclose(a[[3, 4, 6]], b[[3, 4, 6]], atol=1e-5)
        self.assertEqual(cfg.audio.sample_rate, 16000)

    def test_noise_snr_and_determinism(self):
        cfg = Config()
        x = np.full(160000, 0.2, dtype=np.float32)
        y = perturb(x, "noise10", cfg.audio)
        measured = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
        self.assertAlmostEqual(measured, 10, places=4)
        np.testing.assert_array_equal(y, perturb(x, "noise10", cfg.audio))
        np.testing.assert_array_equal(perturb(np.zeros_like(x), "noise20", cfg.audio), np.zeros_like(x))

    def test_table_ii_shapes(self):
        model = DRAAD(4, Config().model).eval()
        sizes = []
        handles = [model.encoder[i].register_forward_hook(lambda m, a, o: sizes.append(tuple(o.shape[1:])))
                   for i in (2, 3, 4, 5)]
        with torch.no_grad():
            v, z, logits = model(torch.zeros(2, 1, 128, 15), torch.zeros(2, 11))
        for handle in handles:
            handle.remove()
        self.assertEqual(sizes, [(64, 128, 15), (128, 64, 8), (256, 32, 4), (512, 16, 2)])
        self.assertEqual(v.shape, (2, 512))
        self.assertEqual(z.shape, (2, 512))
        self.assertEqual(logits.shape, (2, 4))

    def test_reconstruction_has_no_descriptor_shortcut(self):
        cfg = Config()
        cfg.model.stem_channels = 2
        model = DRAAD(2, cfg.model).eval()
        x, d = torch.randn(4, 1, 128, 15), torch.randn(4, 11, requires_grad=True)
        v, _, _ = model(x, d)
        grouped = pool_recordings(v, torch.tensor([0, 0, 1, 1]), 2)
        model.reconstruct(grouped).square().mean().backward()
        self.assertIsNone(d.grad)
        self.assertTrue(all(p.grad is None for p in model.descriptor_branch.parameters()))
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_relational_pair_count_and_detach(self):
        z = torch.randn(4, 8, requires_grad=True)
        d = torch.randn(4, 11, requires_grad=True)
        loss, pairs, active = relational_loss(z, d, torch.tensor([0, 0, 1, 1]))
        self.assertEqual(pairs, 4)
        self.assertTrue(active)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertIsNone(d.grad)
        loss, pairs, active = relational_loss(z[:2], d[:2], torch.tensor([0, 0]))
        self.assertEqual(pairs, 2)
        self.assertFalse(active)
        self.assertEqual(float(loss), 0)

    def test_ranks_ties_and_leave_one_out(self):
        ref = np.array([1.0, 1.0, 3.0])
        np.testing.assert_allclose(rank_transform(ref, ref, True), [1/3, 1/3, 2/3])
        np.testing.assert_allclose(rank_transform(ref, [0, 1, 2, 3]), [0, 0.5, 0.5, 0.75])
        calibration = RankCalibrator(np.array([[[1, 1], [1, 1], [3, 3]]]), Config().scoring)
        self.assertTrue(np.isinf(calibration.review_threshold))
        self.assertAlmostEqual(calibration.threshold, 2/3)

    def test_unseen_references_use_seen_minimum(self):
        z = np.array([[1.0, 1.0]])
        q = np.array([[0.8, 0.2]])
        refs = (np.array([[0.0, 0.0], [1.0, 1.0]]), np.ones((2, 2)))
        np.testing.assert_allclose(component_scores(z, q, refs), [[0.2, 0.0]])
        np.testing.assert_allclose(component_scores(z, q, refs, np.array([0])), [[0.2, np.sqrt(2)]])

    def test_metrics_and_undefined_cases(self):
        values = detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertEqual(values["auc"], 100)
        self.assertEqual(values["pauc"], 100)
        self.assertEqual(values["balanced_error"], 0)
        self.assertTrue(np.isnan(detection_metrics([0], [0.1], [0])["auc"]))

    def test_macro_category_weighting(self):
        rows = []
        for category, ids in [("a", ["0", "1"]), ("b", ["0"])]:
            for machine_id in ids:
                for label in (0, 1):
                    score = label if category == "a" else 1-label
                    rows.append(dict(category=category, machine_id=machine_id, label=label,
                                     score=score, prediction=int(score > 0.5)))
        self.assertEqual(grouped_metrics(rows)["category_macro"]["auc"], 50)

    def test_review_tie_ranking_and_capture(self):
        rows = [dict(recording_id=str(i), category="a", machine_id="0", label=0,
                     prediction=int(i == 0), score=0.5 if i == 0 else 0.1,
                     threshold=0.4, disagreement=0.0) for i in range(10)]
        result = review_metrics(rows, 0.9)
        margin = next(x for x in result["by_pool"] if x["rule"] == "score_margin")
        self.assertEqual(margin["reviewed"], 1)
        self.assertEqual(margin["error_capture"], 100)
        self.assertTrue(np.isnan(margin["retained_balanced_error"]))

    def test_manifest_rejects_training_anomalies(self):
        rows = [dict(recording_id="x", path="x.wav", category="fan", machine_id="00",
                     split="fit", label="1")]
        with self.assertRaises(ValueError):
            validate_manifest(rows)

    def test_all_ablation_classification_gradients(self):
        cfg = Config()
        cfg.model.stem_channels = 2
        cfg.model.descriptor_hidden = 4
        x, d, labels = torch.randn(4, 1, 128, 15), torch.randn(4, 11), torch.tensor([0, 0, 1, 1])
        for variant in ("cnn", "fusion", "relational", "full"):
            model = DRAAD(2, cfg.model, variant)
            loss = classification_loss(model, x, d, labels, cfg.training, np.random.default_rng(42))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.encoder[0].weight.grad)
            self.assertEqual(model.decoder is not None, variant == "full")

    def test_negative_alignment_weight_rejected(self):
        cfg = Config()
        cfg.training.relational_weight = -0.15
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
