"""Log-Mel patches, overlap-aware fitting statistics, and auditable feature caching."""
from dataclasses import asdict
from functools import lru_cache
import importlib.metadata
from pathlib import Path

import librosa
import numpy as np

from .audio import load_waveform, perturb
from .descriptors import extract_descriptors
from .utils import fingerprint, sha256

FEATURE_SCHEMA = 1


def log_mel(wave, cfg):
    mel = librosa.feature.melspectrogram(
        y=wave, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.n_fft, window="hann", center=True, pad_mode="constant",
        power=2.0, n_mels=cfg.n_mels, fmin=0.0, fmax=cfg.sample_rate / 2,
        htk=False, norm="slaney", dtype=np.float32)
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=cfg.top_db)


def select_patches(mel, cfg, fitting=False):
    stride = cfg.train_stride if fitting else cfg.eval_stride
    cap = cfg.train_patches if fitting else cfg.eval_patches
    starts = np.arange(0, mel.shape[1] - cfg.patch_frames + 1, stride)
    if not len(starts):
        raise ValueError("Waveform too short for configured patches")
    if len(starts) > cap:
        starts = starts[np.linspace(0, len(starts) - 1, cap, dtype=int)]
    return np.stack([mel[:, s:s + cfg.patch_frames] for s in starts]).astype(np.float32)


class FeatureStore:
    def __init__(self, root, cache, cfg, condition="clean"):
        self.root, self.cfg, self.condition = Path(root), cfg, condition
        identity = {"audio": asdict(cfg), "schema": FEATURE_SCHEMA,
                    "versions": {name: importlib.metadata.version(name)
                                 for name in ("librosa", "numpy", "scipy", "soundfile")},
                    "implementation": fingerprint({name: (Path(__file__).parent / name).read_text(encoding="utf-8")
                                                   for name in ("features.py", "audio.py", "descriptors.py")})}
        self.folder = Path(cache) / condition / fingerprint(identity)[:16]
        self.folder.mkdir(parents=True, exist_ok=True)

    def prepare(self, rows):
        """Validate content hashes once, before a run; never trust a stale cache silently."""
        for index, row in enumerate(rows):
            source = self.root / row["path"]
            audio_hash = sha256(source)
            if row.get("audio_sha256") and row["audio_sha256"] != audio_hash:
                raise ValueError(f"Audio changed since manifest creation: {source}")
            target = self.folder / (row["recording_id"] + ".npz")
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    if str(saved["audio_sha256"]) == audio_hash:
                        continue
            x = perturb(load_waveform(source, self.cfg), self.condition, self.cfg,
                        row["recording_id"])
            mel = log_mel(x, self.cfg)
            temp = target.with_suffix(".tmp.npz")
            np.savez_compressed(temp, mel=mel, descriptors=extract_descriptors(x),
                                audio_sha256=np.array(audio_hash))
            temp.replace(target)
            if (index + 1) % 100 == 0:
                print(f"Cached {index + 1}/{len(rows)} ({self.condition})", flush=True)

    @lru_cache(maxsize=16)
    def get(self, recording_id):
        with np.load(self.folder / (recording_id + ".npz"), allow_pickle=False) as entry:
            return entry["mel"].copy(), entry["descriptors"].copy()

    def patches(self, row, fitting=False):
        return select_patches(self.get(row["recording_id"])[0], self.cfg, fitting)


def fit_statistics(rows, store):
    """Population statistics; patch overlaps count each retained occurrence."""
    total = np.zeros(store.cfg.n_mels, dtype=np.float64)
    squares = total.copy()
    count = 0
    desc = []
    for row in rows:
        patches = store.patches(row, fitting=True).astype(np.float64)
        total += patches.sum(axis=(0, 2))
        squares += (patches ** 2).sum(axis=(0, 2))
        count += patches.shape[0] * patches.shape[2]
        desc.append(store.get(row["recording_id"])[1])
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    desc = np.asarray(desc, dtype=np.float64)
    return {"mel_mean": mean.astype(np.float32), "mel_std": std.astype(np.float32),
            "descriptor_mean": desc.mean(axis=0).astype(np.float32),
            "descriptor_std": desc.std(axis=0, ddof=0).astype(np.float32)}


def normalized_record(row, store, stats, fitting=False):
    patches = store.patches(row, fitting)
    floor = store.cfg.normalization_floor
    patches = (patches - stats["mel_mean"][None, :, None]) / np.maximum(
        stats["mel_std"], floor)[None, :, None]
    desc = (store.get(row["recording_id"])[1] - stats["descriptor_mean"]) / np.maximum(
        stats["descriptor_std"], floor)
    return patches[:, None].astype(np.float32), desc.astype(np.float32)
