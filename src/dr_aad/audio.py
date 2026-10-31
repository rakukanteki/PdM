"""Waveform preparation and explicit evaluation-only perturbations."""
from math import gcd

import numpy as np
from scipy import signal
import soundfile as sf

CONDITIONS = ("clean", "noise20", "noise10", "gainp6", "gainm6", "lowpass")


def load_waveform(path, cfg):
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    wave = wave.mean(axis=1, dtype=np.float64)
    if sr != cfg.sample_rate:
        divisor = gcd(sr, cfg.sample_rate)
        wave = signal.resample_poly(wave, cfg.sample_rate // divisor, sr // divisor,
                                    window=("kaiser", 5.0), padtype="constant")
    length = round(cfg.duration * cfg.sample_rate)
    wave = wave[:length]
    return np.pad(wave, (0, max(0, length - len(wave)))).astype(np.float32)


def perturb(wave, condition, cfg, recording_id=""):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown acoustic condition: {condition}")
    wave = np.asarray(wave, dtype=np.float64)
    if condition.startswith("noise"):
        seed = cfg.noise_seed
        if cfg.noise_mode == "independent":
            import hashlib
            seed += int(hashlib.sha256(recording_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(wave))
        snr = float(condition[5:])
        rms = np.sqrt(np.mean(wave ** 2))
        noise *= rms / (np.sqrt(np.mean(noise ** 2)) * 10 ** (snr / 20))
        wave = wave + noise
    elif condition in {"gainp6", "gainm6"}:
        gain = 6 if condition == "gainp6" else -6
        wave = np.clip(wave * 10 ** (gain / 20), -1, 1)
    elif condition == "lowpass":
        sos = signal.butter(4, 3000, fs=cfg.sample_rate, output="sos")
        wave = signal.sosfilt(sos, wave)
    return wave.astype(np.float32)
