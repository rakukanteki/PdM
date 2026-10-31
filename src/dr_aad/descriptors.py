"""Eleven recording-level descriptors in the order of Table I."""
import numpy as np
from scipy import signal, stats

DESCRIPTOR_NAMES = (
    "peak_energy_concentration", "envelope_periodicity", "modulation_concentration",
    "log_crest_factor", "log_kurtosis", "spectral_flatness", "normalized_centroid",
    "band_0_250", "band_250_1000", "band_1000_3000", "band_3000_8000",
)


def extract_descriptors(wave, sr=16000):
    x = np.asarray(wave, dtype=np.float64)
    eps = 1e-10
    if x.ndim != 1 or len(x) < 2048 or sr != 16000:
        raise ValueError("Descriptors require a mono waveform >=2048 samples at 16 kHz")
    f, psd = signal.welch(x, fs=sr, window="hann", nperseg=2048, noverlap=1024,
                         nfft=2048, detrend="constant", return_onesided=True,
                         scaling="density", average="mean")
    psd = np.maximum(psd, eps)
    p = psd / psd.sum()
    peaks, _ = signal.find_peaks(psd, distance=4)
    peaks = peaks[np.argsort(psd[peaks])[-8:]]
    peak_energy = p[peaks].sum()
    frames = np.lib.stride_tricks.sliding_window_view(x, 320)[::160]
    envelope = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    envelope = (envelope - envelope.mean()) / (envelope.std(ddof=0) + 1e-8)
    ac = signal.correlate(envelope, envelope, mode="full", method="fft")[len(envelope)-1:]
    ac = ac / (ac[0] + eps)
    periodicity = np.max(ac[2:min(201, len(ac))]) if len(ac) > 2 else 0.0
    segment = min(256, len(envelope))
    mf, mp = signal.welch(envelope, fs=100, window="hann", nperseg=segment,
                          noverlap=segment // 2, nfft=segment, detrend="constant",
                          return_onesided=True, scaling="density", average="mean")
    band = mp[(mf >= 0.5) & (mf <= 40)]
    modulation = band.max() / (band.sum() + eps) if len(band) else 0.0
    rms = np.sqrt(np.mean(x ** 2) + eps)
    kurt = 3.0 if np.ptp(x) == 0 else stats.kurtosis(x, fisher=False, bias=False)
    flatness = np.exp(np.mean(np.log(psd))) / (np.mean(psd) + eps)
    centroid = np.sum(f * p) / (sr / 2)
    energies = [p[(f >= lo) & (f < hi)].sum()
                for lo, hi in [(0, 250), (250, 1000), (1000, 3000), (3000, 8000)]]
    values = [peak_energy, periodicity, modulation,
              np.log1p(np.max(np.abs(x)) / (rms + eps)),
              np.log1p(max(float(kurt), 0)), flatness, centroid, *energies]
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
