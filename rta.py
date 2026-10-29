"""PhaseLab - offline spectrum analyzer (RTA), loudness, spectrogram."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly, spectrogram, welch


def k_weight(x, fs):
    """ITU-R BS.1770 K-weighting, coefficients recomputed for any fs
    (derivation that reproduces exactly the published 48 kHz coefficients)."""
    # stage 1: high shelf
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    K = math.tan(math.pi * fc / fs)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    # stage 2: RLB high-pass
    Q, fc = 0.5003270373238773, 38.13547087602444
    K = math.tan(math.pi * fc / fs)
    d = 1 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (K * K - 1) / d, (1 - K / Q + K * K) / d]
    return lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)


def integrated_lufs(data, fs):
    """Integrated loudness (BS.1770-4): 400 ms blocks, absolute and relative gates."""
    y = k_weight(data, fs)
    blk, hop = int(0.4 * fs), int(0.1 * fs)
    n = (len(y) - blk) // hop + 1
    if n < 1:
        return float("nan")
    cs = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y ** 2, axis=0)])
    idx = np.arange(n) * hop
    z = (cs[idx + blk] - cs[idx]) / blk
    zs = z.sum(axis=1)
    lj = -0.691 + 10 * np.log10(np.maximum(zs, 1e-20))
    g = zs[lj > -70]
    if not len(g):
        return float("-inf")
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g = zs[(lj > -70) & (lj > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(g))) if len(g) else float("-inf")


def true_peak_db(data, chunk=1 << 18):
    """True peak (4x oversampling)."""
    pk = 0.0
    for c in range(data.shape[1]):
        x = data[:, c]
        for i in range(0, len(x), chunk):
            seg = x[max(i - 32, 0):i + chunk + 32]
            pk = max(pk, float(np.max(np.abs(resample_poly(seg, 4, 1)))) if len(seg) else 0.0)
    return 20 * math.log10(max(pk, 1e-12))


class AudioFile:
    """Any audio file (mix, track, reference) for spectrum and waveform views."""

    def __init__(self, path):
        data, fs = sf.read(str(path), always_2d=True, dtype="float32")
        self.path = str(path)
        self.name = Path(path).stem
        self.fs = int(fs)
        self.data = data
        self.mono = data.mean(axis=1).astype(np.float64)
        self.visible = True
        self.color = None
        self._psd = None
        self._spec = None
        d64 = data.astype(np.float64)
        self.duration = len(data) / fs
        self.peak_db = 20 * math.log10(max(float(np.max(np.abs(data))), 1e-12))
        rms = math.sqrt(float(np.mean(d64 ** 2)))
        self.rms_db = 20 * math.log10(max(rms, 1e-12))
        self.crest_db = self.peak_db - self.rms_db
        self.lufs = integrated_lufs(d64, fs)
        self.true_peak = true_peak_db(data)

    def psd(self, nfft=16384):
        if self._psd is None or self._psd[0] != nfft:
            n = min(nfft, 1 << int(math.floor(math.log2(max(len(self.data), 256)))))
            P = None
            for c in range(self.data.shape[1]):
                f, p = welch(self.data[:, c].astype(np.float64), self.fs, window="hann",
                             nperseg=n, noverlap=n // 2, scaling="density")
                P = p if P is None else P + p
            self._psd = (nfft, f, P / self.data.shape[1])
        return self._psd[1], self._psd[2]

    def spectrogram(self, n_bins=320, max_frames=1600, nfft=4096):
        if self._spec is None:
            x = self.mono
            n = min(nfft, 1 << int(math.floor(math.log2(max(len(x), 256)))))
            hop = max(n // 4, int(math.ceil(len(x) / max_frames)))
            f, t, S = spectrogram(x, self.fs, window="hann", nperseg=n, noverlap=max(n - hop, 0),
                                  scaling="spectrum", mode="psd")
            fl = np.geomspace(20, min(20000, 0.48 * self.fs), n_bins)
            Sdb = 10 * np.log10(np.maximum(S, 1e-20))
            img = np.array([np.interp(fl, f, Sdb[:, i]) for i in range(Sdb.shape[1])])
            self._spec = (t, fl, img)
        return self._spec


def rta_curve(af: AudioFile, grid, smoother_cls, frac):
    """Average "per band" spectrum (pink noise = flat), in dB, smoothed."""
    f, P = af.psd()
    sm = smoother_cls(f, grid, frac)
    # power per constant-percentage band: density × f
    return 10 * np.log10(np.maximum(sm(P) * grid, 1e-30))
