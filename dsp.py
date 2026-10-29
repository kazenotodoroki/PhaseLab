"""
PhaseLab - analysis engine.

Methodology (that of MATLAB acoustic measurement toolboxes, e.g. ITA-Toolbox /
Farina-Novak scripts, and of dual-channel analyzers such as Open Sound Meter /
Smaart):

1. Excitation generated here (synchronized sweep, periodic sweeps, noises,
   MLS…) so that the analysis knows exactly what was played.
2. Three measurement methods:
   - sweep: regularized spectral division (Kirkeby inversion); harmonic
     distortion falls into negative time and is discarded;
   - periodic: synchronous averaging of the periods, then exact circular
     deconvolution (periodic noise, MLS, repeated sweeps);
   - dual channel: averaged transfer function H1 = Sxy / Sxx (Welch) with
     coherence, after automatic delay search. Works with any signal (noise,
     music) as long as the reference is available (loopback or signal
     file).
3. Absolute time reference: time 0 is the start of the signal file
   (or the loopback channel if one was recorded).
4. Windowing of the impulse response around its peak (half-Hann).
5. Complex fractional-octave smoothing after removing the own delay.
6. Cross-spectrum comparison: level and phase difference, and joint
   estimation of delay + constant phase rotation.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, hilbert, max_len_seq

KIND_LABELS = {
    "sweep": "Log sweep",
    "sweep_rep": "Repeated sweeps (periodic)",
    "pink": "Pink noise",
    "white": "White noise",
    "brown": "Brown noise",
    "pink_per": "Periodic pink noise",
    "white_per": "Periodic white noise",
    "mls": "MLS (pseudo-random)",
    "sine": "Sine",
    "impulse": "Impulse",
}
PERIODIC_KINDS = {"sweep_rep", "pink_per", "white_per", "mls"}
DECONV_KINDS = {"sweep", "impulse", "inconnu"}
METHOD_LABELS = {"auto": "Automatic", "sweep": "Deconvolution (sweep)",
                 "periodic": "Synchronous average (periodic)", "dual": "Dual channel (any signal)",
                 "combined": "Combination", "live": "Real time (dual channel)"}
SUBTYPES = {"PCM_24": "24 bits", "FLOAT": "32-bit float", "PCM_16": "16 bits"}


# ----------------------------------------------------------------- utilities
def next_pow2(n: int) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(int(n), 1)))))


def speed_of_sound(temp_c: float) -> float:
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


def wrap_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def parse_angle(name: str):
    """Detects an angle in a file name: 'snare_45deg', 'mic -30°'."""
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°|deg|degr[ée]s?)", name, re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def band_weight(f, f1, f2, width_oct=0.5):
    """1 inside the band, cosine transition over half an octave, 0 beyond."""
    f = np.maximum(np.asarray(f, dtype=float), 1e-9)
    lo = np.log2(f / f1)
    hi = np.log2(f / f2)
    w = np.zeros_like(f)
    w[(lo >= 0) & (hi <= 0)] = 1.0
    m = (lo < 0) & (lo > -width_oct)
    w[m] = 0.5 + 0.5 * np.cos(np.pi * lo[m] / width_oct)
    m = (hi > 0) & (hi < width_oct)
    w[m] = 0.5 + 0.5 * np.cos(np.pi * hi[m] / width_oct)
    return w


def _fade(n, n_in, n_out):
    w = np.ones(n)
    n_in, n_out = min(n_in, n), min(n_out, n)
    if n_in > 0:
        w[:n_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    if n_out > 0:
        w[n - n_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_out + 1) / n_out)
    return w


def frac_shift(x, delay_s, fs, n_out=None):
    """Exact fractional delay (frequency domain)."""
    n_out = n_out or len(x)
    N = next_pow2(max(len(x), n_out) + int(abs(delay_s) * fs) + 64)
    f = np.fft.rfftfreq(N, 1 / fs)
    return np.fft.irfft(np.fft.rfft(x, N) * np.exp(-2j * np.pi * f * delay_s), N)[:n_out]


# ------------------------------------------------------------ test signals
@dataclass
class SignalSpec:
    kind: str = "sweep"
    fs: int = 48000
    duration: float = 5.0
    f1: float = 20.0
    f2: float = 20000.0
    level_db: float = -6.0
    pre_silence: float = 0.5
    post_silence: float = 2.0
    subtype: str = "PCM_24"
    seed: int = 1
    period_pow: int = 17          # period = 2^n samples (periodic signals)
    repeats: int = 8              # number of periods
    sine_freq: float = 1000.0


def _shaped_noise(n, fs, f1, f2, slope, rng, flat_mag=False):
    """Band-limited noise; slope 0 = white, 1 = pink (-3 dB/oct), 2 = brown."""
    f = np.fft.rfftfreq(n, 1 / fs)
    if flat_mag:   # periodic: exact magnitude, random phase
        X = np.exp(2j * np.pi * rng.random(len(f)))
    else:
        X = np.fft.rfft(rng.standard_normal(n))
    X *= band_weight(f, f1, f2, 0.25)
    if slope:
        X /= np.maximum(f, f1) ** (slope / 2)
    X[0] = 0
    return np.fft.irfft(X, n)


def _exp_sweep(n, fs, f1, f2):
    T = n / fs
    t = np.arange(n) / fs
    k = math.log(f2 / f1)
    return np.sin(2 * np.pi * f1 * T / k * (np.exp(t / T * k) - 1))


def generate_signal(spec: SignalSpec):
    fs = int(spec.fs)
    f1 = max(float(spec.f1), 1.0)
    f2 = min(float(spec.f2), 0.499 * fs)
    rng = np.random.default_rng(spec.seed)
    dur = float(spec.duration)
    kind = spec.kind
    period = None

    if kind == "sweep":
        # Synchronized exponential sweep (Novak): harmonic phases are
        # coherent and the signal starts/ends cleanly.
        L = max(round(f1 * dur / math.log(f2 / f1)), 1) / f1
        dur = L * math.log(f2 / f1)
        t = np.arange(int(round(dur * fs))) / fs
        core = np.sin(2 * np.pi * f1 * L * np.exp(t / L))
        n_in = int(fs * min(0.1 * dur, 2.0 / f1))
        n_out = int(fs * min(0.005, 0.02 * dur))
        core *= _fade(len(core), n_in, n_out)
    elif kind in ("pink", "white", "brown"):
        n = int(round(dur * fs))
        slope = {"white": 0, "pink": 1, "brown": 2}[kind]
        core = _shaped_noise(n, fs, f1, f2, slope, rng)
        core *= _fade(n, int(0.01 * fs), int(0.01 * fs))
    elif kind in PERIODIC_KINDS:
        if kind == "mls":
            nbits = int(spec.period_pow)
            one = max_len_seq(nbits)[0].astype(float) * 2 - 1
            f1, f2 = max(fs / len(one), 1.0), 0.499 * fs
        else:
            P = 1 << int(spec.period_pow)
            if kind == "sweep_rep":
                ns = int(P * 0.9)
                sw = _exp_sweep(ns, fs, f1, f2)
                sw *= _fade(ns, int(fs * min(0.05, 2.0 / f1)), int(fs * 0.003))
                one = np.concatenate([sw, np.zeros(P - ns)])
            else:
                one = _shaped_noise(P, fs, f1, f2, 1 if kind == "pink_per" else 0, rng, flat_mag=True)
        period = len(one)
        reps = max(int(spec.repeats), 2)
        core = np.tile(one, reps)
        dur = len(core) / fs
    elif kind == "sine":
        n = int(round(dur * fs))
        core = np.sin(2 * np.pi * spec.sine_freq * np.arange(n) / fs)
        core *= _fade(n, int(0.02 * fs), int(0.02 * fs))
        f1 = f2 = float(spec.sine_freq)
    elif kind == "impulse":
        n = max(int(round(dur * fs)), 1)
        core = np.zeros(n)
        core[0] = 1.0
    else:
        raise ValueError(f"Unknown signal type: {kind}")

    core *= 10 ** (spec.level_db / 20) / np.max(np.abs(core))
    pre = int(round(spec.pre_silence * fs))
    post = int(round(spec.post_silence * fs))
    x = np.concatenate([np.zeros(pre), core, np.zeros(post)])
    meta = asdict(spec)
    meta.update(dict(f1=f1, f2=f2, duration=dur, samples=len(x),
                     pre_samples=pre, phaselab_signal=1))
    if period:
        meta.update(period=period, repeats=max(int(spec.repeats), 2))
    return x, meta


def sidecar_path(path) -> Path:
    return Path(path).with_suffix(".phaselab.json")


class TestSignal:
    """The reference signal, as played in Reaper."""

    def __init__(self, data, fs, meta, path=None):
        self.data = np.asarray(data, dtype=float)
        self.fs = int(fs)
        self.meta = dict(meta)
        self.path = str(path) if path else None

    @property
    def kind(self):
        return self.meta.get("kind", "inconnu")

    @property
    def periodic(self):
        return self.kind in PERIODIC_KINDS and "period" in self.meta

    @property
    def label(self):
        return KIND_LABELS.get(self.kind, "External signal")

    @property
    def f1(self):
        return float(self.meta.get("f1", 20.0))

    @property
    def f2(self):
        return float(self.meta.get("f2", min(20000.0, 0.45 * self.fs)))

    @property
    def name(self):
        return Path(self.path).name if self.path else "(not saved)"

    def save(self, path):
        path = Path(path)
        sf.write(str(path), self.data, self.fs, subtype=self.meta.get("subtype", "PCM_24"))
        sidecar_path(path).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        self.path = str(path)

    @classmethod
    def load(cls, path):
        data, fs = sf.read(str(path), always_2d=True, dtype="float64")
        meta = {}
        sc = sidecar_path(path)
        if sc.exists():
            try:
                meta = json.loads(sc.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if not meta:
            meta = dict(kind="inconnu", f1=20.0, f2=min(20000.0, 0.45 * fs))
        return cls(data[:, 0], fs, meta, path)


# ------------------------------------------------------------ three methods
def deconvolve(y, x, fs, f1, f2, keep_s=3.0, neg_s=0.05):
    """Impulse response h such that y = x * h (Kirkeby inversion).

    Returns (ir, t0): t0 is the time (s) of the first sample of ir
    (negative: a little negative time is kept for delays < 0).
    """
    N = next_pow2(len(y) + len(x))
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    f = np.fft.rfftfreq(N, 1 / fs)
    P = X.real ** 2 + X.imag ** 2
    w = band_weight(f, f1, f2)
    inband = w > 0.99
    pmax = P[inband].max() if inband.any() else P.max()
    eps = pmax * (1e-6 + 1e-1 * (1 - w))
    h = np.fft.irfft(Y * np.conj(X) / (P + eps), N)
    nneg = int(round(neg_s * fs))
    nkeep = min(int(keep_s * fs), len(y), N - nneg)
    ir = np.concatenate([h[N - nneg:], h[:nkeep]])
    return ir, -nneg / fs


def _regularize(H_num, P, f, f1, f2):
    w = band_weight(f, f1, f2)
    inband = w > 0.99
    pmax = P[inband].max() if inband.any() else P.max()
    return H_num / (P + pmax * (1e-8 + 1e-2 * (1 - w)))


def measure_periodic(y, xsig, meta, fs, f1, f2, loop=None):
    """Synchronous averaging then circular deconvolution (periodic signal).

    The 1st period is ignored (room transient)."""
    P = int(meta["period"])
    R = int(meta.get("repeats", 2))
    pre = int(meta.get("pre_samples", 0))
    starts = [pre + k * P for k in range(1 if R > 1 else 0, R)]
    starts = [s for s in starts if s + P <= len(y)]
    if not starts:
        raise ValueError("Take too short for the periodic signal.")
    f = np.fft.rfftfreq(P, 1 / fs)
    Ys = np.array([np.fft.rfft(y[s:s + P]) for s in starts])
    if loop is not None:
        Xs = np.array([np.fft.rfft(loop[s:s + P]) for s in starts])
    else:
        Xs = np.tile(np.fft.rfft(xsig[pre:pre + P]), (len(starts), 1))
    Sxy = np.mean(Ys * np.conj(Xs), axis=0)
    Sxx = np.mean(np.abs(Xs) ** 2, axis=0)
    Syy = np.mean(np.abs(Ys) ** 2, axis=0)
    H = _regularize(Sxy, Sxx, f, f1, f2)
    coh = None
    if len(starts) >= 2:
        coh = np.clip(np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30), 0, 1)
    h = np.fft.irfft(H, P)
    nneg = min(int(0.05 * fs), P // 8)
    ir = np.concatenate([h[P - nneg:], h[:P - nneg]])
    return ir, -nneg / fs, f, coh, len(starts)


def find_delay(y, x, fs, f1=20.0, f2=20000.0, max_s=2.0, max_len_s=60.0):
    """Coarse delay of y relative to x (GCC-PHAT), in samples."""
    n = min(len(x), len(y), int(max_len_s * fs))
    maxlag = int(max_s * fs)
    N = next_pow2(n + maxlag)
    X = np.fft.rfft(x[:n], N)
    Y = np.fft.rfft(y[:min(len(y), n + maxlag)], N)
    f = np.fft.rfftfreq(N, 1 / fs)
    C = Y * np.conj(X)
    C = C / (np.abs(C) + 1e-20) * band_weight(f, max(f1, 20.0), min(f2, 0.45 * fs))
    r = np.fft.irfft(C, N)
    cand = np.concatenate([np.arange(0, maxlag + 1), np.arange(N - maxlag, N)])
    j = int(cand[np.argmax(np.abs(r[cand]))])
    return j if j <= N // 2 else j - N


def measure_dual(y, x, fs, f1, f2, nfft=32768, overlap=0.75):
    """Dual-channel transfer function (H1, Welch) + coherence, any signal."""
    d = find_delay(y, x, fs, f1, f2)
    if d >= 0:
        ya, xa = y[d:], x
    else:
        ya, xa = y, x[-d:]
    L = min(len(ya), len(xa))
    ya, xa = ya[:L], xa[:L]
    nfft = int(min(nfft, 1 << int(math.floor(math.log2(max(L, 256))))))
    hop = max(int(nfft * (1 - overlap)), 1)
    win = np.hanning(nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    Sxx = np.zeros(len(f)); Syy = np.zeros(len(f)); Sxy = np.zeros(len(f), complex)
    starts = np.arange(0, L - nfft + 1, hop)
    for i in range(0, len(starts), 64):
        idx = starts[i:i + 64, None] + np.arange(nfft)[None, :]
        X = np.fft.rfft(xa[idx] * win, axis=1)
        Y = np.fft.rfft(ya[idx] * win, axis=1)
        Sxx += np.sum(np.abs(X) ** 2, axis=0)
        Syy += np.sum(np.abs(Y) ** 2, axis=0)
        Sxy += np.sum(Y * np.conj(X), axis=0)
    H = _regularize(Sxy, Sxx, f, f1, f2)
    coh = np.clip(np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30), 0, 1)
    h = np.fft.irfft(H, nfft)
    nneg = nfft // 8
    ir = np.concatenate([h[nfft - nneg:], h[:nfft - nneg]])
    return ir, (d - nneg) / fs, f, coh, len(starts)


# ------------------------------------------------------------ IR analysis
def _sinc_peak(ir, k):
    """Fractional position of the maximum of |ir| by sinc interpolation."""
    lo, hi = max(k - 32, 0), min(k + 33, len(ir))
    n = np.arange(lo, hi)
    seg = ir[lo:hi]
    tt = k + np.linspace(-1.0, 1.0, 129)
    vals = np.abs(np.sinc(tt[:, None] - n[None, :]) @ seg)
    j = int(np.argmax(vals))
    if 0 < j < len(vals) - 1:
        a, b, c = vals[j - 1], vals[j], vals[j + 1]
        d = a - 2 * b + c
        off = 0.5 * (a - c) / d if d != 0 else 0.0
    else:
        off = 0.0
    return tt[j] + off * (tt[1] - tt[0])


def analyze_ir(ir, t0, fs):
    k = int(np.argmax(np.abs(ir)))
    kp = _sinc_peak(ir, k)
    polarity = 1 if ir[k] >= 0 else -1
    lo = max(k - int(0.02 * fs), 0)
    env = np.abs(hilbert(ir[lo:min(k + int(0.02 * fs), len(ir))]))
    kk = k - lo
    thr = env[kk] * 10 ** (-20 / 20)
    j = kk
    while j > 0 and env[j] > thr:
        j -= 1
    onset = t0 + (lo + j) / fs
    tail = ir[int(len(ir) * 0.8):]
    noise = np.sqrt(np.mean(tail ** 2)) + 1e-20
    snr = 20 * np.log10(np.abs(ir[k]) / noise)
    return t0 + kp / fs, polarity, onset, snr


class Measurement:
    """A measurement: impulse response + adjustments (delay, gain, polarity).

    The *_raw fields are the raw measurement; ir, t0, peak_time… include the
    user's adjustments, like Open Sound Meter's stored traces.
    """

    def __init__(self, name, fs, ir, t0, f1, f2, y=None, xref=None, path=None, channel=0,
                 loopback=False, method="sweep", coh_f=None, coh=None, averages=1):
        self.name = name
        self.fs = int(fs)
        self.ir_raw = np.asarray(ir, dtype=float)
        self.t0_raw = float(t0)
        self.f1, self.f2 = float(f1), float(f2)
        self.y = None if y is None else np.asarray(y, dtype=float)
        self.xref = xref
        self.path = str(path) if path else None
        self.channel = channel
        self.loopback = loopback
        self.method = method
        self.coh_f, self.coh = coh_f, coh
        self.averages = averages
        self.angle = parse_angle(name)
        self.visible = True
        self.color = None
        self.virtual = False
        self.recipe = None
        self.offset_ms = 0.0
        self.gain_db = 0.0
        self.invert = False
        self.peak_raw, self.pol_raw, self.onset_raw, self.snr_db = analyze_ir(self.ir_raw, self.t0_raw, self.fs)
        self._env_raw = None
        self._ir_cache = None

    # --- adjustments
    @property
    def adj_key(self):
        return (round(self.offset_ms, 9), round(self.gain_db, 6), bool(self.invert))

    @property
    def g(self):
        return (-1.0 if self.invert else 1.0) * 10 ** (self.gain_db / 20)

    @property
    def adjusted(self):
        return self.adj_key != (0.0, 0.0, False)

    def reset_adjust(self):
        self.offset_ms, self.gain_db, self.invert = 0.0, 0.0, False

    @property
    def ir(self):
        if self._ir_cache is None or self._ir_cache[0] != self.adj_key:
            self._ir_cache = (self.adj_key, self.ir_raw * self.g)
        return self._ir_cache[1]

    @property
    def t0(self):
        return self.t0_raw + self.offset_ms * 1e-3

    @property
    def peak_time(self):
        return self.peak_raw + self.offset_ms * 1e-3

    @property
    def onset_time(self):
        return self.onset_raw + self.offset_ms * 1e-3

    @property
    def polarity(self):
        return -self.pol_raw if self.invert else self.pol_raw

    @property
    def env(self):
        if self._env_raw is None:
            self._env_raw = np.abs(hilbert(self.ir_raw))
        return self._env_raw * abs(self.g)

    @property
    def key(self):
        return (self.path, self.channel) if not self.virtual else ("virtual", self.name)


def _contains(loop, sig, fs):
    """Does the loopback contain the test signal? (normalized correlation)"""
    try:
        d = find_delay(loop, sig, fs, max_len_s=20.0)
    except Exception:
        return False
    a = loop[d:] if d >= 0 else loop
    b = sig if d >= 0 else sig[-d:]
    n = min(len(a), len(b), int(20 * fs))
    a, b = a[:n], b[:n]
    den = math.sqrt(float(a @ a) * float(b @ b))
    return den > 0 and abs(float(a @ b)) / den > 0.3


def load_takes(path, signal: TestSignal | None, loop_channel=0, method="auto", nfft=32768):
    """Loads a WAV of takes. loop_channel: 0 = none, else channel number (1..n)."""
    data, fs = sf.read(str(path), always_2d=True, dtype="float64")
    if signal is not None and fs != signal.fs:
        raise ValueError(f"{Path(path).name}: {fs} Hz, but the signal is at {signal.fs} Hz.")
    nch = data.shape[1]
    chans = list(range(nch))
    loop = None
    if loop_channel and 1 <= loop_channel <= nch and nch > 1:
        loop = data[:, loop_channel - 1].copy()
        chans.remove(loop_channel - 1)
    xref = loop if loop is not None else (signal.data if signal is not None else None)
    if xref is None:
        raise ValueError("no reference: load the test signal, or set the loopback "
                         "(reference) channel for a dual-channel measurement.")
    if method == "auto" and loop is not None and signal is not None and not _contains(loop, signal.data, fs):
        method = "dual"          # something other than the signal was played (music…): dual channel
    if method == "auto":
        if signal is not None and signal.periodic:
            method = "periodic"
        elif signal is not None and signal.kind in DECONV_KINDS:
            method = "sweep"
        else:
            method = "dual"
    if method == "periodic" and (signal is None or not signal.periodic):
        raise ValueError("synchronous averaging needs a periodic signal generated by PhaseLab.")
    f1 = signal.f1 if signal is not None else 20.0
    f2 = signal.f2 if signal is not None else min(20000.0, 0.45 * fs)
    if signal is not None and signal.kind == "sine":
        f1, f2 = 20.0, min(20000.0, 0.45 * fs)
    if method == "dual":
        f1 = max(f1, 2.0 * fs / nfft)
    stem = Path(path).stem
    out = []
    for c in chans:
        y = data[:, c]
        coh_f = coh = None
        avg = 1
        if method == "sweep":
            ir, t0 = deconvolve(y, xref, fs, f1, f2)
        elif method == "periodic":
            ir, t0, coh_f, coh, avg = measure_periodic(y, signal.data, signal.meta, fs, f1, f2, loop)
        else:
            ir, t0, coh_f, coh, avg = measure_dual(y, xref, fs, f1, f2, nfft)
        name = stem if len(chans) == 1 else f"{stem} · c{c + 1}"
        out.append(Measurement(name, fs, ir, t0, f1, f2, y, xref, path, c, loop is not None,
                               method, coh_f, coh, avg))
    return out


# ------------------------------------------------------- combined measurements
def min_phase_from_power(P, N):
    """Minimum-phase response with power spectrum P (real cepstrum)."""
    logm = 0.5 * np.log(np.maximum(P, 1e-20))
    c = np.fft.irfft(logm, N)
    w = np.zeros(N)
    w[0] = 1
    w[1:N // 2] = 2
    w[N // 2] = 1
    return np.fft.irfft(np.exp(np.fft.rfft(c * w, N)), N)


COMBINE_OPS = {"vector": "Vector average (complex)", "power": "Power average (minimum phase)",
               "sum": "Sum (A + B + …)", "diff": "Difference (A − B)"}


def combine(ms: list[Measurement], op: str, name: str) -> Measurement:
    fs = ms[0].fs
    if any(m.fs != fs for m in ms):
        raise ValueError("different sample rates")
    if op == "diff" and len(ms) != 2:
        raise ValueError("the difference needs exactly two measurements")
    t_start = min(m.t0 for m in ms)
    t_end = max(m.t0 + len(m.ir) / fs for m in ms)
    N = next_pow2(int((t_end - t_start) * fs) + 64)
    f = np.fft.rfftfreq(N, 1 / fs)
    Hs = []
    for m in ms:
        s = (m.t0 - t_start) * fs
        k = int(math.floor(s))
        buf = np.zeros(N)
        n = min(len(m.ir), N - k)
        buf[k:k + n] = m.ir[:n]
        Hs.append(np.fft.rfft(buf) * np.exp(-2j * np.pi * f * (s - k) / fs))
    Hs = np.array(Hs)
    t0 = t_start
    if op == "vector":
        H = Hs.mean(axis=0)
    elif op == "sum":
        H = Hs.sum(axis=0)
    elif op == "diff":
        H = Hs[0] - Hs[1]
    elif op == "power":
        h = min_phase_from_power(np.mean(np.abs(Hs) ** 2, axis=0), N)
        kp = int(np.argmax(np.abs(h)))
        t0 = float(np.mean([m.peak_time for m in ms])) - kp / fs
        H = np.fft.rfft(h)
    else:
        raise ValueError(op)
    ir = np.fft.irfft(H, N)
    out = Measurement(name, fs, ir, t0, max(m.f1 for m in ms), min(m.f2 for m in ms), method="combined")
    out.virtual = True
    out.recipe = dict(op=op, sources=[list(m.key) for m in ms])
    return out


# ------------------------------------------------------------- onset
def onset_info(x, fs, t0=0.0, thr_db=-20.0, anchor=None, search_ms=10.0):
    """Analysis of the start of a waveform.

    anchor: index of the main peak (IR); otherwise the first significant
    event above the noise floor is detected.
    Returns dict(onset, motion, peak, rise, polarity) in seconds (t0 included).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8 or not np.any(x):
        return None
    if anchor is None:
        blk = max(int(0.005 * fs), 8)
        nb = n // blk
        if nb >= 4:
            rms = np.sqrt(np.mean(x[:nb * blk].reshape(nb, blk) ** 2, axis=1))
            noise = np.percentile(rms, 10)
        else:
            noise = 0.0
        gpk = np.max(np.abs(x))
        thr0 = max(noise * 10 ** (15 / 20), gpk * 10 ** (-45 / 20))
        above = np.nonzero(np.abs(x) > thr0)[0]
        if not len(above):
            return None
        i0 = int(above[0])
        i1 = min(i0 + int(search_ms * 1e-3 * fs), n)
        anchor = i0 + int(np.argmax(np.abs(x[i0:i1])))
    lo = max(anchor - int(0.03 * fs), 0)
    hi = min(anchor + int(0.005 * fs), n)
    seg = x[lo:hi]
    a = np.abs(seg)
    k = anchor - lo
    lp = a[k]
    if lp <= 0:
        return None
    # Work on the samples themselves (not on the Hilbert envelope, which is
    # non-causal and "smears" before the true start).
    thr = lp * 10 ** (thr_db / 20)
    j = int(np.nonzero(a[:k + 1] >= thr)[0][0])
    if j > 0 and a[j] != a[j - 1]:
        onset = j - 1 + (thr - a[j - 1]) / (a[j] - a[j - 1])
    else:
        onset = float(j)
    # first motion: go back from the threshold to the previous zero crossing,
    # or down to 6 dB below the threshold for
    # slow onsets (sub) that never cross zero.
    sgn = np.sign(seg[j]) or 1.0
    quiet = thr * 0.5
    m = j - 1
    while m > 0 and np.sign(seg[m]) == sgn and a[m] > quiet:
        m -= 1
    if m > 0 and np.sign(seg[m]) == sgn:
        # stopped on the noise level: accept the zero crossing if it is close
        lim, m2 = max(2 * (j - m), 2), m
        while m2 > 0 and np.sign(seg[m2]) == sgn and (m - m2) < lim:
            m2 -= 1
        if np.sign(seg[m2]) != sgn:
            m = m2
    if m >= 0 and np.sign(seg[m]) != sgn and seg[m + 1] != seg[m]:
        motion = m + seg[m] / (seg[m] - seg[m + 1])
    elif m >= 0 and a[m + 1] != a[m]:
        motion = m + (quiet - a[m]) / (a[m + 1] - a[m])
    else:
        motion = float(max(m, 0))
    # polarity: first lobe reaching half of the peak
    big = np.nonzero(a[j:k + 1] >= 0.5 * lp)[0]
    pol = int(np.sign(seg[j + big[0]]) or 1) if len(big) else int(sgn)
    # 10-90 % rise time
    i10 = j + int(np.nonzero(a[j:k + 1] >= 0.1 * lp)[0][0]) if np.any(a[j:k + 1] >= 0.1 * lp) else j
    i90 = j + int(np.nonzero(a[j:k + 1] >= 0.9 * lp)[0][0])
    return dict(onset=t0 + (lo + onset) / fs, motion=t0 + (lo + motion) / fs,
                peak=t0 + anchor / fs, rise=(i90 - i10) / fs, polarity=pol,
                level=float(lp))


def find_arrivals(m: Measurement, max_n=4, min_sep_ms=0.15, min_db=-24.0, span_ms=60.0):
    """Distinct arrivals in a single response (e.g. two speakers playing together)."""
    from scipy.signal import find_peaks
    fs = m.fs
    kp = int(round((m.peak_time - m.t0) * fs))
    lo = max(kp - int(0.02 * fs), 0)
    hi = min(kp + int(span_ms * 1e-3 * fs), len(m.ir))
    env = m.env[lo:hi]
    db = 20 * np.log10(np.maximum(env / (env.max() + 1e-30), 1e-6))
    pk, props = find_peaks(db, height=min_db, distance=max(int(min_sep_ms * 1e-3 * fs), 1), prominence=6)
    if not len(pk):
        return []
    order = np.argsort(db[pk])[::-1][:max_n]
    pk = np.sort(pk[order])
    out = []
    for p in pk:
        out.append(dict(t=m.t0 + (lo + p) / fs, db=float(db[p]),
                        polarity=1 if m.ir[lo + p] >= 0 else -1))
    return out

# ------------------------------------------------------- spectra and smoothing
@dataclass
class Spectrum:
    f: np.ndarray
    H0: np.ndarray      # response with its own delay removed (reference = peak)
    tp: float           # peak time (s)
    window: tuple = (0.0, 0.0)


def fft_size(fs, pre_ms, post_ms):
    return max(next_pow2(int((pre_ms + post_ms) * 1e-3 * fs) + 2), next_pow2(fs))


def windowed_spectrum(m: Measurement, pre_ms, post_ms, N):
    fs = m.fs
    kp = (m.peak_time - m.t0) * fs
    pre, post = pre_ms * 1e-3, post_ms * 1e-3
    i0 = max(int(math.floor(kp - pre * fs)), 0)
    i1 = min(int(math.ceil(kp + post * fs)) + 1, len(m.ir))
    seg = m.ir[i0:i1]
    tt = (np.arange(i0, i1) - kp) / fs
    w = np.ones(len(seg))
    a = (tt >= -pre) & (tt < -pre / 2)
    w[a] = 0.5 - 0.5 * np.cos(np.pi * (tt[a] + pre) / (pre / 2))
    b = tt > 0.75 * post
    w[b] = 0.5 + 0.5 * np.cos(np.pi * (tt[b] - 0.75 * post) / (0.25 * post))
    w[(tt < -pre) | (tt > post)] = 0
    f = np.fft.rfftfreq(N, 1 / fs)
    H0 = np.fft.rfft(seg * w, N) * np.exp(2j * np.pi * f * (kp - i0) / fs)
    return Spectrum(f, H0, m.peak_time, (m.peak_time - pre, m.peak_time + post))


def unit_spectrum(f):
    return Spectrum(f, np.ones(len(f), dtype=complex), 0.0)


def log_grid(fmin, fmax, n=1500):
    return np.geomspace(fmin, fmax, n)


class Smoother:
    """Fractional-octave averaging on a log grid.

    The window edges are fractional (interpolated cumulative sum): the result
    is continuous, without steps, even where the window covers only a few
    FFT bins. Minimum window = 1 bin, which amounts to linear interpolation
    (the "no smoothing" case).
    """

    def __init__(self, f, grid, frac):
        self.f, self.grid = f, grid
        df = f[1] - f[0]
        u = grid / df
        if frac:
            lo, hi = u * 2 ** (-0.5 / frac), u * 2 ** (0.5 / frac)
        else:
            lo, hi = u.copy(), u.copy()
        small = (hi - lo) < 1.0
        lo[small], hi[small] = u[small] - 0.5, u[small] + 0.5
        n = len(f)
        self.lo = np.clip(lo, -0.5, n - 0.5)
        self.hi = np.clip(hi, -0.5, n - 0.5)
        self.width = np.maximum(self.hi - self.lo, 1e-9)
        self.pos = np.arange(n + 1) - 0.5

    def _integ(self, cs, x):
        return np.interp(x, self.pos, cs)

    def __call__(self, Z):
        cs = np.concatenate(([0], np.cumsum(Z)))
        if np.iscomplexobj(cs):
            re = self._integ(cs.real, self.hi) - self._integ(cs.real, self.lo)
            im = self._integ(cs.imag, self.hi) - self._integ(cs.imag, self.lo)
            return (re + 1j * im) / self.width
        return (self._integ(cs, self.hi) - self._integ(cs, self.lo)) / self.width


@dataclass
class Curve:
    Hs: np.ndarray       # smoothed complex, own delay removed
    pw: np.ndarray       # smoothed power
    tp: float
    gd: np.ndarray       # smoothed residual group delay (s)

    @property
    def mag_db(self):
        return 10 * np.log10(self.pw)

    def phase_res_deg(self):
        return np.degrees(np.unwrap(np.angle(self.Hs)))


def make_curve(spec: Spectrum, sm: Smoother) -> Curve:
    H = spec.H0
    P = np.abs(H) ** 2
    pw = np.maximum(sm(P), 1e-30)
    # Group delay bin by bin (phase difference between neighboring bins),
    # then energy-weighted average: robust, without a noisy numerical derivative.
    df = spec.f[1] - spec.f[0]
    d = -np.angle(H[1:] * np.conj(H[:-1])) / (2 * np.pi * df)
    gd_lin = np.concatenate(([d[0]], 0.5 * (d[1:] + d[:-1]), [d[-1]]))
    gd = sm(gd_lin * P) / pw
    return Curve(sm(H), pw, spec.tp, gd)


def unit_curve(grid):
    n = len(grid)
    return Curve(np.ones(n, complex), np.ones(n), 0.0, np.zeros(n))


def group_delay_ms(curve: Curve, grid, extra_s=0.0):
    return (curve.gd + extra_s) * 1e3


# ----------------------------------------------------------------- comparison
@dataclass
class Comparison:
    dmag: np.ndarray          # dB
    dres: np.ndarray          # ° unwrapped, delay dtp removed
    dtp: float                # peak time difference (s)
    delta: float              # fine correction (s)
    tau: float                # total estimated delay (s): measurement is late if > 0
    theta: float              # total rotation (°), polarity included
    inverted: bool
    rotation: float           # remaining rotation after polarity (°)
    gain_db: float
    resid_rms: float          # ° RMS in the band after correction
    notch_hz: float | None
    sum_before: np.ndarray | None = None
    sum_after: np.ndarray | None = None
    sum_ideal: np.ndarray | None = None

    def dphase(self, grid, mode):
        """mode 'indiv': estimated delay removed; otherwise delay included."""
        if mode == "indiv":
            return self.dres + 360 * grid * self.delta
        return self.dres - 360 * grid * self.dtp


def compare(si: Spectrum, sref: Spectrum, ci: Curve, cref: Curve, sm: Smoother,
            fs, flo, fhi, with_sum=True, search_ms=20.0, up=8):
    f, grid = si.f, sm.grid
    dtp = si.tp - sref.tp
    C = si.H0 * np.conj(sref.H0)

    # --- joint delay + rotation (1/f-weighted GCC, analytic envelope)
    mag = np.abs(C)
    band = (f >= flo) & (f <= fhi)
    if band.sum() < 4:
        band = (f >= grid[0]) & (f <= grid[-1])
    thr = np.median(mag[band]) * 1e-3 if band.any() else 0
    band &= mag > thr
    W = np.zeros(len(f))
    W[band] = 1.0 / f[band]
    U = np.zeros(len(f), complex)
    U[band] = C[band] / mag[band]
    Nfft = 2 * (len(f) - 1)
    M = Nfft * up
    Z = np.zeros(M, complex)
    Z[:len(f)] = W * U
    r = np.abs(np.fft.ifft(Z))
    lim = int(search_ms * 1e-3 * fs * up)
    cand = np.concatenate([np.arange(0, lim + 1), np.arange(M - lim, M)])
    j = cand[np.argmax(r[cand])]
    a, b, c = r[(j - 1) % M], r[j], r[(j + 1) % M]
    d = a - 2 * b + c
    off = 0.5 * (a - c) / d if d != 0 else 0.0
    lag = j + off
    if lag > M / 2:
        lag -= M
    delta = lag / (fs * up)
    theta = float(np.degrees(np.angle(np.sum(W * U * np.exp(2j * np.pi * f * delta)))))
    inverted = abs(theta) > 90
    rotation = float(wrap_deg(theta - 180)) if inverted else theta

    # --- difference curves (smoothed cross-spectrum / reference power)
    Cs = sm(C)
    D0 = Cs / cref.pw
    dres = np.degrees(np.unwrap(np.angle(D0)))
    dmag = 10 * np.log10(ci.pw / cref.pw)
    inb = (grid >= flo) & (grid <= fhi)
    if not inb.any():
        inb = np.ones(len(grid), bool)
    fc = math.sqrt(max(flo, grid[0]) * min(fhi, grid[-1]))
    k = int(np.argmin(np.abs(grid - fc)))
    ind = dres + 360 * grid * delta
    dres -= 360 * round((ind[k] - theta) / 360)
    gain_db = float(np.mean(dmag[inb]))
    corr = wrap_deg(dres + 360 * grid * delta - theta)
    resid = float(np.sqrt(np.mean(corr[inb] ** 2)))

    out = Comparison(dmag, dres, dtp, delta, dtp + delta, theta, inverted,
                     rotation, gain_db, resid, None)

    if with_sum:
        A = sref.H0
        B = si.H0 * np.exp(-2j * np.pi * f * dtp)
        Bc = si.H0 * np.exp(2j * np.pi * f * delta) * np.exp(-1j * np.radians(theta)) \
            * 10 ** (-gain_db / 20)
        pa = np.maximum(sm(np.abs(A) ** 2), 1e-30)
        pb = np.maximum(sm(np.abs(B) ** 2), 1e-30)
        sb = np.maximum(sm(np.abs(A + B) ** 2), 1e-30)
        sa = np.maximum(sm(np.abs(A + Bc) ** 2), 1e-30)
        out.sum_before = 10 * np.log10(sb)
        out.sum_after = 10 * np.log10(sa)
        out.sum_ideal = 10 * np.log10(pa) + 20 * np.log10(2)
        rel = 10 * np.log10(sb) - 20 * np.log10(np.sqrt(pa) + np.sqrt(pb))
        idx = np.where(rel < -10)[0]
        out.notch_hz = float(grid[idx[0]]) if len(idx) else None
    return out




# ------------------------------------------------- time-domain comparison
def original_vs_take(m: Measurement, mode="aligned"):
    """Returns (prediction, residual, gain, residual_dB).

    aligned: the original (or the loopback) shifted by the measured delay and scaled.
    model  : the original filtered by the complete measured response; the residual
              then only contains noise + distortion.
    """
    y, x, fs = m.y, m.xref, m.fs
    if mode == "model":
        full = fftconvolve(x, m.ir_raw)
        shift = int(round(m.t0_raw * fs))
        if shift < 0:
            pred = full[-shift:-shift + len(y)]
        else:
            pred = np.concatenate([np.zeros(shift), full])[:len(y)]
        if len(pred) < len(y):
            pred = np.pad(pred, (0, len(y) - len(pred)))
        g = 1.0
    else:
        xs = frac_shift(x, m.peak_raw, fs, len(y))
        den = float(xs @ xs)
        g = float(y @ xs) / den if den > 0 else 0.0
        pred = g * xs
    res = y - pred
    ry = np.sqrt(np.mean(y ** 2)) + 1e-20
    rr = np.sqrt(np.mean(res ** 2)) + 1e-20
    return pred, res, g, 20 * np.log10(rr / ry)
