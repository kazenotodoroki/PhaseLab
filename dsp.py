"""
PhaseLab - moteur d'analyse.

Méthodologie (celle des boîtes à outils de mesure acoustique sous MATLAB,
p. ex. ITA-Toolbox / scripts Farina-Novak) :

1. Excitation : sweep exponentiel synchronisé (Novak 2015) ou bruit à bande
   limitée, généré ici pour que l'analyse connaisse exactement le signal joué.
2. Déconvolution : division spectrale régularisée (inversion de Kirkeby).
   Avec un sweep, les produits de distorsion harmonique tombent en temps
   négatif et sont éliminés automatiquement.
3. Repère temporel absolu : le temps 0 est le début du fichier du signal
   (ou le canal de boucle si un loopback est enregistré).
4. Fenêtrage de la réponse impulsionnelle autour de son pic (demi-Hann).
5. Lissage complexe par fraction d'octave après retrait du retard propre,
   pour que la phase reste exploitable.
6. Comparaison par spectre croisé : différence de niveau, de phase, et
   estimation conjointe retard + rotation de phase constante (GCC pondéré
   1/f, enveloppe analytique, interpolation sous-échantillon).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, hilbert

KIND_LABELS = {
    "sweep": "Sweep logarithmique",
    "pink": "Bruit rose",
    "white": "Bruit blanc",
    "impulse": "Impulsion",
}
SUBTYPES = {"PCM_24": "24 bits", "FLOAT": "32 bits flottant", "PCM_16": "16 bits"}


# ----------------------------------------------------------------- utilitaires
def next_pow2(n: int) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(int(n), 1)))))


def speed_of_sound(temp_c: float) -> float:
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


def wrap_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def parse_angle(name: str):
    """Détecte un angle dans un nom de fichier : 'snare_45deg', 'mic -30°'."""
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°|deg|degr[ée]s?)", name, re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def band_weight(f, f1, f2, width_oct=0.5):
    """1 dans la bande, transition en cosinus sur une demi-octave, 0 au-delà."""
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


# ------------------------------------------------------------ signaux de test
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


def generate_signal(spec: SignalSpec):
    fs = int(spec.fs)
    f1 = max(float(spec.f1), 1.0)
    f2 = min(float(spec.f2), 0.499 * fs)
    rng = np.random.default_rng(spec.seed)
    dur = float(spec.duration)

    if spec.kind == "sweep":
        # Sweep exponentiel synchronisé (Novak) : la phase des harmoniques est
        # cohérente et le signal commence/finit proprement.
        L = max(round(f1 * dur / math.log(f2 / f1)), 1) / f1
        dur = L * math.log(f2 / f1)
        t = np.arange(int(round(dur * fs))) / fs
        core = np.sin(2 * np.pi * f1 * L * np.exp(t / L))
        n_in = int(fs * min(0.1 * dur, 2.0 / f1))
        n_out = int(fs * min(0.005, 0.02 * dur))
        core *= _fade(len(core), n_in, n_out)
    elif spec.kind in ("pink", "white"):
        n = int(round(dur * fs))
        X = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / fs)
        X *= band_weight(f, f1, f2, 0.25)
        if spec.kind == "pink":
            X /= np.sqrt(np.maximum(f, f1))
        core = np.fft.irfft(X, n)
        core *= _fade(n, int(0.01 * fs), int(0.01 * fs))
    elif spec.kind == "impulse":
        n = max(int(round(dur * fs)), 1)
        core = np.zeros(n)
        core[0] = 1.0
    else:
        raise ValueError(f"Type de signal inconnu : {spec.kind}")

    core *= 10 ** (spec.level_db / 20) / np.max(np.abs(core))
    pre = int(round(spec.pre_silence * fs))
    post = int(round(spec.post_silence * fs))
    x = np.concatenate([np.zeros(pre), core, np.zeros(post)])
    meta = asdict(spec)
    meta.update(dict(f1=f1, f2=f2, duration=dur, samples=len(x),
                     pre_samples=pre, phaselab_signal=1))
    return x, meta


def sidecar_path(path) -> Path:
    return Path(path).with_suffix(".phaselab.json")


class TestSignal:
    """Le signal de référence, tel qu'il est joué dans Reaper."""

    def __init__(self, data, fs, meta, path=None):
        self.data = np.asarray(data, dtype=float)
        self.fs = int(fs)
        self.meta = dict(meta)
        self.path = str(path) if path else None

    @property
    def kind(self):
        return self.meta.get("kind", "inconnu")

    @property
    def label(self):
        return KIND_LABELS.get(self.kind, "Signal externe")

    @property
    def f1(self):
        return float(self.meta.get("f1", 20.0))

    @property
    def f2(self):
        return float(self.meta.get("f2", min(20000.0, 0.45 * self.fs)))

    @property
    def name(self):
        return Path(self.path).name if self.path else "(non enregistré)"

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


# -------------------------------------------------------------- déconvolution
def deconvolve(y, x, fs, f1, f2, keep_s=3.0, neg_s=0.05):
    """Réponse impulsionnelle h telle que y = x * h (inversion de Kirkeby).

    Renvoie (ir, t0) : t0 est le temps (s) du premier échantillon de ir
    (négatif : on garde un peu de temps négatif pour les retards < 0).
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


def _sinc_peak(ir, k):
    """Position fractionnaire du maximum de |ir| par interpolation sinc."""
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
    """Une prise (un canal d'un fichier WAV) et sa réponse impulsionnelle."""

    def __init__(self, name, y, xref, fs, f1, f2, path=None, channel=0,
                 loopback=False):
        self.name = name
        self.y = np.asarray(y, dtype=float)
        self.xref = xref
        self.fs = int(fs)
        self.path = str(path) if path else None
        self.channel = channel
        self.loopback = loopback
        self.angle = parse_angle(name)
        self.visible = True
        self.color = None
        self.ir, self.t0 = deconvolve(self.y, xref, fs, f1, f2)
        self.peak_time, self.polarity, self.onset_time, self.snr_db = analyze_ir(
            self.ir, self.t0, self.fs)
        self._env = None

    @property
    def env(self):
        if self._env is None:
            self._env = np.abs(hilbert(self.ir))
        return self._env


def load_takes(path, signal: TestSignal, loop_channel=0):
    """Charge un WAV de prises. loop_channel : 0 = aucun, sinon n° (1..n)."""
    data, fs = sf.read(str(path), always_2d=True, dtype="float64")
    if fs != signal.fs:
        raise ValueError(
            f"{Path(path).name} : {fs} Hz, alors que le signal est à {signal.fs} Hz.")
    nch = data.shape[1]
    xref, loop = signal.data, False
    chans = list(range(nch))
    if loop_channel and 1 <= loop_channel <= nch and nch > 1:
        xref, loop = data[:, loop_channel - 1].copy(), True
        chans.remove(loop_channel - 1)
    stem = Path(path).stem
    out = []
    for c in chans:
        name = stem if len(chans) == 1 else f"{stem} · c{c + 1}"
        out.append(Measurement(name, data[:, c], xref, fs, signal.f1, signal.f2,
                               path, c, loop))
    return out


# ------------------------------------------------------- spectres et lissage
@dataclass
class Spectrum:
    f: np.ndarray
    H0: np.ndarray      # réponse avec le retard propre retiré (référence = pic)
    tp: float           # temps du pic (s)
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
    """Moyenne par fraction d'octave sur une grille log.

    Les bornes de la fenêtre sont fractionnaires (somme cumulée interpolée) :
    le résultat est continu, sans marches, même là où la fenêtre ne couvre
    que quelques raies FFT. Fenêtre minimale = 1 raie, ce qui équivaut à une
    interpolation linéaire (cas « aucun lissage »).
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
    Hs: np.ndarray       # complexe lissé, retard propre retiré
    pw: np.ndarray       # puissance lissée
    tp: float
    gd: np.ndarray       # retard de groupe résiduel lissé (s)

    @property
    def mag_db(self):
        return 10 * np.log10(self.pw)

    def phase_res_deg(self):
        return np.degrees(np.unwrap(np.angle(self.Hs)))


def make_curve(spec: Spectrum, sm: Smoother) -> Curve:
    H = spec.H0
    P = np.abs(H) ** 2
    pw = np.maximum(sm(P), 1e-30)
    # Retard de groupe raie par raie (différence de phase entre raies voisines),
    # puis moyenne pondérée par l'énergie : robuste, sans dérivée numérique bruitée.
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


# ----------------------------------------------------------------- comparaison
@dataclass
class Comparison:
    dmag: np.ndarray          # dB
    dres: np.ndarray          # ° déroulé, retard dtp retiré
    dtp: float                # écart des pics (s)
    delta: float              # correction fine (s)
    tau: float                # retard total estimé (s) : mesure en retard si > 0
    theta: float              # rotation totale (°), polarité incluse
    inverted: bool
    rotation: float           # rotation restante après polarité (°)
    gain_db: float
    resid_rms: float          # ° RMS dans la bande après correction
    notch_hz: float | None
    sum_before: np.ndarray | None = None
    sum_after: np.ndarray | None = None
    sum_ideal: np.ndarray | None = None

    def dphase(self, grid, mode):
        """mode 'indiv' : retard estimé retiré ; sinon retard inclus."""
        if mode == "indiv":
            return self.dres + 360 * grid * self.delta
        return self.dres - 360 * grid * self.dtp


def compare(si: Spectrum, sref: Spectrum, ci: Curve, cref: Curve, sm: Smoother,
            fs, flo, fhi, with_sum=True, search_ms=20.0, up=8):
    f, grid = si.f, sm.grid
    dtp = si.tp - sref.tp
    C = si.H0 * np.conj(sref.H0)

    # --- retard + rotation conjoints (GCC pondéré 1/f, enveloppe analytique)
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

    # --- courbes de différence (spectre croisé lissé / puissance de référence)
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


# ------------------------------------------------- comparaison temporelle
def original_vs_take(m: Measurement, mode="aligned"):
    """Renvoie (prédiction, résidu, gain, résidu_dB).

    aligned : l'original décalé du retard mesuré et mis à l'échelle.
    model   : l'original filtré par la réponse mesurée complète ; le résidu
              ne contient alors que bruit + distorsion.
    """
    y, x, fs = m.y, m.xref, m.fs
    if mode == "model":
        nneg = int(round(-m.t0 * fs))
        pred = fftconvolve(x, m.ir)[nneg:nneg + len(y)]
        if len(pred) < len(y):
            pred = np.pad(pred, (0, len(y) - len(pred)))
        g = 1.0
    else:
        N = next_pow2(len(y) + len(x))
        f = np.fft.rfftfreq(N, 1 / fs)
        xs = np.fft.irfft(np.fft.rfft(x, N) * np.exp(-2j * np.pi * f * m.peak_time), N)[:len(y)]
        den = float(xs @ xs)
        g = float(y @ xs) / den if den > 0 else 0.0
        pred = g * xs
    res = y - pred
    ry = np.sqrt(np.mean(y ** 2)) + 1e-20
    rr = np.sqrt(np.mean(res ** 2)) + 1e-20
    return pred, res, g, 20 * np.log10(rr / ry)
