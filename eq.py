"""PhaseLab - parametric EQ (RBJ biquads), auto-EQ towards a target, targets."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import sosfilt, sosfreqz

TYPES = {
    "peak": "Bell",
    "lowshelf": "Low shelf",
    "highshelf": "High shelf",
    "hp": "High-pass 12 dB/oct",
    "lp": "Low-pass 12 dB/oct",
    "notch": "Notch",
    "allpass": "All-pass",
}
HAS_GAIN = {"peak", "lowshelf", "highshelf"}
REAEQ_NAMES = {"peak": "Band", "lowshelf": "Low Shelf", "highshelf": "High Shelf",
               "hp": "High Pass", "lp": "Low Pass", "notch": "Notch", "allpass": "All Pass"}


@dataclass
class Band:
    type: str = "peak"
    f: float = 1000.0
    gain: float = 0.0
    q: float = 1.0
    on: bool = True


def q_to_bw(q):
    """Q -> bandwidth in octaves (ReaEQ / RBJ convention)."""
    return 2 / math.log(2) * math.asinh(1 / (2 * q))


def band_sos(b: Band, fs):
    f = min(max(b.f, 1.0), 0.49 * fs)
    q = max(b.q, 0.05)
    A = 10 ** (b.gain / 40)
    w0 = 2 * math.pi * f / fs
    c, s = math.cos(w0), math.sin(w0)
    al = s / (2 * q)
    sA = math.sqrt(A)
    t = b.type
    if t == "peak":
        bb = [1 + al * A, -2 * c, 1 - al * A]; aa = [1 + al / A, -2 * c, 1 - al / A]
    elif t == "lowshelf":
        bb = [A * ((A + 1) - (A - 1) * c + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * c),
              A * ((A + 1) - (A - 1) * c - 2 * sA * al)]
        aa = [(A + 1) + (A - 1) * c + 2 * sA * al, -2 * ((A - 1) + (A + 1) * c),
              (A + 1) + (A - 1) * c - 2 * sA * al]
    elif t == "highshelf":
        bb = [A * ((A + 1) + (A - 1) * c + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * c),
              A * ((A + 1) + (A - 1) * c - 2 * sA * al)]
        aa = [(A + 1) - (A - 1) * c + 2 * sA * al, 2 * ((A - 1) - (A + 1) * c),
              (A + 1) - (A - 1) * c - 2 * sA * al]
    elif t == "lp":
        bb = [(1 - c) / 2, 1 - c, (1 - c) / 2]; aa = [1 + al, -2 * c, 1 - al]
    elif t == "hp":
        bb = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; aa = [1 + al, -2 * c, 1 - al]
    elif t == "notch":
        bb = [1, -2 * c, 1]; aa = [1 + al, -2 * c, 1 - al]
    elif t == "allpass":
        bb = [1 - al, -2 * c, 1 + al]; aa = [1 + al, -2 * c, 1 - al]
    else:
        raise ValueError(t)
    return np.array([[bb[0] / aa[0], bb[1] / aa[0], bb[2] / aa[0], 1.0, aa[1] / aa[0], aa[2] / aa[0]]])


def eq_sos(bands, fs):
    act = [band_sos(b, fs) for b in bands if b.on]
    return np.vstack(act) if act else None


def eq_response(bands, f, fs):
    """Complex response of the EQ chain at frequencies f (Hz)."""
    sos = eq_sos(bands, fs)
    if sos is None:
        return np.ones(len(f), complex)
    f = np.minimum(np.asarray(f, float), 0.4999 * fs)
    return sosfreqz(sos, worN=f, fs=fs)[1]


def band_response_db(b, f, fs):
    return 20 * np.log10(np.abs(sosfreqz(band_sos(b, fs), worN=np.minimum(f, 0.4999 * fs), fs=fs)[1]) + 1e-12)


def eq_impulse(bands, fs, n=None):
    n = n or int(fs)
    x = np.zeros(n)
    x[0] = 1.0
    sos = eq_sos(bands, fs)
    return x if sos is None else sosfilt(sos, x)


def auto_eq(f, source_db, target_db, fs, n_bands=6, fmin=40.0, fmax=12000.0,
            max_boost=6.0, max_cut=12.0, q_min=0.5, q_max=8.0):
    """Bell-filter EQ that brings source towards target (average level aligned).

    Greedy placement on the largest deviation, then joint optimization
    (bounded least squares) of all bands.
    """
    sel = (f >= fmin) & (f <= fmax)
    fg = np.geomspace(max(fmin, f[sel][0]), min(fmax, f[sel][-1]), 240)
    lf = np.log10(f)
    src = np.interp(np.log10(fg), lf, source_db)
    tgt = np.interp(np.log10(fg), lf, target_db)
    err = tgt - src
    err -= np.mean(err)
    bands = []
    r = err.copy()
    for _ in range(n_bands):
        cand = np.clip(r, -max_cut, max_boost)
        i = int(np.argmax(np.abs(cand)))
        g = float(cand[i])
        if abs(g) < 1.0:
            break
        half = abs(r[i]) / 2
        lo = i
        while lo > 0 and abs(r[lo]) > half and np.sign(r[lo]) == np.sign(r[i]):
            lo -= 1
        hi = i
        while hi < len(r) - 1 and abs(r[hi]) > half and np.sign(r[hi]) == np.sign(r[i]):
            hi += 1
        bw = max(math.log2(fg[hi] / fg[lo]), 0.1)
        q = float(np.clip(1 / (2 * math.sinh(math.log(2) / 2 * bw)), q_min, q_max))
        b = Band("peak", float(fg[i]), g, q)
        bands.append(b)
        r = r - band_response_db(b, fg, fs)
    if not bands:
        return []

    def unpack(p):
        return [Band("peak", 10 ** p[3 * k], p[3 * k + 1], 10 ** p[3 * k + 2]) for k in range(len(bands))]

    def resid(p):
        tot = np.zeros(len(fg))
        for b in unpack(p):
            tot += band_response_db(b, fg, fs)
        return np.concatenate([tot - err, 0.4 * p[1::3]])

    p0, lb, ub = [], [], []
    for b in bands:
        p0 += [math.log10(b.f), b.gain, math.log10(b.q)]
        lb += [math.log10(fg[0]), -max_cut, math.log10(q_min)]
        ub += [math.log10(fg[-1]), max(max_boost, 1e-3), math.log10(q_max)]
    p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    try:
        sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=400)
        out = unpack(sol.x)
    except Exception:
        out = bands
    out = [b for b in out if abs(b.gain) >= 0.5]
    for b in out:
        b.f, b.gain, b.q = round(b.f, 1), round(b.gain, 2), round(b.q, 2)
    return sorted(out, key=lambda b: b.f)


def reaeq_text(bands):
    lines = ["PhaseLab — settings to enter in ReaEQ (BW = bandwidth in octaves)", ""]
    for i, b in enumerate(bands, 1):
        if not b.on:
            continue
        g = f"{b.gain:+.1f} dB" if b.type in HAS_GAIN else "—"
        lines.append(f"Band {i}: {REAEQ_NAMES[b.type]:10s}  {b.f:8.1f} Hz   gain {g:>9s}   "
                     f"BW {q_to_bw(b.q):.2f} oct  (Q {b.q:.2f})")
    return "\n".join(lines)


# ----------------------------------------------------------------- targets
TARGET_DIR = Path.home() / ".phaselab" / "targets"
LEGACY_DIR = Path.home() / ".phaselab" / "cibles"   # targets saved by earlier (French) versions


@dataclass
class Target:
    name: str
    f: list = field(default_factory=list)
    db: list = field(default_factory=list)
    tol: float = 1.5
    note: str = ""
    builtin: bool = False

    def on_grid(self, grid):
        f = np.asarray(self.f, float)
        d = np.asarray(self.db, float)
        return np.interp(np.log10(grid), np.log10(f), d)


def slope_target(name, slope_db_oct, tol=1.5):
    f = np.geomspace(10, 40000, 64)
    return Target(name, list(f), list(slope_db_oct * np.log2(f / 1000)), tol, builtin=True)


BUILTIN = [slope_target("Flat", 0.0), slope_target("Slope −3 dB/oct (pink noise)", -3.0),
           slope_target("Slope −4.5 dB/oct (average mix)", -4.5)]


def _safe(name):
    return re.sub(r"[^\w\-. ]+", "_", name).strip() or "target"


def list_targets():
    out = list(BUILTIN)
    files = []
    for folder in (TARGET_DIR, LEGACY_DIR):
        if folder.exists():
            files += sorted(folder.glob("*.json"))
    if files:
        for p in files:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                d.pop("builtin", None)
                out.append(Target(**d))
            except Exception:
                continue
    return out


def save_target(t: Target):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    d = asdict(t)
    d.pop("builtin")
    d["f"] = [round(float(v), 3) for v in d["f"]]
    d["db"] = [round(float(v), 3) for v in d["db"]]
    (TARGET_DIR / f"{_safe(t.name)}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def delete_target(name):
    for folder in (TARGET_DIR, LEGACY_DIR):
        p = folder / f"{_safe(name)}.json"
        if p.exists():
            p.unlink()


def import_target(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        d = json.loads(path.read_text(encoding="utf-8"))
        d.pop("builtin", None)
        return Target(**d)
    fs_, ds = [], []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(("*", "#")):
            continue
        if ";" in line:   # European-style CSV: decimal comma
            parts = [p.strip().replace(",", ".") for p in line.split(";")]
        else:
            parts = re.split(r"[\s,]+", line)
        try:
            fv, dv = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        if fv > 0:
            fs_.append(fv); ds.append(dv)
    if len(fs_) < 2:
        raise ValueError("no frequency / dB pair found")
    o = np.argsort(fs_)
    return Target(path.stem, list(np.array(fs_)[o]), list(np.array(ds)[o]))


def export_target(t: Target, path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        d = asdict(t); d.pop("builtin")
        path.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"* PhaseLab - target {t.name}\n* Frequency(Hz) Level(dB)\n")
            for f, d in zip(t.f, t.db):
                fh.write(f"{f:.2f} {d:.3f}\n")
