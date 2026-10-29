"""PhaseLab - real-time engine (dual-channel analyzer, Open Sound Meter style).

The audio stream runs in a separate thread (interface callback): it plays the
generator and stores the input in ring buffers. The GUI calls update() about
10 times per second: new frames are analyzed (Hann-windowed FFT, 75 % overlap),
averaged (FIFO or exponential), giving for each measurement channel: spectrum
(RTA), transfer function H1 = Sxy/Sxx, coherence and impulse response, with
compensation of the automatically found delay.

"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

import dsp

try:  # PortAudio ships with sounddevice on Windows and macOS
    import sounddevice as sd
    SD_ERROR = None
except Exception as e:  # pragma: no cover - depends on the machine
    sd = None
    SD_ERROR = str(e)

GEN_KINDS = {"pink": "Pink noise", "white": "White noise", "brown": "Brown noise",
             "pink_per": "Periodic pink noise", "sweep": "Continuous sweeps", "sine": "Sine"}
# Generators whose content is not broadband at every instant: the frequency
# response builds up over time, but delay / impulse response are not reliable
# (not effective for time-domain measurement).
TIMING_UNRELIABLE = {"sweep", "sine"}
AVERAGES = [("None", ("none", 1)), ("FIFO 2", ("fifo", 2)), ("FIFO 4", ("fifo", 4)), ("FIFO 8", ("fifo", 8)),
            ("FIFO 16", ("fifo", 16)), ("FIFO 32", ("fifo", 32)), ("Exponential fast", ("exp", 0.6)),
            ("Exponential slow", ("exp", 0.9))]


# ------------------------------------------------------------------ generator
class Generator:
    """Continuous source: reads a precomputed periodic buffer (or a sine)."""

    def __init__(self, kind="pink", fs=48000, level_db=-12.0, sine_freq=1000.0, seed=11):
        self.kind, self.fs = kind, int(fs)
        self.gain = 10 ** (level_db / 20)
        self.sine_freq = sine_freq
        self.pos = 0
        self.phase = 0.0
        rng = np.random.default_rng(seed)
        if kind == "sine":
            self.buf = None
        elif kind == "sweep":
            # period 2^16 (1.37 s at 48 kHz): longer than the delay search window
            P = 1 << 16
            ns = int(P * 0.92)
            sw = dsp._exp_sweep(ns, fs, 20.0, min(20000.0, 0.45 * fs))
            sw *= dsp._fade(ns, int(fs * 0.03), int(fs * 0.003))
            self.buf = np.concatenate([sw, np.zeros(P - ns)])
        else:
            P = 1 << 18
            slope = {"white": 0, "pink": 1, "brown": 2, "pink_per": 1}[kind]
            self.buf = dsp._shaped_noise(P, fs, 20.0, min(20000.0, 0.45 * fs), slope, rng,
                                         flat_mag=(kind == "pink_per"))
        if self.buf is not None:
            self.buf = (self.buf / np.max(np.abs(self.buf))).astype(np.float32)

    def set_level(self, level_db):
        self.gain = 10 ** (level_db / 20)

    def next(self, n):
        if self.buf is None:
            w = 2 * math.pi * self.sine_freq / self.fs
            ph = self.phase + w * np.arange(n)
            self.phase = float((self.phase + w * n) % (2 * math.pi))
            return (self.gain * np.sin(ph)).astype(np.float32)
        P = len(self.buf)
        idx = (self.pos + np.arange(n)) % P
        self.pos = (self.pos + n) % P
        return self.gain * self.buf[idx]


# --------------------------------------------------------------- tampons
class Ring:
    def __init__(self, nch, size=1 << 21):
        self.size = size
        self.data = np.zeros((nch, size), np.float32)
        self.count = 0           # total number of samples written
        self.lock = threading.Lock()

    def write(self, block):      # block: (nch, n)
        n = block.shape[1]
        with self.lock:
            i = self.count % self.size
            k = min(n, self.size - i)
            self.data[:, i:i + k] = block[:, :k]
            if k < n:
                self.data[:, :n - k] = block[:, k:]
            self.count += n

    def read(self, ch, end, n):
        """n samples of channel ch ending at absolute index end."""
        with self.lock:
            start = end - n
            if start < self.count - self.size or end > self.count or start < 0:
                return None
            idx = np.arange(start, end) % self.size
            return self.data[ch, idx].astype(np.float64)


# ---------------------------------------------------------------- engine
class Channel:
    """Analysis of one measurement channel against the reference."""

    def __init__(self, name):
        self.name = name
        self.delay = 0            # integer delay (samples) measurement / reference
        self.frames = deque()
        self.S = None             # averaged (Sxx, Syy, Sxy)
        self.n = 0

    def reset(self):
        self.frames.clear()
        self.S = None
        self.n = 0


class LiveEngine:
    REF, A, B = 0, 1, 2

    def __init__(self):
        self.fs = 48000
        self.nfft = 32768
        self.avg = ("fifo", 8)
        self.ring = Ring(3)
        self.gen = None
        self.gen_on = True
        self.out_channels = [0, 1]
        self.in_a, self.in_b, self.in_loop = 0, None, None   # 0-based indices; loop None = internal generator
        self.ch = {self.A: Channel("A"), self.B: Channel("B")}
        self.next_end = None
        self.levels = {}          # input channel -> (peak dBFS, RMS dBFS)
        self.out_level = (-120.0, -120.0)
        self.xruns = 0
        self.backend = None
        self.running = False
        self.last_result = {}

    # ------------------------------------------------------------ config
    def configure(self, fs, nfft, avg):
        changed = (fs != self.fs) or (nfft != self.nfft)
        self.fs, self.nfft, self.avg = int(fs), int(nfft), avg
        if changed:
            self.reset_averages()

    def reset_averages(self):
        for c in self.ch.values():
            c.reset()
        self.next_end = None

    # ------------------------------------------------------ audio thread
    def audio_callback(self, indata, outdata):
        """Called by the audio thread. indata (n, nin), outdata (n, nout)."""
        n = indata.shape[0]
        outdata[:] = 0
        g = self.gen
        out = g.next(n) if (g is not None and self.gen_on) else np.zeros(n, np.float32)
        for c in self.out_channels:
            if c < outdata.shape[1]:
                outdata[:, c] = out
        nin = indata.shape[1]
        ref = indata[:, self.in_loop] if (self.in_loop is not None and self.in_loop < nin) else out
        a = indata[:, self.in_a] if self.in_a is not None and self.in_a < nin else np.zeros(n, np.float32)
        b = indata[:, self.in_b] if self.in_b is not None and self.in_b < nin else np.zeros(n, np.float32)
        self.ring.write(np.vstack([ref, a, b]))
        lv = {}
        for c in range(nin):
            x = indata[:, c]
            pk = float(np.max(np.abs(x))) if n else 0.0
            rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
            lv[c] = (20 * math.log10(max(pk, 1e-6)), 20 * math.log10(max(rms, 1e-6)))
        self.levels = lv
        pk = float(np.max(np.abs(out))) if n else 0.0
        self.out_level = (20 * math.log10(max(pk, 1e-6)), 0.0)

    # ------------------------------------------------------ analysis
    def find_delay(self, key, seconds=2.0, max_s=1.0):
        """Delay of the measurement relative to the reference (GCC-PHAT); returns s or None."""
        fs = self.fs
        end = self.ring.count
        M = int(seconds * fs)
        maxlag = int(max_s * fs)
        x = self.ring.read(self.REF, end - maxlag, M)
        y = self.ring.read(key, end, M + maxlag)
        if x is None or y is None or not np.any(x) or not np.any(y):
            return None
        d = dsp.find_delay(y, x, fs, 20.0, min(20000.0, 0.45 * fs), max_s=max_s, max_len_s=seconds)
        d = max(int(d), -maxlag)
        c = self.ch[key]
        c.delay = d
        c.reset()
        self.next_end = None
        return d / fs

    def set_delay(self, key, seconds):
        self.ch[key].delay = int(round(seconds * self.fs))
        self.ch[key].reset()

    def update(self, keys=(1,)):
        """Analyzes the available frames; returns {key: results}."""
        nfft, fs = self.nfft, self.fs
        hop = nfft // 4
        win = np.hanning(nfft)
        avail = self.ring.count
        maxd = max([max(self.ch[k].delay, 0) for k in keys] + [0])
        if self.next_end is None:
            self.next_end = max(avail - maxd, nfft)
            self.next_end -= self.next_end % hop
        done = 0
        while self.next_end + maxd <= avail and done < 16:
            e = self.next_end
            x = self.ring.read(self.REF, e, nfft)
            if x is None:
                self.next_end = None
                break
            X = np.fft.rfft(x * win)
            for k in keys:
                c = self.ch[k]
                y = self.ring.read(k, e + c.delay, nfft)
                if y is None:
                    continue
                Y = np.fft.rfft(y * win)
                fr = (np.abs(X) ** 2, np.abs(Y) ** 2, Y * np.conj(X))
                self._accumulate(c, fr)
            self.next_end += hop
            done += 1
        if self.next_end is not None and self.next_end + maxd + 4 * hop < avail:
            # analysis lagging too far behind (slow machine): jump forward
            self.next_end = avail - maxd - (avail - maxd) % hop
        out = {}
        f = np.fft.rfftfreq(nfft, 1 / fs)
        for k in keys:
            c = self.ch[k]
            if c.S is None:
                continue
            Sxx, Syy, Sxy = c.S
            H = dsp._regularize(Sxy, Sxx, f, 20.0, min(20000.0, 0.45 * fs))
            coh = np.clip(np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30), 0, 1)
            out[k] = dict(f=f, Sxx=Sxx, Syy=Syy, H=H, coh=coh, n=c.n, delay=c.delay / fs)
        self.last_result = out
        return out

    def _accumulate(self, c, fr):
        mode, p = self.avg
        if mode == "fifo":
            c.frames.append(fr)
            c.S = tuple(np.array(a, copy=True) for a in fr) if c.S is None else tuple(s + a for s, a in zip(c.S, fr))
            while len(c.frames) > p:
                old = c.frames.popleft()
                c.S = tuple(s - a for s, a in zip(c.S, old))
            if c.n % 256 == 255:  # avoids numerical drift from repeated additions / subtractions
                c.S = tuple(np.sum([fr_[i] for fr_ in c.frames], axis=0) for i in range(3))
        elif mode == "exp":
            c.S = fr if c.S is None else tuple(p * s + (1 - p) * a for s, a in zip(c.S, fr))
        else:
            c.S = fr
        c.n += 1

    # ------------------------------------------------------- results
    @staticmethod
    def impulse(res, fs):
        """Impulse response (with a little negative time) and its time origin."""
        H = res["H"]
        nfft = 2 * (len(H) - 1)
        h = np.fft.irfft(H, nfft)
        nneg = nfft // 8
        ir = np.concatenate([h[nfft - nneg:], h[:nfft - nneg]])
        return ir, res["delay"] - nneg / fs

    def precise_delay(self, res, fs):
        ir, t0 = self.impulse(res, fs)
        k = int(np.argmax(np.abs(ir)))
        kp = dsp._sinc_peak(ir, k)
        return t0 + kp / fs, (1 if ir[k] >= 0 else -1)

    def to_measurement(self, res, name, f1=20.0, f2=20000.0):
        ir, t0 = self.impulse(res, self.fs)
        m = dsp.Measurement(name, self.fs, ir, t0, max(f1, 2.0 * self.fs / self.nfft),
                            min(f2, 0.45 * self.fs), method="live", coh_f=res["f"], coh=res["coh"],
                            averages=res["n"])
        m.virtual = True
        return m

    # ------------------------------------------------------- start / stop
    def start(self, backend):
        self.stop()
        self.ring = Ring(3)
        self.reset_averages()
        self.backend = backend
        backend.start(self)
        self.running = True

    def stop(self):
        if self.backend is not None:
            try:
                self.backend.stop()
            except Exception:
                pass
        self.backend = None
        self.running = False


# --------------------------------------------------------------- backends
def list_devices():
    """[(label, (input, output), n inputs, n outputs, default fs)]"""
    out = []
    if sd is None:
        return out
    try:
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception:
        return out
    for api in apis:
        ins = [i for i in api["devices"] if devs[i]["max_input_channels"] > 0]
        outs = [i for i in api["devices"] if devs[i]["max_output_channels"] > 0]
        for i in ins:
            # prefer the same name for input and output (multichannel interface)
            o = next((j for j in outs if devs[j]["name"] == devs[i]["name"]), None)
            if o is None and api.get("default_output_device", -1) in outs:
                o = api["default_output_device"]
            if o is None and outs:
                o = outs[0]
            if o is None:
                continue
            lab = devs[i]["name"] if devs[o]["name"] == devs[i]["name"] else f"{devs[i]['name']} → {devs[o]['name']}"
            out.append((f"{lab}  ({api['name']})", (i, o), devs[i]["max_input_channels"],
                        devs[o]["max_output_channels"], int(devs[i]["default_samplerate"])))
    return out


class DeviceBackend:
    """Real audio interface (PortAudio via sounddevice), full duplex."""

    def __init__(self, device, n_in, n_out, fs, blocksize=1024):
        self.device, self.n_in, self.n_out = device, n_in, n_out
        self.fs, self.blocksize = fs, blocksize
        self.stream = None

    def start(self, engine):
        def cb(indata, outdata, frames, t, status):
            if status:
                engine.xruns += 1
            engine.audio_callback(indata, outdata)
        self.stream = sd.Stream(samplerate=self.fs, blocksize=self.blocksize, device=self.device,
                                channels=(self.n_in, self.n_out), dtype="float32", callback=cb,
                                latency="low")
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class SimBackend:
    """Simulated interface: input 1 = mic A, 2 = mic B, 3 = loopback.

    The generator output goes through a "room" (filters, delays,
    reflection): lets you try the analyzer without hardware.
    """
    N_IN, N_OUT = 3, 2

    def __init__(self, fs=48000, blocksize=1024, delay_a=0.0032, delay_b=0.0044, latency=0.0053,
                 invert_b=False):
        self.fs, self.blocksize = fs, blocksize
        from scipy.signal import butter, sosfilt
        L = int(0.03 * fs)
        imp = np.zeros(L)
        imp[0] = 1.0
        top = sosfilt(butter(2, [70 / (fs / 2), 15000 / (fs / 2)], "band", output="sos"), imp)
        def fir(d, g=1.0, refl=0.3):
            h = g * dsp.frac_shift(top, latency + d, fs, L) + refl * g * dsp.frac_shift(top, latency + d + 0.0047, fs, L)
            return h.astype(np.float32)
        self.irs = [fir(delay_a), fir(delay_b, -0.7 if invert_b else 0.7, 0.15),
                    dsp.frac_shift(imp, latency, fs, L).astype(np.float32)]
        self.hist = np.zeros(L - 1, np.float32)
        self.rng = np.random.default_rng(5)
        self.thread = None
        self.alive = False

    def _block(self, engine):
        n = self.blocksize
        indata = np.zeros((n, self.N_IN), np.float32)
        outdata = np.zeros((n, self.N_OUT), np.float32)
        # this block's output comes back through the room; the previous output is used
        x = np.concatenate([self.hist, self._last_out])
        for c, h in enumerate(self.irs):
            indata[:, c] = np.convolve(x, h, mode="valid")[:n]
        indata[:, :2] += (2e-4 * self.rng.standard_normal((n, 2))).astype(np.float32)
        engine.audio_callback(indata, outdata)
        self.hist = x[-(len(self.hist)):]
        self._last_out = outdata[:, 0].copy()

    def start(self, engine):
        self._last_out = np.zeros(self.blocksize, np.float32)
        self.alive = True

        def run():
            dt = self.blocksize / self.fs
            t_next = time.perf_counter()
            while self.alive:
                self._block(engine)
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def run_blocks(self, engine, nblocks):
        """Used by tests: advances the simulation without waiting."""
        if not hasattr(self, "_last_out"):
            self._last_out = np.zeros(self.blocksize, np.float32)
        for _ in range(nblocks):
            self._block(engine)

    def stop(self):
        self.alive = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
