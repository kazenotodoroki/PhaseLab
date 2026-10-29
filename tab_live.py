"""Real-time tab: live dual-channel analyzer (Open Sound Meter style)."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import dsp
import eq
import live
from style import C
from widgets import (break_wraps, card, dspin, fmt_hz, freq_plot, lin_plot, muted, options_row, pen,
                     plot_area, readout, section, tab_container)

COL_A, COL_B, COL_REF = "#4fc3f7", "#ff8a65", "#7c8590"
SIM_LABEL = "Simulation (no audio interface)"
NFFTS = [8192, 16384, 32768, 65536, 131072]


class Meter(QtWidgets.QProgressBar):
    def __init__(self):
        super().__init__()
        self.setRange(-600, 0)
        self.setTextVisible(True)
        self.setFixedHeight(14)
        self.set(-120)

    def set(self, db):
        db = max(db, -60.0)
        self.setValue(int(db * 10))
        self.setFormat(f"{db:.1f} dB" if db > -59.9 else "—")
        col = C["bad"] if db > -1 else (C["warn"] if db > -6 else C["accent"])
        self.setStyleSheet(f"QProgressBar {{ background: {C['panel2']}; border: 1px solid {C['border']}; "
                           f"border-radius: 1px; color: {C['text']}; font-size: 10px; text-align: center; }}"
                           f"QProgressBar::chunk {{ background: {col}; }}")


class LiveTab(QtWidgets.QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.engine = live.LiveEngine()
        self.devices = live.list_devices()
        self.frozen = []          # frozen spectra (name, color, grid, dB)
        self.stored = []          # stored traces (name, color, grid, dB, phase)
        self.res = {}
        self._sm_cache = {}
        self._t_start = None
        self._last_track = 0.0
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_panel())
        lay.addWidget(self._build_plots(), 1)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self._device_changed()

    # ------------------------------------------------------------ panel
    def _build_panel(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side = QtWidgets.QFrame()
        side.setObjectName("Sidebar")
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(7)

        v.addWidget(section("Audio"))
        self.dev = QtWidgets.QComboBox()
        for lab, *_ in self.devices:
            self.dev.addItem(lab)
        self.dev.addItem(SIM_LABEL)
        self.dev.currentIndexChanged.connect(self._device_changed)
        v.addWidget(self.dev)
        if live.sd is None:
            note = muted("Audio interface unavailable (" + (live.SD_ERROR or "") + "). Install \"sounddevice\": "
                         "the launcher does it automatically. Simulation mode is still available.", wrap=True)
            v.addWidget(note)
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        self.fs = QtWidgets.QComboBox()
        for f in (44100, 48000, 88200, 96000):
            self.fs.addItem(f"{f} Hz", f)
        self.fs.setCurrentIndex(1)
        self.block = QtWidgets.QComboBox()
        for b in (256, 512, 1024, 2048):
            self.block.addItem(f"{b} samples", b)
        self.block.setCurrentIndex(2)
        self.in_a = QtWidgets.QComboBox()
        self.in_b = QtWidgets.QComboBox()
        self.ref = QtWidgets.QComboBox()
        self.ref.setToolTip("Loopback: the interface output fed back into an input (most accurate).\n"
                            "Internal generator: the signal sent is used as reference; the interface "
                            "latency is then included in the delay.")
        for r, (lab, w) in enumerate((("Sample rate", self.fs), ("Block", self.block), ("Measure A", self.in_a),
                                      ("Measure B", self.in_b), ("Reference", self.ref))):
            g.addWidget(muted(lab), r, 0)
            g.addWidget(w, r, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        for w in (self.fs, self.block):
            w.currentIndexChanged.connect(self._restart_if_running)
        for w in (self.in_a, self.in_b, self.ref):
            w.currentIndexChanged.connect(self._routing_changed)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.toggle)
        v.addWidget(self.start_btn)
        self.state = muted("Stopped", wrap=True)
        v.addWidget(self.state)

        v.addSpacing(6)
        v.addWidget(section("Generator"))
        r = QtWidgets.QHBoxLayout()
        self.gen_on = QtWidgets.QCheckBox("On")
        self.gen_on.setChecked(True)
        self.gen_on.toggled.connect(lambda on: setattr(self.engine, "gen_on", on))
        self.gen_kind = QtWidgets.QComboBox()
        for k, lab in live.GEN_KINDS.items():
            self.gen_kind.addItem(lab, k)
        self.gen_kind.currentIndexChanged.connect(self._gen_changed)
        r.addWidget(self.gen_on)
        r.addWidget(self.gen_kind, 1)
        v.addLayout(r)
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(8)
        self.level = dspin(-80, 0, -18, 1, " dBFS", step=1, slot=self._level_changed,
                           tip="Generator peak level. Start low and raise it gradually!")
        self.sine = dspin(10, 20000, 1000, 1, " Hz", slot=self._gen_changed)
        self.outs = QtWidgets.QComboBox()
        self.outs.currentIndexChanged.connect(self._routing_changed)
        g.addWidget(muted("Level"), 0, 0)
        g.addWidget(self.level, 0, 1)
        self.sine_lab = muted("Frequency")
        g.addWidget(self.sine_lab, 1, 0)
        g.addWidget(self.sine, 1, 1)
        g.addWidget(muted("Outputs"), 2, 0)
        g.addWidget(self.outs, 2, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        self.gen_warn = QtWidgets.QLabel(
            "Continuous sweeps: the frequency response builds up with long averaging (FIFO 16 or more), "
            "but they are not effective for time-domain measurement (delay, impulse response, phase). "
            "Use pink noise for those.")
        self.gen_warn.setWordWrap(True)
        self.gen_warn.setStyleSheet(f"color: {C['warn']}; font-size: 11px;")
        self.gen_warn.setVisible(False)
        v.addWidget(self.gen_warn)

        v.addSpacing(6)
        v.addWidget(section("Analysis"))
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(8)
        self.nfft = QtWidgets.QComboBox()
        for n in NFFTS:
            self.nfft.addItem(f"{n // 1024}k  ({n / 48000:.2f} s)", n)
        self.nfft.setCurrentIndex(2)
        self.nfft.setToolTip("FFT size: larger = more accurate in the low end, but slower to react")
        self.avg = QtWidgets.QComboBox()
        for lab, val in live.AVERAGES:
            self.avg.addItem(lab, val)
        self.avg.setCurrentIndex(3)
        for w in (self.nfft, self.avg):
            w.currentIndexChanged.connect(self._analysis_changed)
        g.addWidget(muted("FFT"), 0, 0)
        g.addWidget(self.nfft, 0, 1)
        g.addWidget(muted("Averaging"), 1, 0)
        g.addWidget(self.avg, 1, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g)

        cd = card()
        cv = QtWidgets.QVBoxLayout(cd)
        cv.setContentsMargins(10, 8, 10, 8)
        cv.setSpacing(3)
        self.delay_lab = QtWidgets.QLabel("Delay: —")
        self.delay_lab.setTextFormat(Qt.RichText)
        self.delay_lab.setStyleSheet("font-size: 13px;")
        cv.addWidget(self.delay_lab)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Find delay")
        b.clicked.connect(self.find_delays)
        self.track = QtWidgets.QCheckBox("Auto tracking")
        self.track.setToolTip("Searches the delay every 2 s and updates it if it moved (mic or speaker moved)")
        r.addWidget(b, 1)
        r.addWidget(self.track)
        cv.addLayout(r)
        v.addWidget(cd)

        self.meters = {}
        mg = QtWidgets.QGridLayout()
        mg.setHorizontalSpacing(8)
        mg.setVerticalSpacing(4)
        for i, (k, lab) in enumerate((("a", "Input A"), ("b", "Input B"), ("ref", "Reference"), ("out", "Output"))):
            mg.addWidget(muted(lab), i, 0)
            m = Meter()
            mg.addWidget(m, i, 1)
            self.meters[k] = m
        mg.setColumnStretch(1, 1)
        v.addLayout(mg)

        v.addSpacing(6)
        v.addWidget(section("Capture"))
        r = QtWidgets.QHBoxLayout()
        self.store_a = QtWidgets.QPushButton("Store A")
        self.store_b = QtWidgets.QPushButton("Store B")
        for btn, k in ((self.store_a, 1), (self.store_b, 2)):
            btn.setToolTip("Adds the current measurement to the takes list: every tab "
                           "(Alignment, EQ, Differences…) can then use it")
            btn.clicked.connect(lambda _, k=k: self.store(k))
            r.addWidget(btn)
        v.addLayout(r)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Freeze spectrum")
        b.clicked.connect(self.freeze_rta)
        r.addWidget(b)
        b = QtWidgets.QPushButton("Clear")
        b.setToolTip("Clears the frozen spectra and stored traces shown here")
        b.clicked.connect(self.clear_overlays)
        r.addWidget(b)
        v.addLayout(r)
        b = QtWidgets.QPushButton("Spectrum A as target…")
        b.clicked.connect(self.rta_as_target)
        v.addWidget(b)
        self.hold = QtWidgets.QCheckBox("Hold display")
        v.addWidget(self.hold)
        v.addStretch()
        scroll.setWidget(side)
        return scroll

    def _build_plots(self):
        w, v = tab_container()
        o = options_row()
        self.show = {}
        for k, lab, on in (("rta", "Spectrum", True), ("mag", "Transfer function", True), ("ph", "Phase", True),
                           ("coh", "Coherence", True), ("ir", "Impulse response", False)):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(on)
            cb.toggled.connect(self._relayout)
            o.addWidget(cb)
            self.show[k] = cb
        o.addSpacing(12)
        o.addWidget(muted("Target"))
        self.target = QtWidgets.QComboBox()
        self.target.setMinimumWidth(150)
        o.addWidget(self.target)
        o.addWidget(muted("Hide below"))
        self.coh_thr = dspin(0, 0.99, 0, 2, "", 56, 0.05, tip="Hides frequencies where coherence is below this threshold")
        o.addWidget(self.coh_thr)
        o.addStretch()
        v.addLayout(o)
        self.glw = plot_area()
        self.p_rta = freq_plot("dBFS / band")
        self.p_mag = freq_plot("Transfer (dB)")
        self.p_ph = freq_plot("Phase (°)")
        self.p_coh = freq_plot("Coherence")
        self.p_ir = lin_plot("Response (norm.)", "Time (ms)")
        for p in (self.p_mag, self.p_ph, self.p_coh):
            p.setXLink(self.p_rta)
        self.p_ph.setYRange(-185, 185, padding=0)
        self.p_ph.getAxis("left").setTicks([[(x, f"{x}") for x in range(-180, 181, 90)]])
        self.p_coh.setYRange(0, 1.02, padding=0)
        self.p_rta.setYRange(-100, -10, padding=0)
        self.p_mag.setYRange(-30, 12, padding=0)
        self.p_rta.setXRange(math.log10(20), math.log10(20000), padding=0.01)
        # persistent curves
        self.c = {}
        for k, col in (("a", COL_A), ("b", COL_B)):
            self.c["rta_" + k] = self.p_rta.plot([], [], pen=pen(col, 1.5))
            self.c["mag_" + k] = self.p_mag.plot([], [], pen=pen(col, 1.7), connect="finite")
            self.c["ph_" + k] = self.p_ph.plot([], [], pen=pen(col, 1.5), connect="finite")
            self.c["coh_" + k] = self.p_coh.plot([], [], pen=pen(col, 1.2))
            self.c["ir_" + k] = self.p_ir.plot([], [], pen=pen(col, 1.3))
        self.c["rta_ref"] = self.p_rta.plot([], [], pen=pen(COL_REF, 1.0, Qt.DashLine))
        self.c["tgt"] = self.p_rta.plot([], [], pen=pen(C["text"], 1.3, Qt.DashLine))
        self.overlay_items = []
        v.addWidget(self.glw, 1)
        self.ro = readout()
        v.addWidget(self.ro)
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self._relayout()
        self.reload_targets()
        return w

    def _relayout(self, *_):
        self.glw.ci.clear()
        row = 0
        for k, p in (("rta", self.p_rta), ("mag", self.p_mag), ("ph", self.p_ph), ("coh", self.p_coh),
                     ("ir", self.p_ir)):
            if self.show[k].isChecked():
                self.glw.addItem(p, row, 0)
                self.glw.ci.layout.setRowStretchFactor(row, 1 if k == "coh" else 3)
                row += 1

    def reload_targets(self):
        cur = self.target.currentText()
        self.targets = eq.list_targets()
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("None")
        for t in self.targets:
            self.target.addItem(t.name)
        self.target.setCurrentIndex(max(self.target.findText(cur), 0))
        self.target.blockSignals(False)

    # ------------------------------------------------------------ routing
    def _is_sim(self):
        return self.dev.currentText() == SIM_LABEL

    def _device_info(self):
        if self._is_sim():
            return None, live.SimBackend.N_IN, live.SimBackend.N_OUT, 48000
        lab, dev, nin, nout, fs = self.devices[self.dev.currentIndex()]
        return dev, nin, nout, fs

    def _device_changed(self, *_):
        dev, nin, nout, fs = self._device_info()
        sim = self._is_sim()
        names = ["simulated mic A", "simulated mic B", "simulated loopback"] if sim else []
        for combo, none in ((self.in_a, False), (self.in_b, True)):
            combo.blockSignals(True)
            combo.clear()
            if none:
                combo.addItem("None", None)
            for i in range(min(nin, 32)):
                combo.addItem(f"Input {i + 1}" + (f" ({names[i]})" if i < len(names) else ""), i)
            combo.blockSignals(False)
        self.ref.blockSignals(True)
        self.ref.clear()
        self.ref.addItem("Generator (internal)", None)
        for i in range(min(nin, 32)):
            self.ref.addItem(f"Loopback: input {i + 1}", i)
        self.ref.blockSignals(False)
        self.outs.blockSignals(True)
        self.outs.clear()
        if nout >= 2:
            self.outs.addItem("Outputs 1 + 2", [0, 1])
        for i in range(min(nout, 32)):
            self.outs.addItem(f"Output {i + 1}", [i])
        self.outs.blockSignals(False)
        if sim:
            self.in_b.setCurrentIndex(2)
            self.ref.setCurrentIndex(3)
        else:
            self.in_b.setCurrentIndex(0)
            self.ref.setCurrentIndex(0)
        i = self.fs.findData(fs)
        if i >= 0:
            self.fs.blockSignals(True)
            self.fs.setCurrentIndex(i)
            self.fs.blockSignals(False)
        self._routing_changed()
        self._restart_if_running()

    def _routing_changed(self, *_):
        e = self.engine
        e.in_a = self.in_a.currentData()
        e.in_b = self.in_b.currentData()
        e.in_loop = self.ref.currentData()
        e.out_channels = self.outs.currentData() or [0]
        e.reset_averages()
        self.store_b.setEnabled(e.in_b is not None)

    def _gen_changed(self, *_):
        k = self.gen_kind.currentData()
        self.sine.setVisible(k == "sine")
        self.sine_lab.setVisible(k == "sine")
        self.gen_warn.setVisible(k in live.TIMING_UNRELIABLE)
        if k == "sweep" and self.avg.currentData()[0] == "fifo" and self.avg.currentData()[1] < 16:
            self.avg.setCurrentIndex(self.avg.findText("FIFO 16"))
        self.engine.gen = live.Generator(k, self.fs.currentData(), self.level.value(), self.sine.value())
        self.engine.reset_averages()

    def _level_changed(self, *_):
        if self.engine.gen is not None:
            self.engine.gen.set_level(self.level.value())

    def _analysis_changed(self, *_):
        self.engine.configure(self.fs.currentData(), self.nfft.currentData(), self.avg.currentData())
        self.engine.reset_averages()
        self._sm_cache.clear()

    # ------------------------------------------------------------ run
    def keys(self):
        return (1, 2) if self.engine.in_b is not None else (1,)

    def toggle(self):
        if self.engine.running:
            self.stop()
        else:
            self.start()

    def start(self):
        fs = self.fs.currentData()
        self.engine.configure(fs, self.nfft.currentData(), self.avg.currentData())
        self._gen_changed()
        self._routing_changed()
        dev, nin, nout, _ = self._device_info()
        try:
            if self._is_sim():
                backend = live.SimBackend(fs, self.block.currentData())
            else:
                backend = live.DeviceBackend(dev, nin, nout, fs, self.block.currentData())
            self.engine.start(backend)
        except Exception as e:
            self.engine.stop()
            self.mw.error("Cannot start audio", f"{e}\n\nCheck that the interface is not used "
                          "in exclusive mode by Reaper, and check the sample rate.")
            return
        self._t_start = time.time()
        self._delay_found = False
        self.start_btn.setText("Stop")
        self.timer.start()
        self.state.setText(f"Running · {fs} Hz · block {self.block.currentData()}"
                           + (" · simulation" if self._is_sim() else ""))
        self.mw.status("Real time started: the delay is searched automatically after 2 s.")

    def stop(self):
        self.timer.stop()
        self.engine.stop()
        self.start_btn.setText("Start")
        self.state.setText("Stopped")
        for m in self.meters.values():
            m.set(-120)

    def _restart_if_running(self, *_):
        if self.engine.running:
            self.stop()
            self.start()

    # ------------------------------------------------------------ delay
    def find_delays(self):
        if not self.engine.running:
            return
        for k in self.keys():
            self.engine.find_delay(k)
        self._delay_found = True
        self._sm_cache.clear()

    def _track(self):
        e = self.engine
        for k in self.keys():
            old = e.ch[k].delay
            fs = e.fs
            end = e.ring.count
            M, maxlag = int(2 * fs), int(fs)
            x = e.ring.read(e.REF, end - maxlag, M)
            y = e.ring.read(k, end, M + maxlag)
            if x is None or y is None or not np.any(x):
                continue
            new = dsp.find_delay(y, x, fs, 20.0, min(20000.0, 0.45 * fs), max_s=1.0, max_len_s=2.0)
            if abs(new - old) > 2:
                e.set_delay(k, new / fs)
                self.mw.status(f"Delay {'AB'[k - 1]} updated: {new / fs * 1e3:.3f} ms")

    # ------------------------------------------------------------ loop
    def tick(self):
        e = self.engine
        if not e.running:
            return
        lv = e.levels
        for k, ch in (("a", e.in_a), ("b", e.in_b), ("ref", e.in_loop)):
            self.meters[k].set(lv.get(ch, (-120, 0))[0] if ch is not None else -120)
        self.meters["out"].set(e.out_level[0])
        if self._t_start and not self._delay_found and time.time() - self._t_start > 2.2:
            self.find_delays()
        if self.track.isChecked() and time.time() - self._last_track > 2.0:
            self._last_track = time.time()
            self._track()
        res = e.update(self.keys())
        if res:
            self.res = res
        if not self.hold.isChecked() and self.isVisible():
            self.draw()
        if e.xruns:
            self.state.setText(f"Running · {e.fs} Hz · {e.xruns} audio dropout(s): increase the block size")

    def _smoother(self, f):
        frac = self.mw.smooth_combo.currentData() or 48
        k = (len(f), frac, self.engine.fs)
        if k not in self._sm_cache:
            fs = self.engine.fs
            grid = dsp.log_grid(20, min(20000, 0.45 * fs), 700)
            self._sm_cache[k] = dsp.Smoother(f, grid, frac)
        return self._sm_cache[k]

    def curves(self, r):
        """Smoothed curves of a result: grid, measurement RTA, ref RTA, |H|, phase, coherence."""
        e = self.engine
        f = r["f"]
        sm = self._smoother(f)
        grid = sm.grid
        nfft = 2 * (len(f) - 1)
        win = np.hanning(nfft)
        mode, p = e.avg
        navg = len(e.ch[1].frames) if mode == "fifo" else 1
        scale = 2.0 / (e.fs * np.sum(win ** 2) * max(navg, 1))
        frac = self.mw.smooth_combo.currentData() or 48
        bw = 2 ** (0.5 / frac) - 2 ** (-0.5 / frac)
        rta = 10 * np.log10(np.maximum(sm(r["Syy"] * scale) * grid * bw, 1e-20))
        rref = 10 * np.log10(np.maximum(sm(r["Sxx"] * scale) * grid * bw, 1e-20))
        H = r["H"]
        mag = 10 * np.log10(np.maximum(sm(np.abs(H) ** 2), 1e-20))
        # phase: residual (fractional) delay removed for a stable reading
        tp, _ = e.precise_delay(r, e.fs)
        Hc = H * np.exp(2j * np.pi * f * (tp - r["delay"]))
        ph = np.degrees(np.angle(sm(Hc)))
        coh = np.clip(sm(r["coh"]), 0, 1)
        return grid, rta, rref, mag, ph, coh

    def draw(self):
        e = self.engine
        thr = self.coh_thr.value()
        txt = []
        self._last_curves = {}
        for k, key in ((1, "a"), (2, "b")):
            r = self.res.get(k)
            if r is None or (k == 2 and e.in_b is None):
                for p in ("rta_", "mag_", "ph_", "coh_", "ir_"):
                    self.c[p + key].setData([], [])
                continue
            grid, rta, rref, mag, ph, coh = self.curves(r)
            self._last_curves[key] = (grid, rta, mag, ph, coh)
            m2, p2 = mag.copy(), break_wraps(ph)
            if thr > 0:
                m2[coh < thr] = np.nan
                p2[coh < thr] = np.nan
            self.c["rta_" + key].setData(grid, rta)
            self.c["mag_" + key].setData(grid, m2)
            self.c["ph_" + key].setData(grid, p2)
            self.c["coh_" + key].setData(grid, coh)
            if k == 1:
                self.c["rta_ref"].setData(grid, rref)
            if self.show["ir"].isChecked():
                ir, t0 = e.impulse(r, e.fs)
                kp = int(np.argmax(np.abs(ir)))
                a, b = max(kp - int(0.004 * e.fs), 0), min(kp + int(0.03 * e.fs), len(ir))
                t = (t0 + np.arange(a, b) / e.fs) * 1e3
                self.c["ir_" + key].setData(t, ir[a:b] / (np.max(np.abs(ir)) or 1))
            tp, pol = e.precise_delay(r, e.fs)
            txt.append((key, tp, pol, r["n"]))
        # target on the spectrum
        i = self.target.currentIndex() - 1
        if 0 <= i < len(self.targets) and "a" in self._last_curves:
            grid, rta = self._last_curves["a"][:2]
            tgt = self.targets[i].on_grid(grid)
            sel = (grid >= 200) & (grid <= 5000)
            tgt = tgt - np.mean(tgt[sel]) + np.mean(rta[sel])
            self.c["tgt"].setData(grid, tgt)
        else:
            self.c["tgt"].setData([], [])
        # delays
        c_air = dsp.speed_of_sound(self.mw.temp_spin.value())
        if txt:
            parts = []
            for key, tp, pol, n in txt:
                col = COL_A if key == "a" else COL_B
                parts.append(f"<span style='color:{col}'>■</span> {key.upper()}: <b>{tp * 1e3:.3f} ms</b> "
                             f"· {tp * c_air * 100:.1f} cm · {'+' if pol > 0 else '−'}")
            if len(txt) == 2:
                d = txt[1][1] - txt[0][1]
                parts.append(f"B − A: <b style='color:{C['accent']}'>{d * 1e3:+.3f} ms</b> · {d * c_air * 100:+.1f} cm"
                             + (" · opposite polarities" if txt[0][2] != txt[1][2] else ""))
            ref = "loopback" if e.in_loop is not None else "generator (latency included)"
            if self.gen_kind.currentData() in live.TIMING_UNRELIABLE:
                parts.append(f"<span style='color:{C['warn']}'>{self.gen_kind.currentText()} generator: "
                             "delay not reliable</span>")
            parts.append(f"<span style='color:{C['muted']}'>ref.: {ref} · {txt[0][3]} frames</span>")
            self.delay_lab.setText("<br>".join(parts))

    # ------------------------------------------------------------ capture
    def store(self, k):
        r = self.res.get(k)
        if r is None:
            return self.mw.error("Nothing to store", "Start the analysis and wait a few seconds.")
        name = f"Live {'AB'[k - 1]} {time.strftime('%H:%M:%S')}"
        m = self.engine.to_measurement(r, name)
        m.live_snapshot = True
        self.mw.add_measurement(m)
        grid, rta, rref, mag, ph, coh = self.curves(r)
        self.stored.append((name, m.color, grid, mag))
        it = self.p_mag.plot(grid, mag, pen=pen(m.color, 1.1, Qt.DashLine))
        self.overlay_items.append((self.p_mag, it))
        self.mw.status(f"\"{name}\" added to the takes (rename it by double-clicking in the list).")

    def freeze_rta(self):
        r = self.res.get(1)
        if r is None:
            return
        grid, rta = self.curves(r)[:2]
        col = ["#e0e0e0", "#ba68c8", "#ffd54f", "#aed581", "#f06292"][len(self.frozen) % 5]
        self.frozen.append((time.strftime("%H:%M:%S"), col, grid, rta))
        it = self.p_rta.plot(grid, rta, pen=pen(col, 1.1, Qt.DotLine))
        self.overlay_items.append((self.p_rta, it))

    def clear_overlays(self):
        for p, it in self.overlay_items:
            p.removeItem(it)
        self.overlay_items, self.frozen, self.stored = [], [], []

    def rta_as_target(self):
        r = self.res.get(1)
        if r is None:
            return self.mw.error("Nothing to save", "Start the analysis.")
        grid, rta = self.curves(r)[:2]
        name, ok = QtWidgets.QInputDialog.getText(self, "New target", "Target name:",
                                                  text=f"Live spectrum {time.strftime('%H:%M')}")
        if not ok or not name.strip():
            return
        sel = np.arange(0, len(grid), 3)
        eq.save_target(eq.Target(name.strip(), list(grid[sel]), list((rta - np.mean(rta))[sel]), 1.5,
                                 note="Captured in real time"))
        self.mw.rta_tab.reload_targets()
        self.reload_targets()
        self.mw.status(f"Target \"{name.strip()}\" saved.")

    # ------------------------------------------------------------- mouse
    def _hover(self, pos):
        if not getattr(self, "_last_curves", None):
            return
        for p in (self.p_rta, self.p_mag, self.p_ph, self.p_coh):
            if p.isVisible() and p.scene() is not None and p.sceneBoundingRect().contains(pos):
                x = p.vb.mapSceneToView(pos).x()
                break
        else:
            return
        parts = [f"<b>{fmt_hz(10 ** x)}</b>"]
        for key, (grid, rta, mag, ph, coh) in self._last_curves.items():
            lg = np.log10(grid)
            col = COL_A if key == "a" else COL_B
            parts.append(f"<span style='color:{col}'>■</span> {key.upper()}: spectrum {np.interp(x, lg, rta):.1f} dB · "
                         f"transfer {np.interp(x, lg, mag):+.1f} dB · {np.interp(x, lg, ph):+.0f}° · "
                         f"coherence {np.interp(x, lg, coh):.2f}")
        self.ro.setText("&nbsp;&nbsp;&nbsp;".join(parts))
