"""Onset tab: zoom down to the sample on the start of the waveform."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from scipy.signal import resample_poly

import dsp
from style import C
from widgets import (cell, dspin, lin_plot, muted, options_row, pen, plot_area,
                     small_table, tab_container)

SOURCES = [("Impulse response", "ir"), ("Step response", "step"),
           ("Raw take (recorded wave)", "raw"), ("Audio files (Spectrum tab)", "files")]
ALIGNS = [("Absolute time", "abs"), ("Aligned on first motion", "onset"), ("Aligned on peak", "peak")]
UP = 16            # reconstruction factor
DETAIL_MAX = 3000  # max visible samples for the reconstruction
DOTS_MAX = 600     # max visible samples to show the dots


class AttackTab(QtWidgets.QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.items = []
        self._info_cache = {}
        self._frame_key = None
        w, v = tab_container()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(w)

        o = options_row()
        self.src = QtWidgets.QComboBox()
        for lab, k in SOURCES:
            self.src.addItem(lab, k)
        self.src.currentIndexChanged.connect(self.refresh)
        self.align = QtWidgets.QComboBox()
        for lab, k in ALIGNS:
            self.align.addItem(lab, k)
        self.align.currentIndexChanged.connect(self.refresh)
        self.norm = QtWidgets.QCheckBox("Normalize")
        self.norm.setChecked(True)
        self.norm.toggled.connect(self.refresh)
        self.recon = QtWidgets.QCheckBox("Reconstructed wave")
        self.recon.setChecked(True)
        self.recon.setToolTip("Reconstruction of the real wave between samples (sinc interpolation ×16),\n"
                              "as the converter renders it. The dots are the samples.")
        self.recon.toggled.connect(self._update_detail)
        self.thr = dspin(-60, -3, -20, 0, " dB", 62, tip="Onset detection threshold, below the attack peak",
                         slot=self.refresh)
        o.addWidget(self.src)
        o.addWidget(self.align)
        o.addSpacing(8)
        o.addWidget(self.norm)
        o.addWidget(self.recon)
        o.addSpacing(8)
        o.addWidget(muted("Threshold"))
        o.addWidget(self.thr)
        o.addStretch()
        for lab, fn, tip in (("Frame the onset", self.frame_attack, "Zooms onto the start of the wave (double-click)"),
                             ("−", lambda: self.zoom(2.0), "Zoom out"), ("+", lambda: self.zoom(0.5), "Zoom in"),
                             ("All", self.show_all, "Show the whole duration")):
            b = QtWidgets.QPushButton(lab)
            b.setToolTip(tip)
            if lab in ("−", "+"):
                b.setFixedWidth(30)
            b.clicked.connect(fn)
            o.addWidget(b)
        v.addLayout(o)
        self.info = muted("Wheel: horizontal zoom · drag: pan · double-click: frame the onset. "
                          "When zooming in, samples appear and the real wave is reconstructed between them.",
                          wrap=True)
        self.info.setContentsMargins(14, 4, 14, 0)
        v.addWidget(self.info)

        split = QtWidgets.QSplitter(Qt.Vertical)
        split.setHandleWidth(1)
        self.glw = plot_area()
        self.p = lin_plot("Amplitude", "Time (ms)")
        self.p.setMouseEnabled(x=True, y=False)
        self.p.vb.setAutoVisible(y=True)
        self.glw.addItem(self.p)
        split.addWidget(self.glw)
        self.table = small_table(["Item", "First motion (ms)", "Threshold crossed (ms)", "Peak (ms)",
                                  "Rise 10–90 % (µs)", "Polarity", "Offset vs 1st row (µs)",
                                  "Offset (samples)"], stretch_col=0)
        split.addWidget(self.table)
        split.setSizes([640, 170])
        v.addWidget(split, 1)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._update_detail)
        self.p.sigXRangeChanged.connect(lambda *_: self._timer.start())
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self.glw.scene().sigMouseClicked.connect(self._clicked)

    # ---------------------------------------------------------------- data
    def _collect(self):
        mw = self.mw
        src = self.src.currentData()
        out = []
        if src == "files":
            for af in mw.audio_files:
                if af.visible:
                    out.append(dict(key=("f", af.path), name=af.name, color=af.color, x=af.mono,
                                    fs=af.fs, t0=0.0, anchor=None, info_x=af.mono))
            return out
        for m in mw.visible():
            if src == "raw":
                if m.y is None:
                    continue
                x = m.y * m.g
                out.append(dict(key=("r", m.key, m.adj_key), name=m.name, color=m.color, x=x, fs=m.fs,
                                t0=m.offset_ms * 1e-3, anchor=None, info_x=x))
            else:
                anchor = int(round((m.peak_time - m.t0) * m.fs))
                x = m.ir if src == "ir" else np.cumsum(m.ir)
                out.append(dict(key=("i", m.key, m.adj_key), name=m.name, color=m.color, x=x, fs=m.fs,
                                t0=m.t0, anchor=anchor, info_x=m.ir))
        return out

    def _info(self, it):
        k = (it["key"], self.thr.value())
        if k not in self._info_cache:
            self._info_cache[k] = dsp.onset_info(it["info_x"], it["fs"], it["t0"], self.thr.value(), it["anchor"])
        return self._info_cache[k]

    # --------------------------------------------------------------- drawing
    def refresh(self, *_):
        p = self.p
        p.clear()
        self.items = self._collect()
        self.table.setRowCount(0)
        if not self.items:
            src = self.src.currentData()
            self.info.setText("Add files in the Spectrum tab to see their onset." if src == "files"
                              else "No take to display (combined measurements have no raw take).")
            return
        mode = self.align.currentData()
        ref = None
        for it in self.items:
            it["inf"] = self._info(it)
            if ref is None and it["inf"]:
                ref = it["inf"]
        gmax = max(float(np.max(np.abs(it["x"]))) for it in self.items) or 1.0
        for it in self.items:
            inf = it["inf"]
            shift = 0.0
            if inf and ref and mode == "onset":
                shift = ref["motion"] - inf["motion"]
            elif inf and ref and mode == "peak":
                shift = ref["peak"] - inf["peak"]
            it["shift"] = shift
            if self.norm.isChecked():
                lvl = inf["level"] if inf else float(np.max(np.abs(it["x"])))
                if self.src.currentData() == "step":
                    lvl = float(np.max(np.abs(it["x"])))
                it["scale"] = 1.0 / (lvl or 1.0)
            else:
                it["scale"] = 1.0 / gmax
            fs = it["fs"]
            t = (it["t0"] + shift + np.arange(len(it["x"])) / fs) * 1e3
            main = p.plot(t, it["x"] * it["scale"], pen=pen(it["color"], 1.2))
            main.setClipToView(True)
            main.setDownsampling(auto=True, method="peak")
            it["main"] = main
            it["recon"] = p.plot([], [], pen=pen(it["color"], 1.7))
            it["dots"] = p.plot([], [], pen=None, symbol="o", symbolSize=5,
                                symbolBrush=it["color"], symbolPen=pg.mkPen(C["plot_bg"], width=1))
            if inf:
                tm = (inf["motion"] + shift) * 1e3
                ln = pg.InfiniteLine(pos=tm, angle=90, movable=False,
                                     pen=pen(it["color"], 1, Qt.DashLine))
                p.addItem(ln, ignoreBounds=True)
        p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C["faint"], style=Qt.DotLine)), ignoreBounds=True)
        self.p.setLabel("left", "Amplitude (normalized)" if self.norm.isChecked() else "Amplitude (relative)")
        self._fill_table(ref)
        fk = (self.src.currentData(), mode, tuple(it["key"] for it in self.items), self.thr.value())
        if fk != self._frame_key:
            self._frame_key = fk
            self.frame_attack()
        else:
            self._update_detail()
        src = self.src.currentData()
        txt = {"ir": "Impulse response: how the system (speaker + room + mic) starts.",
               "step": "Step response: integral of the response, direct reading of the direction of the first motion "
                       "and of the time alignment of a speaker's drivers.",
               "raw": "Wave as actually recorded.",
               "files": "Audio files from the Spectrum tab."}[src]
        self.info.setText(txt + "  Wheel: zoom · drag: pan · double-click: frame the onset. "
                                "Dashed lines: detected first motion.")

    def _fill_table(self, ref):
        rows = []
        for it in self.items:
            inf = it["inf"]
            if not inf:
                rows.append((it, [it["name"]] + ["—"] * 7))
                continue
            d = (inf["motion"] - ref["motion"]) if ref else 0.0
            rows.append((it, [it["name"], f"{inf['motion'] * 1e3:.4f}", f"{inf['onset'] * 1e3:.4f}",
                              f"{inf['peak'] * 1e3:.4f}", f"{inf['rise'] * 1e6:.0f}",
                              "+ (push)" if inf["polarity"] > 0 else "− (pull)",
                              f"{d * 1e6:+.1f}", f"{d * it['fs']:+.2f}"]))
        self.table.setRowCount(len(rows))
        for r, (it, vals) in enumerate(rows):
            for c, txt in enumerate(vals):
                col = None
                if c == 5 and txt.startswith("−"):
                    col = C["warn"]
                ic = cell(txt, c > 0, col)
                if c == 0:
                    from widgets import color_icon
                    ic.setIcon(color_icon(it["color"]))
                self.table.setItem(r, c, ic)

    # --------------------------------------------------------------- zoom
    def _bounds(self):
        ts = []
        for it in self.items:
            t0 = (it["t0"] + it["shift"]) * 1e3
            ts += [t0, t0 + len(it["x"]) / it["fs"] * 1e3]
        return (min(ts), max(ts)) if ts else (0, 1)

    def frame_attack(self):
        infos = [(it["inf"], it["shift"]) for it in self.items if it.get("inf")]
        if not infos:
            self.show_all()
            return
        a = min((inf["motion"] + s) for inf, s in infos) * 1e3
        b = max((inf["peak"] + s) for inf, s in infos) * 1e3
        span = max(b - a, 0.5)
        lo = a - max(0.25 * span, 0.2)
        hi = b + max(0.6 * span, 0.5)
        self.p.setXRange(lo, min(hi, lo + 40), padding=0)
        self._update_detail()

    def zoom(self, k):
        (x0, x1), _ = self.p.vb.viewRange()
        c, w = (x0 + x1) / 2, (x1 - x0) * k / 2
        self.p.setXRange(c - w, c + w, padding=0)

    def show_all(self):
        lo, hi = self._bounds()
        self.p.setXRange(lo, hi, padding=0.01)

    def _update_detail(self):
        if not self.items:
            return
        (x0, x1), _ = self.p.vb.viewRange()
        on = self.recon.isChecked()
        for it in self.items:
            if "main" not in it:
                continue
            fs, x = it["fs"], it["x"]
            base = it["t0"] + it["shift"]
            na = int(math.floor((x0 * 1e-3 - base) * fs)) - 1
            nb = int(math.ceil((x1 * 1e-3 - base) * fs)) + 1
            na, nb = max(na, 0), min(nb, len(x))
            cnt = nb - na
            if cnt <= 1 or cnt > DETAIL_MAX:
                it["recon"].setData([], [])
                it["dots"].setData([], [])
                it["main"].setVisible(True)
                continue
            sc = it["scale"]
            if cnt <= DOTS_MAX:
                ts = (base + np.arange(na, nb) / fs) * 1e3
                it["dots"].setData(ts, x[na:nb] * sc)
            else:
                it["dots"].setData([], [])
            if on:
                m = 24
                a, b = max(na - m, 0), min(nb + m, len(x))
                up = resample_poly(x[a:b], UP, 1)
                tu = (base + a / fs + np.arange(len(up)) / (fs * UP)) * 1e3
                it["recon"].setData(tu, up * sc)
                it["main"].setVisible(False)
            else:
                it["recon"].setData([], [])
                it["main"].setVisible(True)

    # ------------------------------------------------------------- mouse
    def _hover(self, pos):
        if not self.p.sceneBoundingRect().contains(pos) or not self.items:
            return
        pt = self.p.vb.mapSceneToView(pos)
        fs = self.items[0]["fs"]
        self.mw.cursor_lab.setText(f"t = {pt.x():.4f} ms  ({pt.x() * 1e-3 * fs:.2f} samples)   y = {pt.y():.4g}")

    def _clicked(self, ev):
        if ev.double() and self.p.sceneBoundingRect().contains(ev.scenePos()):
            self.frame_attack()
