"""Alignment tab: instant offset between two speakers (or two mics)."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import dsp
from style import C
from widgets import (break_wraps, card, color_icon, dspin, fmt_hz, freq_plot, lin_plot, muted,
                     options_row, pen, plot_area, short_hz, tab_container)

METHODS = [("Phase at crossover (recommended for two-way)", "phase"),
           ("Response peak", "peak"), ("First motion (onset)", "onset")]


def align_at_crossover(f, SA, SB, lo, hi, fc, tau0, n=1601):
    """Delay + polarity maximizing the sum A + B in the crossover band.

    Energy of the sum = |A|² + |B|² + 2·pol·Re Σ B·A*·e^(j2πfτ): we look for τ
    (around the group-delay estimate tau0, ±0.6 period) that maximizes
    |Re Σ|, and pol = sign of Re Σ. This is "phase on phase" alignment at the
    crossover, as done with a dual-channel analyzer.
    """
    band = (f >= lo) & (f <= hi)
    fb = f[band]
    C = (SB * np.conj(SA))[band] * dsp.band_weight(fb, lo * 2 ** 0.25, hi * 2 ** -0.25, 0.25)
    taus = tau0 + np.linspace(-0.6 / fc, 0.6 / fc, n)
    R = np.empty(n, complex)
    for i in range(0, n, 200):
        R[i:i + 200] = np.exp(2j * np.pi * taus[i:i + 200, None] * fb[None, :]) @ C
    k = int(np.argmax(np.abs(R.real)))
    tau = float(taus[k])
    inv = bool(R[k].real < 0)
    ang = math.degrees(math.atan2(R[k].imag, R[k].real))
    rot = float(dsp.wrap_deg(ang - 180)) if inv else float(ang)
    return tau, inv, rot


class AlignTab(QtWidgets.QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.res = None
        w, v = tab_container()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(w)

        o = options_row()
        self.a = QtWidgets.QComboBox()
        self.b = QtWidgets.QComboBox()
        for c in (self.a, self.b):
            c.setMinimumWidth(150)
            c.currentIndexChanged.connect(self.refresh)
        self.method = QtWidgets.QComboBox()
        for lab, k in METHODS:
            self.method.addItem(lab, k)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.fc = dspin(10, 20000, 100, 0, " Hz", 80, tip="Crossover frequency between the two sources",
                        slot=self.refresh)
        self.width = dspin(0.3, 4, 1.0, 1, " oct", 66, 0.1, tip="Width of the band analyzed around the crossover",
                           slot=self.refresh)
        self.multi = QtWidgets.QCheckBox("Multiple arrivals in A")
        self.multi.setToolTip("Both speakers play at the same time in a single take: detects each arrival "
                              "in A's response and measures their offset.")
        self.multi.toggled.connect(self.refresh)
        o.addWidget(muted("A"))
        o.addWidget(self.a)
        o.addWidget(muted("B"))
        o.addWidget(self.b)
        o.addSpacing(10)
        o.addWidget(self.method)
        o.addWidget(self.fc)
        o.addWidget(self.width)
        o.addSpacing(10)
        o.addWidget(self.multi)
        o.addStretch()
        v.addLayout(o)

        cd = card()
        ch = QtWidgets.QHBoxLayout(cd)
        ch.setContentsMargins(16, 12, 16, 12)
        col = QtWidgets.QVBoxLayout()
        self.big = QtWidgets.QLabel("—")
        self.big.setStyleSheet(f"font-size: 22px; font-weight: 600; color: {C['text']};")
        self.detail = muted("")
        self.detail.setTextFormat(Qt.RichText)
        self.reco = QtWidgets.QLabel("")
        self.reco.setStyleSheet(f"color: {C['accent']}; font-weight: 600;")
        col.addWidget(self.big)
        col.addWidget(self.detail)
        col.addWidget(self.reco)
        ch.addLayout(col, 1)
        bcol = QtWidgets.QVBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply correction")
        self.apply_btn.setObjectName("Primary")
        self.apply_btn.setToolTip("Sets the delay and polarity of the takes (Adjustments panel): "
                                  "every tab then shows the aligned result.")
        self.apply_btn.clicked.connect(self.apply)
        self.reset_btn = QtWidgets.QPushButton("Undo adjustments")
        self.reset_btn.clicked.connect(self.reset)
        bcol.addWidget(self.apply_btn)
        bcol.addWidget(self.reset_btn)
        ch.addLayout(bcol)
        wrap = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(wrap)
        wl.setContentsMargins(14, 8, 14, 4)
        wl.addWidget(cd)
        v.addWidget(wrap)

        self.glw = plot_area()
        self.p_ir = lin_plot("Response (norm.)", "Time (ms)")
        self.p_sum = freq_plot("Sum A + B (dB)")
        self.p_dph = freq_plot("Δ phase B − A (°)")
        self.glw.addItem(self.p_ir, 0, 0, 1, 2)
        self.glw.addItem(self.p_sum, 1, 0)
        self.glw.addItem(self.p_dph, 1, 1)
        self.p_dph.setXLink(self.p_sum)
        v.addWidget(self.glw, 1)
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self._method_changed()

    # ------------------------------------------------------------------
    def rebuild_lists(self):
        mw = self.mw
        for combo, default in ((self.a, 0), (self.b, 1)):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for m in mw.meas:
                combo.addItem(color_icon(m.color), m.name, repr(m.key))
            idx = combo.findData(cur) if cur is not None else -1
            if idx < 0:
                idx = min(default, combo.count() - 1)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _method_changed(self):
        ph = self.method.currentData() == "phase"
        self.fc.setEnabled(ph)
        self.width.setEnabled(ph)
        self.refresh()

    def _get(self, combo):
        k = combo.currentData()
        return next((m for m in self.mw.meas if repr(m.key) == k), None)

    # ------------------------------------------------------------------
    def refresh(self, *_):
        for p in (self.p_ir, self.p_sum, self.p_dph):
            p.clear()
        self.res = None
        mw = self.mw
        A = mw.A
        a = self._get(self.a)
        self.b.setEnabled(not self.multi.isChecked())
        self.apply_btn.setEnabled(False)
        if A is None or a is None:
            self.big.setText("Import at least one take")
            self.detail.setText("Measure speaker A then speaker B with the same signal (or both together, "
                                "then tick \"Multiple arrivals\").")
            self.reco.setText("")
            return
        if self.multi.isChecked():
            self._multi(a)
            return
        b = self._get(self.b)
        if b is None or b is a:
            self.big.setText("Choose two different takes")
            self.detail.setText("")
            self.reco.setText("")
            self._plot_ir([a], {})
            return
        self._pair(a, b)

    def _pair(self, a, b):
        mw, A = self.mw, self.mw.A
        fs, grid, sm = A["fs"], A["grid"], A["sm"]
        sA, sB = A["specs"][id(a)], A["specs"][id(b)]
        cA, cB = A["curves"][id(a)], A["curves"][id(b)]
        method = self.method.currentData()
        fc = self.fc.value()
        rot = None
        marks = {}
        f = sA.f
        SA = sA.H0 * np.exp(-2j * np.pi * f * sA.tp)
        SB = sB.H0 * np.exp(-2j * np.pi * f * sB.tp)
        if method == "phase":
            w = self.width.value()
            lo, hi = fc * 2 ** (-w / 2), fc * 2 ** (w / 2)
            comp = dsp.compare(sB, sA, cB, cA, sm, fs, lo, hi, with_sum=False,
                               search_ms=max(20.0, abs(b.peak_time - a.peak_time) * 1e3 + 10))
            tau, inv, rot = align_at_crossover(f, SA, SB, lo, hi, fc, float(comp.tau))
            marks = {id(a): a.peak_time, id(b): a.peak_time + tau}
        elif method == "peak":
            tau = b.peak_time - a.peak_time
            inv = bool(a.polarity != b.polarity)
            marks = {id(a): a.peak_time, id(b): b.peak_time}
        else:
            ia = dsp.onset_info(a.ir, fs, a.t0, -20, int(round((a.peak_time - a.t0) * fs)))
            ib = dsp.onset_info(b.ir, fs, b.t0, -20, int(round((b.peak_time - b.t0) * fs)))
            tau = ib["motion"] - ia["motion"]
            inv = bool(ia["polarity"] != ib["polarity"])
            marks = {id(a): ia["motion"], id(b): ib["motion"]}

        # summation before / after
        pol = -1.0 if inv else 1.0
        SBc = pol * SB * np.exp(2j * np.pi * f * tau)
        pa, pb = sm(np.abs(SA) ** 2), sm(np.abs(SB) ** 2)
        before = 10 * np.log10(np.maximum(sm(np.abs(SA + SB) ** 2), 1e-30))
        after = 10 * np.log10(np.maximum(sm(np.abs(SA + SBc) ** 2), 1e-30))
        ideal = 20 * np.log10(np.sqrt(np.maximum(pa, 1e-30)) + np.sqrt(np.maximum(pb, 1e-30)))
        dph_b = np.degrees(np.angle(sm(SB * np.conj(SA))))
        dph_a = np.degrees(np.angle(sm(SBc * np.conj(SA))))
        self.p_sum.plot(grid, ideal, pen=pen(C["faint"], 1.2, Qt.DashLine))
        self.p_sum.plot(grid, before, pen=pen(b.color, 1.3))
        self.p_sum.plot(grid, after, pen=pen(C["accent"], 2.0))
        self.p_dph.plot(grid, break_wraps(dph_b), pen=pen(b.color, 1.3), connect="finite")
        self.p_dph.plot(grid, break_wraps(dph_a), pen=pen(C["accent"], 2.0), connect="finite")
        self.p_dph.setYRange(-185, 185, padding=0)
        self.p_dph.getAxis("left").setTicks([[(v_, f"{v_}") for v_ in range(-180, 181, 90)]])
        for p in (self.p_dph,):
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C["faint"], style=Qt.DotLine)))
        if method == "phase":
            for p in (self.p_sum, self.p_dph):
                p.addItem(pg.InfiniteLine(pos=math.log10(fc), angle=90,
                                          pen=pg.mkPen(C["warn"], width=1, style=Qt.DashLine)), ignoreBounds=True)
        self.p_sum.setTitle(f"color: raw · <span style='color:{C['accent']}'>green: corrected</span> · "
                            "dashed: ideal summation", color=C["muted"], size="9pt")
        self.p_dph.setTitle("0° = sources in phase", color=C["muted"], size="9pt")
        self.p_sum.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self._plot_ir([a, b], marks, b_corr=(b, tau, pol))

        c_air = dsp.speed_of_sound(mw.temp_spin.value())
        lg = np.log10(grid)
        f_read = fc if method == "phase" else math.sqrt(A["band"][0] * A["band"][1])
        gb = float(np.interp(math.log10(f_read), lg, before))
        ga = float(np.interp(math.log10(f_read), lg, after))
        word = "after" if tau >= 0 else "before"
        self.big.setText(f"B arrives {abs(tau) * 1e3:.3f} ms {word} A")
        det = (f"{abs(tau) * c_air * 100:.1f} cm · {abs(tau) * fs:.1f} samples · polarity "
               f"<b>{'inverted' if inv else 'same'}</b>")
        if rot is not None:
            det += f" · residual rotation {rot:+.0f}° around {short_hz(round(fc))} Hz"
        det += f" · sum at {fmt_hz(f_read)}: {gb:+.1f} dB → <span style='color:{C['accent']}'>{ga:+.1f} dB</span>"
        self.detail.setText(det)
        early, late = (a, b) if tau >= 0 else (b, a)
        r = f"→ Delay {early.name} by {abs(tau) * 1e3:.3f} ms"
        if inv:
            r += f" and invert the polarity of {b.name}"
        if abs(tau) < 1e-6 and not inv:
            r = "→ Already aligned"
        self.reco.setText(r)
        self.res = dict(a=a, b=b, tau=tau, inv=inv)
        self.apply_btn.setEnabled(bool(abs(tau) >= 1e-6 or inv))
        self._data = (grid, before, after, dph_b, dph_a)

    def _multi(self, a):
        arr = dsp.find_arrivals(a)
        fs = a.fs
        c_air = dsp.speed_of_sound(self.mw.temp_spin.value())
        self._plot_ir([a], {}, arrivals=arr)
        if len(arr) < 2:
            self.big.setText("Only one arrival detected")
            self.detail.setText("No 2nd arrival within 24 dB of the peak in the first 60 ms.")
            self.reco.setText("")
            return
        t1 = arr[0]["t"]
        self.big.setText(f"Offset between arrivals: {(arr[1]['t'] - t1) * 1e3:.3f} ms")
        lines = []
        for i, x in enumerate(arr):
            d = x["t"] - t1
            lines.append(f"#{i + 1}: {x['t'] * 1e3:.3f} ms ({x['db']:+.1f} dB, "
                         f"{'+' if x['polarity'] > 0 else '−'}) · Δ {d * 1e3:+.3f} ms = {d * c_air * 100:+.1f} cm "
                         f"= {d * fs:+.1f} samples")
        self.detail.setText("<br>".join(lines) + "<br><span style='color:#7c8590'>Strong reflections "
                            "also show up as arrivals: mute a speaker to identify them.</span>")
        self.reco.setText(f"→ Delay the closest source by {(arr[1]['t'] - t1) * 1e3:.3f} ms "
                          "(identify it by muting one speaker)")

    def _plot_ir(self, ms, marks, b_corr=None, arrivals=None):
        p = self.p_ir
        if not ms:
            return
        t_lo = min(m.peak_time for m in ms) - 0.003
        t_hi = max(m.peak_time for m in ms) + 0.008
        if arrivals:
            t_hi = max(t_hi, arrivals[-1]["t"] + 0.004)
        for m in ms:
            gmax = float(np.max(np.abs(m.ir))) or 1.0
            fs = m.fs
            i0 = max(int((t_lo - m.t0) * fs), 0)
            i1 = min(int((t_hi - m.t0) * fs) + 1, len(m.ir))
            t = (m.t0 + np.arange(i0, i1) / fs) * 1e3
            p.plot(t, m.ir[i0:i1] / gmax, pen=pen(m.color, 1.3))
            if id(m) in marks:
                p.addItem(pg.InfiniteLine(pos=marks[id(m)] * 1e3, angle=90,
                                          pen=pen(m.color, 1, Qt.DashLine)), ignoreBounds=True)
        if b_corr is not None:
            b, tau, pol = b_corr
            fs = b.fs
            i0 = max(int((t_lo + tau - b.t0) * fs), 0)
            i1 = min(int((t_hi + tau - b.t0) * fs) + 1, len(b.ir))
            t = (b.t0 - tau + np.arange(i0, i1) / fs) * 1e3
            p.plot(t, pol * b.ir[i0:i1] / (float(np.max(np.abs(b.ir))) or 1.0),
                   pen=pen(C["accent"], 1.2, Qt.DashLine))
            p.setTitle(f"green dashed: {b.name} after correction", color=C["muted"], size="9pt")
        else:
            p.setTitle("")
        if arrivals:
            for i, x in enumerate(arrivals):
                p.addItem(pg.InfiniteLine(pos=x["t"] * 1e3, angle=90, pen=pen(C["warn"], 1, Qt.DashLine),
                                          label=f"#{i + 1}", labelOpts=dict(position=0.92, color=C["warn"])),
                          ignoreBounds=True)
        p.setXRange(t_lo * 1e3, t_hi * 1e3, padding=0)
        p.enableAutoRange(axis="y")

    # ------------------------------------------------------------------
    def apply(self):
        r = self.res
        if not r:
            return
        a, b, tau, inv = r["a"], r["b"], r["tau"], r["inv"]
        early = a if tau >= 0 else b
        early.offset_ms += abs(tau) * 1e3
        if inv:
            b.invert = not b.invert
        self.mw.adjustments_changed()
        self.mw.status(f"{early.name} delayed by {abs(tau) * 1e3:.3f} ms"
                       + (f", polarity of {b.name} inverted" if inv else "") + ".")

    def reset(self):
        for m in (self._get(self.a), self._get(self.b)):
            if m is not None:
                m.reset_adjust()
        self.mw.adjustments_changed()

    def _hover(self, pos):
        if self.p_ir.sceneBoundingRect().contains(pos):
            pt = self.p_ir.vb.mapSceneToView(pos)
            self.mw.cursor_lab.setText(f"t = {pt.x():.4f} ms")
            return
        for p in (self.p_sum, self.p_dph):
            if p.sceneBoundingRect().contains(pos) and getattr(self, "_data", None):
                pt = p.vb.mapSceneToView(pos)
                grid, before, after, dpb, dpa = self._data
                x = pt.x()
                lg = np.log10(grid)
                self.mw.cursor_lab.setText(
                    f"f = {fmt_hz(10 ** x)}   sum {np.interp(x, lg, before):+.1f} → {np.interp(x, lg, after):+.1f} dB"
                    f"   Δφ {np.interp(x, lg, dpb):+.0f}° → {np.interp(x, lg, dpa):+.0f}°")
                return
