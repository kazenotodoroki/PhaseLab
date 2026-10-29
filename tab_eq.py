"""EQ & target tab: parametric EQ with draggable nodes on top of a measured curve."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import dsp
import eq
from style import C, PALETTE
from widgets import (break_wraps, color_icon, dspin, fmt_hz, freq_plot, muted, options_row, pen,
                     plot_area, readout, tab_container)

NODE_COLORS = PALETTE


class EqNode(pg.TargetItem):
    """Band handle: drag = frequency / gain, wheel = Q."""
    sigWheel = QtCore.Signal(object, float)

    def wheelEvent(self, ev):
        d = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        self.sigWheel.emit(self, d)
        ev.accept()


class EqTab(QtWidgets.QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.bands: list[eq.Band] = []
        self.nodes = []
        self.src = None
        self._lock = False
        w, v = tab_container()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(w)

        o = options_row()
        self.source = QtWidgets.QComboBox()
        self.source.setMinimumWidth(190)
        self.source.currentIndexChanged.connect(self.refresh)
        self.target = QtWidgets.QComboBox()
        self.target.setMinimumWidth(170)
        self.target.currentIndexChanged.connect(self.refresh)
        o.addWidget(muted("Curve"))
        o.addWidget(self.source)
        o.addWidget(muted("Target"))
        o.addWidget(self.target)
        o.addSpacing(10)
        self.fmin = dspin(10, 20000, 40, 0, " Hz", 74, slot=self.refresh, tip="Band corrected by the auto-EQ")
        self.fmax = dspin(100, 24000, 12000, 0, " Hz", 80, slot=self.refresh, tip="Band corrected by the auto-EQ")
        o.addWidget(muted("Band"))
        o.addWidget(self.fmin)
        o.addWidget(muted("–"))
        o.addWidget(self.fmax)
        o.addSpacing(10)
        self.nb = QtWidgets.QSpinBox()
        self.nb.setRange(1, 16)
        self.nb.setValue(6)
        self.nb.setSuffix(" bands")
        self.nb.setFixedWidth(84)
        self.cut_only = QtWidgets.QCheckBox("Cuts only")
        self.cut_only.setToolTip("Recommended for live sound: cut the bumps rather than fill the dips")
        o.addWidget(self.nb)
        o.addWidget(self.cut_only)
        b = QtWidgets.QPushButton("Auto-EQ")
        b.setObjectName("Primary")
        b.setToolTip("Computes the bands that bring the curve to the target (average level aligned)")
        b.clicked.connect(self.auto)
        o.addWidget(b)
        o.addStretch()
        b = QtWidgets.QPushButton("Reset")
        b.clicked.connect(self.clear_bands)
        o.addWidget(b)
        exp = QtWidgets.QPushButton("Export")
        m = QtWidgets.QMenu(exp)
        m.addAction("Copy ReaEQ settings", self.copy_reaeq)
        m.addAction("ReaEQ settings (TXT)…", self.save_reaeq)
        m.addAction("EQ as WAV for ReaVerb (convolution)…", self.save_wav)
        m.addAction("Curves (CSV)…", self.save_csv)
        m.addSeparator()
        m.addAction("Save the corrected curve as a target…", self.result_as_target)
        exp.setMenu(m)
        o.addWidget(exp)
        v.addLayout(o)

        split = QtWidgets.QSplitter(Qt.Vertical)
        split.setHandleWidth(1)
        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)
        self.glw = plot_area()
        self.p = freq_plot("Level (dB)")
        self.p.setMouseEnabled(x=False, y=False)
        self.pph = freq_plot("Phase (°)")
        self.pph.setXLink(self.p)
        self.glw.addItem(self.p, 0, 0)
        self.glw.addItem(self.pph, 1, 0)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        tv.addWidget(self.glw, 1)
        self.ro = readout()
        tv.addWidget(self.ro)
        split.addWidget(top)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["", "On", "Type", "Frequency", "Gain", "Q", "Width", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        hh = self.table.horizontalHeader()
        for c in range(8):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(2, 190)
        hh.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        split.addWidget(self.table)
        split.setSizes([620, 190])
        v.addWidget(split, 1)

        # persistent plot items
        self.c_src = self.p.plot([], [], pen=pen(C["muted"], 1.4))
        self.tol_up = pg.PlotDataItem([], [], pen=pg.mkPen(None))
        self.tol_dn = pg.PlotDataItem([], [], pen=pg.mkPen(None))
        self.p.addItem(self.tol_up)
        self.p.addItem(self.tol_dn)
        self.tol_fill = pg.FillBetweenItem(self.tol_up.curve, self.tol_dn.curve, brush=pg.mkBrush(214, 218, 224, 22))
        self.p.addItem(self.tol_fill)
        self.c_tgt = self.p.plot([], [], pen=pen(C["text"], 1.4, Qt.DashLine))
        self.c_eq = self.p.plot([], [], pen=pen(C["warn"], 1.5))
        self.c_res = self.p.plot([], [], pen=pen(C["accent"], 2.2))
        self.p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C["faint"], style=Qt.DotLine)), ignoreBounds=True)
        self.ph_eq = self.pph.plot([], [], pen=pen(C["warn"], 1.4))
        self.ph_src = self.pph.plot([], [], pen=pen(C["muted"], 1.0), connect="finite")
        self.ph_res = self.pph.plot([], [], pen=pen(C["accent"], 1.6), connect="finite")
        self.pph.setYRange(-185, 185, padding=0)
        self.pph.getAxis("left").setTicks([[(x, f"{x}") for x in range(-180, 181, 90)]])
        self.p.setYRange(-18, 18, padding=0)
        self.glw.scene().sigMouseClicked.connect(self._clicked)
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self.reload_targets()
        self.rebuild_sources()

    # --------------------------------------------------------------- lists
    def rebuild_sources(self):
        cur = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem("Flat (0 dB) — draw a target", "flat")
        for m in self.mw.meas:
            self.source.addItem(color_icon(m.color), f"Measurement · {m.name}", f"meas|{m.key!r}")
        for a in self.mw.audio_files:
            self.source.addItem(color_icon(a.color), f"Audio · {a.name}", f"file|{a.path}")
        i = self.source.findData(cur) if cur else -1
        if i < 0 and self.source.count() > 1:
            i = 1
        self.source.setCurrentIndex(max(i, 0))
        self.source.blockSignals(False)

    def reload_targets(self):
        cur = self.target.currentText()
        self.targets = eq.list_targets()
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("No target")
        for t in self.targets:
            self.target.addItem(t.name)
        i = self.target.findText(cur)
        self.target.setCurrentIndex(max(i, 0))
        self.target.blockSignals(False)

    def set_source(self, data, target_name=None):
        self.rebuild_sources()
        i = self.source.findData(data)
        if i >= 0:
            self.source.setCurrentIndex(i)
        if target_name:
            self.reload_targets()
            j = self.target.findText(target_name)
            if j >= 0:
                self.target.setCurrentIndex(j)
        self.refresh()

    def current_target(self):
        i = self.target.currentIndex() - 1
        return self.targets[i] if 0 <= i < len(self.targets) else None

    # --------------------------------------------------------------- source
    def _source_curve(self):
        kind, _, key = (self.source.currentData() or "flat").partition("|")
        mw = self.mw
        phase = None
        if kind == "meas" and mw.A is not None:
            m = next((x for x in mw.meas if repr(x.key) == key), None)
            if m is not None:
                c = mw.A["curves"][id(m)]
                return mw.A["grid"], c.mag_db, c.phase_res_deg(), m.fs, m.name
        if kind == "file":
            af = next((a for a in mw.audio_files if a.path == key), None)
            if af is not None:
                grid = mw.rta_tab.grid()
                return grid, mw.rta_tab.curve(af, grid), None, af.fs, af.name
        grid = dsp.log_grid(20, 20000, 1000)
        return grid, np.zeros(len(grid)), phase, 48000, "Flat"

    def _band_mean(self, grid, db):
        sel = (grid >= self.fmin.value()) & (grid <= self.fmax.value())
        return float(np.mean(db[sel])) if sel.any() else float(np.mean(db))

    # --------------------------------------------------------------- drawing
    def refresh(self, *_):
        grid, db, ph, fs, name = self._source_curve()
        db = db - self._band_mean(grid, db)
        t = self.current_target()
        tgt = None
        if t:
            tgt = t.on_grid(grid)
            tgt = tgt - self._band_mean(grid, tgt)
        self.src = dict(grid=grid, db=db, ph=ph, fs=fs, name=name, tgt=tgt, tol=t.tol if t else 0)
        col = C["muted"]
        kind, _, key = (self.source.currentData() or "flat").partition("|")
        for m in self.mw.meas:
            if kind == "meas" and repr(m.key) == key:
                col = m.color
        for a in self.mw.audio_files:
            if kind == "file" and a.path == key:
                col = a.color
        self.c_src.setPen(pen(col, 1.4))
        self.c_src.setData(grid, db)
        if tgt is not None:
            self.c_tgt.setData(grid, tgt)
            self.tol_up.setData(grid, tgt + self.src["tol"])
            self.tol_dn.setData(grid, tgt - self.src["tol"])
        else:
            for c in (self.c_tgt, self.tol_up, self.tol_dn):
                c.setData([], [])
        if ph is not None:
            self.ph_src.setData(grid, break_wraps(dsp.wrap_deg(ph)))
        else:
            self.ph_src.setData([], [])
        lo, hi = math.log10(grid[0]), math.log10(grid[-1])
        self.p.setXRange(lo, hi, padding=0.01)
        self.update_eq()
        self.rebuild_nodes()
        self.rebuild_table()

    def update_eq(self):
        s = self.src
        if s is None:
            return
        grid, fs = s["grid"], s["fs"]
        H = eq.eq_response(self.bands, grid, fs)
        edb = 20 * np.log10(np.abs(H) + 1e-12)
        res = s["db"] + edb
        self.c_eq.setData(grid, edb)
        self.c_res.setData(grid, res)
        eph = np.degrees(np.angle(H))
        self.ph_eq.setData(grid, eph)
        if s["ph"] is not None:
            self.ph_res.setData(grid, break_wraps(dsp.wrap_deg(s["ph"] + np.degrees(np.unwrap(np.angle(H))))))
        else:
            self.ph_res.setData([], [])
        s["edb"], s["res"] = edb, res
        lo = min(np.min(s["db"]), np.min(res), -6)
        hi = max(np.max(s["db"]), np.max(res), 6)
        if s["tgt"] is not None:
            lo, hi = min(lo, np.min(s["tgt"])), max(hi, np.max(s["tgt"]))
        lo, hi = max(lo, -40), min(hi, 40)
        self.p.setYRange(math.floor(lo / 3) * 3 - 3, math.ceil(hi / 3) * 3 + 3, padding=0)
        txt = f"<span style='color:{C['muted']}'>{s['name']} · level brought to 0 dB on average · "
        if s["tgt"] is not None:
            sel = (grid >= self.fmin.value()) & (grid <= self.fmax.value())
            e0 = np.sqrt(np.mean((s["db"][sel] - s["tgt"][sel]) ** 2))
            e1 = np.sqrt(np.mean((res[sel] - s["tgt"][sel]) ** 2))
            txt += f"deviation from target {e0:.1f} → <b style='color:{C['accent']}'>{e1:.1f} dB RMS</b> · "
        txt += "double-click: add a band · drag: frequency/gain · wheel on a node: Q</span>"
        self.ro.setText(txt)

    # --------------------------------------------------------------- nodes
    def rebuild_nodes(self):
        for n in self.nodes:
            self.p.removeItem(n)
        self.nodes = []
        for i, b in enumerate(self.bands):
            col = NODE_COLORS[i % len(NODE_COLORS)]
            y = b.gain if b.type in eq.HAS_GAIN else 0.0
            n = EqNode(pos=(math.log10(b.f), y), size=13, symbol="o", movable=True,
                       pen=pg.mkPen(col, width=2), brush=pg.mkBrush(col if b.on else C["panel2"]),
                       hoverBrush=pg.mkBrush(C["text"]), label=str(i + 1),
                       labelOpts=dict(offset=(9, -9), color=col))
            n.band_index = i
            n.sigPositionChanged.connect(self._node_moved)
            n.sigPositionChangeFinished.connect(lambda *_: self.rebuild_table())
            n.sigWheel.connect(self._node_wheel)
            self.p.addItem(n)
            self.nodes.append(n)

    def _node_moved(self, n):
        if self._lock:
            return
        b = self.bands[n.band_index]
        p = n.pos()
        fs = self.src["fs"] if self.src else 48000
        b.f = float(np.clip(10 ** p.x(), 10, 0.49 * fs))
        if b.type in eq.HAS_GAIN:
            b.gain = float(np.clip(p.y(), -24, 24))
        elif abs(p.y()) > 1e-9:
            self._lock = True
            n.setPos((p.x(), 0.0))
            self._lock = False
        self.update_eq()
        self._update_row(n.band_index)
        self.mw.cursor_lab.setText(f"Band {n.band_index + 1}: {fmt_hz(b.f)}  {b.gain:+.1f} dB  Q {b.q:.2f}")

    def _node_wheel(self, n, delta):
        b = self.bands[n.band_index]
        b.q = float(np.clip(b.q * (1.1 ** (delta / 120)), 0.1, 30))
        self.update_eq()
        self._update_row(n.band_index)
        self.mw.cursor_lab.setText(f"Band {n.band_index + 1}: Q {b.q:.2f}")

    def _clicked(self, ev):
        if not ev.double() or not self.p.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = self.p.vb.mapSceneToView(ev.scenePos())
        fs = self.src["fs"] if self.src else 48000
        self.bands.append(eq.Band("peak", float(np.clip(10 ** pt.x(), 10, 0.49 * fs)),
                                  float(np.clip(pt.y(), -18, 18)), 1.4))
        self.after_bands_changed()

    def after_bands_changed(self):
        self.update_eq()
        self.rebuild_nodes()
        self.rebuild_table()

    # --------------------------------------------------------------- table
    def rebuild_table(self):
        t = self.table
        t.setRowCount(len(self.bands))
        for i, b in enumerate(self.bands):
            col = NODE_COLORS[i % len(NODE_COLORS)]
            it = QtWidgets.QTableWidgetItem(str(i + 1))
            it.setIcon(color_icon(col))
            t.setItem(i, 0, it)
            chk = QtWidgets.QCheckBox()
            chk.setChecked(b.on)
            chk.toggled.connect(lambda v, i=i: self._set(i, "on", v))
            t.setCellWidget(i, 1, chk)
            cb = QtWidgets.QComboBox()
            for k, lab in eq.TYPES.items():
                cb.addItem(lab, k)
            cb.setCurrentIndex(cb.findData(b.type))
            cb.currentIndexChanged.connect(lambda _, i=i, cb=cb: self._set(i, "type", cb.currentData()))
            t.setCellWidget(i, 2, cb)
            sf_ = dspin(10, 24000, b.f, 1, " Hz", 96, slot=lambda v, i=i: self._set(i, "f", v))
            sg = dspin(-24, 24, b.gain, 1, " dB", 76, 0.5, slot=lambda v, i=i: self._set(i, "gain", v))
            sg.setEnabled(b.type in eq.HAS_GAIN)
            sq = dspin(0.1, 30, b.q, 2, "", 64, 0.1, slot=lambda v, i=i: self._set(i, "q", v))
            t.setCellWidget(i, 3, sf_)
            t.setCellWidget(i, 4, sg)
            t.setCellWidget(i, 5, sq)
            t.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{eq.q_to_bw(b.q):.2f} oct"))
            rm = QtWidgets.QPushButton("✕")
            rm.setObjectName("Flat")
            rm.setFixedWidth(28)
            rm.clicked.connect(lambda _, i=i: self._remove(i))
            t.setCellWidget(i, 7, rm)

    def _update_row(self, i):
        b = self.bands[i]
        for c, val in ((3, b.f), (4, b.gain), (5, b.q)):
            w = self.table.cellWidget(i, c)
            if w is not None:
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)
        it = self.table.item(i, 6)
        if it:
            it.setText(f"{eq.q_to_bw(b.q):.2f} oct")

    def _set(self, i, attr, val):
        if i >= len(self.bands):
            return
        setattr(self.bands[i], attr, val)
        if attr in ("type", "on"):
            self.after_bands_changed()
        else:
            self.update_eq()
            self._lock = True
            b = self.bands[i]
            if i < len(self.nodes):
                self.nodes[i].setPos((math.log10(b.f), b.gain if b.type in eq.HAS_GAIN else 0.0))
            self._lock = False
            self._update_row(i)

    def _remove(self, i):
        if i < len(self.bands):
            del self.bands[i]
            self.after_bands_changed()

    def clear_bands(self):
        self.bands = []
        self.after_bands_changed()

    # --------------------------------------------------------------- auto
    def auto(self):
        s = self.src
        if s is None:
            return
        tgt = s["tgt"] if s["tgt"] is not None else np.zeros(len(s["grid"]))
        fmin, fmax = self.fmin.value(), self.fmax.value()
        if fmax <= fmin * 1.2:
            return self.mw.error("Invalid band", "The upper frequency must be above the lower one.")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.bands = eq.auto_eq(s["grid"], s["db"], tgt, s["fs"], self.nb.value(), fmin, fmax,
                                    max_boost=0.0 if self.cut_only.isChecked() else 6.0, max_cut=12.0)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.after_bands_changed()
        self.mw.status(f"Auto-EQ: {len(self.bands)} band(s).")

    # --------------------------------------------------------------- export
    def copy_reaeq(self):
        QtWidgets.QApplication.clipboard().setText(eq.reaeq_text(self.bands))
        self.mw.status("ReaEQ settings copied to the clipboard.")

    def save_reaeq(self):
        p = self.mw._ask_save("ReaEQ settings", "eq_reaeq.txt", "Text (*.txt)")
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(eq.reaeq_text(self.bands) + "\n")
            self.mw.status(f"Exported: {p}")

    def save_wav(self):
        if not self.bands:
            return self.mw.error("No band", "Add at least one band.")
        fs = self.src["fs"] if self.src else 48000
        p = self.mw._ask_save("EQ as impulse response", f"eq_{fs // 1000}k.wav", "WAV (*.wav)")
        if p:
            h = eq.eq_impulse(self.bands, fs, int(fs))
            sf.write(p, h.astype(np.float32), fs, subtype="FLOAT")
            self.mw.status(f"EQ exported as WAV ({fs} Hz): load it in ReaVerb, Wet 0 dB, Dry off.")

    def save_csv(self):
        s = self.src
        if not s:
            return
        p = self.mw._ask_save("EQ curves", "eq_curves.csv", "CSV (*.csv)")
        if not p:
            return
        hdr = ["Frequency (Hz)", f"{s['name']} (dB)", "EQ (dB)", "Result (dB)"]
        cols = [s["grid"], s["db"], s["edb"], s["res"]]
        if s["tgt"] is not None:
            hdr.append("Target (dB)")
            cols.append(s["tgt"])
        self.mw._write_csv(p, hdr, cols)
        self.mw.status(f"Exported: {p}")

    def result_as_target(self):
        s = self.src
        if not s:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New target", "Target name:",
                                                  text=f"{s['name']} + EQ")
        if not ok or not name.strip():
            return
        sel = np.arange(0, len(s["grid"]), 4)
        t = eq.Target(name.strip(), list(s["grid"][sel]), list((s["res"] - np.mean(s["res"]))[sel]), 1.5,
                      note="Created in the EQ tab")
        eq.save_target(t)
        self.mw.rta_tab.reload_targets()
        self.reload_targets()
        j = self.target.findText(t.name)
        if j >= 0:
            self.target.setCurrentIndex(j)
        self.mw.status(f"Target \"{t.name}\" saved.")

    # ------------------------------------------------------------- mouse
    def _hover(self, pos):
        s = self.src
        if not s or "res" not in s:
            return
        for p in (self.p, self.pph):
            if p.sceneBoundingRect().contains(pos):
                x = p.vb.mapSceneToView(pos).x()
                break
        else:
            return
        lg = np.log10(s["grid"])
        parts = [f"<b>{fmt_hz(10 ** x)}</b>", f"{s['name']} {np.interp(x, lg, s['db']):+.1f} dB",
                 f"<span style='color:{C['warn']}'>EQ {np.interp(x, lg, s['edb']):+.1f} dB</span>",
                 f"<span style='color:{C['accent']}'>result {np.interp(x, lg, s['res']):+.1f} dB</span>"]
        if s["tgt"] is not None:
            parts.append(f"target {np.interp(x, lg, s['tgt']):+.1f} dB")
        self.ro.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    # --------------------------------------------------------------- project
    def state(self):
        from dataclasses import asdict
        return dict(bands=[asdict(b) for b in self.bands], source=self.source.currentData() or "flat",
                    target=self.target.currentText(), fmin=self.fmin.value(), fmax=self.fmax.value())

    def load_state(self, st):
        self.bands = [eq.Band(**b) for b in st.get("bands", [])]
        self.fmin.setValue(st.get("fmin", 40))
        self.fmax.setValue(st.get("fmax", 12000))
        self.set_source(st.get("source") or "flat", st.get("target"))
