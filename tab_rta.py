"""Spectrum tab: analyzer (average spectrum, spectrogram), loudness, mix targets."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import dsp
import eq
import rta
from style import C
from widgets import (FreqAxis, busy, cell, color_icon, dspin, fmt_hz, freq_plot, muted, options_row,
                     pen, plot_area, section, small_table, style_plot, tab_container)

AUDIO_FILTER = "Audio (*.wav *.flac *.aif *.aiff *.ogg *.mp3)"
NORM_BAND = (200.0, 5000.0)


class RtaTab(QtWidgets.QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self._curve_cache = {}
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ---------- left panel
        left = QtWidgets.QFrame()
        left.setObjectName("Sidebar")
        left.setFixedWidth(330)
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(14, 12, 14, 12)
        lv.setSpacing(8)
        lv.addWidget(section("Audio files"))
        self.table = small_table(["", "Name", "LUFS", "dBTP", "Crest"], stretch_col=1)
        self.table.setColumnWidth(0, 34)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self.refresh)
        self.table.setToolTip("LUFS: integrated loudness (BS.1770) · dBTP: true peak · "
                              "Crest: crest factor (peak − RMS, dB)")
        lv.addWidget(self.table, 1)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Add files…")
        b.clicked.connect(self.add_dialog)
        r.addWidget(b, 1)
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self.remove_selected)
        r.addWidget(b)
        lv.addLayout(r)
        lv.addWidget(muted("Mixes, stems, references: any audio file. "
                           "Drag them here from the Finder / Explorer.", wrap=True))
        lv.addSpacing(8)
        lv.addWidget(section("Target"))
        self.target = QtWidgets.QComboBox()
        self.target.currentIndexChanged.connect(self.refresh)
        lv.addWidget(self.target)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(muted("Tolerance ±"))
        self.tol = dspin(0, 12, 1.5, 1, " dB", 70, 0.5, slot=self.refresh)
        r.addWidget(self.tol)
        r.addStretch()
        lv.addLayout(r)
        r = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Create a target")
        m = QtWidgets.QMenu(save)
        m.addAction("From the selected file…", lambda: self.save_target(False))
        m.addAction("From the average of the visible files…", lambda: self.save_target(True))
        save.setMenu(m)
        r.addWidget(save, 1)
        more = QtWidgets.QPushButton("Manage")
        m = QtWidgets.QMenu(more)
        m.addAction("Import a target (JSON / TXT / CSV)…", self.import_target)
        m.addAction("Export the target…", self.export_target)
        m.addSeparator()
        m.addAction("Delete the target", self.delete_target)
        more.setMenu(m)
        r.addWidget(more)
        lv.addLayout(r)
        self.match_btn = QtWidgets.QPushButton("Fit an EQ to the target →")
        self.match_btn.setObjectName("Primary")
        self.match_btn.setToolTip("Opens the EQ tab with the selected file as source and this target")
        self.match_btn.clicked.connect(self.send_to_eq)
        lv.addWidget(self.match_btn)
        lay.addWidget(left)

        # ---------- plots
        w, v = tab_container()
        o = options_row()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Average spectrum", "Spectrogram (selected file)"])
        self.mode.currentIndexChanged.connect(self.refresh)
        self.align = QtWidgets.QCheckBox("Align levels (200 Hz – 5 kHz)")
        self.align.setChecked(True)
        self.align.setToolTip("Compares tonal balance regardless of loudness")
        self.align.toggled.connect(self.refresh)
        o.addWidget(self.mode)
        o.addSpacing(10)
        o.addWidget(self.align)
        o.addStretch()
        self.info = muted("\"Per band\" spectrum: pink noise looks flat (like a 1/N-octave RTA).")
        o.addWidget(self.info)
        v.addLayout(o)
        self.glw = plot_area()
        self.p_spec = freq_plot("Level (dB)")
        self.p_diff = freq_plot("Deviation (dB)")
        self.p_sg = style_plot(pg.PlotItem(axisItems={"left": FreqAxis("left")}), "Frequency", "Time (s)")
        self.p_sg.setLogMode(x=False, y=True)
        self.glw.addItem(self.p_spec, 0, 0)
        self.glw.addItem(self.p_diff, 1, 0)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.p_diff.setXLink(self.p_spec)
        v.addWidget(self.glw, 1)
        self.sg_glw = plot_area()
        self.sg_glw.addItem(self.p_sg)
        self.sg_glw.setVisible(False)
        v.addWidget(self.sg_glw, 1)
        from widgets import readout
        self.ro = readout()
        v.addWidget(self.ro)
        lay.addWidget(w, 1)
        self.glw.scene().sigMouseMoved.connect(self._hover)
        self.sg_glw.scene().sigMouseMoved.connect(self._hover_sg)
        self.reload_targets()

    # ------------------------------------------------------------ files
    def add_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add audio files", "", AUDIO_FILTER)
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        mw = self.mw
        have = {a.path for a in mw.audio_files}
        errs = []
        with busy():
            for p in paths:
                if str(p) in have:
                    continue
                try:
                    af = rta.AudioFile(p)
                except Exception as e:
                    errs.append(f"{Path(p).name}: {e}")
                    continue
                af.color = mw._next_color()
                mw.audio_files.append(af)
                mw.status(f"Analyzing {af.name}…")
                QtWidgets.QApplication.processEvents()
        self.rebuild_table(select_last=True)
        mw.lists_changed()
        if errs:
            mw.error("Unreadable files", "\n".join(errs))

    def remove_selected(self):
        af = self.selected()
        if af:
            self.mw.audio_files.remove(af)
            self.rebuild_table()
            self.mw.lists_changed()

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        files = self.mw.audio_files
        if rows and rows[0].row() < len(files):
            return files[rows[0].row()]
        vis = [a for a in files if a.visible]
        return vis[0] if vis else (files[0] if files else None)

    def rebuild_table(self, select_last=False):
        files = self.mw.audio_files
        sel = self.selected()
        self.table.blockSignals(True)
        self.table.setRowCount(len(files))
        for r, af in enumerate(files):
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if af.visible else Qt.Unchecked)
            it.setIcon(color_icon(af.color))
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, cell(af.name, tip=f"{af.path}\n{af.fs} Hz · {af.data.shape[1]} channel(s) · "
                                                         f"{af.duration:.1f} s · RMS {af.rms_db:.1f} dBFS"))
            self.table.setItem(r, 2, cell(f"{af.lufs:.1f}" if math.isfinite(af.lufs) else "—", True))
            self.table.setItem(r, 3, cell(f"{af.true_peak:+.1f}", True, C["warn"] if af.true_peak > -1 else None))
            self.table.setItem(r, 4, cell(f"{af.crest_db:.1f}", True))
        self.table.blockSignals(False)
        target_row = len(files) - 1 if select_last else (files.index(sel) if sel in files else -1)
        if target_row >= 0:
            self.table.selectRow(target_row)

    def _item_changed(self, it):
        if it.column() == 0 and it.row() < len(self.mw.audio_files):
            self.mw.audio_files[it.row()].visible = it.checkState() == Qt.Checked
            self.refresh()

    # ------------------------------------------------------------ curves
    def grid(self):
        fs = max([a.fs for a in self.mw.audio_files] or [48000])
        return dsp.log_grid(20, min(20000, 0.48 * fs), 1000)

    def frac(self):
        return self.mw.smooth_combo.currentData() or 48

    def curve(self, af, grid=None):
        grid = self.grid() if grid is None else grid
        k = (af.path, self.frac(), len(grid), grid[0], grid[-1])
        if k not in self._curve_cache:
            self._curve_cache[k] = rta.rta_curve(af, grid, dsp.Smoother, self.frac())
        return self._curve_cache[k]

    @staticmethod
    def band_mean(grid, db, band=NORM_BAND):
        sel = (grid >= band[0]) & (grid <= band[1])
        return float(np.mean(db[sel])) if sel.any() else float(np.mean(db))

    # ------------------------------------------------------------ targets
    def reload_targets(self, select=None):
        cur = select or self.target.currentText()
        self.targets = eq.list_targets()
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("No target")
        for t in self.targets:
            self.target.addItem(t.name)
        i = self.target.findText(cur) if cur else -1
        self.target.setCurrentIndex(max(i, 0))
        self.target.blockSignals(False)
        if hasattr(self.mw, "eq_tab"):
            self.mw.eq_tab.reload_targets()
        if hasattr(self.mw, "live_tab"):
            self.mw.live_tab.reload_targets()

    def current_target(self):
        i = self.target.currentIndex() - 1
        return self.targets[i] if 0 <= i < len(self.targets) else None

    def save_target(self, average):
        grid = self.grid()
        if average:
            files = [a for a in self.mw.audio_files if a.visible]
            if not files:
                return self.mw.error("No file", "Tick at least one file.")
            curves = [self.curve(a, grid) for a in files]
            db = np.mean([c - self.band_mean(grid, c) for c in curves], axis=0)
            default = "Average " + ", ".join(a.name for a in files[:3])
        else:
            af = self.selected()
            if not af:
                return self.mw.error("No file", "Add an audio file first.")
            c = self.curve(af, grid)
            db = c - self.band_mean(grid, c)
            default = af.name
        name, ok = QtWidgets.QInputDialog.getText(self, "New target", "Target name:", text=default[:60])
        if not ok or not name.strip():
            return
        sel = np.arange(0, len(grid), 4)
        t = eq.Target(name.strip(), list(grid[sel]), list(db[sel]), self.tol.value(),
                      note=f"Smoothing 1/{self.frac()} oct")
        eq.save_target(t)
        self.reload_targets(select=t.name)
        self.refresh()
        self.mw.status(f"Target \"{t.name}\" saved (available in all your projects).")

    def import_target(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import a target", "",
                                                     "Target (*.json *.txt *.csv *.frd)")
        if not p:
            return
        try:
            t = eq.import_target(p)
        except Exception as e:
            return self.mw.error("Unreadable target", str(e))
        eq.save_target(t)
        self.reload_targets(select=t.name)
        self.refresh()

    def export_target(self):
        t = self.current_target()
        if not t:
            return self.mw.error("No target", "Choose a target.")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the target", f"{t.name}.json",
                                                     "PhaseLab (*.json);;REW text (*.txt)")
        if p:
            eq.export_target(t, p)

    def delete_target(self):
        t = self.current_target()
        if not t or t.builtin:
            return self.mw.error("Not possible", "Built-in targets cannot be deleted.")
        eq.delete_target(t.name)
        self.reload_targets()
        self.refresh()

    def send_to_eq(self):
        af = self.selected()
        if not af:
            return self.mw.error("No file", "Add and select an audio file.")
        t = self.current_target()
        self.mw.eq_tab.set_source(f"file|{af.path}", t.name if t else None)
        self.mw.goto_tab("eq")

    # --------------------------------------------------------------- drawing
    def refresh(self, *_):
        sg = self.mode.currentIndex() == 1
        self.glw.setVisible(not sg)
        self.sg_glw.setVisible(sg)
        for p in (self.p_spec, self.p_diff, self.p_sg):
            p.clear()
        files = [a for a in self.mw.audio_files if a.visible]
        self.data = None
        if sg:
            return self._draw_sg()
        if not files:
            self.ro.setText(f"<span style='color:{C['muted']}'>Add a mix, a track or a reference "
                            "to see its spectrum and loudness and compare it to a target.</span>")
            self.p_diff.setTitle("")
            return
        grid = self.grid()
        align = self.align.isChecked()
        sel = self.selected()
        curves = []
        for af in files:
            c = self.curve(af, grid)
            if align:
                c = c - self.band_mean(grid, c)
            curves.append((af, c))
            self.p_spec.plot(grid, c, pen=pen(af.color, 2.0 if af is sel else 1.3))
        t = self.current_target()
        ref_c = dict(curves).get(sel, curves[0][1])
        tgt = None
        if t:
            tgt = t.on_grid(grid)
            tgt = tgt - self.band_mean(grid, tgt) + (0.0 if align else self.band_mean(grid, ref_c))
            tol = self.tol.value()
            up = pg.PlotDataItem(grid, tgt + tol, pen=pg.mkPen(None))
            dn = pg.PlotDataItem(grid, tgt - tol, pen=pg.mkPen(None))
            self.p_spec.addItem(up)
            self.p_spec.addItem(dn)
            fill = pg.FillBetweenItem(up.curve, dn.curve, brush=pg.mkBrush(214, 218, 224, 22))
            self.p_spec.addItem(fill)
            self.p_spec.plot(grid, tgt, pen=pen(C["text"], 1.4, Qt.DashLine))
        base = tgt if tgt is not None else ref_c
        diffs = []
        for af, c in curves:
            if tgt is None and af is sel:
                continue
            d = c - base
            diffs.append((af, d))
            self.p_diff.plot(grid, d, pen=pen(af.color, 2.0 if af is sel else 1.3))
        if tgt is not None:
            self.p_diff.addItem(pg.LinearRegionItem([-self.tol.value(), self.tol.value()], orientation="horizontal",
                                                    movable=False, brush=pg.mkBrush(214, 218, 224, 18),
                                                    pen=pg.mkPen(None)), ignoreBounds=True)
        self.p_diff.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C["faint"], style=Qt.DotLine)))
        self.p_diff.setTitle(f"Deviation from target \"{t.name}\"" if t else
                             (f"Deviation from {sel.name}" if sel else ""), color=C["muted"], size="9pt")
        self.p_spec.setLabel("left", "Relative level (dB)" if align else "Level per band (dB)")
        self.p_spec.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self.data = (grid, curves, tgt, diffs)
        self.ro.setText(f"<span style='color:{C['muted']}'>Hover over the plot to read values · "
                        f"smoothing 1/{self.frac()} oct (top bar)</span>")

    def _draw_sg(self):
        af = self.selected()
        if not af:
            return
        with busy():
            t, fl, img = af.spectrogram()
        im = pg.ImageItem(img.T)
        top = float(np.max(img))
        im.setLevels((top - 90, top))
        for name in ("inferno", "magma", "viridis"):
            try:
                im.setLookupTable(pg.colormap.get(name).getLookupTable(nPts=256))
                break
            except Exception:
                continue
        y0, y1 = math.log10(fl[0]), math.log10(fl[-1])
        x1 = float(t[-1]) if len(t) else af.duration
        im.setRect(QtCore.QRectF(0, y0, x1 or 1, y1 - y0))
        self.p_sg.addItem(im)
        self.p_sg.setXRange(0, x1, padding=0)
        self.p_sg.setYRange(y0, y1, padding=0)
        self.p_sg.setTitle(f"{af.name} — {af.duration:.1f} s", color=C["muted"], size="9pt")
        self._sg = (t, fl, img)
        self.ro.setText(f"<span style='color:{C['muted']}'>Spectrogram of {af.name} · 90 dB range</span>")

    def _hover(self, pos):
        if not self.data:
            return
        for p in (self.p_spec, self.p_diff):
            if p.sceneBoundingRect().contains(pos):
                x = p.vb.mapSceneToView(pos).x()
                break
        else:
            return
        grid, curves, tgt, diffs = self.data
        lg = np.log10(grid)
        parts = [f"<b>{fmt_hz(10 ** x)}</b>"]
        for af, c in curves:
            parts.append(f"<span style='color:{af.color}'>■</span> {af.name} {np.interp(x, lg, c):+.1f} dB")
        if tgt is not None:
            parts.append(f"target {np.interp(x, lg, tgt):+.1f} dB")
        self.ro.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _hover_sg(self, pos):
        if not getattr(self, "_sg", None) or not self.p_sg.sceneBoundingRect().contains(pos):
            return
        pt = self.p_sg.vb.mapSceneToView(pos)
        t, fl, img = self._sg
        i = int(np.clip(np.searchsorted(t, pt.x()), 0, len(t) - 1))
        j = int(np.clip(np.searchsorted(np.log10(fl), pt.y()), 0, len(fl) - 1))
        self.mw.cursor_lab.setText(f"t = {pt.x():.2f} s   f = {fmt_hz(10 ** pt.y())}   {img[i, j]:.1f} dB")
