#!/usr/bin/env python3
"""PhaseLab - phase, delay and response analysis for mic takes and speakers.

Run:  python phaselab.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

import dsp
from style import C, PALETTE, QSS
from widgets import (break_wraps, busy, card, color_icon, dspin, fmt_hz, freq_plot, lin_plot, muted, nz,
                     plot_area, readout, safe_name, section, short_hz)
from tab_attack import AttackTab
from tab_align import AlignTab
from tab_rta import RtaTab
from tab_eq import EqTab
from tab_live import LiveTab

APP = "PhaseLab"
VERSION = "3.0"
pg.setConfigOptions(antialias=True, background=C["plot_bg"], foreground=C["muted"],
                    imageAxisOrder="row-major")

SMOOTHING = [("None", 0), ("1/48 oct", 48), ("1/24 oct", 24), ("1/12 oct", 12),
             ("1/6 oct", 6), ("1/3 oct", 3)]
DELAY_MODES = [("Delay included", "none"), ("Reference delay removed", "common"),
               ("Own delay removed", "indiv")]
TIME_MODES = [("Impulse response", "ir"), ("ETC envelope (dB)", "etc"),
              ("Take vs aligned original / loopback", "aligned"),
              ("Take vs original × response (residual = noise + distortion)", "model")]
FREQ_COLORS = ["#4fc3f7", "#aed581", "#ffd54f", "#ff8a65", "#f06292", "#ba68c8", "#4db6ac", "#e0e0e0"]
TABS = [("live", "Real time"), ("resp", "Response"), ("time", "Time"), ("attack", "Onset"), ("diff", "Differences"),
        ("align", "Alignment"), ("rot", "Rotation"), ("rta", "Spectrum"), ("eq", "EQ && target")]
MEASURE_TABS = {"resp", "time", "diff", "rot"}
NFFTS = [8192, 16384, 32768, 65536, 131072]


# ---------------------------------------------------------------- dialogues
class GenerateDialog(QtWidgets.QDialog):
    HINTS = {
        "sweep": "Reference method (Farina / Novak). Rejects harmonic distortion, best signal-to-noise "
                 "ratio. Longer = more accurate.",
        "sweep_rep": "Back-to-back sweeps: the 1st period is ignored, the following ones are averaged "
                     "(3 dB more signal-to-noise each time the count doubles) and coherence is computed.",
        "pink": "Pink noise (−3 dB/oct), equal energy per octave. Analysed as dual channel, as in "
                "Open Sound Meter: transfer function + coherence.",
        "white": "White noise, equal energy per Hz (lots of treble: careful with tweeters).",
        "brown": "Brown noise (−6 dB/oct), mostly low end: useful for subs.",
        "pink_per": "Periodic pink noise: exact spectrum, no random fluctuation. Synchronous averaging: "
                    "the method of pro measurement systems for a stable reading in little time.",
        "white_per": "Periodic white noise (exactly flat spectrum), synchronous averaging.",
        "mls": "Maximum length sequence (MLS): the historical method (MLSSA). Sensitive to "
               "distortion; prefer the sweep or periodic noise.",
        "sine": "Pure sine: level calibration, polarity check, distortion (Spectrum tab).",
        "impulse": "Single impulse (Dirac). Little energy: only for digital chain tests.",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate a test signal")
        self.setMinimumWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        self.form = form
        self.kind = QtWidgets.QComboBox()
        for k, v in dsp.KIND_LABELS.items():
            self.kind.addItem(v, k)
        self.fs = QtWidgets.QComboBox()
        for f in (44100, 48000, 88200, 96000, 192000):
            self.fs.addItem(f"{f} Hz", f)
        self.fs.setCurrentIndex(1)
        self.subtype = QtWidgets.QComboBox()
        for k, v in dsp.SUBTYPES.items():
            self.subtype.addItem(v, k)
        self.duration = dspin(0.5, 600, 5, 1, " s", step=0.5)
        self.f1 = dspin(1, 2000, 20, 1, " Hz")
        self.f2 = dspin(100, 96000, 20000, 0, " Hz", step=100)
        self.sine = dspin(1, 40000, 1000, 1, " Hz")
        self.period = QtWidgets.QComboBox()
        for n in range(14, 20):
            self.period.addItem(f"2^{n} samples", n)
        self.period.setCurrentIndex(3)
        self.repeats = QtWidgets.QSpinBox()
        self.repeats.setRange(2, 256)
        self.repeats.setValue(8)
        self.level = dspin(-60, 0, -6, 1, " dBFS", step=0.5)
        self.pre = dspin(0, 10, 0.5, 2, " s", step=0.1)
        self.post = dspin(0, 20, 2, 2, " s", step=0.5)
        self.rows = {}
        for lab, w in (("Type", self.kind), ("Sample rate", self.fs), ("Format", self.subtype),
                       ("Duration", self.duration), ("Frequency", self.sine), ("Start frequency", self.f1),
                       ("End frequency", self.f2), ("Period", self.period), ("Repeats", self.repeats),
                       ("Peak level", self.level), ("Silence before", self.pre), ("Silence after", self.post)):
            form.addRow(lab, w)
            self.rows[lab] = w
        lay.addLayout(form)
        self.hint = muted("", wrap=True)
        lay.addWidget(self.hint)
        self.calc = muted("", wrap=True)
        lay.addWidget(self.calc)
        lay.addWidget(muted("Use the same sample rate as your Reaper project. The silence after "
                            "must cover the room's decay.", wrap=True))
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        cancel = QtWidgets.QPushButton("Cancel")
        ok = QtWidgets.QPushButton("Generate and save…")
        ok.setObjectName("Primary")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)
        for w in (self.kind, self.fs, self.period):
            w.currentIndexChanged.connect(self._kind_changed)
        self.repeats.valueChanged.connect(self._kind_changed)
        self._kind_changed()

    def _show(self, lab, on):
        w = self.rows[lab]
        w.setVisible(on)
        lb = self.form.labelForField(w)
        if lb:
            lb.setVisible(on)

    def _kind_changed(self, *_):
        k = self.kind.currentData()
        per = k in dsp.PERIODIC_KINDS
        self.hint.setText(self.HINTS[k])
        self._show("Duration", not per and k != "impulse")
        self._show("Frequency", k == "sine")
        self._show("Start frequency", k not in ("impulse", "sine", "mls"))
        self._show("End frequency", k not in ("impulse", "sine", "mls"))
        self._show("Period", per)
        self._show("Repeats", per)
        if per:
            n = self.period.currentData()
            P = (2 ** n - 1) if k == "mls" else 2 ** n
            fs = self.fs.currentData()
            self.calc.setText(f"Period {P / fs:.2f} s (response up to {P / fs * 1000:.0f} ms, resolution "
                              f"{fs / P:.2f} Hz) · total duration {P * self.repeats.value() / fs:.1f} s · "
                              f"{self.repeats.value() - 1} periods averaged")
        else:
            self.calc.setText("")
        self.adjustSize()

    def spec(self):
        k = self.kind.currentData()
        fs = self.fs.currentData()
        return dsp.SignalSpec(
            kind=k, fs=fs, duration=self.duration.value() if k != "impulse" else 0.01,
            f1=self.f1.value() if k != "impulse" else 10.0,
            f2=self.f2.value() if k != "impulse" else 0.45 * fs,
            level_db=self.level.value(), pre_silence=self.pre.value(),
            post_silence=self.post.value(), subtype=self.subtype.currentData(),
            period_pow=self.period.currentData(), repeats=self.repeats.value(), sine_freq=self.sine.value())


class CombineDialog(QtWidgets.QDialog):
    def __init__(self, meas, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine measurements")
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)
        self.op = QtWidgets.QComboBox()
        for k, v in dsp.COMBINE_OPS.items():
            self.op.addItem(v, k)
        lay.addWidget(muted("Operation"))
        lay.addWidget(self.op)
        lay.addWidget(muted("Measurements (in order: A, B, …)"))
        self.list = QtWidgets.QListWidget()
        for m in meas:
            it = QtWidgets.QListWidgetItem(color_icon(m.color), m.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if m.visible else Qt.Unchecked)
            self.list.addItem(it)
        lay.addWidget(self.list)
        self.name = QtWidgets.QLineEdit("Average")
        lay.addWidget(muted("Name"))
        lay.addWidget(self.name)
        lay.addWidget(muted("Power average: average level of several mic positions (the classic basis "
                            "for a room EQ). Sum: what two speakers or two mics give together, "
                            "delays and polarities included.", wrap=True))
        self.op.currentIndexChanged.connect(
            lambda: self.name.setText({"vector": "Average", "power": "Power average", "sum": "Sum",
                                       "diff": "Difference"}[self.op.currentData()]))
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        c = QtWidgets.QPushButton("Cancel")
        c.clicked.connect(self.reject)
        ok = QtWidgets.QPushButton("Create")
        ok.setObjectName("Primary")
        ok.clicked.connect(self.accept)
        btns.addWidget(c)
        btns.addWidget(ok)
        lay.addLayout(btns)

    def chosen(self):
        return [i for i in range(self.list.count()) if self.list.item(i).checkState() == Qt.Checked]


GUIDE_HTML = f"""
<style>
h3 {{ color: {C['text']}; font-weight: 600; margin: 14px 0 6px 0; }}
p, li {{ color: {C['text']}; line-height: 145%; }}
.m {{ color: {C['muted']}; }}
b {{ color: {C['accent']}; font-weight: 600; }}
</style>
<h3>Real time</h3>
<p><b>Real time</b> tab: choose your audio interface, the mic input (A, and B for a 2nd mic),
the reference (<b>loopback</b> recommended, otherwise the internal generator) and the generator output
(pink noise recommended, <b>start at a low level</b>). <b>Start</b>: spectrum, transfer function,
phase, coherence and impulse response are shown live; the delay is found automatically
after 2 s (<b>Find delay</b> to run it again, <b>Auto tracking</b> if you move the mic).
<b>Store A / B</b> adds the measurement to the takes list: Alignment, EQ, Differences… work on it.
Close Reaper (or free the interface) if it uses it in exclusive mode. <b>Simulation</b> mode lets you
try everything without hardware.</p>
<p class="m">Continuous sweeps are available in real time: the frequency response builds up with long
averaging, but they are <b>not effective for time-domain measurement</b> (delay, impulse response,
phase). Use pink noise for those.</p>

<h3>1 · Generate</h3>
<p><b>Generate…</b> button: sweep (most accurate), repeated sweeps, white / pink / brown noise,
periodic noises, MLS, sine. Choose your Reaper project's sample rate.
A small <span class="m">.phaselab.json</span> file is created next to the WAV: keep them together.</p>

<h3>2 · Record in Reaper</h3>
<ol>
<li>Project at the <b>same sample rate</b> as the signal.</li>
<li>Drag the signal WAV onto a track, <b>at the very start of the project (0:00)</b>, routed to the speaker.</li>
<li>One track per mic, armed, <b>no input FX</b>. Record from 0:00 to the end of the file.</li>
<li>Name the takes with the angle (<span class="m">snare_45deg</span>…): it is read automatically.</li>
<li>Export: <span class="m">File › Render</span>, <b>Stems (selected tracks)</b>, bounds <b>Entire project</b>,
no normalization or dither.</li>
</ol>

<h3>Loopback (dual channel, like Open Sound Meter)</h3>
<p>Connect an interface output to an input and record it together with the mics
(<span class="m">Render › Multichannel</span>). Enter its number in <b>Loopback</b>.
Each take is then measured against what actually left the interface: interface latency and coloration
disappear. With a loopback, <b>any signal works</b>, even music, and the test signal becomes optional.
Coherence (Response tab) shows how reliable each frequency is.</p>

<h3>3 · Read the results</h3>
<p><b>Response</b>: level, phase, group delay, coherence.
<b>Time</b>: impulse response, envelope, comparison with the original or the loopback.
<b>Onset</b>: zoom down to the sample, wave reconstructed between samples, first motion
and polarity of each take.
<b>Differences</b>: level and phase deviations, summation, settings table.
<b>Alignment</b>: instant offset between two speakers (peak, onset or phase at crossover) and a button
to apply it. Two speakers playing together: tick "Multiple arrivals".
<b>Rotation</b>: phase, level and time versus mic angle.</p>

<h3>Mixes, targets and EQ</h3>
<p><b>Spectrum</b>: drag in your mixes and references (any audio file): average spectrum,
spectrogram, LUFS loudness, true peak. <b>Create a target</b> from a reference or from the average
of several: it is kept for all your projects.
<b>EQ & target</b>: put a parametric EQ on any curve (measurement or mix). Double-click to
add a band, drag for frequency and gain, mouse wheel for Q, or <b>Auto-EQ</b> to the target.
Export to ReaEQ (settings) or ReaVerb (WAV of the exact EQ).</p>

<h3>Adjustments</h3>
<p>Below the takes list: delay, gain and polarity of the selected take. Every tab takes them into
account, so you can simulate an alignment before doing it for real.</p>
<p class="m">Tip: hover over a plot to read exact values.</p>
"""


class GuideDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide")
        self.resize(660, 720)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 18)
        tb = QtWidgets.QTextBrowser()
        tb.setHtml(GUIDE_HTML)
        lay.addWidget(tb)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        lay.addLayout(row)


# ------------------------------------------------------------ main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1480, 900)
        self.setAcceptDrops(True)

        self.signal: dsp.TestSignal | None = None
        self.meas: list[dsp.Measurement] = []
        self.audio_files = []
        self.ref: dsp.Measurement | None = None     # None = original signal
        self.sel: dsp.Measurement | None = None
        self.project_path = None
        self._spec_cache = {}
        self._orig_cache = {}
        self._updating = False
        self._force_tabs = False
        self.A = None   # analysis results
        self._color_i = 0
        self._dirty = {k for k, _ in TABS}

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh_all)

        self._build_ui()
        self._build_menu()
        self._update_signal_card()
        self._update_adjust_card()
        self._rebuild_ref_combo()
        self._update_welcome()

    # ------------------------------------------------------------- interface
    def _build_ui(self):
        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        right = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(right)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_toolbar())
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_welcome())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.attack_tab = AttackTab(self)
        self.align_tab = AlignTab(self)
        self.rta_tab = RtaTab(self)
        self.eq_tab = EqTab(self)
        self.live_tab = LiveTab(self)
        widgets = dict(live=self.live_tab, resp=self._build_resp_tab(), time=self._build_time_tab(), attack=self.attack_tab,
                       diff=self._build_diff_tab(), align=self.align_tab, rot=self._build_rot_tab(),
                       rta=self.rta_tab, eq=self.eq_tab)
        for k, lab in TABS:
            self.tabs.addTab(widgets[k], lab)
        self.tabs.currentChanged.connect(self._draw_current)
        self.stack.addWidget(self.tabs)
        v.addWidget(self.stack, 1)
        h.addWidget(right, 1)
        self.setCentralWidget(central)

        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.status_msg = QtWidgets.QLabel("Ready")
        self.cursor_lab = QtWidgets.QLabel("")
        self.cursor_lab.setObjectName("Value")
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.cursor_lab)

    def _build_sidebar(self):
        side = QtWidgets.QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(310)
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(APP)
        t.setObjectName("Title")
        top.addWidget(t)
        top.addStretch()
        help_btn = QtWidgets.QPushButton("Guide")
        help_btn.setObjectName("Flat")
        help_btn.clicked.connect(self.show_guide)
        top.addWidget(help_btn)
        v.addLayout(top)

        v.addWidget(section("Test signal"))
        cd = card()
        cv = QtWidgets.QVBoxLayout(cd)
        cv.setContentsMargins(10, 8, 10, 8)
        cv.setSpacing(2)
        self.sig_name = QtWidgets.QLabel("No signal")
        self.sig_info = muted("", wrap=True)
        cv.addWidget(self.sig_name)
        cv.addWidget(self.sig_info)
        v.addWidget(cd)
        row = QtWidgets.QHBoxLayout()
        self.btn_gen = QtWidgets.QPushButton("Generate…")
        self.btn_gen.setToolTip("Sweep, noises, periodic signals, MLS, sine to play in Reaper")
        self.btn_gen.clicked.connect(self.generate_signal)
        self.btn_load_sig = QtWidgets.QPushButton("Load…")
        self.btn_load_sig.setToolTip("Loads a previously generated signal (WAV)")
        self.btn_load_sig.clicked.connect(self.load_signal_dialog)
        row.addWidget(self.btn_gen)
        row.addWidget(self.btn_load_sig)
        v.addLayout(row)

        v.addSpacing(4)
        v.addWidget(section("Takes"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Name", "Angle", "Peak ms", "SNR"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in (2, 3, 4):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 34)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                   | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table.itemChanged.connect(self._table_item_changed)
        self.table.itemSelectionChanged.connect(self._table_selection)
        self.table.setToolTip("Tick to show · double-click to rename or enter the angle")
        v.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Import…")
        self.btn_import.clicked.connect(self.import_dialog)
        b = QtWidgets.QPushButton("Combine…")
        b.setToolTip("Average (vector or power), sum or difference of measurements")
        b.clicked.connect(self.combine_dialog)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected)
        row.addWidget(self.btn_import, 1)
        row.addWidget(b)
        row.addWidget(self.btn_remove)
        v.addLayout(row)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.loop_combo = QtWidgets.QComboBox()
        self.loop_combo.addItem("None", 0)
        for i in range(1, 17):
            self.loop_combo.addItem(f"Canal {i}", i)
        self.loop_combo.setToolTip(
            "Channel carrying the direct return of the interface output (dual-channel reference). "
            "With a loopback, any signal (noise, music) can be measured.")
        self.loop_combo.currentIndexChanged.connect(self._loop_changed)
        self.method_combo = QtWidgets.QComboBox()
        for k in ("auto", "sweep", "periodic", "dual"):
            self.method_combo.addItem(dsp.METHOD_LABELS[k], k)
        self.method_combo.setToolTip(
            "Automatic: sweep → deconvolution, periodic signal → synchronous averaging, "
            "noise or music → dual channel.")
        self.method_combo.currentIndexChanged.connect(self._loop_changed)
        self.nfft_combo = QtWidgets.QComboBox()
        for n in NFFTS:
            self.nfft_combo.addItem(f"{n // 1024}k", n)
        self.nfft_combo.setCurrentIndex(2)
        self.nfft_combo.setToolTip("Dual-channel FFT size: larger = better low-end resolution, "
                                   "fewer averages")
        self.nfft_combo.currentIndexChanged.connect(self._loop_changed)
        grid.addWidget(muted("Loopback"), 0, 0)
        grid.addWidget(self.loop_combo, 0, 1, 1, 2)
        grid.addWidget(muted("Method"), 1, 0)
        grid.addWidget(self.method_combo, 1, 1)
        grid.addWidget(self.nfft_combo, 1, 2)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)

        v.addSpacing(4)
        v.addWidget(section("Take adjustments"))
        cd = card()
        g = QtWidgets.QGridLayout(cd)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        self.adj_delay = dspin(-1000, 1000, 0, 3, " ms", step=0.01, slot=self._adjust_edited,
                               tip="Delay added to the take (positive = later)")
        self.adj_gain = dspin(-40, 40, 0, 2, " dB", step=0.5, slot=self._adjust_edited)
        self.adj_inv = QtWidgets.QCheckBox("Polarity inverted")
        self.adj_inv.toggled.connect(self._adjust_edited)
        self.adj_reset = QtWidgets.QPushButton("Reset")
        self.adj_reset.setObjectName("Flat")
        self.adj_reset.clicked.connect(self._adjust_reset)
        g.addWidget(muted("Delay"), 0, 0)
        g.addWidget(self.adj_delay, 0, 1)
        g.addWidget(muted("Gain"), 1, 0)
        g.addWidget(self.adj_gain, 1, 1)
        g.addWidget(self.adj_inv, 2, 0, 1, 2)
        g.addWidget(self.adj_reset, 2, 1, Qt.AlignRight)
        g.setColumnStretch(1, 1)
        v.addWidget(cd)
        self.adj_card = cd
        return side

    def _labeled(self, lay, text, w):
        lay.addWidget(muted(text))
        lay.addWidget(w)
        lay.addSpacing(10)

    def _build_toolbar(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName("Toolbar")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(6)

        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.setMinimumWidth(170)
        self.ref_combo.setToolTip("All differences are computed against this reference")
        self.ref_combo.currentIndexChanged.connect(self._ref_changed)
        self._labeled(h, "Reference", self.ref_combo)

        self.smooth_combo = QtWidgets.QComboBox()
        for lab, v in SMOOTHING:
            self.smooth_combo.addItem(lab, v)
        self.smooth_combo.setCurrentIndex(3)
        self.smooth_combo.setToolTip("Fractional-octave smoothing (complex for measurements: preserves phase)")
        self.smooth_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Smoothing", self.smooth_combo)

        self.wrap_combo = QtWidgets.QComboBox()
        self.wrap_combo.addItems(["Wrapped ±180°", "Unwrapped"])
        self.wrap_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Phase", self.wrap_combo)

        self.delay_combo = QtWidgets.QComboBox()
        for lab, v in DELAY_MODES:
            self.delay_combo.addItem(lab, v)
        self.delay_combo.setCurrentIndex(1)
        self.delay_combo.setToolTip(
            "Included: raw phase (the delay makes the phase spin very fast).\n"
            "Reference removed: time offsets between takes are kept.\n"
            "Own removed: each take is re-centered, only the phase shape is shown.")
        self.delay_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Display", self.delay_combo)

        self.pre_spin = dspin(0.2, 200, 5, 1, " ms", 84, slot=self.schedule,
                              tip="Duration kept before the impulse response peak")
        self.post_spin = dspin(1, 3000, 100, 1, " ms", 84, slot=self.schedule,
                               tip="Duration kept after the peak. Shorter = ignores reflections, "
                                   "but low-frequency resolution ≈ 1000 / duration Hz")
        h.addWidget(muted("Window"))
        h.addWidget(self.pre_spin)
        h.addWidget(muted("/"))
        h.addWidget(self.post_spin)
        h.addStretch()

        exp = QtWidgets.QPushButton("Export")
        menu = QtWidgets.QMenu(exp)
        menu.addAction("Response curves (CSV)…", self.export_response_csv)
        menu.addAction("Difference curves (CSV)…", self.export_diff_csv)
        menu.addAction("Settings table (CSV)…", self.export_settings_csv)
        menu.addAction("EQ correction curves (TXT)…", self.export_eq)
        menu.addAction("Rotation data (CSV)…", self.export_rotation_csv)
        menu.addSeparator()
        menu.addAction("Impulse responses (WAV)…", self.export_irs)
        menu.addAction("Tab image (PNG)…", self.export_png)
        exp.setMenu(menu)
        h.addWidget(exp)
        return bar

    def _build_welcome(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.addStretch(2)
        col = QtWidgets.QWidget()
        col.setMaximumWidth(580)
        v = QtWidgets.QVBoxLayout(col)
        v.setSpacing(12)
        t = QtWidgets.QLabel("Measure, compare, align")
        t.setObjectName("Title")
        v.addWidget(t)
        v.addWidget(muted("The signal is created here, played and recorded in Reaper, then analyzed here.", wrap=True))
        v.addSpacing(6)

        def step(num, title, desc, buttons):
            fr = card()
            hl = QtWidgets.QHBoxLayout(fr)
            hl.setContentsMargins(14, 12, 14, 12)
            n = QtWidgets.QLabel(num)
            n.setFixedWidth(26)
            n.setStyleSheet(f"color:{C['accent']}; font-size:16px; font-weight:600;")
            hl.addWidget(n, 0, Qt.AlignTop)
            tv = QtWidgets.QVBoxLayout()
            tl = QtWidgets.QLabel(title)
            tl.setStyleSheet("font-weight:600;")
            tv.addWidget(tl)
            tv.addWidget(muted(desc, wrap=True))
            hl.addLayout(tv, 1)
            for b in buttons:
                hl.addWidget(b, 0, Qt.AlignVCenter)
            v.addWidget(fr)
            return tl

        b1 = QtWidgets.QPushButton("Generate")
        b1.setObjectName("Primary")
        b1.clicked.connect(self.generate_signal)
        b1b = QtWidgets.QPushButton("Load")
        b1b.clicked.connect(self.load_signal_dialog)
        self.w_step1 = step("1", "Create the test signal",
                            "Sweep, pink noise, periodic noise, MLS, sine…", [b1, b1b])
        b2 = QtWidgets.QPushButton("Reaper guide")
        b2.clicked.connect(self.show_guide)
        step("2", "Play and record it in Reaper",
             "Signal placed at 0:00. With a loopback, any signal (even music) works.", [b2])
        self.w_b3 = QtWidgets.QPushButton("Import")
        self.w_b3.setObjectName("Primary")
        self.w_b3.clicked.connect(self.import_dialog)
        step("3", "Import the takes", "Or simply drag the WAV files into the window.", [self.w_b3])
        b5 = QtWidgets.QPushButton("Real time")
        b5.setObjectName("Primary")
        b5.clicked.connect(lambda: self.goto_tab("live"))
        step("◆", "Measure in real time",
             "Like Open Sound Meter: generator, spectrum, transfer function, coherence and delay "
             "live. Simulation mode to try it without an interface.", [b5])
        b4 = QtWidgets.QPushButton("Open the analyzer")
        b4.clicked.connect(lambda: self.goto_tab("rta"))
        step("◆", "Analyze a mix, create a target, apply an EQ",
             "Spectrum, spectrogram, LUFS, mix targets and parametric EQ, no test signal needed.", [b4])
        h = QtWidgets.QHBoxLayout()
        h.addStretch()
        h.addWidget(col)
        h.addStretch()
        outer.addLayout(h)
        outer.addStretch(3)
        return w

    def _plot_area(self):
        return plot_area()

    def _readout(self):
        return readout()

    def _build_resp_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        opts = QtWidgets.QHBoxLayout()
        opts.setContentsMargins(14, 6, 14, 0)
        self.norm_check = QtWidgets.QCheckBox("Level normalized at 1 kHz")
        self.norm_check.toggled.connect(self.schedule)
        self.gd_check = QtWidgets.QCheckBox("Group delay")
        self.gd_check.setChecked(True)
        self.gd_check.toggled.connect(self.schedule)
        self.coh_check = QtWidgets.QCheckBox("Coherence")
        self.coh_check.setChecked(True)
        self.coh_check.setToolTip("Reliability of the measurement at each frequency (dual-channel and periodic measurements)")
        self.coh_check.toggled.connect(self.schedule)
        self.coh_thr = dspin(0, 0.99, 0, 2, "", 56, 0.05, slot=self.schedule,
                             tip="Hides the areas where coherence is below this threshold (0 = never)")
        for wdg in (self.norm_check, self.gd_check, self.coh_check):
            opts.addWidget(wdg)
            opts.addSpacing(14)
        opts.addWidget(muted("Hide below"))
        opts.addWidget(self.coh_thr)
        opts.addStretch()
        v.addLayout(opts)
        self.resp_glw = self._plot_area()
        self.p_mag = freq_plot("Level (dB)")
        self.p_ph = freq_plot("Phase (°)")
        self.p_gd = freq_plot("Group delay (ms)")
        self.p_coh = freq_plot("Coherence")
        self.resp_glw.addItem(self.p_mag, 0, 0)
        self.resp_glw.addItem(self.p_ph, 1, 0)
        self.resp_glw.addItem(self.p_gd, 2, 0)
        self.resp_glw.addItem(self.p_coh, 3, 0)
        self.resp_glw.ci.layout.setRowStretchFactor(0, 3)
        self.resp_glw.ci.layout.setRowStretchFactor(1, 3)
        self.resp_glw.ci.layout.setRowStretchFactor(2, 2)
        self.resp_glw.ci.layout.setRowStretchFactor(3, 1)
        for p in (self.p_ph, self.p_gd, self.p_coh):
            p.setXLink(self.p_mag)
        v.addWidget(self.resp_glw, 1)
        self.resp_readout = self._readout()
        v.addWidget(self.resp_readout)
        self.resp_lines = []
        self.resp_data = []
        self.resp_glw.scene().sigMouseMoved.connect(
            lambda pos: self._freq_hover(pos, [self.p_mag, self.p_ph, self.p_gd, self.p_coh], "resp"))
        return w

    def _build_time_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        opts = QtWidgets.QHBoxLayout()
        opts.setContentsMargins(14, 6, 14, 0)
        self.time_combo = QtWidgets.QComboBox()
        for lab, k in TIME_MODES:
            self.time_combo.addItem(lab, k)
        self.time_combo.currentIndexChanged.connect(self.schedule)
        opts.addWidget(self.time_combo)
        opts.addSpacing(12)
        self.time_info = muted("")
        opts.addWidget(self.time_info, 1)
        v.addLayout(opts)
        self.time_glw = self._plot_area()
        self.p_time = lin_plot("Amplitude", "Time (ms)")
        self.time_glw.addItem(self.p_time)
        v.addWidget(self.time_glw, 1)
        self.time_glw.scene().sigMouseMoved.connect(self._time_hover)
        return w

    def _build_diff_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        opts = QtWidgets.QHBoxLayout()
        opts.setContentsMargins(14, 6, 14, 0)

        def fspin(val, tip):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(1, 96000); s.setDecimals(0); s.setValue(val); s.setSuffix(" Hz")
            s.setFixedWidth(84); s.setKeyboardTracking(False)
            s.setToolTip(tip)
            s.valueChanged.connect(self.schedule)
            return s

        self.fit_lo = fspin(100, "Band used to compute delay, rotation and gain")
        self.fit_hi = fspin(10000, "Band used to compute delay, rotation and gain")
        opts.setSpacing(6)
        opts.addWidget(muted("Fit band"))
        opts.addWidget(self.fit_lo)
        opts.addWidget(muted("–"))
        opts.addWidget(self.fit_hi)
        opts.addSpacing(18)
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(-20, 50); self.temp_spin.setValue(20); self.temp_spin.setSuffix(" °C")
        self.temp_spin.setFixedWidth(70)
        self.temp_spin.setToolTip("Air temperature, to convert delays into distance")
        self.temp_spin.valueChanged.connect(self.schedule)
        opts.addWidget(muted("Air"))
        opts.addWidget(self.temp_spin)
        opts.addStretch()
        copy = QtWidgets.QPushButton("Copy settings")
        copy.clicked.connect(self.copy_settings)
        opts.addWidget(copy)
        v.addLayout(opts)

        split = QtWidgets.QSplitter(Qt.Vertical)
        split.setHandleWidth(1)
        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)
        self.diff_glw = self._plot_area()
        self.p_dmag = freq_plot("Δ level (dB)")
        self.p_dph = freq_plot("Δ phase (°)")
        self.p_sum = freq_plot("Sum (dB)")
        self.diff_glw.addItem(self.p_dmag, 0, 0)
        self.diff_glw.addItem(self.p_dph, 1, 0)
        self.diff_glw.addItem(self.p_sum, 2, 0)
        for p in (self.p_dph, self.p_sum):
            p.setXLink(self.p_dmag)
        tv.addWidget(self.diff_glw, 1)
        self.diff_readout = self._readout()
        tv.addWidget(self.diff_readout)
        split.addWidget(top)

        self.set_table = QtWidgets.QTableWidget(0, 10)
        self.set_table.setHorizontalHeaderLabels(
            ["Take", "Delay (ms)", "Samples", "Distance (cm)", "Polarity",
             "Rotation (°)", "Gain (dB)", "Residual deviation", "1st notch (sum)", "Correction to apply"])
        self.set_table.verticalHeader().setVisible(False)
        self.set_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.set_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.set_table.setShowGrid(False)
        self.set_table.verticalHeader().setDefaultSectionSize(24)
        hh = self.set_table.horizontalHeader()
        for c in range(9):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(9, QtWidgets.QHeaderView.Stretch)
        self.set_table.setToolTip(
            "Delay > 0: the take arrives after the reference.\n"
            "Rotation: constant phase shift remaining once delay and polarity are corrected.\n"
            "Residual deviation: what delay + rotation do not explain (° RMS in the band).")
        split.addWidget(self.set_table)
        split.setSizes([620, 200])
        v.addWidget(split, 1)
        self.diff_lines = []
        self.diff_glw.scene().sigMouseMoved.connect(
            lambda pos: self._freq_hover(pos, [self.p_dmag, self.p_dph, self.p_sum], "diff"))
        return w

    def _build_rot_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        opts = QtWidgets.QHBoxLayout()
        opts.setContentsMargins(14, 6, 14, 0)
        opts.addWidget(muted("Frequencies"))
        self.rot_freqs = QtWidgets.QLineEdit("125, 500, 1000, 4000, 10000")
        self.rot_freqs.setFixedWidth(260)
        self.rot_freqs.setToolTip("Frequencies tracked versus angle, comma-separated")
        self.rot_freqs.editingFinished.connect(self.schedule)
        opts.addWidget(self.rot_freqs)
        opts.addSpacing(14)
        self.rot_unwrap = QtWidgets.QCheckBox("Unwrapped phase")
        self.rot_unwrap.setChecked(True)
        self.rot_unwrap.setToolTip("Unwrapped: the phase accumulates turns from the low end (no ambiguity).\n"
                                   "Untick to show it between −180° and +180°.")
        self.rot_unwrap.toggled.connect(self.schedule)
        opts.addWidget(self.rot_unwrap)
        opts.addSpacing(14)
        self.rot_info = muted("")
        opts.addWidget(self.rot_info, 1)
        v.addLayout(opts)
        self.rot_glw = self._plot_area()
        self.p_rph = lin_plot("Δ phase (°)", "Angle (°)")
        self.p_rmag = lin_plot("Δ level (dB)", "Angle (°)")
        self.p_rt = lin_plot("Δ time (µs)", "Angle (°)")
        self.p_rmap = freq_plot("Angle (°)")
        self.p_rmap.setTitle("Δ phase versus frequency and angle", color=C["muted"], size="9pt")
        self.rot_glw.addItem(self.p_rph, 0, 0)
        self.rot_glw.addItem(self.p_rmag, 0, 1)
        self.rot_glw.addItem(self.p_rt, 1, 0)
        self.rot_glw.addItem(self.p_rmap, 1, 1)
        self.p_rmag.setXLink(self.p_rph)
        self.p_rt.setXLink(self.p_rph)
        self.rot_img = pg.ImageItem()
        self.rot_cbar = None
        v.addWidget(self.rot_glw, 1)
        self.rot_glw.scene().sigMouseMoved.connect(self._rot_hover)
        return w

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("File")
        m.addAction("New project", self.new_project, QtGui.QKeySequence.New)
        m.addAction("Open project…", self.open_project, QtGui.QKeySequence.Open)
        m.addAction("Save project", self.save_project, QtGui.QKeySequence.Save)
        m.addAction("Save as…", lambda: self.save_project(True), QtGui.QKeySequence.SaveAs)
        m.addSeparator()
        m.addAction("Quit", self.close, QtGui.QKeySequence.Quit)
        m = mb.addMenu("Signal")
        m.addAction("Generate a signal…", self.generate_signal, "Ctrl+G")
        m.addAction("Load a signal…", self.load_signal_dialog)
        m = mb.addMenu("Takes")
        m.addAction("Import takes…", self.import_dialog, "Ctrl+I")
        m.addAction("Combine measurements…", self.combine_dialog, "Ctrl+M")
        m.addAction("Remove the selected take", self.remove_selected, QtGui.QKeySequence.Delete)
        m.addAction("Set as reference", self.set_selected_as_ref, "Ctrl+R")
        m.addAction("Reset all adjustments", self._adjust_reset_all)
        m.addSeparator()
        m.addAction("Show all", lambda: self._set_all_visible(True))
        m.addAction("Hide all", lambda: self._set_all_visible(False))
        m = mb.addMenu("Audio")
        m.addAction("Add audio files (mixes, references)…", self.rta_tab.add_dialog, "Ctrl+Shift+A")
        m.addAction("Open the analyzer", lambda: self.goto_tab("rta"))
        m.addAction("Real time: start / stop", self._toggle_live, "Ctrl+L")
        m.addAction("Open the EQ", lambda: self.goto_tab("eq"))
        m = mb.addMenu("Help")
        m.addAction("Guide", self.show_guide, QtGui.QKeySequence.HelpContents)

    # ------------------------------------------------------------- state / UI
    def status(self, text):
        self.status_msg.setText(text)

    def schedule(self, *_):
        self._timer.start()

    def goto_tab(self, key):
        self._force_tabs = True
        self._update_welcome()
        idx = [k for k, _ in TABS].index(key)
        self.tabs.setCurrentIndex(idx)
        self._draw_current()

    def _update_signal_card(self):
        s = self.signal
        if not s:
            self.sig_name.setText("No signal")
            self.sig_info.setText("Generate or load the signal played in Reaper "
                                  "(optional with a loopback).")
            self.btn_gen.setObjectName("Primary")
        else:
            self.sig_name.setText(s.name)
            dur = len(s.data) / s.fs
            extra = ""
            if s.periodic:
                extra = f" · {s.meta['repeats']} × {s.meta['period'] / s.fs:.2f} s"
            band = f"Band {short_hz(round(s.f1, 1))} – {short_hz(round(s.f2))} Hz" if s.kind != "sine" \
                else f"{s.f1:g} Hz"
            self.sig_info.setText(f"{s.label} · {s.fs} Hz · {dur:.2f} s{extra}\n{band}")
            self.btn_gen.setObjectName("")
        self.btn_gen.style().unpolish(self.btn_gen)
        self.btn_gen.style().polish(self.btn_gen)

    def _update_welcome(self):
        ok = self.signal is not None
        self.w_step1.setText("Create the test signal" + ("   ✓ " + self.signal.name if ok else ""))
        show = bool(self.meas or self.audio_files or self._force_tabs)
        self.stack.setCurrentIndex(1 if show else 0)

    def _next_color(self):
        c = PALETTE[self._color_i % len(PALETTE)]
        self._color_i += 1
        return c

    def _rebuild_table(self):
        self._updating = True
        self.table.setRowCount(len(self.meas))
        for r, m in enumerate(self.meas):
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if m.visible else Qt.Unchecked)
            it.setIcon(color_icon(m.color))
            self.table.setItem(r, 0, it)
            name = m.name + ("  ★" if m is self.ref else "") + ("  ±" if m.adjusted else "")
            it = QtWidgets.QTableWidgetItem(name)
            tip = f"{m.path or 'combined measurement'}"
            if not m.virtual:
                tip += f"\ncanal {m.channel + 1}" + (" · loopback" if m.loopback else "")
            tip += f"\n{dsp.METHOD_LABELS.get(m.method, m.method)}"
            if m.averages > 1:
                tip += f" · {m.averages} averages"
            if m.adjusted:
                tip += f"\nadjusted: {m.offset_ms:+.3f} ms, {m.gain_db:+.1f} dB" + (", polarity inverted" if m.invert else "")
            it.setToolTip(tip)
            self.table.setItem(r, 1, it)
            it = QtWidgets.QTableWidgetItem("" if m.angle is None else f"{m.angle:g}")
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 2, it)
            for c, txt in ((3, f"{m.peak_time * 1e3:.3f}"), (4, f"{m.snr_db:.0f}")):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 4 and m.snr_db < 40:
                    it.setForeground(QtGui.QColor(C["warn"]))
                    it.setToolTip("Low signal-to-noise ratio: raise the level or lengthen the signal")
                self.table.setItem(r, c, it)
            if m is self.sel:
                self.table.selectRow(r)
        self._updating = False

    def _rebuild_ref_combo(self):
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItem("Original signal / loopback")
        for m in self.meas:
            self.ref_combo.addItem(color_icon(m.color), m.name)
        idx = self.meas.index(self.ref) + 1 if self.ref in self.meas else 0
        self.ref_combo.setCurrentIndex(idx)
        self.ref_combo.blockSignals(False)

    def _refresh_lists(self):
        if self.ref is not None and self.ref not in self.meas:
            self.ref = None
        if self.sel is not None and self.sel not in self.meas:
            self.sel = None
        self._rebuild_table()
        self._rebuild_ref_combo()
        self.lists_changed()

    def lists_changed(self):
        """The takes / files lists changed: updates the tabs that depend on them."""
        self.align_tab.rebuild_lists()
        self.eq_tab.rebuild_sources()
        self._update_welcome()
        self._update_adjust_card()
        self._dirty = {k for k, _ in TABS}
        self.schedule()

    def _table_item_changed(self, it):
        if self._updating:
            return
        r, c = it.row(), it.column()
        if r >= len(self.meas):
            return
        m = self.meas[r]
        if c == 0:
            m.visible = it.checkState() == Qt.Checked
            self.schedule()
        elif c == 1:
            m.name = it.text().replace("★", "").strip() or m.name
            self._refresh_lists()
        elif c == 2:
            txt = it.text().strip().replace(",", ".").replace("°", "")
            try:
                m.angle = float(txt) if txt else None
            except ValueError:
                self.status("Invalid angle: enter a number in degrees.")
            self._refresh_lists()

    def _table_selection(self):
        if self._updating:
            return
        rows = self.table.selectionModel().selectedRows()
        self.sel = self.meas[rows[0].row()] if rows else None
        self._update_adjust_card()
        self._dirty = {k for k, _ in TABS}
        self.schedule()

    def _set_all_visible(self, v):
        for m in self.meas:
            m.visible = v
        self._refresh_lists()

    def _ref_changed(self, idx):
        self.ref = None if idx <= 0 else self.meas[idx - 1]
        self._rebuild_table()
        self.schedule()

    def set_selected_as_ref(self):
        if self.sel:
            self.ref = self.sel
            self._refresh_lists()

    # ------------------------------------------------------------ adjustments
    def _update_adjust_card(self):
        m = self.sel
        self.adj_card.setEnabled(m is not None)
        for w, val in ((self.adj_delay, m.offset_ms if m else 0.0), (self.adj_gain, m.gain_db if m else 0.0)):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.adj_inv.blockSignals(True)
        self.adj_inv.setChecked(bool(m and m.invert))
        self.adj_inv.blockSignals(False)

    def _adjust_edited(self, *_):
        m = self.sel
        if m is None:
            return
        m.offset_ms = self.adj_delay.value()
        m.gain_db = self.adj_gain.value()
        m.invert = self.adj_inv.isChecked()
        self.adjustments_changed(update_card=False)

    def _adjust_reset(self):
        if self.sel:
            self.sel.reset_adjust()
            self.adjustments_changed()

    def _adjust_reset_all(self):
        for m in self.meas:
            m.reset_adjust()
        self.adjustments_changed()

    def adjustments_changed(self, update_card=True):
        self._orig_cache.clear()
        if update_card:
            self._update_adjust_card()
        self._rebuild_table()
        self._dirty = {k for k, _ in TABS}
        self.schedule()

    # --------------------------------------------------------------- signal
    def generate_signal(self):
        dlg = GenerateDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        spec = dlg.spec()
        if spec.kind in dsp.PERIODIC_KINDS:
            tag = f"{spec.repeats}x2e{spec.period_pow}"
        elif spec.kind == "sine":
            tag = f"{spec.sine_freq:g}Hz_{spec.duration:g}s"
        else:
            tag = f"{spec.duration:g}s"
        default = f"phaselab_{spec.kind}_{spec.fs // 1000}k_{tag}.wav"
        start = str(Path(self.signal.path).parent / default) if self.signal and self.signal.path else default
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the signal", start, "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        with busy():
            x, meta = dsp.generate_signal(spec)
            sig = dsp.TestSignal(x, spec.fs, meta)
            sig.save(path)
        self.set_signal(sig)
        self.status(f"Signal saved: {Path(path).name}. Drag it to 0:00 in Reaper.")

    def load_signal_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load the test signal", "", "WAV (*.wav)")
        if path:
            self.load_signal(path)

    def load_signal(self, path):
        try:
            sig = dsp.TestSignal.load(path)
        except Exception as e:
            self.error("Cannot read the signal", str(e))
            return
        if sig.kind == "inconnu":
            self.status("Signal without a .phaselab.json file: default analysis band 20 Hz – 20 kHz.")
        self.set_signal(sig)

    def set_signal(self, sig):
        self.signal = sig
        self._update_signal_card()
        if self.meas:
            self.reload_measurements()
        self._update_welcome()
        lo, hi = max(sig.f1, 20), min(sig.f2, 0.45 * sig.fs)
        if sig.kind != "sine":
            self.fit_lo.setValue(max(100, lo))
            self.fit_hi.setValue(min(10000, hi))
        self.schedule()

    # --------------------------------------------------------------- takes
    def _can_measure(self):
        return self.signal is not None or self.loop_combo.currentData() > 0

    def import_dialog(self):
        if not self._can_measure():
            self.error("Missing reference",
                       "Generate or load the test signal first, or set the loopback channel "
                       "(dual-channel measurement with any signal).")
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Import takes", "",
                                                          "Audio (*.wav *.flac *.aif *.aiff)")
        if paths:
            self.import_takes(paths)

    def _load(self, path):
        return dsp.load_takes(path, self.signal, self.loop_combo.currentData(),
                              self.method_combo.currentData(), self.nfft_combo.currentData())

    def import_takes(self, paths):
        if not self._can_measure():
            self.error("Missing reference", "Load the test signal or set the loopback channel.")
            return
        existing = {m.key for m in self.meas}
        fs0 = self.meas[0].fs if self.meas else None
        was_empty = not self.meas
        errors, added = [], 0
        with busy():
            for p in paths:
                if self.signal and self.signal.path and Path(p).resolve() == Path(self.signal.path).resolve():
                    continue
                try:
                    new = self._load(p)
                except Exception as e:
                    errors.append(f"{Path(p).name}: {e}")
                    continue
                for m in new:
                    if m.key in existing:
                        continue
                    if fs0 and m.fs != fs0:
                        errors.append(f"{m.name}: {m.fs} Hz instead of {fs0} Hz")
                        continue
                    fs0 = m.fs
                    m.color = self._next_color()
                    self.meas.append(m)
                    added += 1
                self.status(f"Analyzing… {added} take(s)")
                QtWidgets.QApplication.processEvents()
        if was_empty and self.meas:
            self.ref = self.meas[0]
            self.sel = self.meas[1] if len(self.meas) > 1 else self.meas[0]
        self._spec_cache.clear()
        self._refresh_lists()
        meth = {dsp.METHOD_LABELS.get(m.method) for m in self.meas[-added:]} if added else set()
        self.status(f"{added} take(s) imported" + (f" · {', '.join(sorted(meth))}" if meth else "") + ".")
        if errors:
            self.error("Some takes could not be read", "\n".join(errors))

    def reload_measurements(self):
        """Recomputes all takes (after a change of signal, loopback or method)."""
        old = {m.key: m for m in self.meas}
        ref_key = self.ref.key if self.ref else None
        sel_key = self.sel.key if self.sel else None
        paths = list(dict.fromkeys(m.path for m in self.meas if not m.virtual))
        virtuals = [m for m in self.meas if m.virtual]
        new, errors = [], []
        with busy():
            for p in paths:
                try:
                    for m in self._load(p):
                        o = old.get(m.key)
                        if o:
                            m.name, m.angle, m.visible, m.color = o.name, o.angle, o.visible, o.color
                            m.offset_ms, m.gain_db, m.invert = o.offset_ms, o.gain_db, o.invert
                        else:
                            m.color = self._next_color()
                        new.append(m)
                except Exception as e:
                    errors.append(f"{Path(p).name}: {e}")
        for vm in virtuals:
            nm = self._rebuild_virtual(vm, new)
            if nm is not None:
                new.append(nm)
        self.meas = new
        self._spec_cache.clear()
        self._orig_cache.clear()
        self.ref = next((m for m in new if m.key == ref_key), None)
        self.sel = next((m for m in new if m.key == sel_key), None)
        self._refresh_lists()
        if errors:
            self.error("Some takes could not be recomputed", "\n".join(errors))

    def _rebuild_virtual(self, vm, pool):
        if vm.recipe is None:      # trace stored in real time: nothing to recompute
            return vm
        try:
            srcs = [next(m for m in pool if list(m.key) == list(k)) for k in vm.recipe["sources"]]
            nm = dsp.combine(srcs, vm.recipe["op"], vm.name)
        except Exception:
            return None
        nm.color, nm.visible, nm.angle = vm.color, vm.visible, vm.angle
        nm.offset_ms, nm.gain_db, nm.invert = vm.offset_ms, vm.gain_db, vm.invert
        return nm

    def _loop_changed(self):
        if self.meas and self._can_measure():
            self.reload_measurements()

    def remove_selected(self):
        if self.sel in self.meas:
            self.meas.remove(self.sel)
            self._spec_cache.pop(id(self.sel), None)
            self.sel = None
            self._refresh_lists()

    def add_measurement(self, m):
        """Adds a measurement (e.g. stored in real time) to the takes list."""
        if self.meas and m.fs != self.meas[0].fs:
            return self.error("Different sample rate", f"{m.fs} Hz instead of {self.meas[0].fs} Hz.")
        base, k = m.name, 2
        while any(x.name == m.name for x in self.meas):
            m.name = f"{base} {k}"
            k += 1
        m.color = self._next_color()
        self.meas.append(m)
        if self.ref is None and len(self.meas) == 1:
            self.ref = m
        self.sel = m
        self._refresh_lists()

    def _toggle_live(self):
        self.goto_tab("live")
        self.live_tab.toggle()

    def closeEvent(self, ev):
        self.live_tab.stop()
        super().closeEvent(ev)

    def combine_dialog(self):
        if len(self.meas) < 2:
            return self.error("Not enough measurements", "At least two measurements are needed.")
        dlg = CombineDialog(self.meas, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        ms = [self.meas[i] for i in dlg.chosen()]
        try:
            with busy():
                nm = dsp.combine(ms, dlg.op.currentData(), dlg.name.text().strip() or "Combination")
        except Exception as e:
            return self.error("Combination not possible", str(e))
        base, k = nm.name, 2
        while any(m.name == nm.name for m in self.meas):
            nm.name = f"{base} {k}"
            k += 1
        nm.color = self._next_color()
        self.meas.append(nm)
        self.sel = nm
        self._refresh_lists()
        self.status(f"Measurement \"{nm.name}\" created from {len(ms)} measurements.")

    # ------------------------------------------------------------- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        exts = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")
        projects = [p for p in paths if p.endswith(".phaselab")]
        if projects:
            return self.open_project(projects[0])
        paths = [p for p in paths if Path(p).suffix.lower() in exts]
        if not paths:
            return
        cur = TABS[self.tabs.currentIndex()][0] if self.stack.currentIndex() == 1 else None
        if cur in ("rta", "eq"):
            return self.rta_tab.add_files(paths)
        sigs = [p for p in paths if dsp.sidecar_path(p).exists()]
        takes = [p for p in paths if p not in sigs]
        if sigs:
            self.load_signal(sigs[0])
        if self.signal is not None and self.loop_combo.currentData() == 0:
            # a file much longer than the test signal is a mix, not a take
            limit = len(self.signal.data) / self.signal.fs * 1.5 + 2
            longs = []
            for p in takes:
                try:
                    if sf.info(p).duration > limit:
                        longs.append(p)
                except Exception:
                    pass
            if longs:
                takes = [p for p in takes if p not in longs]
                self.rta_tab.add_files(longs)
                self.status(f"{len(longs)} long file(s) added to the spectrum analyzer.")
        if takes:
            if self._can_measure():
                self.import_takes(takes)
            else:
                self.rta_tab.add_files(takes)
                self.goto_tab("rta")
                self.status("No test signal and no loopback: files added to the spectrum analyzer.")

    # ------------------------------------------------------------- analysis
    def analyze(self):
        if not self.meas:
            self.A = None
            return
        fs = self.meas[0].fs
        fmin = max(min(m.f1 for m in self.meas), 10.0)
        fmax = min(max(m.f2 for m in self.meas), 0.48 * fs)
        if fmax <= fmin * 2:
            fmin, fmax = 20.0, min(20000.0, 0.48 * fs)
        grid = dsp.log_grid(fmin, fmax, 1500)
        pre, post = self.pre_spin.value(), self.post_spin.value()
        N = dsp.fft_size(fs, pre, post)
        specs = {}
        for m in self.meas:
            key = (pre, post, N, m.adj_key)
            cached = self._spec_cache.get(id(m))
            if not cached or cached[0] != key or cached[2] is not m:
                self._spec_cache[id(m)] = (key, dsp.windowed_spectrum(m, pre, post, N), m)
            specs[id(m)] = self._spec_cache[id(m)][1]
        f = next(iter(specs.values())).f
        frac = self.smooth_combo.currentData()
        sm = dsp.Smoother(f, grid, frac)
        curves = {k: dsp.make_curve(sp, sm) for k, sp in specs.items()}
        coh = {}
        for m in self.meas:
            if m.coh is not None:
                coh[id(m)] = np.clip(dsp.Smoother(m.coh_f, grid, frac or 48)(m.coh), 0, 1)
        ref = self.ref
        if ref is not None:
            sref, cref = specs[id(ref)], curves[id(ref)]
        else:
            sref, cref = dsp.unit_spectrum(f), dsp.unit_curve(grid)
        lo = max(self.fit_lo.value(), fmin)
        hi = min(self.fit_hi.value(), fmax)
        if hi <= lo * 1.05:
            lo, hi = fmin, fmax
        comps = {}
        for m in self.meas:
            if m is ref:
                continue
            comps[id(m)] = dsp.compare(specs[id(m)], sref, curves[id(m)], cref, sm, fs, lo, hi,
                                       with_sum=ref is not None,
                                       search_ms=max(20.0, self.post_spin.value()))
        self.A = dict(grid=grid, specs=specs, curves=curves, comps=comps, band=(lo, hi), coh=coh, sm=sm,
                      wrapped=self.wrap_combo.currentIndex() == 0,
                      dmode=self.delay_combo.currentData(), fs=fs)

    def refresh_all(self):
        self._update_welcome()
        if self.stack.currentIndex() == 0:
            return
        try:
            with busy():
                self.analyze()
        except Exception as e:  # keeps the app alive whatever happens
            import traceback
            traceback.print_exc()
            self.status(f"Analysis error: {e}")
            self.A = None
        self._dirty = {k for k, _ in TABS}
        self._draw_current()

    def _draw_current(self, *_):
        if self.stack.currentIndex() == 0:
            return
        key = TABS[self.tabs.currentIndex()][0]
        if key not in self._dirty:
            return
        self._dirty.discard(key)
        try:
            with busy():
                self._ensure_analysis()
                if key in MEASURE_TABS:
                    if self.A is None:
                        self._clear_measure_tabs()
                    else:
                        dict(resp=self.draw_response, time=self.draw_time, diff=self.draw_diff,
                             rot=self.draw_rotation)[key]()
                elif key == "attack":
                    self.attack_tab.refresh()
                elif key == "align":
                    self.align_tab.refresh()
                elif key == "rta":
                    self.rta_tab.refresh()
                elif key == "eq":
                    self.eq_tab.refresh()
                elif key == "live":
                    self.live_tab.reload_targets()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status(f"Display error: {e}")

    def _ensure_analysis(self):
        """Recomputes the analysis if the takes list changed since last time."""
        ids = {id(m) for m in self.meas}
        if (self.A is None and self.meas) or (self.A is not None and set(self.A["specs"]) != ids):
            self.analyze()

    def draw_all(self):
        """Draws every tab (useful for exports and tests)."""
        for i in range(self.tabs.count()):
            self._dirty.add(TABS[i][0])
            cur = self.tabs.currentIndex()
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(i)
            self._draw_current()
            self.tabs.setCurrentIndex(cur)
            self.tabs.blockSignals(False)

    def _clear_measure_tabs(self):
        for p in (self.p_mag, self.p_ph, self.p_gd, self.p_coh, self.p_time, self.p_dmag, self.p_dph,
                  self.p_sum, self.p_rph, self.p_rmag, self.p_rt, self.p_rmap):
            p.clear()
        self.set_table.setRowCount(0)
        msg = "No take: import measured takes (test signal or loopback)."
        self.resp_readout.setText(f"<span style='color:{C['muted']}'>{msg}</span>")
        self.diff_readout.setText(f"<span style='color:{C['muted']}'>{msg}</span>")
        self.time_info.setText(msg)
        self.rot_info.setText(msg)
        self.resp_data, self.diff_data, self.sum_target, self.rot_rows = [], [], None, ([], [])

    def visible(self):
        return [m for m in self.meas if m.visible]

    def _phase(self, ph):
        return break_wraps(dsp.wrap_deg(ph)) if self.A["wrapped"] else ph

    def _pen(self, color, width=1.4, style=None):
        p = pg.mkPen(color, width=width)
        if style:
            p.setStyle(style)
        return p

    def _cursor_lines(self, plots):
        lines = []
        for p in plots:
            ln = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen(C["faint"], width=1, style=Qt.DashLine))
            ln.setVisible(False)
            p.addItem(ln, ignoreBounds=True)
            lines.append(ln)
        return lines

    def _band_region(self, p):
        lo, hi = self.A["band"]
        reg = pg.LinearRegionItem([math.log10(lo), math.log10(hi)], movable=False,
                                  brush=pg.mkBrush(56, 201, 178, 14), pen=pg.mkPen(None))
        reg.setZValue(-10)
        p.addItem(reg, ignoreBounds=True)

    # -- Response
    def draw_response(self):
        A = self.A
        grid = A["grid"]
        for p in (self.p_mag, self.p_ph, self.p_gd, self.p_coh):
            p.clear()
        self.p_gd.setVisible(self.gd_check.isChecked())
        show_coh = self.coh_check.isChecked() and bool(A["coh"])
        self.p_coh.setVisible(show_coh)
        ref_tp = A["curves"][id(self.ref)].tp if self.ref else 0.0
        k1 = int(np.argmin(np.abs(grid - 1000)))
        thr = self.coh_thr.value()
        self.resp_data = []
        for m in self.visible():
            c = A["curves"][id(m)]
            tau_rm = {"none": 0.0, "common": ref_tp, "indiv": c.tp}[A["dmode"]]
            mag = c.mag_db
            if self.norm_check.isChecked():
                mag = mag - mag[k1]
            ph = c.phase_res_deg() - 360 * grid * (c.tp - tau_rm)
            gd = dsp.group_delay_ms(c, grid, c.tp - tau_rm)
            coh = A["coh"].get(id(m))
            mask = None
            if coh is not None and thr > 0:
                mask = coh < thr
            w = 2.0 if m is self.sel else 1.3
            pm, pp, pgd = mag.copy(), self._phase(ph), gd.copy()
            if mask is not None:
                pm[mask] = np.nan
                pp = np.where(mask, np.nan, pp)
                pgd[mask] = np.nan
            self.p_mag.plot(grid, pm, pen=self._pen(m.color, w), connect="finite")
            self.p_ph.plot(grid, pp, pen=self._pen(m.color, w), connect="finite")
            if self.gd_check.isChecked():
                self.p_gd.plot(grid, pgd, pen=self._pen(m.color, w), connect="finite")
            if show_coh and coh is not None:
                self.p_coh.plot(grid, coh, pen=self._pen(m.color, 1.1))
            self.resp_data.append((m, mag, dsp.wrap_deg(ph) if A["wrapped"] else ph, gd))
        if show_coh:
            self.p_coh.setYRange(0, 1.02, padding=0)
            if thr > 0:
                self.p_coh.addItem(pg.InfiniteLine(pos=thr, angle=0, pen=pg.mkPen(C["warn"], style=Qt.DotLine)))
        if A["wrapped"]:
            self.p_ph.setYRange(-185, 185, padding=0)
            self.p_ph.getAxis("left").setTicks([[(v, f"{v}") for v in range(-180, 181, 90)]])
        else:
            self.p_ph.enableAutoRange(axis="y")
            self.p_ph.getAxis("left").setTicks(None)
        if self.gd_check.isChecked() and self.resp_data:
            allgd = np.concatenate([d[3] for d in self.resp_data])
            lo, hi = np.nanpercentile(allgd, [2, 98])
            pad = max((hi - lo) * 0.15, 0.05)
            self.p_gd.setYRange(lo - pad, hi + pad)
        self.resp_lines = self._cursor_lines([self.p_mag, self.p_ph, self.p_gd, self.p_coh])
        self.p_mag.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self.resp_readout.setText(f"<span style='color:{C['muted']}'>Hover over a plot to read the values.</span>")

    # -- Time
    def draw_time(self):
        p = self.p_time
        p.clear()
        mode = self.time_combo.currentData()
        vis = self.visible()
        if not vis:
            return
        pre, post = self.pre_spin.value() * 1e-3, self.post_spin.value() * 1e-3
        p.setDownsampling(auto=True, mode="peak")
        p.setClipToView(True)
        if mode in ("ir", "etc"):
            gmax = max(np.max(np.abs(m.ir)) for m in vis)
            t_lo = min(m.peak_time for m in vis) - max(pre, 0.002) - 0.001
            t_hi = max(m.peak_time for m in vis) + post
            for m in vis:
                fs = m.fs
                i0 = max(int((t_lo - m.t0) * fs), 0)
                i1 = min(int((t_hi - m.t0) * fs) + 1, len(m.ir))
                t = (m.t0 + np.arange(i0, i1) / fs) * 1e3
                if mode == "ir":
                    y = m.ir[i0:i1] / gmax
                else:
                    y = 20 * np.log10(np.maximum(m.env[i0:i1] / gmax, 1e-7))
                p.plot(t, y, pen=self._pen(m.color, 1.8 if m is self.sel else 1.1))
            p.setLabel("left", "Amplitude (normalized)" if mode == "ir" else "Level (dB)")
            focus = self.sel if self.sel in vis else vis[0]
            reg = pg.LinearRegionItem([(focus.peak_time - pre) * 1e3, (focus.peak_time + post) * 1e3],
                                      movable=False, brush=pg.mkBrush(56, 201, 178, 18),
                                      pen=pg.mkPen(C["accent_dim"]))
            reg.setZValue(-10)
            p.addItem(reg, ignoreBounds=True)
            p.setXRange(t_lo * 1e3, min(t_hi, max(m.peak_time for m in vis) + 0.015) * 1e3, padding=0)
            if mode == "etc":
                p.setYRange(-90, 3, padding=0)
            else:
                p.enableAutoRange(axis="y")
            lab = "Analysis window shown for: " + focus.name
            pk = " · ".join(f"{m.name} {m.peak_time * 1e3:.3f} ms" for m in vis[:3])
            self.time_info.setText(f"{lab}   ·   Peaks: {pk}{' …' if len(vis) > 3 else ''}")
        else:
            m = self.sel if self.sel in vis else vis[0]
            if m.y is None:
                self.time_info.setText(f"{m.name} is a combined measurement: no raw take to compare.")
                return
            key = (id(m), mode)
            if key not in self._orig_cache:
                self._orig_cache[key] = dsp.original_vs_take(m, mode)
            pred, res, g, rd = self._orig_cache[key]
            t = np.arange(len(m.y)) / m.fs * 1e3
            p.plot(t, m.y, pen=self._pen(m.color, 1.0), name="Take")
            p.plot(t, pred, pen=self._pen(C["text"], 0.8), name="Original")
            p.plot(t, res, pen=self._pen(C["bad"], 0.8), name="Residual")
            p.setLabel("left", "Amplitude (FS)")
            p.enableAutoRange()
            src = "loopback" if m.loopback else "original"
            if mode == "aligned":
                self.time_info.setText(
                    f"{m.name} · {src} shifted by {m.peak_raw * 1e3:.3f} ms and scaled "
                    f"({20 * np.log10(abs(g) + 1e-20):+.2f} dB{', polarity inverted' if g < 0 else ''}) · "
                    f"residual {rd:.1f} dB  (white: {src} · red: residual)")
            else:
                self.time_info.setText(
                    f"{m.name} · {src} filtered by the measured response · residual (noise + distortion) "
                    f"{rd:.1f} dB below the take")

    # -- Differences
    def draw_diff(self):
        A = self.A
        grid = A["grid"]
        for p in (self.p_dmag, self.p_dph, self.p_sum):
            p.clear()
        self._band_region(self.p_dmag)
        self._band_region(self.p_dph)
        mode = "indiv" if A["dmode"] == "indiv" else "incl"
        self.diff_data = []
        for m in self.visible():
            if m is self.ref:
                continue
            c = A["comps"][id(m)]
            dph = c.dphase(grid, mode)
            w = 2.0 if m is self.sel else 1.3
            self.p_dmag.plot(grid, c.dmag, pen=self._pen(m.color, w))
            self.p_dph.plot(grid, self._phase(dph), pen=self._pen(m.color, w), connect="finite")
            self.diff_data.append((m, c.dmag, dsp.wrap_deg(dph) if A["wrapped"] else dph))
        if self.ref is not None:
            for ln in (self.p_dmag, self.p_dph):
                ln.addItem(pg.InfiniteLine(pos=0, angle=0, pen=self._pen(self.ref.color, 1, Qt.DotLine)))
        if A["wrapped"]:
            self.p_dph.setYRange(-185, 185, padding=0)
            self.p_dph.getAxis("left").setTicks([[(v, f"{v}") for v in range(-180, 181, 90)]])
        else:
            self.p_dph.enableAutoRange(axis="y")
            self.p_dph.getAxis("left").setTicks(None)

        target = self.sel if (self.sel is not None and self.sel is not self.ref) else \
            next((m for m in self.visible() if m is not self.ref), None)
        self.sum_target = None
        if self.ref is not None and target is not None:
            c = A["comps"][id(target)]
            self.p_sum.plot(grid, c.sum_ideal, pen=self._pen(C["faint"], 1.2, Qt.DashLine))
            self.p_sum.plot(grid, c.sum_before, pen=self._pen(target.color, 1.3))
            self.p_sum.plot(grid, c.sum_after, pen=self._pen(C["accent"], 1.8))
            self.p_sum.setTitle(
                f"Sum {self.ref.name} + {target.name} — color: raw · "
                f"<span style='color:{C['accent']}'>green: corrected</span> · dashed: ideal (+6 dB)",
                color=C["muted"], size="9pt")
            self.sum_target = (target, c)
        else:
            self.p_sum.setTitle("Choose a take as reference to see the summation",
                                color=C["faint"], size="9pt")
        self.diff_lines = self._cursor_lines([self.p_dmag, self.p_dph, self.p_sum])
        self.p_dmag.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self._fill_settings_table()
        lo, hi = A["band"]
        ref = self.ref.name if self.ref else "the original signal"
        self.diff_readout.setText(
            f"<span style='color:{C['muted']}'>Reference: {ref} · fit band "
            f"{short_hz(round(lo))}–{short_hz(round(hi))} Hz (shaded area)</span>")

    def settings_rows(self):
        A = self.A
        fs = A["fs"]
        c_air = dsp.speed_of_sound(self.temp_spin.value())
        rows = []
        for m in self.meas:
            if m is self.ref:
                rows.append([m.name, "0", "0", "0", "—", "—", "—", "—", "—", "Reference"])
                continue
            c = A["comps"][id(m)]
            tau_ms = c.tau * 1e3
            parts = []
            if abs(tau_ms) >= 0.0005:
                parts.append(f"{'advance' if tau_ms > 0 else 'delay'} by {abs(tau_ms):.3f} ms "
                             f"({abs(c.tau) * fs:.1f} samples)")
            if c.inverted:
                parts.append("invert the polarity")
            if abs(c.rotation) >= 1:
                parts.append(f"phase rotation {-c.rotation:+.0f}°")
            if abs(c.gain_db) >= 0.1:
                parts.append(f"gain {-c.gain_db:+.1f} dB")
            corr = ", ".join(parts) or "none"
            corr = corr[0].upper() + corr[1:]
            rows.append([
                m.name, f"{tau_ms:+.3f}", f"{c.tau * fs:+.2f}", f"{c.tau * c_air * 100:+.2f}",
                "inverted" if c.inverted else "normal", f"{nz(c.rotation):+.1f}", f"{nz(c.gain_db):+.2f}",
                f"{c.resid_rms:.1f}° RMS", fmt_hz(c.notch_hz) if c.notch_hz else "—", corr])
        return rows

    def _fill_settings_table(self):
        rows = self.settings_rows()
        t = self.set_table
        t.setRowCount(len(rows))
        for r, (m, row) in enumerate(zip(self.meas, rows)):
            for c, txt in enumerate(row):
                it = QtWidgets.QTableWidgetItem(txt)
                if c == 0:
                    it.setIcon(color_icon(m.color))
                elif c < 9:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 4 and txt == "inverted":
                    it.setForeground(QtGui.QColor(C["warn"]))
                if c == 7 and m is not self.ref:
                    if self.A["comps"][id(m)].resid_rms > 20:
                        it.setForeground(QtGui.QColor(C["warn"]))
                        it.setToolTip("Delay + rotation do not explain everything: the difference "
                                      "depends on frequency (filtering, reflections, directivity).")
                if c == 9 and m is not self.ref:
                    it.setForeground(QtGui.QColor(C["accent"]))
                    it.setToolTip(txt)
                t.setItem(r, c, it)

    def copy_settings(self):
        if not self.A:
            return
        hdr = [self.set_table.horizontalHeaderItem(i).text() for i in range(self.set_table.columnCount())]
        lines = ["\t".join(hdr)] + ["\t".join(r) for r in self.settings_rows()]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.status("Settings copied to the clipboard.")

    # -- Rotation
    def _rot_freqs(self):
        out = []
        for tok in re.split(r"[,;\s]+", self.rot_freqs.text()):
            tok = tok.strip().lower().replace("hz", "")
            if not tok:
                continue
            try:
                v = float(tok[:-1]) * 1000 if tok.endswith("k") else float(tok)
                out.append(v)
            except ValueError:
                pass
        g = self.A["grid"]
        return [f for f in out if g[0] <= f <= g[-1]][:8]

    def rotation_data(self):
        A = self.A
        grid = A["grid"]
        lg = np.log10(grid)
        mode = "indiv" if A["dmode"] == "indiv" else "incl"
        ms = sorted([m for m in self.visible() if m.angle is not None], key=lambda m: m.angle)
        freqs = self._rot_freqs()
        rows = []
        for m in ms:
            if m is self.ref:
                dmag = np.zeros_like(grid); dph = np.zeros_like(grid); tau = 0.0
            else:
                c = A["comps"][id(m)]
                dmag, dph, tau = c.dmag, c.dphase(grid, mode), c.tau
            # phase unwrapped over frequency (from the low end): no turn ambiguity
            # between two angles, even when the delay changes a lot
            ph_at = [float(np.interp(math.log10(f), lg, dph)) for f in freqs]
            if not self.rot_unwrap.isChecked():
                ph_at = [float(dsp.wrap_deg(v)) for v in ph_at]
            mg_at = [float(np.interp(math.log10(f), lg, dmag)) for f in freqs]
            rows.append(dict(m=m, angle=m.angle, tau=tau, ph=ph_at, mag=mg_at, dph=dsp.wrap_deg(dph)))
        return freqs, rows

    def draw_rotation(self):
        for p in (self.p_rph, self.p_rmag, self.p_rt, self.p_rmap):
            p.clear()
            if p.legend is not None:
                p.legend.scene().removeItem(p.legend) if p.legend.scene() else None
                p.legend = None
        freqs, rows = self.rotation_data()
        self.rot_rows = (freqs, rows)
        if not rows:
            self.rot_info.setText("No visible take has an angle: enter it in the \"Angle\" column.")
            return
        ref = self.ref.name if self.ref else "l'original"
        tip = "" if self.A["dmode"] == "indiv" else \
            " · tip: Display › \"Own delay removed\" isolates the phase rotation from the delay"
        self.rot_info.setText(f"{len(rows)} take(s) with angle · deviations from {ref}{tip}")
        ang = np.array([r["angle"] for r in rows])
        leg = self.p_rph.addLegend(offset=(8, 8), brush=pg.mkBrush(C["panel"]), pen=pg.mkPen(C["border"]),
                                   labelTextColor=C["text"])
        for j, f in enumerate(freqs):
            col = FREQ_COLORS[j % len(FREQ_COLORS)]
            sym = dict(symbol="s", symbolSize=6, symbolBrush=col, symbolPen=None)
            self.p_rph.plot(ang, [r["ph"][j] for r in rows], pen=self._pen(col, 1.4), name=f"{short_hz(f)} Hz", **sym)
            self.p_rmag.plot(ang, [r["mag"][j] for r in rows], pen=self._pen(col, 1.4), **sym)
        self.p_rt.plot(ang, [r["tau"] * 1e6 for r in rows], pen=self._pen(C["accent"], 1.6),
                       symbol="s", symbolSize=6, symbolBrush=C["accent"], symbolPen=None)
        if not self.rot_unwrap.isChecked():
            self.p_rph.setYRange(-185, 185, padding=0)
        else:
            self.p_rph.enableAutoRange(axis="y")
        for p in (self.p_rph, self.p_rmag, self.p_rt):
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C["faint"], style=Qt.DotLine)))

        # angle × frequency map
        grid = self.A["grid"]
        self.p_rmap.setTitle("Δ phase versus frequency and angle", color=C["muted"], size="9pt")
        if len(rows) >= 2:
            na = 181
            ag = np.linspace(ang.min(), ang.max(), na)
            idx = np.array([int(np.argmin(np.abs(ang - a))) for a in ag])
            img = np.array([rows[i]["dph"] for i in idx])
            self.rot_img = pg.ImageItem(img)
            self.rot_img.setLevels((-180, 180))
            cmap = None
            for name in ("CET-C2", "CET-C1", "CET-C6", "viridis"):
                try:
                    cmap = pg.colormap.get(name)
                    break
                except Exception:
                    continue
            if cmap is not None:
                self.rot_img.setLookupTable(cmap.getLookupTable(nPts=256))
            lx0, lx1 = math.log10(grid[0]), math.log10(grid[-1])
            a0, a1 = float(ang.min()), float(ang.max())
            self.rot_img.setRect(QtCore.QRectF(lx0, a0, lx1 - lx0, (a1 - a0) or 1.0))
            self.p_rmap.addItem(self.rot_img)
            self.p_rmap.setXRange(lx0, lx1, padding=0)
            self.p_rmap.setYRange(a0, a1 if a1 > a0 else a0 + 1, padding=0)
            self.rot_map = (ag, img, lx0, lx1)
        else:
            self.rot_map = None
            self.p_rmap.setTitle("At least two angles needed", color=C["faint"], size="9pt")

    # ------------------------------------------------------------ mouse readout
    def _freq_hover(self, pos, plots, which):
        if not self.A:
            return
        for p in plots:
            if p.isVisible() and p.sceneBoundingRect().contains(pos):
                pt = p.vb.mapSceneToView(pos)
                f = 10 ** pt.x()
                break
        else:
            return
        grid = self.A["grid"]
        if not (grid[0] <= f <= grid[-1]):
            return
        lines = self.resp_lines if which == "resp" else self.diff_lines
        for ln in lines:
            ln.setPos(math.log10(f))
            ln.setVisible(True)
        lg = np.log10(grid)
        x = math.log10(f)
        self.cursor_lab.setText(f"f = {fmt_hz(f)}   y = {pt.y():.3f}")
        parts = [f"<b>{fmt_hz(f)}</b>"]
        if which == "resp":
            for m, mag, ph, gd in self.resp_data:
                parts.append(f"<span style='color:{m.color}'>■</span> {m.name}&nbsp; "
                             f"{np.interp(x, lg, mag):+.2f} dB&nbsp; {np.interp(x, lg, ph):+.1f}°&nbsp; "
                             f"{np.interp(x, lg, gd):.3f} ms")
            self.resp_readout.setText("&nbsp;&nbsp;&nbsp;".join(parts))
        else:
            for m, dmag, dph in self.diff_data:
                parts.append(f"<span style='color:{m.color}'>■</span> {m.name}&nbsp; "
                             f"Δ {np.interp(x, lg, dmag):+.2f} dB&nbsp; {np.interp(x, lg, dph):+.1f}°")
            if self.sum_target:
                t, c = self.sum_target
                parts.append(f"sum {np.interp(x, lg, c.sum_before):+.1f} → "
                             f"<span style='color:{C['accent']}'>{np.interp(x, lg, c.sum_after):+.1f} dB</span>")
            self.diff_readout.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _time_hover(self, pos):
        p = self.p_time
        if not p.sceneBoundingRect().contains(pos) or not self.A:
            return
        pt = p.vb.mapSceneToView(pos)
        fs = self.A["fs"]
        self.cursor_lab.setText(f"t = {pt.x():.4f} ms  ({pt.x() * 1e-3 * fs:.1f} samples)   y = {pt.y():.4g}")

    def _rot_hover(self, pos):
        for p in (self.p_rph, self.p_rmag, self.p_rt):
            if p.sceneBoundingRect().contains(pos):
                pt = p.vb.mapSceneToView(pos)
                self.cursor_lab.setText(f"angle = {pt.x():.1f}°   y = {pt.y():.3f}")
                return
        if self.p_rmap.sceneBoundingRect().contains(pos) and getattr(self, "rot_map", None):
            pt = self.p_rmap.vb.mapSceneToView(pos)
            ag, img, lx0, lx1 = self.rot_map
            grid = self.A["grid"]
            j = int(np.clip(np.searchsorted(np.log10(grid), pt.x()), 0, len(grid) - 1))
            i = int(np.clip(np.argmin(np.abs(ag - pt.y())), 0, len(ag) - 1))
            self.cursor_lab.setText(f"f = {fmt_hz(10 ** pt.x())}   angle = {pt.y():.1f}°   "
                                    f"Δ phase = {img[i, j]:+.1f}°")

    # ----------------------------------------------------------------- export
    def _ask_save(self, title, default, filt):
        base = Path(self.project_path).parent if self.project_path else \
            (Path(self.signal.path).parent if self.signal and self.signal.path else Path.home())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, str(base / default), filt)
        return path

    def _write_csv(self, path, header, cols):
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(header)
            for row in zip(*cols):
                w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in row])

    def _need_data(self):
        if not self.A:
            self.error("Nothing to export", "Import takes first.")
            return False
        return True

    def export_response_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Export response curves", "responses.csv", "CSV (*.csv)")
        if not path:
            return
        hdr, cols = ["Frequency (Hz)"], [self.A["grid"]]
        for m, mag, ph, gd in self.resp_data:
            hdr += [f"{m.name} level (dB)", f"{m.name} phase (°)", f"{m.name} group delay (ms)"]
            cols += [mag, ph, gd]
        self._write_csv(path, hdr, cols)
        self.status(f"Exported: {Path(path).name}")

    def export_diff_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Export differences", "differences.csv", "CSV (*.csv)")
        if not path:
            return
        hdr, cols = ["Frequency (Hz)"], [self.A["grid"]]
        for m, dmag, dph in self.diff_data:
            hdr += [f"{m.name} Δ level (dB)", f"{m.name} Δ phase (°)"]
            cols += [dmag, dph]
        if self.sum_target:
            t, c = self.sum_target
            hdr += [f"Raw sum {t.name} (dB)", f"Corrected sum {t.name} (dB)"]
            cols += [c.sum_before, c.sum_after]
        self._write_csv(path, hdr, cols)
        self.status(f"Exported: {Path(path).name}")

    def export_settings_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Export settings", "settings.csv", "CSV (*.csv)")
        if not path:
            return
        hdr = [self.set_table.horizontalHeaderItem(i).text() for i in range(self.set_table.columnCount())]
        rows = self.settings_rows()
        self._write_csv(path, hdr, list(zip(*rows)) if rows else [[] for _ in hdr])
        self.status(f"Exported: {Path(path).name}")

    def export_eq(self):
        if not self._need_data():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the correction curves")
        if not folder:
            return
        n = 0
        for m, dmag, dph in self.diff_data:
            corr = np.clip(-dmag, -20, 20)
            p = Path(folder) / f"{safe_name(m.name)}_correction_EQ.txt"
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"* PhaseLab - level correction for {m.name} towards "
                         f"{self.ref.name if self.ref else 'original'}\n* Frequency(Hz) Gain(dB)\n")
                for f, g in zip(self.A["grid"][::3], corr[::3]):
                    fh.write(f"{f:.2f} {g:.3f}\n")
            n += 1
        self.status(f"{n} correction curve(s) exported (REW / curve-based equalizer format).")

    def export_rotation_csv(self):
        if not self._need_data():
            return
        freqs, rows = self.rot_rows
        if not rows:
            self.error("Nothing to export", "No visible take has an angle.")
            return
        path = self._ask_save("Export rotation", "rotation.csv", "CSV (*.csv)")
        if not path:
            return
        hdr = ["Angle (°)", "Take", "Δ time (µs)"]
        for f in freqs:
            hdr += [f"Δ phase {short_hz(f)} Hz (°)", f"Δ level {short_hz(f)} Hz (dB)"]
        cols = [[r["angle"] for r in rows], [r["m"].name for r in rows], [r["tau"] * 1e6 for r in rows]]
        for j in range(len(freqs)):
            cols += [[r["ph"][j] for r in rows], [r["mag"][j] for r in rows]]
        self._write_csv(path, hdr, cols)
        self.status(f"Exported: {Path(path).name}")

    def export_irs(self):
        if not self._need_data():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the impulse responses")
        if not folder:
            return
        vis = self.visible()
        segs = [m.ir[int(round(-m.t0 * m.fs)):] for m in vis]
        peak = max(np.max(np.abs(s)) for s in segs) if segs else 1
        scale = 0.9 / peak if peak > 0.9 else 1.0
        for m, s in zip(vis, segs):
            sf.write(str(Path(folder) / f"{safe_name(m.name)}_IR.wav"), s * scale, m.fs, subtype="FLOAT")
        self.status(f"{len(vis)} impulse response(s) exported, time 0 = start of the signal"
                    + (f", common gain {20 * math.log10(scale):.1f} dB" if scale != 1 else ""))

    def export_png(self):
        w = self.tabs.currentWidget()
        name = self.tabs.tabText(self.tabs.currentIndex()).lower()
        path = self._ask_save("Export image", f"phaselab_{name}.png", "PNG (*.png)")
        if path:
            w.grab().save(path)
            self.status(f"Image saved: {Path(path).name}")

    # ----------------------------------------------------------------- project
    def new_project(self):
        self.meas, self.ref, self.sel, self.signal, self.A = [], None, None, None, None
        self.audio_files = []
        self.project_path = None
        self._force_tabs = False
        self._spec_cache.clear()
        self._orig_cache.clear()
        self.eq_tab.bands = []
        self._update_signal_card()
        self.rta_tab.rebuild_table()
        self._refresh_lists()
        self.setWindowTitle(APP)

    def save_project(self, ask=False):
        if not (self.signal or self.meas or self.audio_files):
            self.error("Empty project", "Nothing to save yet.")
            return
        path = self.project_path
        if ask or not path:
            path = self._ask_save("Save project", "measurements.phaselab", "PhaseLab project (*.phaselab)")
            if not path:
                return

        def mdict(m):
            d = dict(name=m.name, angle=m.angle, visible=m.visible, color=m.color,
                     offset_ms=m.offset_ms, gain_db=m.gain_db, invert=m.invert)
            if m.virtual and m.recipe is None:
                folder = Path(path).with_name(Path(path).stem + "_traces")
                folder.mkdir(exist_ok=True)
                f = folder / f"{safe_name(m.name)}.wav"
                sf.write(str(f), m.ir_raw.astype(np.float32), m.fs, subtype="FLOAT")
                d.update(live=True, file=str(f), t0=m.t0_raw, f1=m.f1, f2=m.f2, averages=m.averages)
            elif m.virtual:
                d.update(virtual=True, recipe=m.recipe)
            else:
                d.update(path=m.path, channel=m.channel)
            return d

        data = dict(
            app=APP, version=VERSION, signal=self.signal.path if self.signal else None,
            loop_channel=self.loop_combo.currentData(), method=self.method_combo.currentData(),
            nfft=self.nfft_combo.currentData(),
            ref=self.meas.index(self.ref) if self.ref in self.meas else -1,
            settings=dict(smoothing=self.smooth_combo.currentIndex(), wrap=self.wrap_combo.currentIndex(),
                          delay=self.delay_combo.currentIndex(), pre=self.pre_spin.value(),
                          post=self.post_spin.value(), fit_lo=self.fit_lo.value(),
                          fit_hi=self.fit_hi.value(), temp=self.temp_spin.value(),
                          rot_freqs=self.rot_freqs.text()),
            measurements=[mdict(m) for m in self.meas],
            audio=[dict(path=a.path, visible=a.visible, color=a.color) for a in self.audio_files],
            rta_target=self.rta_tab.target.currentText(),
            eq=self.eq_tab.state())
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.project_path = path
        self.setWindowTitle(f"{APP} — {Path(path).stem}")
        self.status(f"Project saved: {Path(path).name}")

    def open_project(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a project", "",
                                                            "PhaseLab project (*.phaselab)")
            if not path:
                return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            self.error("Unreadable project", str(e))
            return
        self.new_project()
        st = data.get("settings", {})
        for w, k in ((self.smooth_combo, "smoothing"), (self.wrap_combo, "wrap"), (self.delay_combo, "delay")):
            if k in st:
                w.setCurrentIndex(st[k])
        for w, k in ((self.pre_spin, "pre"), (self.post_spin, "post"), (self.temp_spin, "temp")):
            if k in st:
                w.setValue(st[k])
        if "rot_freqs" in st:
            self.rot_freqs.setText(st["rot_freqs"])
        for combo, k, default in ((self.loop_combo, "loop_channel", 0), (self.method_combo, "method", "auto"),
                                  (self.nfft_combo, "nfft", 32768)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(data.get(k, default)), 0))
            combo.blockSignals(False)
        missing = []
        if data.get("signal"):
            if Path(data["signal"]).exists():
                self.load_signal(data["signal"])
            else:
                missing.append(data["signal"])
        for k in ("fit_lo", "fit_hi"):
            if k in st:
                getattr(self, k).setValue(st[k])
        items = data.get("measurements", [])
        real = [it for it in items if not it.get("virtual") and not it.get("live")]
        paths = list(dict.fromkeys(it["path"] for it in real))
        missing += [p for p in paths if not Path(p).exists()]
        ok_paths = [p for p in paths if Path(p).exists()]
        if ok_paths and self._can_measure():
            self.import_takes(ok_paths)
        lookup = {(it["path"], it["channel"]): it for it in real}
        order = {}
        for i, it in enumerate(items):
            virt = it.get("virtual") or it.get("live")
            order[("virtual", it["name"]) if virt else (it["path"], it["channel"])] = i
        kept = []
        for m in self.meas:
            it = lookup.get(m.key)
            if it:
                m.name, m.angle, m.visible = it["name"], it["angle"], it["visible"]
                m.color = it.get("color") or m.color
                m.offset_ms, m.gain_db, m.invert = it.get("offset_ms", 0), it.get("gain_db", 0), it.get("invert", False)
                kept.append(m)
        for it in items:
            if it.get("live"):
                try:
                    ir, fs_ = sf.read(it["file"], dtype="float64")
                except Exception:
                    missing.append(it.get("file", it["name"]))
                    continue
                nm = dsp.Measurement(it["name"], fs_, ir, it["t0"], it["f1"], it["f2"], method="live",
                                     averages=it.get("averages", 1))
                nm.virtual = True
                nm.color, nm.visible, nm.angle = it.get("color"), it["visible"], it.get("angle")
                nm.offset_ms, nm.gain_db, nm.invert = it.get("offset_ms", 0), it.get("gain_db", 0), it.get("invert", False)
                kept.append(nm)
            elif it.get("virtual"):
                try:
                    srcs = [next(m for m in kept if list(m.key) == list(k)) for k in it["recipe"]["sources"]]
                    nm = dsp.combine(srcs, it["recipe"]["op"], it["name"])
                except Exception:
                    continue
                nm.color, nm.visible, nm.angle = it.get("color"), it["visible"], it.get("angle")
                nm.offset_ms, nm.gain_db, nm.invert = it.get("offset_ms", 0), it.get("gain_db", 0), it.get("invert", False)
                kept.append(nm)
        self.meas = sorted(kept, key=lambda m: order.get(m.key, 1e9))
        r = data.get("ref", -1)
        self.ref = self.meas[r] if 0 <= r < len(self.meas) else None
        self._spec_cache.clear()
        afiles = [a for a in data.get("audio", []) if Path(a["path"]).exists()]
        missing += [a["path"] for a in data.get("audio", []) if not Path(a["path"]).exists()]
        if afiles:
            self.rta_tab.add_files([a["path"] for a in afiles])
            for af, a in zip(self.audio_files, afiles):
                af.visible, af.color = a.get("visible", True), a.get("color") or af.color
            self.rta_tab.rebuild_table()
        if data.get("rta_target"):
            i = self.rta_tab.target.findText(data["rta_target"])
            if i >= 0:
                self.rta_tab.target.setCurrentIndex(i)
        self.project_path = path
        self.setWindowTitle(f"{APP} — {Path(path).stem}")
        self._refresh_lists()
        if data.get("eq"):
            self.eq_tab.load_state(data["eq"])
        if missing:
            self.error("Missing files", "\n".join(missing))

    # ----------------------------------------------------------------- misc
    def show_guide(self):
        GuideDialog(self).exec()

    def error(self, title, text):
        QtWidgets.QMessageBox.warning(self, title, text)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle("Fusion")
    font = QtGui.QFont()
    for fam in ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial"):
        if fam in QtGui.QFontDatabase.families():
            font.setFamily(fam)
            break
    font.setPointSizeF(9.5)
    app.setFont(font)
    pal = app.palette()
    for role, col in ((QtGui.QPalette.Window, C["bg"]), (QtGui.QPalette.Base, C["panel2"]),
                      (QtGui.QPalette.Text, C["text"]), (QtGui.QPalette.WindowText, C["text"]),
                      (QtGui.QPalette.Button, C["panel2"]), (QtGui.QPalette.ButtonText, C["text"]),
                      (QtGui.QPalette.Highlight, C["accent_dim"]),
                      (QtGui.QPalette.HighlightedText, C["text"])):
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    for a in sys.argv[1:]:
        if a.endswith(".phaselab"):
            win.open_project(a)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
