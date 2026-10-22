#!/usr/bin/env python3
"""PhaseLab - analyse de phase, de retard et de réponse pour prises micro.

Lancer :  python phaselab.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

import dsp
from style import C, PALETTE, QSS

APP = "PhaseLab"
VERSION = "1.0"
pg.setConfigOptions(antialias=True, background=C["plot_bg"], foreground=C["muted"],
                    imageAxisOrder="row-major")

SMOOTHING = [("Aucun", 0), ("1/48 oct", 48), ("1/24 oct", 24), ("1/12 oct", 12),
             ("1/6 oct", 6), ("1/3 oct", 3)]
DELAY_MODES = [("Retard inclus", "none"), ("Retard de la référence retiré", "common"),
               ("Retard propre retiré", "indiv")]
TIME_MODES = [("Réponse impulsionnelle", "ir"), ("Enveloppe ETC (dB)", "etc"),
              ("Prise vs original aligné", "aligned"),
              ("Prise vs original × réponse (résidu = bruit + distorsion)", "model")]
FREQ_COLORS = ["#4fc3f7", "#aed581", "#ffd54f", "#ff8a65", "#f06292", "#ba68c8", "#4db6ac", "#e0e0e0"]


# ----------------------------------------------------------------- utilitaires
def fmt_hz(f):
    return f"{f:.1f} Hz" if f < 1000 else f"{f / 1000:.3f} kHz"


def nz(v, eps=0.05):
    """Évite l'affichage de « -0.0 »."""
    return 0.0 if abs(v) < eps else v


def short_hz(f):
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


def safe_name(s):
    return re.sub(r"[^\w\-. ]+", "_", s).strip() or "mesure"


def break_wraps(y, limit=180.0):
    """Insère des NaN aux sauts de phase pour ne pas tracer de traits verticaux."""
    y = np.array(y, dtype=float)
    jumps = np.where(np.abs(np.diff(y)) > limit)[0]
    y[jumps + 1] = np.nan
    return y


def color_icon(color, size=10):
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor(color))
    return QtGui.QIcon(pm)


@contextmanager
def busy():
    QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
    QtWidgets.QApplication.processEvents()
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


class FreqAxis(pg.AxisItem):
    """Axe fréquentiel log avec graduations lisibles (20, 50, 100, 1k…)."""
    MAJ = [10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4]
    MIN = [k * d for d in (10, 100, 1000, 10000) for k in range(1, 10)]

    def tickValues(self, minVal, maxVal, size):
        if not self.logMode:
            return super().tickValues(minVal, maxVal, size)
        maj = [math.log10(v) for v in self.MAJ if minVal <= math.log10(v) <= maxVal]
        mi = [math.log10(v) for v in self.MIN if v not in self.MAJ and minVal <= math.log10(v) <= maxVal]
        if len(maj) < 2:
            lo, hi = 10 ** minVal, 10 ** maxVal
            maj = [math.log10(v) for v in np.geomspace(lo, hi, 5)]
        return [(1.0, maj), (0.5, mi)]

    def tickStrings(self, values, scale, spacing):
        if not self.logMode:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            f = 10 ** v
            out.append(short_hz(float(f"{f:.3g}")) if spacing == 1.0 else "")
        return out


def style_plot(p, ylabel, xlabel=None):
    p.showGrid(x=True, y=True, alpha=0.10)
    p.hideButtons()
    p.setMenuEnabled(False)
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(C["border"]))
        a.setTextPen(pg.mkPen(C["muted"]))
    p.getAxis("left").setWidth(58)
    p.setLabel("left", ylabel, color=C["muted"], **{"font-size": "9pt"})
    if xlabel:
        p.setLabel("bottom", xlabel, color=C["muted"], **{"font-size": "9pt"})
    return p


def freq_plot(ylabel):
    p = pg.PlotItem(axisItems={"bottom": FreqAxis("bottom")})
    p.setLogMode(x=True, y=False)
    return style_plot(p, ylabel)


def lin_plot(ylabel, xlabel=None):
    return style_plot(pg.PlotItem(), ylabel, xlabel)


def section(text):
    lab = QtWidgets.QLabel(text.upper())
    lab.setObjectName("Section")
    return lab


def muted(text, wrap=False):
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("Muted")
    lab.setWordWrap(wrap)
    return lab


# ---------------------------------------------------------------- dialogues
class GenerateDialog(QtWidgets.QDialog):
    HINTS = {
        "sweep": "Méthode de référence (Farina / Novak). Rejette la distorsion harmonique, "
                 "meilleur rapport signal/bruit. Plus long = plus de précision.",
        "pink": "Bruit rose à bande limitée. Utile pour vérifier une mesure, moins précis "
                "que le sweep pour la phase.",
        "white": "Bruit blanc à bande limitée. Énergie égale par Hz (beaucoup d'aigus).",
        "impulse": "Impulsion unique (Dirac). Peu d'énergie : à réserver aux tests de chaîne.",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Générer un signal de test")
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)

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

        def dspin(lo, hi, val, dec, suffix, step=1.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
            s.setSuffix(suffix); s.setSingleStep(step)
            return s

        self.duration = dspin(0.5, 120, 5, 1, " s", 0.5)
        self.f1 = dspin(1, 2000, 20, 1, " Hz")
        self.f2 = dspin(100, 96000, 20000, 0, " Hz", 100)
        self.level = dspin(-60, 0, -6, 1, " dBFS", 0.5)
        self.pre = dspin(0, 10, 0.5, 2, " s", 0.1)
        self.post = dspin(0, 20, 2, 2, " s", 0.5)
        form.addRow("Type", self.kind)
        form.addRow("Échantillonnage", self.fs)
        form.addRow("Format", self.subtype)
        form.addRow("Durée", self.duration)
        form.addRow("Fréquence début", self.f1)
        form.addRow("Fréquence fin", self.f2)
        form.addRow("Niveau crête", self.level)
        form.addRow("Silence avant", self.pre)
        form.addRow("Silence après", self.post)
        lay.addLayout(form)
        self.hint = muted("", wrap=True)
        lay.addWidget(self.hint)
        tip = muted("Le silence après doit couvrir la décroissance de la pièce. "
                    "Utilise la même fréquence d'échantillonnage que ton projet Reaper.", wrap=True)
        lay.addWidget(tip)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        cancel = QtWidgets.QPushButton("Annuler")
        ok = QtWidgets.QPushButton("Générer et enregistrer…")
        ok.setObjectName("Primary")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()

    def _kind_changed(self):
        k = self.kind.currentData()
        self.hint.setText(self.HINTS[k])
        self.duration.setEnabled(k != "impulse")
        self.f1.setEnabled(k != "impulse")
        self.f2.setEnabled(k != "impulse")

    def spec(self):
        k = self.kind.currentData()
        fs = self.fs.currentData()
        return dsp.SignalSpec(
            kind=k, fs=fs, duration=self.duration.value() if k != "impulse" else 0.01,
            f1=self.f1.value() if k != "impulse" else 10.0,
            f2=self.f2.value() if k != "impulse" else 0.45 * fs,
            level_db=self.level.value(), pre_silence=self.pre.value(),
            post_silence=self.post.value(), subtype=self.subtype.currentData())


GUIDE_HTML = f"""
<style>
h3 {{ color: {C['text']}; font-weight: 600; margin: 14px 0 6px 0; }}
p, li {{ color: {C['text']}; line-height: 145%; }}
.m {{ color: {C['muted']}; }}
b {{ color: {C['accent']}; font-weight: 600; }}
</style>
<h3>1 · Générer</h3>
<p>Bouton <b>Générer…</b> : choisis le type (le sweep est recommandé), la fréquence
d'échantillonnage de ton projet Reaper, puis enregistre le WAV. Un petit fichier
<span class="m">.phaselab.json</span> est créé à côté : il décrit exactement le signal.
Garde-les ensemble.</p>

<h3>2 · Enregistrer dans Reaper</h3>
<ol>
<li>Projet à la <b>même fréquence d'échantillonnage</b> que le signal.</li>
<li>Glisse le WAV du signal sur une piste, <b>au tout début du projet (0:00)</b>,
routée vers l'enceinte. Aucun effet sur cette piste.</li>
<li>Une piste par micro, armée, entrée = le micro, <b>sans effet en entrée</b>.</li>
<li>Démarre l'enregistrement depuis 0:00 et laisse tourner jusqu'à la fin du fichier.</li>
<li>Rotation : une prise par angle, en ne changeant <b>que l'angle</b>
(même gain, même position). Nomme les prises avec l'angle :
<span class="m">snare_45deg</span>, <span class="m">snare_90deg</span>… l'angle est lu automatiquement.</li>
<li>Exporte : <span class="m">File › Render</span>, source <b>Stems (selected tracks)</b>,
bornes <b>Entire project</b>, même fréquence, sans normalisation ni dither.
Le rendu depuis 0:00 garantit un alignement exact sur le signal.</li>
</ol>

<h3>Option pro : loopback</h3>
<p>Relie une sortie de l'interface à une entrée et enregistre-la en même temps que
les micros, dans un rendu multicanal (<span class="m">Render › Multichannel</span>).
Indique ensuite son numéro dans <b>Canal loopback</b>. Chaque prise est alors
mesurée par rapport à ce qui est réellement sorti de l'interface (méthode à
double canal) : latence et coloration de l'interface disparaissent.</p>

<h3>3 · Lire les résultats</h3>
<p><b>Réponse</b> : niveau, phase et retard de groupe de chaque prise.
<b>Temps</b> : réponse impulsionnelle, enveloppe, et comparaison directe avec l'original.
<b>Différences</b> : écart de niveau et de phase par rapport à la référence, sommation
avant/après correction, et le tableau des <b>réglages</b> (retard, polarité, rotation
de phase, gain) à appliquer pour aligner chaque prise.
<b>Rotation</b> : évolution de la phase, du niveau et du retard selon l'angle du micro.</p>
<p class="m">Astuce : survole un graphique pour lire les valeurs exactes à une fréquence.</p>
"""


class GuideDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide")
        self.resize(620, 640)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 18)
        tb = QtWidgets.QTextBrowser()
        tb.setHtml(GUIDE_HTML)
        tb.setOpenExternalLinks(True)
        lay.addWidget(tb)
        close = QtWidgets.QPushButton("Fermer")
        close.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        lay.addLayout(row)


# ------------------------------------------------------------ fenêtre principale
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1440, 880)
        self.setAcceptDrops(True)

        self.signal: dsp.TestSignal | None = None
        self.meas: list[dsp.Measurement] = []
        self.ref: dsp.Measurement | None = None     # None = signal original
        self.sel: dsp.Measurement | None = None
        self.project_path = None
        self._spec_cache = {}
        self._orig_cache = {}
        self._updating = False
        self.A = None   # résultats d'analyse
        self._color_i = 0

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh_all)

        self._build_ui()
        self._build_menu()
        self._update_signal_card()
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
        self.tabs.addTab(self._build_resp_tab(), "Réponse")
        self.tabs.addTab(self._build_time_tab(), "Temps")
        self.tabs.addTab(self._build_diff_tab(), "Différences")
        self.tabs.addTab(self._build_rot_tab(), "Rotation")
        self.stack.addWidget(self.tabs)
        v.addWidget(self.stack, 1)
        h.addWidget(right, 1)
        self.setCentralWidget(central)

        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.status_msg = QtWidgets.QLabel("Prêt")
        self.cursor_lab = QtWidgets.QLabel("")
        self.cursor_lab.setObjectName("Value")
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.cursor_lab)

    def _build_sidebar(self):
        side = QtWidgets.QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(310)
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(16, 16, 16, 14)
        v.setSpacing(10)

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
        v.addSpacing(6)

        v.addWidget(section("Signal de test"))
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        cv = QtWidgets.QVBoxLayout(card)
        cv.setContentsMargins(10, 8, 10, 8)
        cv.setSpacing(2)
        self.sig_name = QtWidgets.QLabel("Aucun signal")
        self.sig_info = muted("", wrap=True)
        cv.addWidget(self.sig_name)
        cv.addWidget(self.sig_info)
        v.addWidget(card)
        row = QtWidgets.QHBoxLayout()
        self.btn_gen = QtWidgets.QPushButton("Générer…")
        self.btn_gen.setToolTip("Crée un sweep ou un bruit à jouer dans Reaper")
        self.btn_gen.clicked.connect(self.generate_signal)
        self.btn_load_sig = QtWidgets.QPushButton("Charger…")
        self.btn_load_sig.setToolTip("Charge un signal déjà généré (WAV)")
        self.btn_load_sig.clicked.connect(self.load_signal_dialog)
        row.addWidget(self.btn_gen)
        row.addWidget(self.btn_load_sig)
        v.addLayout(row)

        v.addSpacing(8)
        v.addWidget(section("Prises"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Nom", "Angle", "Pic ms", "SNR"])
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
        self.table.setToolTip("Coche pour afficher · double-clic pour renommer ou saisir l'angle")
        v.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Importer des prises…")
        self.btn_import.clicked.connect(self.import_dialog)
        self.btn_remove = QtWidgets.QPushButton("Retirer")
        self.btn_remove.clicked.connect(self.remove_selected)
        row.addWidget(self.btn_import, 1)
        row.addWidget(self.btn_remove)
        v.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(muted("Canal loopback"))
        self.loop_combo = QtWidgets.QComboBox()
        self.loop_combo.addItem("Aucun", 0)
        for i in range(1, 17):
            self.loop_combo.addItem(f"Canal {i}", i)
        self.loop_combo.setToolTip(
            "Si tes fichiers multicanaux contiennent un retour direct de la sortie de "
            "l'interface, indique son canal : il sert alors de référence temporelle.")
        self.loop_combo.currentIndexChanged.connect(self._loop_changed)
        row.addStretch()
        row.addWidget(self.loop_combo)
        v.addLayout(row)
        v.addWidget(muted("Glisse des WAV n'importe où dans la fenêtre.", wrap=True))
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
        self.ref_combo.setToolTip("Toutes les différences sont calculées par rapport à cette référence")
        self.ref_combo.currentIndexChanged.connect(self._ref_changed)
        self._labeled(h, "Référence", self.ref_combo)

        self.smooth_combo = QtWidgets.QComboBox()
        for lab, v in SMOOTHING:
            self.smooth_combo.addItem(lab, v)
        self.smooth_combo.setCurrentIndex(3)
        self.smooth_combo.setToolTip("Lissage complexe par fraction d'octave (préserve la phase)")
        self.smooth_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Lissage", self.smooth_combo)

        self.wrap_combo = QtWidgets.QComboBox()
        self.wrap_combo.addItems(["Enroulée ±180°", "Déroulée"])
        self.wrap_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Phase", self.wrap_combo)

        self.delay_combo = QtWidgets.QComboBox()
        for lab, v in DELAY_MODES:
            self.delay_combo.addItem(lab, v)
        self.delay_combo.setCurrentIndex(1)
        self.delay_combo.setToolTip(
            "Inclus : phase brute (le retard fait tourner la phase très vite).\n"
            "Référence retirée : on garde les écarts de temps entre prises.\n"
            "Propre retiré : chaque prise est recentrée, on ne voit que la forme de phase.")
        self.delay_combo.currentIndexChanged.connect(self.schedule)
        self._labeled(h, "Affichage", self.delay_combo)

        def spin(lo, hi, val, tip):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val); s.setDecimals(1)
            s.setSuffix(" ms"); s.setFixedWidth(84)
            s.setKeyboardTracking(False)
            s.setToolTip(tip)
            s.valueChanged.connect(self.schedule)
            return s

        self.pre_spin = spin(0.2, 200, 5, "Durée gardée avant le pic de la réponse impulsionnelle")
        self.post_spin = spin(1, 3000, 100,
                              "Durée gardée après le pic. Plus court = ignore les réflexions, "
                              "mais résolution basse fréquence ≈ 1000 / durée Hz")
        h.addWidget(muted("Fenêtre"))
        h.addWidget(self.pre_spin)
        h.addWidget(muted("/"))
        h.addWidget(self.post_spin)
        h.addStretch()

        exp = QtWidgets.QPushButton("Exporter")
        menu = QtWidgets.QMenu(exp)
        menu.addAction("Courbes de réponse (CSV)…", self.export_response_csv)
        menu.addAction("Courbes de différence (CSV)…", self.export_diff_csv)
        menu.addAction("Tableau des réglages (CSV)…", self.export_settings_csv)
        menu.addAction("Courbes de correction EQ (TXT)…", self.export_eq)
        menu.addAction("Données de rotation (CSV)…", self.export_rotation_csv)
        menu.addSeparator()
        menu.addAction("Réponses impulsionnelles (WAV)…", self.export_irs)
        menu.addAction("Image de l'onglet (PNG)…", self.export_png)
        exp.setMenu(menu)
        h.addWidget(exp)
        return bar

    def _build_welcome(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.addStretch(2)
        col = QtWidgets.QWidget()
        col.setMaximumWidth(560)
        v = QtWidgets.QVBoxLayout(col)
        v.setSpacing(14)
        t = QtWidgets.QLabel("Mesurer la phase de tes prises")
        t.setObjectName("Title")
        v.addWidget(t)
        v.addWidget(muted("Trois étapes. Le signal est créé ici, joué et enregistré dans "
                          "Reaper, puis analysé ici.", wrap=True))
        v.addSpacing(8)

        def step(num, title, desc, buttons):
            fr = QtWidgets.QFrame()
            fr.setObjectName("Card")
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

        b1 = QtWidgets.QPushButton("Générer")
        b1.setObjectName("Primary")
        b1.clicked.connect(self.generate_signal)
        b1b = QtWidgets.QPushButton("Charger")
        b1b.clicked.connect(self.load_signal_dialog)
        self.w_step1 = step("1", "Créer le signal de test",
                            "Un sweep logarithmique est recommandé.", [b1, b1b])
        b2 = QtWidgets.QPushButton("Guide Reaper")
        b2.clicked.connect(self.show_guide)
        step("2", "Le jouer et l'enregistrer dans Reaper",
             "Signal posé à 0:00, une prise par position ou angle de micro.", [b2])
        self.w_b3 = QtWidgets.QPushButton("Importer")
        self.w_b3.setObjectName("Primary")
        self.w_b3.clicked.connect(self.import_dialog)
        step("3", "Importer les prises",
             "Ou glisse simplement les fichiers WAV dans la fenêtre.", [self.w_b3])
        h = QtWidgets.QHBoxLayout()
        h.addStretch()
        h.addWidget(col)
        h.addStretch()
        outer.addLayout(h)
        outer.addStretch(3)
        return w

    def _plot_area(self):
        glw = pg.GraphicsLayoutWidget()
        glw.ci.setContentsMargins(6, 10, 12, 6)
        glw.ci.setSpacing(6)
        return glw

    def _readout(self):
        lab = QtWidgets.QLabel(" ")
        lab.setObjectName("Readout")
        lab.setTextFormat(Qt.RichText)
        lab.setWordWrap(True)
        return lab

    def _build_resp_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        opts = QtWidgets.QHBoxLayout()
        opts.setContentsMargins(14, 6, 14, 0)
        self.gd_check = QtWidgets.QCheckBox("Retard de groupe")
        self.gd_check.setChecked(True)
        self.gd_check.toggled.connect(self.schedule)
        self.norm_check = QtWidgets.QCheckBox("Niveau normalisé à 1 kHz")
        self.norm_check.toggled.connect(self.schedule)
        opts.addWidget(self.norm_check)
        opts.addSpacing(16)
        opts.addWidget(self.gd_check)
        opts.addStretch()
        v.addLayout(opts)
        self.resp_glw = self._plot_area()
        self.p_mag = freq_plot("Niveau (dB)")
        self.p_ph = freq_plot("Phase (°)")
        self.p_gd = freq_plot("Retard de groupe (ms)")
        self.resp_glw.addItem(self.p_mag, 0, 0)
        self.resp_glw.addItem(self.p_ph, 1, 0)
        self.resp_glw.addItem(self.p_gd, 2, 0)
        for p in (self.p_ph, self.p_gd):
            p.setXLink(self.p_mag)
        v.addWidget(self.resp_glw, 1)
        self.resp_readout = self._readout()
        v.addWidget(self.resp_readout)
        self.resp_lines = []
        self.resp_glw.scene().sigMouseMoved.connect(
            lambda pos: self._freq_hover(pos, [self.p_mag, self.p_ph, self.p_gd], "resp"))
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
        self.p_time = lin_plot("Amplitude", "Temps (ms)")
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

        self.fit_lo = fspin(100, "Bande utilisée pour calculer retard, rotation et gain")
        self.fit_hi = fspin(10000, "Bande utilisée pour calculer retard, rotation et gain")
        opts.setSpacing(6)
        opts.addWidget(muted("Bande de calcul"))
        opts.addWidget(self.fit_lo)
        opts.addWidget(muted("–"))
        opts.addWidget(self.fit_hi)
        opts.addSpacing(18)
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(-20, 50); self.temp_spin.setValue(20); self.temp_spin.setSuffix(" °C")
        self.temp_spin.setFixedWidth(70)
        self.temp_spin.setToolTip("Température de l'air, pour convertir les retards en distance")
        self.temp_spin.valueChanged.connect(self.schedule)
        opts.addWidget(muted("Air"))
        opts.addWidget(self.temp_spin)
        opts.addStretch()
        copy = QtWidgets.QPushButton("Copier les réglages")
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
        self.p_dmag = freq_plot("Δ niveau (dB)")
        self.p_dph = freq_plot("Δ phase (°)")
        self.p_sum = freq_plot("Somme (dB)")
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
            ["Prise", "Retard (ms)", "Échantillons", "Distance (cm)", "Polarité",
             "Rotation (°)", "Gain (dB)", "Écart résiduel", "1er creux (somme)", "Correction à appliquer"])
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
            "Retard > 0 : la prise arrive après la référence.\n"
            "Rotation : déphasage constant restant une fois retard et polarité corrigés.\n"
            "Écart résiduel : ce que retard + rotation n'expliquent pas (° RMS dans la bande).")
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
        opts.addWidget(muted("Fréquences"))
        self.rot_freqs = QtWidgets.QLineEdit("125, 500, 1000, 4000, 10000")
        self.rot_freqs.setFixedWidth(260)
        self.rot_freqs.setToolTip("Fréquences suivies en fonction de l'angle, séparées par des virgules")
        self.rot_freqs.editingFinished.connect(self.schedule)
        opts.addWidget(self.rot_freqs)
        opts.addSpacing(14)
        self.rot_unwrap = QtWidgets.QCheckBox("Phase déroulée")
        self.rot_unwrap.setChecked(True)
        self.rot_unwrap.setToolTip("Déroulée : la phase cumule les tours depuis le grave (aucune ambiguïté).\n"
                                   "Décoche pour l'afficher entre −180° et +180°.")
        self.rot_unwrap.toggled.connect(self.schedule)
        opts.addWidget(self.rot_unwrap)
        opts.addSpacing(14)
        self.rot_info = muted("")
        opts.addWidget(self.rot_info, 1)
        v.addLayout(opts)
        self.rot_glw = self._plot_area()
        self.p_rph = lin_plot("Δ phase (°)", "Angle (°)")
        self.p_rmag = lin_plot("Δ niveau (dB)", "Angle (°)")
        self.p_rt = lin_plot("Δ temps (µs)", "Angle (°)")
        self.p_rmap = freq_plot("Angle (°)")
        self.p_rmap.setTitle("Δ phase selon fréquence et angle", color=C["muted"], size="9pt")
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
        m = mb.addMenu("Fichier")
        m.addAction("Nouveau projet", self.new_project, QtGui.QKeySequence.New)
        m.addAction("Ouvrir un projet…", self.open_project, QtGui.QKeySequence.Open)
        m.addAction("Enregistrer le projet", self.save_project, QtGui.QKeySequence.Save)
        m.addAction("Enregistrer sous…", lambda: self.save_project(True), QtGui.QKeySequence.SaveAs)
        m.addSeparator()
        m.addAction("Quitter", self.close, QtGui.QKeySequence.Quit)
        m = mb.addMenu("Signal")
        m.addAction("Générer un signal…", self.generate_signal, "Ctrl+G")
        m.addAction("Charger un signal…", self.load_signal_dialog)
        m = mb.addMenu("Prises")
        m.addAction("Importer des prises…", self.import_dialog, "Ctrl+I")
        m.addAction("Retirer la prise sélectionnée", self.remove_selected, QtGui.QKeySequence.Delete)
        m.addAction("Définir comme référence", self.set_selected_as_ref, "Ctrl+R")
        m.addAction("Tout afficher", lambda: self._set_all_visible(True))
        m.addAction("Tout masquer", lambda: self._set_all_visible(False))
        m = mb.addMenu("Aide")
        m.addAction("Guide", self.show_guide, QtGui.QKeySequence.HelpContents)

    # ------------------------------------------------------------- état / UI
    def status(self, text):
        self.status_msg.setText(text)

    def schedule(self, *_):
        self._timer.start()

    def _update_signal_card(self):
        s = self.signal
        if not s:
            self.sig_name.setText("Aucun signal")
            self.sig_info.setText("Génère ou charge le signal joué dans Reaper.")
            self.btn_gen.setObjectName("Primary")
        else:
            self.sig_name.setText(s.name)
            dur = len(s.data) / s.fs
            self.sig_info.setText(
                f"{s.label} · {s.fs} Hz · {dur:.2f} s\n"
                f"Bande {short_hz(round(s.f1, 1))} – {short_hz(round(s.f2))} Hz")
            self.btn_gen.setObjectName("")
        self.btn_gen.style().unpolish(self.btn_gen)
        self.btn_gen.style().polish(self.btn_gen)
        self.btn_import.setEnabled(s is not None)

    def _update_welcome(self):
        ok = self.signal is not None
        self.w_step1.setText("Créer le signal de test" + ("   ✓ " + self.signal.name if ok else ""))
        self.w_b3.setEnabled(ok)
        self.stack.setCurrentIndex(1 if self.meas else 0)

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
            name = m.name + ("  ★" if m is self.ref else "")
            it = QtWidgets.QTableWidgetItem(name)
            it.setToolTip(f"{m.path}\ncanal {m.channel + 1}" + (" · loopback" if m.loopback else ""))
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
                    it.setToolTip("Rapport signal/bruit faible : monte le niveau ou allonge le sweep")
                self.table.setItem(r, c, it)
            if m is self.sel:
                self.table.selectRow(r)
        self._updating = False

    def _rebuild_ref_combo(self):
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItem("Signal original")
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
        self._update_welcome()
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
                self.status("Angle invalide : saisis un nombre en degrés.")
            self._refresh_lists()

    def _table_selection(self):
        if self._updating:
            return
        rows = self.table.selectionModel().selectedRows()
        self.sel = self.meas[rows[0].row()] if rows else None
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

    # --------------------------------------------------------------- signal
    def generate_signal(self):
        dlg = GenerateDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        spec = dlg.spec()
        default = f"phaselab_{spec.kind}_{spec.fs // 1000}k_{spec.duration:g}s.wav"
        start = str(Path(self.signal.path).parent / default) if self.signal and self.signal.path else default
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer le signal", start, "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        with busy():
            x, meta = dsp.generate_signal(spec)
            sig = dsp.TestSignal(x, spec.fs, meta)
            sig.save(path)
        self.set_signal(sig)
        self.status(f"Signal enregistré : {Path(path).name}. Glisse-le à 0:00 dans Reaper.")

    def load_signal_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Charger le signal de test", "", "WAV (*.wav)")
        if path:
            self.load_signal(path)

    def load_signal(self, path):
        try:
            sig = dsp.TestSignal.load(path)
        except Exception as e:
            self.error("Impossible de lire le signal", str(e))
            return
        if sig.kind == "inconnu":
            self.status("Signal sans fichier .phaselab.json : bande d'analyse par défaut 20 Hz – 20 kHz.")
        self.set_signal(sig)

    def set_signal(self, sig):
        self.signal = sig
        self._update_signal_card()
        if self.meas:
            self.reload_measurements()
        self._update_welcome()
        lo, hi = max(sig.f1, 20), min(sig.f2, 0.45 * sig.fs)
        self.fit_lo.setValue(max(100, lo))
        self.fit_hi.setValue(min(10000, hi))
        self.schedule()

    # --------------------------------------------------------------- prises
    def import_dialog(self):
        if not self.signal:
            self.error("Signal manquant", "Génère ou charge d'abord le signal de test.")
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Importer des prises", "", "Audio (*.wav *.flac *.aif *.aiff)")
        if paths:
            self.import_takes(paths)

    def import_takes(self, paths):
        if not self.signal:
            self.error("Signal manquant", "Génère ou charge d'abord le signal de test.")
            return
        loop = self.loop_combo.currentData()
        existing = {(m.path, m.channel) for m in self.meas}
        was_empty = not self.meas
        errors, added = [], 0
        with busy():
            for p in paths:
                if self.signal.path and Path(p).resolve() == Path(self.signal.path).resolve():
                    continue
                try:
                    new = dsp.load_takes(p, self.signal, loop)
                except Exception as e:
                    errors.append(f"{Path(p).name} : {e}")
                    continue
                for m in new:
                    if (m.path, m.channel) in existing:
                        continue
                    m.color = self._next_color()
                    self.meas.append(m)
                    added += 1
                self.status(f"Analyse… {added} prise(s)")
                QtWidgets.QApplication.processEvents()
        if was_empty and self.meas:
            self.ref = self.meas[0]
            self.sel = self.meas[1] if len(self.meas) > 1 else self.meas[0]
        self._refresh_lists()
        self.status(f"{added} prise(s) importée(s).")
        if errors:
            self.error("Certaines prises n'ont pas pu être lues", "\n".join(errors))

    def reload_measurements(self):
        """Recalcule toutes les prises (après changement de signal ou de loopback)."""
        old = {(m.path, m.channel): m for m in self.meas}
        ref_key = (self.ref.path, self.ref.channel) if self.ref else None
        sel_key = (self.sel.path, self.sel.channel) if self.sel else None
        paths = list(dict.fromkeys(m.path for m in self.meas))
        loop = self.loop_combo.currentData()
        new, errors = [], []
        with busy():
            for p in paths:
                try:
                    for m in dsp.load_takes(p, self.signal, loop):
                        o = old.get((m.path, m.channel))
                        if o:
                            m.name, m.angle, m.visible, m.color = o.name, o.angle, o.visible, o.color
                        else:
                            m.color = self._next_color()
                        new.append(m)
                except Exception as e:
                    errors.append(f"{Path(p).name} : {e}")
        self.meas = new
        self._spec_cache.clear()
        self._orig_cache.clear()
        self.ref = next((m for m in new if (m.path, m.channel) == ref_key), None)
        self.sel = next((m for m in new if (m.path, m.channel) == sel_key), None)
        self._refresh_lists()
        if errors:
            self.error("Certaines prises n'ont pas pu être recalculées", "\n".join(errors))

    def _loop_changed(self):
        if self.meas and self.signal:
            self.reload_measurements()

    def remove_selected(self):
        if self.sel in self.meas:
            self.meas.remove(self.sel)
            self._spec_cache.pop(id(self.sel), None)
            self.sel = None
            self._refresh_lists()

    # ------------------------------------------------------------- glisser
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if Path(p).suffix.lower() in (".wav", ".flac", ".aif", ".aiff")]
        if not paths:
            return
        sigs = [p for p in paths if dsp.sidecar_path(p).exists()]
        takes = [p for p in paths if p not in sigs]
        if sigs:
            self.load_signal(sigs[0])
        if takes:
            self.import_takes(takes)

    # ------------------------------------------------------------- analyse
    def analyze(self):
        s = self.signal
        fs = s.fs
        fmin = max(s.f1, 10.0)
        fmax = min(s.f2, 0.48 * fs)
        grid = dsp.log_grid(fmin, fmax, 1500)
        pre, post = self.pre_spin.value(), self.post_spin.value()
        N = dsp.fft_size(fs, pre, post)
        specs = {}
        for m in self.meas:
            key = (pre, post, N)
            cached = self._spec_cache.get(id(m))
            if not cached or cached[0] != key or cached[2] is not m:
                self._spec_cache[id(m)] = (key, dsp.windowed_spectrum(m, pre, post, N), m)
            specs[id(m)] = self._spec_cache[id(m)][1]
        f = next(iter(specs.values())).f
        sm = dsp.Smoother(f, grid, self.smooth_combo.currentData())
        curves = {k: dsp.make_curve(sp, sm) for k, sp in specs.items()}
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
        self.A = dict(grid=grid, specs=specs, curves=curves, comps=comps, band=(lo, hi),
                      wrapped=self.wrap_combo.currentIndex() == 0,
                      dmode=self.delay_combo.currentData(), fs=fs)

    def refresh_all(self):
        if not self.signal or not self.meas:
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)
        try:
            with busy():
                self.analyze()
                self.draw_response()
                self.draw_time()
                self.draw_diff()
                self.draw_rotation()
        except Exception as e:  # garde l'appli vivante quoi qu'il arrive
            import traceback
            traceback.print_exc()
            self.status(f"Erreur d'analyse : {e}")

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

    # -- Réponse
    def draw_response(self):
        A = self.A
        grid = A["grid"]
        for p in (self.p_mag, self.p_ph, self.p_gd):
            p.clear()
        self.p_gd.setVisible(self.gd_check.isChecked())
        ref_tp = A["curves"][id(self.ref)].tp if self.ref else 0.0
        k1 = int(np.argmin(np.abs(grid - 1000)))
        self.resp_data = []
        for m in self.visible():
            c = A["curves"][id(m)]
            tau_rm = {"none": 0.0, "common": ref_tp, "indiv": c.tp}[A["dmode"]]
            mag = c.mag_db
            if self.norm_check.isChecked():
                mag = mag - mag[k1]
            ph = c.phase_res_deg() - 360 * grid * (c.tp - tau_rm)
            gd = dsp.group_delay_ms(c, grid, c.tp - tau_rm)
            w = 2.0 if m is self.sel else 1.3
            self.p_mag.plot(grid, mag, pen=self._pen(m.color, w))
            self.p_ph.plot(grid, self._phase(ph), pen=self._pen(m.color, w), connect="finite")
            if self.gd_check.isChecked():
                self.p_gd.plot(grid, gd, pen=self._pen(m.color, w))
            self.resp_data.append((m, mag, dsp.wrap_deg(ph) if A["wrapped"] else ph, gd))
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
        self.resp_lines = self._cursor_lines([self.p_mag, self.p_ph, self.p_gd])
        self.p_mag.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self.resp_readout.setText(f"<span style='color:{C['muted']}'>Survole un graphique pour lire les valeurs.</span>")

    # -- Temps
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
            p.setLabel("left", "Amplitude (normalisée)" if mode == "ir" else "Niveau (dB)")
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
            lab = "Fenêtre d'analyse affichée pour : " + focus.name
            pk = " · ".join(f"{m.name} {m.peak_time * 1e3:.3f} ms" for m in vis[:3])
            self.time_info.setText(f"{lab}   ·   Pics : {pk}{' …' if len(vis) > 3 else ''}")
        else:
            m = self.sel if self.sel in vis else vis[0]
            key = (id(m), mode)
            if key not in self._orig_cache:
                self._orig_cache[key] = dsp.original_vs_take(m, mode)
            pred, res, g, rd = self._orig_cache[key]
            t = np.arange(len(m.y)) / m.fs * 1e3
            p.plot(t, m.y, pen=self._pen(m.color, 1.0), name="Prise")
            p.plot(t, pred, pen=self._pen(C["text"], 0.8), name="Original")
            p.plot(t, res, pen=self._pen(C["bad"], 0.8), name="Résidu")
            p.setLabel("left", "Amplitude (FS)")
            p.enableAutoRange()
            src = "boucle" if m.loopback else "original"
            if mode == "aligned":
                self.time_info.setText(
                    f"{m.name} · {src} décalé de {m.peak_time * 1e3:.3f} ms et mis à l'échelle "
                    f"({20 * np.log10(abs(g) + 1e-20):+.2f} dB{', polarité inversée' if g < 0 else ''}) · "
                    f"résidu {rd:.1f} dB  (blanc : original · rouge : résidu)")
            else:
                self.time_info.setText(
                    f"{m.name} · {src} filtré par la réponse mesurée · résidu (bruit + distorsion) "
                    f"{rd:.1f} dB sous la prise")

    # -- Différences
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
                f"Somme {self.ref.name} + {target.name} — couleur : brute · "
                f"<span style='color:{C['accent']}'>vert : corrigée</span> · pointillé : idéal (+6 dB)",
                color=C["muted"], size="9pt")
            self.sum_target = (target, c)
        else:
            self.p_sum.setTitle("Choisis une prise comme référence pour voir la sommation",
                                color=C["faint"], size="9pt")
        self.diff_lines = self._cursor_lines([self.p_dmag, self.p_dph, self.p_sum])
        self.p_dmag.setXRange(math.log10(grid[0]), math.log10(grid[-1]), padding=0.01)
        self._fill_settings_table()
        lo, hi = A["band"]
        ref = self.ref.name if self.ref else "le signal original"
        self.diff_readout.setText(
            f"<span style='color:{C['muted']}'>Référence : {ref} · bande de calcul "
            f"{short_hz(round(lo))}–{short_hz(round(hi))} Hz (zone teintée)</span>")

    def settings_rows(self):
        A = self.A
        fs = A["fs"]
        c_air = dsp.speed_of_sound(self.temp_spin.value())
        rows = []
        for m in self.meas:
            if m is self.ref:
                rows.append([m.name, "0", "0", "0", "—", "—", "—", "—", "—", "Référence"])
                continue
            c = A["comps"][id(m)]
            tau_ms = c.tau * 1e3
            parts = []
            if abs(tau_ms) >= 0.0005:
                parts.append(f"{'avancer' if tau_ms > 0 else 'retarder'} de {abs(tau_ms):.3f} ms "
                             f"({abs(c.tau) * fs:.1f} éch.)")
            if c.inverted:
                parts.append("inverser la polarité")
            if abs(c.rotation) >= 1:
                parts.append(f"rotation de phase {-c.rotation:+.0f}°")
            if abs(c.gain_db) >= 0.1:
                parts.append(f"gain {-c.gain_db:+.1f} dB")
            corr = ", ".join(parts) or "aucune"
            corr = corr[0].upper() + corr[1:]
            rows.append([
                m.name, f"{tau_ms:+.3f}", f"{c.tau * fs:+.2f}", f"{c.tau * c_air * 100:+.2f}",
                "inversée" if c.inverted else "normale", f"{nz(c.rotation):+.1f}", f"{nz(c.gain_db):+.2f}",
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
                if c == 4 and txt == "inversée":
                    it.setForeground(QtGui.QColor(C["warn"]))
                if c == 7 and m is not self.ref:
                    if self.A["comps"][id(m)].resid_rms > 20:
                        it.setForeground(QtGui.QColor(C["warn"]))
                        it.setToolTip("Retard + rotation n'expliquent pas tout : la différence "
                                      "dépend de la fréquence (filtrage, réflexions, directivité).")
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
        self.status("Réglages copiés dans le presse-papiers.")

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
            # phase déroulée en fréquence (depuis le grave) : pas d'ambiguïté de
            # tour entre deux angles, même quand le retard change beaucoup
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
            self.rot_info.setText("Aucune prise visible n'a d'angle : saisis-le dans la colonne « Angle ».")
            return
        ref = self.ref.name if self.ref else "l'original"
        tip = "" if self.A["dmode"] == "indiv" else \
            " · astuce : Affichage › « Retard propre retiré » isole la rotation de phase du retard"
        self.rot_info.setText(f"{len(rows)} prise(s) avec angle · écarts par rapport à {ref}{tip}")
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

        # carte angle × fréquence
        grid = self.A["grid"]
        self.p_rmap.setTitle("Δ phase selon fréquence et angle", color=C["muted"], size="9pt")
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
            self.p_rmap.setTitle("Au moins deux angles nécessaires", color=C["faint"], size="9pt")

    # ------------------------------------------------------------ lecture souris
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
                parts.append(f"somme {np.interp(x, lg, c.sum_before):+.1f} → "
                             f"<span style='color:{C['accent']}'>{np.interp(x, lg, c.sum_after):+.1f} dB</span>")
            self.diff_readout.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _time_hover(self, pos):
        p = self.p_time
        if not p.sceneBoundingRect().contains(pos) or not self.A:
            return
        pt = p.vb.mapSceneToView(pos)
        fs = self.A["fs"]
        self.cursor_lab.setText(f"t = {pt.x():.4f} ms  ({pt.x() * 1e-3 * fs:.1f} éch.)   y = {pt.y():.4g}")

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
            self.error("Rien à exporter", "Importe d'abord des prises.")
            return False
        return True

    def export_response_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Exporter les courbes de réponse", "reponses.csv", "CSV (*.csv)")
        if not path:
            return
        hdr, cols = ["Fréquence (Hz)"], [self.A["grid"]]
        for m, mag, ph, gd in self.resp_data:
            hdr += [f"{m.name} niveau (dB)", f"{m.name} phase (°)", f"{m.name} retard de groupe (ms)"]
            cols += [mag, ph, gd]
        self._write_csv(path, hdr, cols)
        self.status(f"Exporté : {Path(path).name}")

    def export_diff_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Exporter les différences", "differences.csv", "CSV (*.csv)")
        if not path:
            return
        hdr, cols = ["Fréquence (Hz)"], [self.A["grid"]]
        for m, dmag, dph in self.diff_data:
            hdr += [f"{m.name} Δ niveau (dB)", f"{m.name} Δ phase (°)"]
            cols += [dmag, dph]
        if self.sum_target:
            t, c = self.sum_target
            hdr += [f"Somme brute {t.name} (dB)", f"Somme corrigée {t.name} (dB)"]
            cols += [c.sum_before, c.sum_after]
        self._write_csv(path, hdr, cols)
        self.status(f"Exporté : {Path(path).name}")

    def export_settings_csv(self):
        if not self._need_data():
            return
        path = self._ask_save("Exporter les réglages", "reglages.csv", "CSV (*.csv)")
        if not path:
            return
        hdr = [self.set_table.horizontalHeaderItem(i).text() for i in range(self.set_table.columnCount())]
        rows = self.settings_rows()
        self._write_csv(path, hdr, list(zip(*rows)) if rows else [[] for _ in hdr])
        self.status(f"Exporté : {Path(path).name}")

    def export_eq(self):
        if not self._need_data():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier pour les courbes de correction")
        if not folder:
            return
        n = 0
        for m, dmag, dph in self.diff_data:
            corr = np.clip(-dmag, -20, 20)
            p = Path(folder) / f"{safe_name(m.name)}_correction_EQ.txt"
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"* PhaseLab - correction de niveau pour {m.name} vers "
                         f"{self.ref.name if self.ref else 'original'}\n* Fréquence(Hz) Gain(dB)\n")
                for f, g in zip(self.A["grid"][::3], corr[::3]):
                    fh.write(f"{f:.2f} {g:.3f}\n")
            n += 1
        self.status(f"{n} courbe(s) de correction exportée(s) (format REW / égaliseurs à courbe).")

    def export_rotation_csv(self):
        if not self._need_data():
            return
        freqs, rows = self.rot_rows
        if not rows:
            self.error("Rien à exporter", "Aucune prise visible n'a d'angle.")
            return
        path = self._ask_save("Exporter la rotation", "rotation.csv", "CSV (*.csv)")
        if not path:
            return
        hdr = ["Angle (°)", "Prise", "Δ temps (µs)"]
        for f in freqs:
            hdr += [f"Δ phase {short_hz(f)} Hz (°)", f"Δ niveau {short_hz(f)} Hz (dB)"]
        cols = [[r["angle"] for r in rows], [r["m"].name for r in rows], [r["tau"] * 1e6 for r in rows]]
        for j in range(len(freqs)):
            cols += [[r["ph"][j] for r in rows], [r["mag"][j] for r in rows]]
        self._write_csv(path, hdr, cols)
        self.status(f"Exporté : {Path(path).name}")

    def export_irs(self):
        if not self._need_data():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier pour les réponses impulsionnelles")
        if not folder:
            return
        vis = self.visible()
        segs = [m.ir[int(round(-m.t0 * m.fs)):] for m in vis]
        peak = max(np.max(np.abs(s)) for s in segs) if segs else 1
        scale = 0.9 / peak if peak > 0.9 else 1.0
        for m, s in zip(vis, segs):
            sf.write(str(Path(folder) / f"{safe_name(m.name)}_IR.wav"), s * scale, m.fs, subtype="FLOAT")
        self.status(f"{len(vis)} réponse(s) impulsionnelle(s) exportée(s), temps 0 = début du signal"
                    + (f", gain commun {20 * math.log10(scale):.1f} dB" if scale != 1 else ""))

    def export_png(self):
        w = self.tabs.currentWidget()
        name = self.tabs.tabText(self.tabs.currentIndex()).lower()
        path = self._ask_save("Exporter l'image", f"phaselab_{name}.png", "PNG (*.png)")
        if path:
            w.grab().save(path)
            self.status(f"Image enregistrée : {Path(path).name}")

    # ----------------------------------------------------------------- projet
    def new_project(self):
        self.meas, self.ref, self.sel, self.signal, self.A = [], None, None, None, None
        self.project_path = None
        self._spec_cache.clear()
        self._orig_cache.clear()
        self._update_signal_card()
        self._refresh_lists()

    def save_project(self, ask=False):
        if not self.signal:
            self.error("Projet vide", "Rien à enregistrer pour l'instant.")
            return
        path = self.project_path
        if ask or not path:
            path = self._ask_save("Enregistrer le projet", "mesures.phaselab", "Projet PhaseLab (*.phaselab)")
            if not path:
                return
        data = dict(
            app=APP, version=VERSION, signal=self.signal.path,
            loop_channel=self.loop_combo.currentData(),
            ref=self.meas.index(self.ref) if self.ref in self.meas else -1,
            settings=dict(smoothing=self.smooth_combo.currentIndex(), wrap=self.wrap_combo.currentIndex(),
                          delay=self.delay_combo.currentIndex(), pre=self.pre_spin.value(),
                          post=self.post_spin.value(), fit_lo=self.fit_lo.value(),
                          fit_hi=self.fit_hi.value(), temp=self.temp_spin.value(),
                          rot_freqs=self.rot_freqs.text()),
            measurements=[dict(path=m.path, channel=m.channel, name=m.name, angle=m.angle,
                               visible=m.visible, color=m.color) for m in self.meas])
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.project_path = path
        self.setWindowTitle(f"{APP} — {Path(path).stem}")
        self.status(f"Projet enregistré : {Path(path).name}")

    def open_project(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ouvrir un projet", "",
                                                            "Projet PhaseLab (*.phaselab)")
            if not path:
                return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            self.error("Projet illisible", str(e))
            return
        st = data.get("settings", {})
        for w, k in ((self.smooth_combo, "smoothing"), (self.wrap_combo, "wrap"), (self.delay_combo, "delay")):
            if k in st:
                w.setCurrentIndex(st[k])
        for w, k in ((self.pre_spin, "pre"), (self.post_spin, "post"), (self.temp_spin, "temp")):
            if k in st:
                w.setValue(st[k])
        if "rot_freqs" in st:
            self.rot_freqs.setText(st["rot_freqs"])
        self.loop_combo.blockSignals(True)
        self.loop_combo.setCurrentIndex(max(self.loop_combo.findData(data.get("loop_channel", 0)), 0))
        self.loop_combo.blockSignals(False)
        self.meas = []
        if not data.get("signal") or not Path(data["signal"]).exists():
            self.error("Signal introuvable", f"Le fichier du signal est introuvable :\n{data.get('signal')}")
            return
        self.load_signal(data["signal"])
        for k in ("fit_lo", "fit_hi"):
            if k in st:
                getattr(self, k).setValue(st[k])
        items = data.get("measurements", [])
        paths = list(dict.fromkeys(it["path"] for it in items))
        missing = [p for p in paths if not Path(p).exists()]
        self.import_takes([p for p in paths if Path(p).exists()])
        lookup = {(it["path"], it["channel"]): it for it in items}
        kept = []
        for m in self.meas:
            it = lookup.get((m.path, m.channel))
            if it:
                m.name, m.angle, m.visible = it["name"], it["angle"], it["visible"]
                m.color = it.get("color") or m.color
                kept.append(m)
        order = {(it["path"], it["channel"]): i for i, it in enumerate(items)}
        self.meas = sorted(kept, key=lambda m: order[(m.path, m.channel)])
        r = data.get("ref", -1)
        self.ref = self.meas[r] if 0 <= r < len(self.meas) else None
        self.project_path = path
        self.setWindowTitle(f"{APP} — {Path(path).stem}")
        self._refresh_lists()
        if missing:
            self.error("Prises introuvables", "\n".join(missing))

    # ----------------------------------------------------------------- divers
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
