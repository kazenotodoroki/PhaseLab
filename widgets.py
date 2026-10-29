"""Shared UI building blocks used by every tab."""
from __future__ import annotations

import math
import re
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from style import C


def fmt_hz(f):
    return f"{f:.1f} Hz" if f < 1000 else f"{f / 1000:.3f} kHz"


def nz(v, eps=0.05):
    """Avoids displaying "-0.0"."""
    return 0.0 if abs(v) < eps else v


def short_hz(f):
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


def safe_name(s):
    return re.sub(r"[^\w\-. ]+", "_", s).strip() or "measurement"


def break_wraps(y, limit=180.0):
    """Inserts NaNs at phase wraps so that no vertical lines are drawn."""
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
    """Log frequency axis with readable ticks (20, 50, 100, 1k…)."""
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
    for ax in ("left", "bottom"):
        p.getAxis(ax).enableAutoSIPrefix(False)
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


def pen(color, width=1.4, style=None):
    p = pg.mkPen(color, width=width)
    if style:
        p.setStyle(style)
    return p


def plot_area():
    glw = pg.GraphicsLayoutWidget()
    glw.ci.setContentsMargins(6, 10, 12, 6)
    glw.ci.setSpacing(6)
    return glw


def readout():
    lab = QtWidgets.QLabel(" ")
    lab.setObjectName("Readout")
    lab.setTextFormat(Qt.RichText)
    lab.setWordWrap(True)
    return lab


def dspin(lo, hi, val, dec=1, suffix="", width=None, step=None, slot=None, tip=None):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    if step:
        s.setSingleStep(step)
    if width:
        s.setFixedWidth(width)
    if tip:
        s.setToolTip(tip)
    if slot:
        s.valueChanged.connect(slot)
    return s


def options_row():
    h = QtWidgets.QHBoxLayout()
    h.setContentsMargins(14, 6, 14, 0)
    h.setSpacing(6)
    return h


def tab_container():
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    return w, v


def small_table(headers, stretch_col=None):
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    t.setShowGrid(False)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    hh = t.horizontalHeader()
    for c in range(len(headers)):
        hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
    if stretch_col is not None:
        hh.setSectionResizeMode(stretch_col, QtWidgets.QHeaderView.Stretch)
    return t


def cell(text, align_right=False, color=None, tip=None):
    it = QtWidgets.QTableWidgetItem(text)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if color:
        it.setForeground(QtGui.QColor(color))
    if tip:
        it.setToolTip(tip)
    return it


def card():
    fr = QtWidgets.QFrame()
    fr.setObjectName("Card")
    return fr
