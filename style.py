"""Thème sombre, minimal, angles quasi droits."""

C = dict(
    bg="#0c0e11",
    panel="#121519",
    panel2="#171b20",
    hover="#1d2228",
    border="#242a31",
    text="#d6dae0",
    muted="#7c8590",
    faint="#4a525c",
    accent="#38c9b2",
    accent_dim="#173f39",
    plot_bg="#0e1114",
    warn="#e3a857",
    bad="#e46f6f",
)

PALETTE = ["#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#ffd54f", "#4db6ac",
           "#f06292", "#9fa8da", "#e57373", "#81c784", "#ffb74d", "#90a4ae"]

QSS = f"""
* {{ outline: 0; }}
QWidget {{ background: {C['bg']}; color: {C['text']}; font-size: 12px; }}
QMainWindow::separator {{ background: {C['border']}; width: 1px; height: 1px; }}
QFrame#Sidebar {{ background: {C['panel']}; border-right: 1px solid {C['border']}; }}
QFrame#Sidebar QWidget {{ background: {C['panel']}; }}
QFrame#Toolbar {{ background: {C['bg']}; border-bottom: 1px solid {C['border']}; }}
QFrame#Card {{ background: {C['panel2']}; border: 1px solid {C['border']}; border-radius: 2px; }}
QFrame#Card QLabel {{ background: transparent; }}
QLabel#Section {{ color: {C['muted']}; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; }}
QLabel#Title {{ font-size: 20px; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#Muted {{ color: {C['muted']}; }}
QLabel#Value {{ font-family: "JetBrains Mono", "Consolas", "Menlo", monospace; }}
QLabel#Readout {{ background: {C['panel']}; border-top: 1px solid {C['border']};
                  padding: 6px 10px; font-family: "JetBrains Mono", "Consolas", "Menlo", monospace; }}

QPushButton {{ background: {C['panel2']}; border: 1px solid {C['border']}; border-radius: 2px;
              padding: 6px 12px; }}
QPushButton:hover {{ border-color: {C['accent']}; }}
QPushButton:pressed {{ background: {C['hover']}; }}
QPushButton:disabled {{ color: {C['faint']}; border-color: {C['border']}; }}
QPushButton#Primary {{ background: {C['accent']}; color: #06120f; border: 1px solid {C['accent']};
                      font-weight: 600; }}
QPushButton#Primary:hover {{ background: #4fd8c2; }}
QPushButton#Flat {{ background: transparent; border: none; color: {C['muted']}; padding: 4px 6px; }}
QPushButton#Flat:hover {{ color: {C['text']}; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {C['panel2']}; border: 1px solid {C['border']}; border-radius: 2px;
    padding: 3px 6px; min-height: 20px; selection-background-color: {C['accent_dim']}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {C['faint']}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {C['accent']}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px; border: none; }}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox::down-arrow {{ width: 0; height: 0; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {C['muted']}; margin-right: 6px; }}
QComboBox QAbstractItemView {{ background: {C['panel2']}; border: 1px solid {C['border']};
    selection-background-color: {C['accent_dim']}; padding: 2px; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ background: {C['bg']}; }}
QTabBar::tab {{ background: transparent; color: {C['muted']}; padding: 8px 16px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {C['text']}; }}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom: 2px solid {C['accent']}; }}

QTableWidget {{ background: {C['panel']}; border: 1px solid {C['border']}; gridline-color: {C['border']};
               selection-background-color: {C['accent_dim']}; selection-color: {C['text']}; }}
QTableWidget::item {{ padding: 2px 4px; }}
QHeaderView {{ background: {C['panel']}; }}
QHeaderView::section {{ background: {C['panel']}; color: {C['muted']}; border: none;
    border-bottom: 1px solid {C['border']}; padding: 4px 6px; font-size: 11px; }}
QTableCornerButton::section {{ background: {C['panel']}; border: none; }}

QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator, QTableWidget::indicator {{ width: 11px; height: 11px; border: 1px solid {C['faint']};
    background: {C['panel2']}; border-radius: 1px; }}
QCheckBox::indicator:checked, QTableWidget::indicator:checked {{ background: {C['accent']};
    border-color: {C['accent']}; }}

QMenuBar {{ background: {C['bg']}; border-bottom: 1px solid {C['border']}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {C['hover']}; }}
QMenu {{ background: {C['panel2']}; border: 1px solid {C['border']}; padding: 3px; }}
QMenu::item {{ padding: 5px 22px 5px 12px; }}
QMenu::item:selected {{ background: {C['accent_dim']}; }}
QMenu::separator {{ height: 1px; background: {C['border']}; margin: 3px 6px; }}

QStatusBar {{ background: {C['panel']}; border-top: 1px solid {C['border']}; color: {C['muted']}; }}
QStatusBar QLabel {{ background: transparent; color: {C['muted']}; }}
QSplitter::handle {{ background: {C['border']}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
QScrollBar::handle {{ background: {C['border']}; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {C['faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {C['panel2']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 4px; }}
QDialog {{ background: {C['panel']}; }}
QDialog QWidget {{ background: {C['panel']}; }}
QTextBrowser {{ background: {C['panel']}; border: none; }}
"""
