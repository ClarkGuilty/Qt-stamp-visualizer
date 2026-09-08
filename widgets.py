# This Python file uses the following encoding: utf-8
"""Shared, generic Qt widgets used by both mosaic.py and single_viewer.py.
"""

import copy
import glob
import json
import os
import re
from functools import partial
from os.path import basename, join, splitext

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIntValidator


class AlignDelegate(QtWidgets.QStyledItemDelegate):
    "https://stackoverflow.com/a/54262963/10555034"
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class ClickableComboBox(QtWidgets.QComboBox):
    "QComboBox that opens its dropdown on a click anywhere in its body, not just the arrow."
    def mousePressEvent(self, event):
        self.showPopup()
        super().mousePressEvent(event)


class LabelledIntField(QtWidgets.QWidget):
    "Widget for the page number."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value,  total_pages):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.fontsize = 18
        self.label = QtWidgets.QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial",self.fontsize,weight=QFont.Bold))
        layout.addWidget(self.label)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setFocusPolicy(Qt.ClickFocus)
        self.lineEdit.setFixedWidth(50)
        self.lineEdit.setValidator(QIntValidator(1,total_pages))
        self.lineEdit.setText(str(initial_value+1))
        self.lineEdit.setFont(QFont("Arial",self.fontsize))
        self.lineEdit.setStyleSheet('background-color: black; color: gray')
        self.lineEdit.setAlignment(Qt.AlignRight)
        layout.addWidget(self.lineEdit)

        self.total_pages = QtWidgets.QLabel()
        self.total_pages.setText("/ "+str(total_pages))
        self.total_pages.setFont(QFont("Arial",self.fontsize))
        layout.addWidget(self.total_pages)

    def setInputText(self, input):
        self.lineEdit.setText(str(input+1))

    def getValue(self):
        return int(self.lineEdit.text())-1


class NamedLabel(QtWidgets.QWidget):
    "Widget to show unclickable label."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.fontsize = 18

        self.name = QtWidgets.QLabel()
        self.name.setText(title)
        self.name.setFont(QFont("Arial",self.fontsize,weight=QFont.Bold))
        layout.addWidget(self.name)

        self.label = QtWidgets.QLineEdit(self)
        self.label.setFixedWidth(50)
        self.label.setEnabled(False)
        self.label.setText(str(initial_value))
        self.label.setFont(QFont("Arial",self.fontsize))
        self.label.setStyleSheet('background-color: black; color: gray')
        self.label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label)

    def setText(self, input):
        self.label.setText(str(input))

    def getValue(self):
        return int(self.lineEdit.text())-1


def add_checkable_menu_action(menu, label, checked=False):
    "Adds a plain checkable item (QWidgetAction-wrapped QCheckBox) to a QMenu, returns the checkbox."
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(6,2,6,2)
    checkbox = QtWidgets.QCheckBox(label)
    checkbox.setChecked(checked)
    layout.addWidget(checkbox)
    action = QtWidgets.QWidgetAction(menu)
    action.setDefaultWidget(widget)
    menu.addAction(action)
    return checkbox


class CheckableSubMenu(QtWidgets.QMenu):
    "One checkable-item list inside a dropdown menu (or submenu)."
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self._checkboxes = {}

    def add_check_item(self, key, label, checked=False):
        checkbox = add_checkable_menu_action(self, label, checked)
        self._checkboxes[key] = checkbox
        return checkbox

    def checked_keys(self):
        return [key for key, checkbox in self._checkboxes.items() if checkbox.isChecked()]

    def set_checked(self, key, checked):
        checkbox = self._checkboxes[key]
        checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        checkbox.blockSignals(False)


class PanelOrderPicker(QtWidgets.QWidget):
    "Dropdown letting the user choose which of the fixed per-thumbnail panels are shown."
    selectionChanged = Signal()

    def __init__(self, panel_keys, panel_labels, parent=None):
        super().__init__(parent)
        self.panel_keys = panel_keys
        self.button = QtWidgets.QToolButton()
        self.button.setText("Panels")
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = CheckableSubMenu("Panels", self.button)
        self.button.setMenu(self.menu)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.button)
        for panel_key in panel_keys:
            checkbox = self.menu.add_check_item(panel_key, panel_labels[panel_key], checked=True)
            checkbox.toggled.connect(lambda checked: self.selectionChanged.emit())

    def selected_panels(self):
        "Currently-checked panels, in the fixed canonical order."
        checked = self.menu.checked_keys()
        return [key for key in self.panel_keys if key in checked]


class PanelRowPicker(QtWidgets.QWidget):
    "Dropdown letting the user assign fixed panels to display rows, one row per menu."
    rowChanged = Signal(str, str, bool) #row_key, panel_key, checked

    def __init__(self, panel_keys, panel_labels, n_rows, parent=None):
        super().__init__(parent)
        self.panel_keys = panel_keys
        self.row_keys = [f"row_{i+1}" for i in range(n_rows)]
        self.button = QtWidgets.QToolButton()
        self.button.setText("Panels")
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = QtWidgets.QMenu(self.button)
        self.button.setMenu(self.menu)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.button)

        self.rows = {}
        for row_key in self.row_keys:
            submenu = CheckableSubMenu(row_key.replace('_',' ').title(), self.menu)
            self.menu.addMenu(submenu)
            for panel_key in panel_keys:
                checkbox = submenu.add_check_item(panel_key, panel_labels[panel_key])
                checkbox.toggled.connect(partial(self._on_toggled, row_key, panel_key))
            self.rows[row_key] = submenu

    def _on_toggled(self, row_key, panel_key, checked):
        if checked:
            for other_row_key, submenu in self.rows.items():
                if other_row_key != row_key:
                    submenu.set_checked(panel_key, False)
        self.rowChanged.emit(row_key, panel_key, checked)

    def row_panels(self, row_key):
        return self.rows[row_key].checked_keys()

    def set_row_panels(self, row_key, panel_keys):
        submenu = self.rows[row_key]
        for panel_key in self.panel_keys:
            submenu.set_checked(panel_key, panel_key in panel_keys)

    def add_toggle(self, label, checked=False):
        "Adds a plain (non-row) checkable setting to the bottom of the menu, e.g. 'Large FoV'."
        self.menu.addSeparator()
        return add_checkable_menu_action(self.menu, label, checked)


class SettingsMenu(QtWidgets.QWidget):
    "Dropdown of independent checkable settings that aren't mutually exclusive (no rows)."
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.button = QtWidgets.QToolButton()
        self.button.setText(title)
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = QtWidgets.QMenu(self.button)
        self.button.setMenu(self.menu)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.button)

    def add_toggle(self, label, checked=False):
        return add_checkable_menu_action(self.menu, label, checked)


def join_nested(lines, sep=' | ', line_sep='\n'):
    return line_sep.join(sep.join(map(str, sub)) for sub in lines)


class BandNamesLabel(QtWidgets.QLabel):
    def updateText(self, status_plot_rows):
        self.setText(join_nested(status_plot_rows))


def _sanitize_preset_name(name):
    "Strips path separators so a preset name can't escape its directory."
    return re.sub(r'[\\/]+', '_', name.strip())


class PredefinedConfigBar(QtWidgets.QWidget):
    """Row of controls for saving/loading named presets of a tool's config dict.

    Presets are plain JSON files under `directory`, one per name. Picking one
    from the dropdown immediately applies it via `apply_config`; the config
    active right before that (fetched via `get_config`) is kept in memory so
    "Restore previous" can undo the swap without needing its own saved file.
    """

    PLACEHOLDER = "(current, unsaved)"

    def __init__(self, directory, get_config, apply_config, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.get_config = get_config
        self.apply_config = apply_config
        self._pre_snapshot = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Preset:"))

        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(160)
        self.combo.setMaxVisibleItems(12)
        self.combo.activated.connect(self._on_activated)
        layout.addWidget(self.combo, 1)

        self.save_btn = QtWidgets.QPushButton("Save as...")
        self.save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self.save_btn)

        self.restore_btn = QtWidgets.QPushButton("Restore previous")
        self.restore_btn.setEnabled(False)
        self.restore_btn.setToolTip("Go back to the configuration held before a preset was loaded.")
        self.restore_btn.clicked.connect(self._on_restore_clicked)
        layout.addWidget(self.restore_btn)

        self.refresh()

    def refresh(self):
        "Rescans `directory` for *.json presets, preserving the current selection if still valid."
        current_name = self.combo.currentText() if self.combo.count() else None
        names = self._list_names()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(self.PLACEHOLDER)
        self.combo.addItems(names)
        self.combo.setCurrentText(current_name if current_name in names else self.PLACEHOLDER)
        self.combo.blockSignals(False)

    def _list_names(self):
        if not os.path.isdir(self.directory):
            return []
        return sorted(splitext(basename(f))[0]
                      for f in glob.glob(join(self.directory, "*.json")))

    def _on_activated(self, index):
        if index <= 0:
            return
        name = self.combo.itemText(index)
        path = join(self.directory, f"{name}.json")
        try:
            with open(path) as f:
                preset = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Load preset failed", f"Could not load '{name}':\n{exc}")
            self.refresh()
            return
        self._pre_snapshot = copy.deepcopy(self.get_config())
        self.restore_btn.setEnabled(True)
        self.apply_config(preset)

    def _on_restore_clicked(self):
        if self._pre_snapshot is None:
            return
        self.apply_config(self._pre_snapshot)
        self._pre_snapshot = None
        self.restore_btn.setEnabled(False)
        self.combo.blockSignals(True)
        self.combo.setCurrentText(self.PLACEHOLDER)
        self.combo.blockSignals(False)

    def _on_save_clicked(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok:
            return
        name = _sanitize_preset_name(name)
        if not name:
            QtWidgets.QMessageBox.warning(self, "Save preset failed", "Preset name can't be empty.")
            return
        path = join(self.directory, f"{name}.json")
        if os.path.exists(path):
            reply = QtWidgets.QMessageBox.question(
                self, "Overwrite preset?",
                f"A preset named '{name}' already exists. Overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        os.makedirs(self.directory, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_config(), f, ensure_ascii=False, indent=4)
        self.refresh()
        self.combo.blockSignals(True)
        self.combo.setCurrentText(name)
        self.combo.blockSignals(False)
