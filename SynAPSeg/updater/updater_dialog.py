"""
Updater Dialog Module for PyQt6 Application.

Provides a modal UI that displays warnings, preserves configuration files,
executes pip upgrade asynchronously via QProcess, restores configs, and prompts for restart.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Target Package Constants
PACKAGE_NAME = "SynAPSeg"
MODULE_NAME = "SynAPSeg"
PRESERVED_CONFIG_FILES = ["config.json", "config/user_settings.yaml"]


def find_package_root(module_name: str) -> Path | None:
    """
    Locate the root directory of a module or package using importlib.util.find_spec.
    Returns the parent path of origin (e.g., path/to/module for a package),
    or the first submodule search location, or None if spec cannot be resolved.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        if spec.origin:
            return Path(spec.origin).parent
        if spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0])
    except Exception:
        return None
    return None


class PipUpgradeDialog(QDialog):
    """
    Modal dialog executing pip upgrade with config backup and restoration.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        package_name: str = PACKAGE_NAME,
        module_name: str = MODULE_NAME,
        preserved_config_files: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.package_name = package_name
        self.module_name = module_name
        self.preserved_config_files = (
            list(preserved_config_files)
            if preserved_config_files is not None
            else list(PRESERVED_CONFIG_FILES)
        )

        self.temp_dir: str | None = None
        self.process: QProcess | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle(f"Update - {self.package_name}")
        self.resize(600, 420)
        self.setMinimumSize(520, 360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Status Label
        self.status_label = QLabel("Ready to update. Click 'Start Update' to begin.")
        self.status_label.setWordWrap(True)
        font = self.status_label.font()
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # Indeterminate Progress Bar (visible during pip execution)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Terminal Console (Read-only QTextEdit)
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setFontFamily("Consolas, Courier New, monospace")
        self.console.setStyleSheet(
            "QTextEdit { "
            "background-color: #1e1e1e; "
            "color: #d4d4d4; "
            "border: 1px solid #333333; "
            "border-radius: 4px; "
            "padding: 6px; "
            "font-size: 12px; "
            "}"
        )
        layout.addWidget(self.console)

        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.start_btn = QPushButton("Start Update", self)
        self.start_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #007acc; "
            "color: white; "
            "font-weight: bold; "
            "padding: 6px 14px; "
            "border-radius: 4px; "
            "} "
            "QPushButton:hover { background-color: #0098ff; } "
            "QPushButton:disabled { background-color: #555555; color: #888888; }"
        )
        self.start_btn.clicked.connect(self.start_update)
        button_layout.addWidget(self.start_btn)

        layout.addLayout(button_layout)

    def start_update(self) -> None:
        """
        Execute the full upgrade workflow:
        1. Pre-flight warning.
        2. Configuration backup.
        3. Asynchronous QProcess pip upgrade.
        """
        # 1. Pre-Flight Warning Dialog
        warning_text = (
            "Running an automated update will overwrite installed source files. "
            "Any direct modifications made to the source code will be permanently replaced by the new release. "
            "Personal configuration files will be automatically preserved. "
            "If you have modified the source code and wish to maintain changes, please cancel and use Git. "
            "Proceed with update?"
        )
        reply = QMessageBox.warning(
            self,
            "Confirm Update",
            warning_text,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            self.status_label.setText("Update cancelled by user.")
            return

        # 2. Configuration Backup Phase
        self.status_label.setText("Backing up local configuration files...")
        self.console.clear()
        self._append_console("Starting automated update...\n")

        self.temp_dir = tempfile.mkdtemp(prefix="synapseg_update_")
        self._append_console(f"Created temporary backup directory: {self.temp_dir}\n")

        pkg_root = find_package_root(self.module_name)
        if pkg_root and pkg_root.exists():
            self._append_console(f"Package root detected: {pkg_root}\n")
            for rel_str in self.preserved_config_files:
                src_file = pkg_root / rel_str
                if src_file.exists() and src_file.is_file():
                    dst_file = Path(self.temp_dir) / rel_str
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    self._append_console(f"Preserved config: {rel_str}\n")
                else:
                    self._append_console(f"Config not found on disk (skipped): {rel_str}\n")
        else:
            self._append_console(
                f"Notice: Package root for '{self.module_name}' could not be resolved from importlib.\n"
            )

        # 3. Subprocess Execution Phase (QProcess)
        self.status_label.setText(f"Upgrading '{self.package_name}' via pip...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_process_finished)

        cmd_args = ["-m", "pip", "install", "--upgrade", self.package_name]
        self._append_console(f"Executing: {sys.executable} {' '.join(cmd_args)}\n\n")
        self.process.start(sys.executable, cmd_args)

    def _on_stdout(self) -> None:
        if self.process:
            data = self.process.readAllStandardOutput().data()
            text = data.decode("utf-8", errors="replace")
            self._append_console(text)

    def _on_stderr(self) -> None:
        if self.process:
            data = self.process.readAllStandardError().data()
            text = data.decode("utf-8", errors="replace")
            self._append_console(text)

    def _append_console(self, text: str) -> None:
        self.console.insertPlainText(text)
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """
        4. Process Completion & Restoration Phase:
           Restores configuration files, cleans temporary storage, and triggers restart prompt.
        """
        if exit_code != 0:
            self._append_console(f"\n[ERROR] Pip process exited with failure code {exit_code}.\n")

            # Clean up temp_dir
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None

            # Re-enable UI
            self.status_label.setText("Update failed.")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)

            QMessageBox.critical(
                self,
                "Update Failed",
                f"Failed to update {self.package_name} (Exit code: {exit_code}).\n\n"
                "Please review the terminal console above for detailed error messages.",
            )
            return

        # Success (exit_code == 0)
        self._append_console("\nPip upgrade completed successfully (exit code 0).\n")
        self.status_label.setText("Restoring preserved configuration files...")

        # Re-resolve package root after installation
        importlib.invalidate_caches()
        pkg_root = find_package_root(self.module_name)

        if self.temp_dir and Path(self.temp_dir).exists():
            temp_path = Path(self.temp_dir)
            if pkg_root and pkg_root.exists():
                for rel_str in self.preserved_config_files:
                    backup_file = temp_path / rel_str
                    if backup_file.exists() and backup_file.is_file():
                        target_file = pkg_root / rel_str
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, target_file)
                        self._append_console(f"Restored configuration: {rel_str}\n")
            # Delete temp_dir via shutil.rmtree
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

        self.status_label.setText("Update complete.")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)

        # 5. Application Relaunch
        reply = QMessageBox.question(
            self,
            "Update Complete",
            f"{self.package_name} has been successfully updated!\n\n"
            "Would you like to restart the application now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            QProcess.startDetached(sys.executable, sys.argv)
            QApplication.quit()
        else:
            self.accept()

    def closeEvent(self, event) -> None:
        """
        Prevent closing while pip process is actively running.
        """
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            QMessageBox.warning(
                self,
                "Update in Progress",
                "The update process is currently running. Please wait for it to finish.",
            )
            event.ignore()
        else:
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None
            event.accept()
