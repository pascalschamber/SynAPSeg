"""
Version Checker Module for PyQt6 Application.

Provides background check against PyPI JSON API and non-blocking update prompts.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from packaging.version import parse as parse_version
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


class VersionCheckWorker(QThread):
    """
    Background worker thread querying PyPI JSON API to check for available updates.
    """

    update_available = pyqtSignal(str, str)  # (current_version, latest_version)
    check_failed = pyqtSignal(str)  # error_message

    def __init__(
        self,
        package_name: str = "SynAPSeg",
        timeout: float = 3.0,
        parent=None,
    ) -> None:
        if parent is not None and isinstance(parent, QObject):
            super().__init__(parent)
        else:
            super().__init__()
        self.package_name = package_name
        self.timeout = timeout

    def run(self) -> None:
        # 1. Resolve local version
        try:
            current = version(self.package_name)
        except PackageNotFoundError:
            self.check_failed.emit(f"Package '{self.package_name}' not found locally.")
            return
        except Exception as e:
            self.check_failed.emit(f"Failed to resolve version for '{self.package_name}': {e}")
            return

        # 2. Construct request to PyPI endpoint with a custom User-Agent
        url = f"https://pypi.org/pypi/{self.package_name}/json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"{self.package_name}-updater/{current}"},
        )

        # 3. Execute urlopen with configured timeout, catching errors silently
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_data = resp.read()
            data = json.loads(raw_data.decode("utf-8"))
            # 4. Extract latest version
            latest = data["info"]["version"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            self.check_failed.emit(f"PyPI check failed: {e}")
            return
        except Exception as e:
            self.check_failed.emit(f"Unexpected error querying PyPI: {e}")
            return

        # 5. Compare semantic versions
        try:
            if parse_version(latest) > parse_version(current):
                self.update_available.emit(current, latest)
        except Exception as e:
            self.check_failed.emit(f"Failed comparing versions ('{current}' vs '{latest}'): {e}")


def check_for_updates_on_startup(
    parent_window: QWidget | None,
    package_name: str = "SynAPSeg",
    timeout: float = 3.0,
) -> VersionCheckWorker:
    """
    Launch a background update check on application startup.

    If an update is available, prompts the user via a modal QMessageBox.
    If confirmed, opens the PipUpgradeDialog.
    """
    from SynAPSeg.updater.updater_dialog import PipUpgradeDialog

    worker = VersionCheckWorker(package_name=package_name, timeout=timeout, parent=parent_window)

    # Assign worker to parent_window instance attribute to prevent GC
    if parent_window is not None:
        parent_window._version_worker = worker

    def on_update_available(current_version: str, latest_version: str) -> None:
        msg_box = QMessageBox(parent_window)
        msg_box.setWindowTitle("Update Available")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(
            f"A new version of {package_name} is available!\n\n"
            f"Installed version: {current_version}\n"
            f"Latest version: {latest_version}\n\n"
            "Would you like to update now?"
        )
        update_btn = msg_box.addButton("Update Now", QMessageBox.ButtonRole.AcceptRole)
        remind_btn = msg_box.addButton("Remind Me Later", QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(update_btn)

        msg_box.exec()

        if msg_box.clickedButton() == update_btn:
            dialog = PipUpgradeDialog(parent=parent_window, package_name=package_name)
            dialog.exec()

    worker.update_available.connect(on_update_available)
    worker.start()
    return worker
