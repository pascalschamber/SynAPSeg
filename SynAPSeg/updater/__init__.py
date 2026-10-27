"""
Updater package for SynAPSeg.
"""

from .updater_dialog import (
    MODULE_NAME,
    PACKAGE_NAME,
    PRESERVED_CONFIG_FILES,
    PipUpgradeDialog,
    find_package_root,
)
from .version_checker import VersionCheckWorker, check_for_updates_on_startup

__all__ = [
    "MODULE_NAME",
    "PACKAGE_NAME",
    "PRESERVED_CONFIG_FILES",
    "PipUpgradeDialog",
    "VersionCheckWorker",
    "check_for_updates_on_startup",
    "find_package_root",
]
