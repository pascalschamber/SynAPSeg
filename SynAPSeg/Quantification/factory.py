#!/usr/bin/env python3
import os
from typing import Any, Dict, List

from SynAPSeg.config import constants
from SynAPSeg.Plugins.base import BasePluginFactory

PLUGINS_DIRS = [
    os.path.join(constants.SYNAPSEG_BASE_DIR, 'Quantification', 'plugins'), # system plugins
    os.path.join(constants.SYNAPSEG_BASE_DIR, 'Plugins') # user plugins
]
PLUGINS_DEFAULT_PARAMETERS_PATH = os.path.join(PLUGINS_DIRS[0], "default_parameters.yaml") # global default params that specifc modules may override
PLUGIN_BASE_CLASS = 'BasePipelineStage'
REQUIRED_SIGNAL = {'__plugin_group__': 'quantification'}
CORE_PLUGINS = ['roi_handling', 'object_detection', 'colocalization'] # display in this order
PLUGIN_PATTERN = '.*\.py$' # if filename.endswith(".py")

QuantificationPluginFactory = BasePluginFactory(
    PLUGINS_DIRS, CORE_PLUGINS, PLUGINS_DEFAULT_PARAMETERS_PATH, PLUGIN_PATTERN, REQUIRED_SIGNAL
)


if __name__ == "__main__":
    print(QuantificationPluginFactory.PLUGINS)
    plug = QuantificationPluginFactory.get_plugin('ABBA_quantification_plugin')
