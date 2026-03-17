#!/usr/bin/env python3
"""
BasePipelineStage

This module defines an abstract base class for all pipeline stages.
It enforces the implementation of a `run()` method and provides
a logger (`self.logger`) to facilitate consistent logging across stages.
    The init method creates a logger with the child class's name. 
    A default StreamHandler with a formatter is added if no handlers are present.
    Child classes can log messages simply via self.logger.info(), self.logger.debug(), etc.

each child class module must define the following variables in the public scope:
__plugin_group__ = 'quantification'    : (str) required for quant plugin factory to find this stage
__plugin__ = 'ROIHandlingStage'        : (str) name of class object implemented in this plugin
__parameters__ = 'roi_handling.yaml'   : (str) filename of yaml which defines plugin's parameter specs
__stage_key__ = 'roi_handling'         : (str) name for this plugin (modules filestem), also identifies config parameters owned by this stage


if neccesary, each child class should define the following attributes:
__runOrderPreferences__ = {'before': [], 'after': []}     : (dict) order for _execute method execution
__compileOrderPreferences__ = {'before': [], 'after': []} : (dict) order for _compile method execution
__blocksCompile__ = []                                    : (list) stage names this stage will prevent from executing thier _compile method
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Optional

from SynAPSeg.IO.BasePipelineComponent import BasePipelineComponent
from SynAPSeg.Quantification.validation import DataRequirement, ConfigRequirement
import SynAPSeg.utils.utils_general as ug  

class BasePipelineStage(BasePipelineComponent):
    """
    Abstract base class for a pipeline stage with dependency and config validation.
    Supports both legacy string-list and new declarative requirements.

    Attributes:
        - __runOrderPreferences__ : dict like {'before': [], 'after': []}
        - __compileOrderPreferences__ : dict like {'before': [], 'after': []} 
        - __blocksCompile__ : Declarative list of stage names this stage will prevent from compiling
    """
    __runOrderPreferences__ = {'before': [], 'after': []}
    __compileOrderPreferences__ = {'before': [], 'after': []}
    __blocksCompile__ = []      

    def __init__(self, pipeline: Optional[object] = None, logger: Optional[logging.Logger] = None, **kwargs):
        # ref to parent pipeline
        self.pipeline = pipeline

        # Legacy support
        self.dependencies = getattr(self, 'dependencies', [])
        self.outputs = getattr(self, 'outputs', [])

        # Track timing and completion
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0
        self.completed = False
        self.exit_flag = False

        # Logger setup
        self.attach_logger(logger)

        # Internal config cache for this stage
        self._config_cache = {}

        # parse kwargs
        self.name = ug.popget(kwargs, 'name', self.__class__.__name__)

        # stage component order priority during pipeline run and compile_results
        raw_prefs = getattr(self, '__runOrderPreferences__', {})
        self.run_order_preferences = {
            'before': raw_prefs.get('before', []),
            'after': raw_prefs.get('after', [])
        }
        raw_prefs = getattr(self, '__compileOrderPreferences__', {})
        self.compile_order_preferences = {
            'before': raw_prefs.get('before', []),
            'after': raw_prefs.get('after', [])
        }

        # NEW: Parse what this stage wants to block
        self.blocks_compile_targets = getattr(self, '__blocksCompile__', [])
        # The flag that gets flipped if another stage blocks THIS stage
        self.is_compile_blocked = False    

        # store other kwargs passed during init
        self._init_kwargs = kwargs

    # ATTRIBUTES SET BY CHILD CLASS
    ##################################################################
    # Properties for new-style plugins to override
    @property
    def config_requirements(self):
        """Return list of ConfigRequirement for config validation."""
        return []

    @property
    def input_requirements(self):
        """Return list of DataRequirement for input validation."""
        return []

    @property
    def output_specifications(self):
        """Return list of keys this stage adds to the data dict."""
        return []

    @abstractmethod
    def _execute(self, data: dict, config: dict) -> dict:
        """
        Subclasses implement all processing here.
        Must return the `data` (dict), anything created should be added to data.
        """
        pass

    def compile(self, data:dict, config: dict) -> dict:
        """ 
        public hook for calling _compile method with proper orchestration
            - allows a stages compile method to be blocked from executing if it interferes 
               with the intended output
            - this behavior can be accessed through 
        
        """
        # Check if compile is blocked
        if self.is_compile_blocked:
            if self.logger:
                self.logger.debug(f"Skipping compile for {self.name}: blocked by pipeline configuration.")
            return data

        # else _compile method executes
        data = self._compile(data, config)
        return data


    def _compile(self, data: dict, config: dict) -> dict:
        """
        Optional private hook: Child classes can override this to contritube to summary_df.
            like _execute, must return the `data` (dict). 
            anything created should be added to the dataframe @ data['summary_df'], 
            but should check if it is None first. 
        """
        return data
    
    def set_is_compile_blocked(self, val:bool):
        self.is_compile_blocked = val

    # BASE METHODS
    ##################################################################
    # The main run() method -> calls validation and self._execute
    def run(self, data: dict, config: dict) -> dict:
        self.logger.info(f"Starting...")
        self.set_start_time()

        # Validate inputs/config
        self.validate_inputs(data)
        stage_config = self.ingest_config(config)

        try:
            data = self._execute(data, config) # main stage-specific processing logic should be in _execute

        except Exception as e:
            self.logger.error(f"{str(e)}")
            raise

        self.set_complete()
        return data

    # Validation methods (ported from QuantificationStage)
    def validate_inputs(self, data: dict):
        errors = []
        for req in self.input_requirements:
            is_valid, error_msg = req.validate(data)
            if not is_valid:
                errors.append(error_msg)
        if errors:
            raise ValueError(f"Input validation failed for {self.__class__.__name__}:\n" +
                             "\n".join(f"  - {err}" for err in errors))
        # Fallback for legacy dependencies
        missing = [key for key in self.dependencies if key not in data]
        if missing:
            raise ValueError(f"Missing required dependencies: {missing}")

    def check_dependencies(self, data: dict):
        # Legacy, prefer validate_inputs for new plugins
        missing = [key for key in self.dependencies if key not in data]
        return missing

    def validate_config(self, config: dict):
        errors = []
        for req in self.config_requirements:
            if req.required and req.key not in config and req.default_value is None:
                errors.append(f"Required config parameter '{req.key}' missing")
        if errors:
            raise ValueError(f"Config validation failed for {self.__class__.__name__}:\n" +
                             "\n".join(f"  - {err}" for err in errors))

    def ingest_config(self, config: dict):
        stage_config = {}
        for req in self.config_requirements:
            try:
                stage_config[req.key] = req.extract(config)
            except ValueError as e:
                raise ValueError(f"Config requirement error in {self.__class__.__name__}: {e}")
        self._config_cache = stage_config
        return stage_config

    def raise_exit_flag(self, msg, data):
        self.logger.error(msg)
        data["EXIT_FLAG"] = msg
        return data

    def get_stage_config(self, config, stage_key):
        """
        Parses the config for params specific to this stage and returns it.
            Back-compatibility: if STAGE_PARAMS is not present, returns the entire config.
        """
        stage_config = config.get("STAGE_PARAMS", {}).get(stage_key)

        # back-compatibility
        if stage_config is None:
            stage_config = config

        return stage_config
