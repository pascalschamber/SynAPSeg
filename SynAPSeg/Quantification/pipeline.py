#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod
import pandas as pd
import os
import sys
import importlib.util
from typing import Any, Dict, List

from SynAPSeg.config import constants
from SynAPSeg.IO.BasePipelineComponent import BasePipelineComponent
from SynAPSeg.Quantification.dispatcher import ExampleDispatcher
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.Quantification.validation import DataRequirement, ConfigRequirement
from SynAPSeg.Plugins.base import get_available_plugins, get_plugin_module, get_plugin_default_parameters
from SynAPSeg.Quantification.factory import QuantificationPluginFactory

__output_data_vars__ = ['whole_rpdf', 'roi_df', 'summary_df']


# def find_stage_objects(directory=None):
#     """
#     Look for quantification pipeline plugins (as before)
#     """
#     if directory is None:
#         directory = os.path.join(constants.SYNAPSEG_BASE_DIR, 'Quantification', 'plugins')
#     stage_to_object = {}

#     # Ensure the directory is in sys.path so imports will work
#     sys.path.append(directory)

#     for filename in os.listdir(directory):
#         if filename.endswith(".py") and not filename.startswith("__"):
#             filepath = os.path.join(directory, filename)
#             module_name = os.path.splitext(filename)[0]

#             spec = importlib.util.spec_from_file_location(module_name, filepath)
#             if spec and spec.loader:
#                 module = importlib.util.module_from_spec(spec)
#                 try:
#                     spec.loader.exec_module(module)

#                     # Check if __plugin__ exists and is a string
#                     stage_name = getattr(module, "__plugin__", None)
#                     if isinstance(stage_name, str):
#                         # Check if the object exists with the same name
#                         obj = getattr(module, stage_name, None)
#                         if obj is not None:
#                             stage_to_object[stage_name] = obj
#                 except Exception as e:
#                     print(f"Failed to import {module_name}: {e}")

#     return stage_to_object


class Pipeline(BasePipelineComponent):
    """
    The Pipeline class manages the execution of multiple PipelineStage instances
        in order defined in PIPELINE_STAGE_NAMES
        `technically` supports validation-driven plugins but this is still # TODO item
    """

    def __init__(self, config, logger=None):
        """
        Initializes the pipeline with an ordered list of stages.
        """
        self.attach_logger(logger)
        self.factory = QuantificationPluginFactory
        self.discovered_plugins = list(self.factory.PLUGINS.keys())

        stages = config.PIPELINE_STAGE_NAMES
        assert isinstance(stages, list) and len(stages) > 0, f"ValueError 'config.PIPELINE_STAGE_NAMES': {stages}"

        self.stages = []
        for stage_name in stages:
            if stage_name in self.discovered_plugins:
                stageObj = self.factory.get_plugin(stage_name, pipeline=self, logger=logger)
                self.stages.append(stageObj)
            else:
                raise ValueError(f"Cannot find {stage_name} in {self.discovered_plugins}")
        
        # Resolve all declared compile blocks globally
        self._resolve_compile_blocks()

        # Optionally run setup-time validation after building stages. This is largely TODO 
        self.validate_pipeline_config(config)
        self.validate_pipeline_dependencies()
        
    
    def _resolve_compile_blocks(self):
        """
        Cross-references all stages to see if any active stage has declared 
        a compile block on another active stage, and updates the target's flag.
        """
        # 1. Gather a set of all stage names that are marked for blocking
        blocked_targets = set()
        for stage in self.stages:
            blocked_targets.update(stage.blocks_compile_targets)

        # 2. Apply the block to any stage whose name is in that set
        if not blocked_targets:
            return

        for stage in self.stages:
            if stage.name in blocked_targets:
                stage.set_is_compile_blocked(True)
                self.logger.info(f"Pipeline Config: Compile step for '{stage.name}' disabled.")

    # << ADDED FOR PLUGIN VALIDATION
    def validate_pipeline_config(self, config):
        """Validate config for all pipeline stages at setup time."""
        errors = []
        for stage in self.stages:
            try:
                if hasattr(stage, "validate_config"):
                    stage.validate_config(config)
            except Exception as e:
                errors.append(f"{stage.__class__.__name__}: {str(e)}")
        if errors:
            raise ValueError("Pipeline configuration validation failed:\n" + "\n".join(errors))

    def validate_pipeline_dependencies(self):
        """
        Validate that each stage's PIPELINE dependencies are satisfied by previous stages' outputs.
        """
        # Use input_requirements/output_specifications if present, else fall back to dependencies/outputs
        available_outputs = set()
        for stage in self.stages:
            reqs = getattr(stage, "input_requirements", [])
            if reqs:
                missing_pipeline_deps = []
                for req in reqs:
                    if getattr(req, "source", None) == "pipeline" or getattr(req, "source", None).name.upper() == "PIPELINE":
                        if req.required and req.key not in available_outputs:
                            missing_pipeline_deps.append(req.key)
                if missing_pipeline_deps:
                    raise ValueError(
                        f"Stage '{stage.__class__.__name__}' requires pipeline outputs {missing_pipeline_deps} but no previous stage provides them. "
                        f"Available pipeline outputs: {sorted(available_outputs)}"
                    )
            else:
                # Fallback for legacy support
                for dep in getattr(stage, "dependencies", []):
                    if dep not in available_outputs:
                        raise ValueError(
                            f"Stage '{stage.__class__.__name__}' requires pipeline output '{dep}', but it is not available."
                        )
            # Update available outputs
            outs = getattr(stage, "output_specifications", []) or getattr(stage, "outputs", [])
            available_outputs.update(outs)
        self.logger.info(f"Pipeline dependency validation passed. Final outputs: {sorted(available_outputs)}")
    # >>

    def get_stage(self, name):
        """ get stage by name """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(
            f"Stage name '{name}' not found in pipeline. Available stages: {[s.name for s in self.stages]}")
    
    def get_stage_names(self) -> list[str]:
        """ return name of each stage in self.stages """
        return [stage.name for stage in self.stages]

    def get_run_order(self):
        components = [Component(s.name, s.run_order_preferences) for s in self.stages]
        return self._get_order(components)
    
    def get_compile_order(self):
        components = [Component(s.name, s.compile_order_preferences) for s in self.stages]
        return self._get_order(components)

    def _get_order(self, components):
        ordered_stage_names = determine_order(components)
        return [c.name for c in ordered_stage_names] # need to extract name from component
    
    def run(self, data: dict, config) -> dict:
        """
        Runs the data through each stage in sequence, updating the data dictionary.
        """
        self.set_start_time()
        for i, stage_name in enumerate(self.get_run_order()):
            info_str = f"Running stage {i + 1}/{len(self.stages)}: {stage_name}\n"
            self.logger.info(info_str + "`"*(len(info_str)*2))
            stage = self.get_stage(stage_name)

            try:
                data = stage.run(data, config)

                if data.get('EXIT_FLAG'):                                   # TODO need to fix error message handling here since the error is logged but isn't included in the trace
                    return data

            except Exception as e:
                self.logger.error(f"Error in stage {stage_name}: {e}", exc_info=True)
                raise
            

        # post run activity
        ###################
        data = self.compile_results(data, config)

        self.logger.info("Pipeline execution completed successfully.")
        self.set_complete()
        return data

    def compile_results(self, data, config):
        """ 
        run functions after completion of all stages, generates summary_df
            - add example identification info to both rpdf and roi_df (using assign_md_attrs)

            - 
        
        """

        # add example identification info to both rpdf and roi_df
        #################################################################
                
        extracted_fn_groups = config.get('extracted_fn_groups') or {}
        extracted_groupping_vars = list(extracted_fn_groups.keys())
        
        assign_md_attrs = {
            'ex_i': config.ex_i, 
            **{k:data['metadata'].get(k) for k in (config.get('ASSIGN_MD_ATTRS') or [])},
        }
        grouping_cols = extracted_groupping_vars + list(assign_md_attrs.keys()) + ['roi_i', 'colocal_id']
        
        
        # init outputs
        # TODO assign above to data directly
        data['assign_md_attrs'] = assign_md_attrs
        data['extracted_groupping_vars'] = extracted_groupping_vars
        data['extracted_fn_groups'] = extracted_fn_groups
        data['grouping_cols'] = grouping_cols
        data['summary_df'] = None

        # generate summary_df using stage's compile method
        ################################################################
        self.logger.debug(f"compile order: {self.get_compile_order()}")
        for i, stage_name in enumerate(self.get_compile_order()):
            stage = self.get_stage(stage_name)
            data = stage.compile(data, config)
        
        if 'summary_df' in data and data['summary_df'] is None:
            del data['summary_df']
            
        return data





def run_pipeline_step_by_step(pipeline, data, disp):
    """ 
    returns a generator which executes each runs each stage individually 
        NOT a complete impl, but can be useful for debugging
    usage:
        pipeline_gen = run_pipeline_step_by_step(pipeline, data, disp)
        stage1_results = next(pipeline_gen)
        stage2_results = next(pipeline_gen)
        ... 
    """
    from typing import Generator

    def run_step_by_step(self, data: dict, config) -> Generator[dict, None, dict]:
        """
        Executes stages one by one, yielding the updated data dictionary after each stage.
        Returns the final data dictionary when complete.
        """
        self.set_start_time()
        run_order = self.get_run_order()
        
        for i, stage_name in enumerate(run_order):
            info_str = f"Running stage {i + 1}/{len(run_order)}: {stage_name}\n"
            self.logger.info(info_str + "`" * (len(info_str) * 2))
            
            stage = self.get_stage(stage_name)

            try:
                data = stage.run(data, config)
                
                # Yield control back to the caller with the current state of data
                yield data

                if data.get('EXIT_FLAG'):
                    self.logger.info(f"Exit flag detected at stage: {stage_name}")
                    return data

            except Exception as e:
                self.logger.error(f"Error in stage {stage_name}: {e}", exc_info=True)
                raise

        return data

    # monkey patch for now
    pipeline_gen = run_step_by_step(pipeline, data, disp.config)

    return pipeline_gen


import heapq
from collections import defaultdict

def determine_order(components):
    # 1. Map names to objects
    registry = {comp.name: comp for comp in components}
    adj = defaultdict(set)
    in_degree = {comp.name: 0 for comp in components}
    
    # 2. Build the Graph
    for comp in components:
        # Access the dictionary safely
        prefs = getattr(comp, 'order_preferences', {})
        
        # 'before' logic: current component must come before these targets
        for target_name in prefs.get('before', []):
            if target_name in registry:
                if target_name not in adj[comp.name]:
                    adj[comp.name].add(target_name)
                    in_degree[target_name] += 1
                
        # 'after' logic: current component must come after these targets
        for target_name in prefs.get('after', []):
            if target_name in registry:
                if comp.name not in adj[target_name]:
                    adj[target_name].add(comp.name)
                    in_degree[comp.name] += 1

    # 3. Kahn's Algorithm (Topological Sort)
    # Using a heap to handle ties (alphabetical order)
    queue = [name for name, deg in in_degree.items() if deg == 0]
    heapq.heapify(queue)
    
    ordered_names = []
    while queue:
        curr = heapq.heappop(queue)
        ordered_names.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(queue, neighbor)

    if len(ordered_names) != len(components):
        raise ValueError("Circular dependency detected!")

    return [registry[name] for name in ordered_names]

class Component:
    def __init__(self, name, prefs=None):
        self.name = name
        # Ensure it always has the required keys
        self.order_preferences = prefs or {'before': [], 'after': []}
        
    def __repr__(self):
        return f"{self.name}"


def test_determine_order():
    # --- Test Case ---
    # D -> A -> (B, C)
    A = Component("A", {'before': ['C'], 'after': []})
    B = Component("B", {'before': [], 'after': ['A']})
    C = Component("C", {'before': [], 'after': []})
    D = Component("D", {'before': ['A'], 'after': []})

    components_list = [A, B, C, D]

    import random
    random.shuffle(components_list)
    print(f"Input Order: {components_list}")

    try:
        result = determine_order(components_list)
        print(f"Final Order: {result}")
        # Expected Result: [Component(D), Component(A), Component(B), Component(C)]
    except ValueError as e:
        print(e)

# test_determine_order()