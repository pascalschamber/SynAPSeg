#!/usr/bin/env python3
"""
main.py

This script runs a small-scale quantification pipeline across a set of examples.
It includes:

• PROJECT_NAME and RUN_CONFIG_PATH to define experiment and config
• A QuantConfig class that loads a YAML-based configuration into a dictionary
• Discovery of input examples using utils.general
• Creation of dispatcher objects to manage each example
• A configurable Pipeline built from sequential stages:
    - PreprocessingStage
    - ROIHandlingStage
    - ObjectDetectionStage
• Collection and compilation of results (region and ROI properties)

Each dispatcher loads its data, runs through the pipeline, and returns result dictionaries.
"""


import os
import sys
import pandas as pd
import logging
import traceback
import numpy as np
import gc
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
console = Console()
def rich_box(text, title=None):
    console.print(Panel(text, title=title, expand=False))

from SynAPSeg.config import constants 
from SynAPSeg.common.Logging import get_logger
from SynAPSeg.IO.BaseConfig import BaseConfig, read_config
from SynAPSeg.IO.project import Project, Example
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.Quantification.pipeline import Pipeline
from SynAPSeg.Quantification.dispatcher import DispatcherCollection
from SynAPSeg.Quantification.output_handler import OutputHandler
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.utils.utils_ImgDB import ImgDB

# TODO issue with: PROJECT_NAME = "2025_0928_hpc_psd95andrbPV_zstacks"
# look into why annotated_file_names = project.get_dir_progress(proj=project, FILE_MAP=FILE_MAP) has duplicate annotated filenames


###############################################################################################################################################################
# main
###############################################################################################################################################################

def main(config_key, config_path=None, default_parameters_path=None, dispatchers_slice=None, outdir_path=None):
    """
    Main quantification pipeline.
    
    Args:
        config_key (str): Key for the quantification configuration.
        config_path (str, optional): Path to the quantification configuration file. 
            if not provided, the default quantification configuration will be used.
        default_parameters_path (str, optional): Path to the default parameters file.
            if not provided, the default parameters file will be used.
        dispatchers_slice (slice, optional): Slice to select a subset of dispatchers.
        outdir_path (str, optional): Path to an (potentially non-existing) output directory.
            useful for resuming an interrupted run.
            if not provided, the output directory will be created in the examples directory outputs folder.
    """

    # Config
    ####################################################################################
    # Load quantification configuration    
    QUANT_CONFIG = BaseConfig(
        config_key, 
        config_path or constants.QUANT_CONFIG_PATH, 
        default_parameters_path or constants.QUANT_DEFAULT_PARAMETERS_PATH
    )
    QUANT_CONFIG.params['OUTPUT_DIR_BASE'] = os.path.join(QUANT_CONFIG.EXAMPLES_BASE_DIR, QUANT_CONFIG.EXAMPLE_PROJ)
    QUANT_CONFIG.params['EXAMPLES_DIR'] = os.path.join(QUANT_CONFIG.OUTPUT_DIR_BASE, QUANT_CONFIG.EXAMPLE_DIRNAME)
    QUANT_CONFIG.params['PIPELINE_STAGE_NAMES'] = list(QUANT_CONFIG.STAGE_PARAMS.keys()) 

    proj = Project(QUANT_CONFIG.OUTPUT_DIR_BASE)
    proj.setup_logger("Quantification") # setup logging -
   
    # optionally add formatting information for data arrays
    ###########################################################
    # proj.batch_attach_data([{'key':'ROI_0', 'data_formats':'YX'}])
    
    # option to set a custom filter to fetch specific examples 
    ###########################################################
    # filter_str = 'PSD95_488'
    # proj.filter_examples(lambda ex: Path(ex.get_image_path()).stem.__contains__(filter_str))


    # setup pipeline components
    ####################################################################################
    # Initialize Dispatchers + pipeline
    dispatchers = DispatcherCollection(QUANT_CONFIG, proj, proj.logger) 
    print(f"n dispatchers: {len(dispatchers)}")
    
    # init pipeline
    pipeline = Pipeline(QUANT_CONFIG, proj.logger)

    # init output handler 
    outputHandler = OutputHandler(pipeline.stages, QUANT_CONFIG, proj.logger,
        outdir_path=outdir_path
    )
    
    ####################################################################################
    # MAIN LOOP # Process Each Dispatcher
    ####################################################################################
    
    for disp in dispatchers[dispatchers_slice or slice(None, None)]:
        data = disp.load(data_loaders=None) # Load example's data
        data = disp.process_example(pipeline, data, outputHandler) # Run pipeline, passing data to outputHandler
    
                
    # Compile Results
    ####################################################################################
    outputHandler.concat_outputs(QUANT_CONFIG)
    outputHandler.log_quant_config(proj, dispatchers, QUANT_CONFIG)
    outputHandler.log_colocal_ids(data)
    outputHandler.print_main_container_head()
    # outputHandler.delete_tempfolder()             # run to delete tempdir and it's contents -- only delete if all dispatchers have been run else concat outputs will fail
    rich_box(f':)', title='ALL DONE') 
    
    return True


if __name__ == "__main__":
    
    # Set environment variables
    ####################################################################################
    env_vars = dict(
        # ROOT_DIR = r"R:\Confocal data archive\Molly\Btbd11 WT_KO ICC\2.9.25 ICC PFA Btbd11 KO TARP and PSD95",
        # PROJECTS_ROOT_DIR = "R:\\Confocal data archive\\Alexei\\SEGMENTATION_DATASETS"
    )
    verify_and_set_env_dirs(env_vars)
    
    # set run parameters
    ####################################################################################
    config_key = "2025_0928_hpc_psd95andrbPV_zstacks"
    # config_key = 'VHL_VglutHomer'
    # config_key = 'test_ABBA_QUANT'
    config_path=constants.QUANT_CONFIG_PATH
    default_parameters_path=constants.QUANT_DEFAULT_PARAMETERS_PATH
    dispatchers_slice=None
    outdir_path=None

    # run main
    ####################################################################################
    result = main(
        config_key,
        config_path=config_path,
        default_parameters_path=default_parameters_path,
        dispatchers_slice=dispatchers_slice,
        outdir_path=outdir_path,
    )


