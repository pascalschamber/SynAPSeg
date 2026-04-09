#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to parse images and run segmentation models
Creates a folder for the results called "examples"
"""

import os
import sys
import gc
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List, Callable

import matplotlib
# Force Matplotlib to use the non-interactive 'Agg' backend to prevent 
# it from trying to open a GUI window when called from the UI
matplotlib.use('Agg')

# Internal modules
from SynAPSeg.IO.BaseConfig import BaseConfig
from SynAPSeg.IO.project import Project
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.common.Logging import get_logger, rich_console
from SynAPSeg.config import constants
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_ML as uML
from SynAPSeg.IO.image_parser import ImageParser
import SynAPSeg.Segmentation.Pipeline as segpipe
import SynAPSeg.Segmentation.OutputHandler as OutputHandler
from SynAPSeg.Segmentation.Processing import (
    verify_input_parsed,
    check_skip_input_file,
    format_prediction_input,
    maybe_log_run,
    DispatcherCollection,
)
from SynAPSeg.Segmentation.config_parser import ConfigParser, interpret_run_config
from SynAPSeg.UI.widgets.thread_worker import SubProgress

torch = ug.try_import('torch')


def main(
    CONFIG_KEY: Optional[str]=None, 
    built_config: Optional[Dict]=None, 
    disp_i_slice:Optional[Tuple[int|Any, int|Any]]=(None, None),
    progress_callback: Optional[Callable[[int, str], None]] = None
    ):
    """
    Run segmetation script 

    Args:
        CONFIG_KEY: str -- key in user config file at constants.SEG_CONFIG_PATH
        built_config: dict with config attrs, values <-- provided when invoked by UI.Segmentation 
        disp_i_slice: tuple -- run subset of dispatchers (e.g. interupted run); indicies to slice into dispatchers list
            internally gets converted to a slice object
        progress_callback: Callable[[int, str], None] -- callback function to send progress updates to e.g. UI
    """
    if CONFIG_KEY is None and built_config is None:
        raise ValueError()
    
    # auto setup
    ####################################################################################
    # setup logging and env vars
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = constants.TF_FORCE_GPU_ALLOW_GROWTH
    logger = get_logger("segmentation_logger", log_filename="Segmentation.log")
    console = rich_console()
    

    # Config
    ####################################################################################
    # default implementation when invoking from script
    if CONFIG_KEY is not None:
        SEG_CONFIG = interpret_run_config(CONFIG_KEY)
    
    else: # when passing config from the gui
        SEG_CONFIG = BaseConfig(None, None, default_parameters_path=constants.SEG_DEFAULT_PARAMETERS_PATH, params=built_config)

    # parse config and set attributes for creating dispatchers: file_paths, models, etc.
    SEG_CONFIG = ConfigParser(SEG_CONFIG).get_config()

    # setup dispatchers and pipeline
    #############################################
    dispatchers = DispatcherCollection(SEG_CONFIG)
    pipeline = segpipe.build_segmentation_pipeline(SEG_CONFIG)
            
    _disp_slice = slice(*disp_i_slice)
    for disp in dispatchers[_disp_slice]:     
        print(dispatchers.get_progress(disp))
        
        # Define the 'slice' of the progress bar for THIS specific image
        # If we have 4 images, Image 1 owns 0-25%, Image 2 owns 25-50%, etc.
        start_pct = int((disp.image_i / dispatchers.n_disp) * 100)
        end_pct = int(((disp.image_i + 1) / dispatchers.n_disp) * 100)
        
        if progress_callback:
            # Create a "Sub-Callback" that translates 0-100% within the pipeline
            # into the start_pct -> end_pct range on the real UI bar.
            pipe_cb = SubProgress(progress_callback, start_pct, end_pct)

        else:
            # pipe_cb = None
            pipe_cb = SubProgress(lambda *args: None, start_pct, end_pct)  # dummy callback 
        
        # 1. Loading phase (First 10% of this item's slice)
        pipe_cb(0, f"Loading {disp.image_i}...")

        # load and init
        #########################################################################################
        # fetch parameters for input image and get input image parser to handle this type of image
        image_parser = disp.get_image_parser()
        img_obj, arr, ex_md = disp.image_parser.run()
        

        # init metadata dict # check if should skip this b/c it is completed 
        if (not verify_input_parsed(image_parser, ex_md, SEG_CONFIG)) or (check_skip_input_file(SEG_CONFIG, ex_md)):
            continue
        
        # slice array for expected input and arrange channels in wavelength order, if chs deviate from expected they will be included into array and index info is in ex_md['data_metadata']['inserted_null_chs']
        arr, arr_mip = format_prediction_input(image_parser, img_obj, arr, ex_md, SEG_CONFIG) # arr fmt = "STCZYX"
        
        # display input
        if bool(0): 
            up.show_ch(uip.norm_percentile(np.moveaxis(arr_mip, 0, -1), (1,99.8), ch_axis=-1), axis=-1)
        


        # Predictions
        #########################################################################################
        predictions = pipeline.run(
            arr, 
            data_state={'path_to_example':ex_md['output_dir']}, 
            progress_callback=pipe_cb if progress_callback else None
        )

        # display results
        if bool(0):
            OutputHandler.show_pipeline_results(predictions, pipeline)          
            
                
        # write outputs
        #########################################################################################
        if SEG_CONFIG.get('WRITE_OUTPUT', True) and (not SEG_CONFIG.get('TESTING', False)): 
            if pipe_cb is not None:
                pipe_cb(99, "Writing outputs...")

            OutputHandler.write_output(
                ex_md, arr, predictions, pipeline, SEG_CONFIG, img_obj,
                generate_summary = bool(1),
                generate_mip = not SEG_CONFIG['USE_EXISTING'],
                overwrite_raw_img = not SEG_CONFIG['USE_EXISTING'],
            )
            dispatchers.edited_examples.append(disp.image_i)
            
        
        # track disp run time        
        console(f"dispatcher (i={disp.image_i}) completed in {disp.get_elapsed_time()} seconds.\n{ug.get_datetime()}", title=" Dispatcher Finished ")

        if pipe_cb is not None:
            pipe_cb(100, "Finished!")
        
        
        # clear gpu memory
        uML.clear_gpu_memory(torch=torch)
                
        if SEG_CONFIG.get('RETURN_OUTPUT', False): # for ui test
            return arr, predictions, ex_md
        
        del(predictions, arr)
        gc.collect()
        
        
    
    maybe_log_run(SEG_CONFIG, dispatchers)
    console(f'run complete.\n\ttook {ug.dt()-dispatchers.t0} seconds.', title=" RUN COMPLETE ")
    
    return None
    





####################################################################################
    # MAIN 
####################################################################################
if __name__ == '__main__':            
    
    verify_and_set_env_dirs(
        # dict(
        # PROJECTS_ROOT_DIR = r"J:\SEGMENTATION_DATASETS",
        # ROOT_DIR = r"J:",
        # ),
        # override=False
    )

    disp_i_slice = (None, None)
 
    # run
    config_keys = [ 
        '2026_0108_synapseg_supFig1',
    ]

    for CONFIG_KEY in config_keys:
        print(CONFIG_KEY)

        main(CONFIG_KEY=CONFIG_KEY, built_config=None, disp_i_slice=disp_i_slice)

