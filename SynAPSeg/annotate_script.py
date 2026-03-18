from typing import Optional
import napari
import tifffile
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import skimage
import scipy

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.Annotation.annotation_IO import load_example_images
from SynAPSeg.Annotation.napari_utils import dat
from SynAPSeg.Annotation.annotation_core import get_example_path, build_image_list, create_napari_viewer
from SynAPSeg.common.Logging import get_logger
from SynAPSeg.IO.project import Project, Example
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.config import constants

"""
annotation script for validation of model predictions and defining custom ROIs
"""

"""
######################################################################################################################################################################
MAIN
######################################################################################################################################################################
"""



if __name__ == '__main__':


    
    ######################################################################################################################################################################
    # PARAMS
    ######################################################################################################################################################################
    # Set base project directory
    verify_and_set_env_dirs(
        dict(
            # PROJECTS_ROOT_DIR = r"J:\SEGMENTATION_DATASETS"
        )
    )
    
    # load specific project example  with this var
    EXAMPLE_PROJ = "2025_0928_hpc_psd95andrbPV_zstacks"
    EXAMPLE_I = '0004'
    

    # load example
    ###################################################################################
    project = Project(os.path.join(os.environ['PROJECTS_ROOT_DIR'], EXAMPLE_PROJ))
    project.setup_logger("Quantification")
    
    if bool(0): # check progress
        res = project.get_dir_progress(project, FILE_MAP={'dends': ['annotated_dends_filt_final.tiff']})

    if bool(0):# add new data metadata formats
        project.batch_attach_data([{'key':'annotated_dends_filt_final', 'data_formats':'ZYX'}])


    ex = project.get_example(EXAMPLE_I)
    
    LABEL_INT_MAP, FILE_MAP, image_dict, get_image_list = load_example_images(
        ex,
        include_only=["annotated_dends_filt_final.tiff", "raw_img.tiff", "pred_stardist_n2v_3d_v310.tiff", "pred_n2v2.tiff"][:3],
        fail_on_format_error=False,
        get_label_int_map=False, # currently has some issues with if raw_img format is not found. not-implemented/used
        use_prefix_as_key=False,
    )
    
    exmd, path_to_example = image_dict.pop('metadata'), ex.path_to_example
    
    # create napari viewer
    viewer, widget_objects = create_napari_viewer(
        exmd,
        path_to_example,
        FILE_MAP,
        image_dict,
        get_image_list,
        set_lbl_contours=0,
        LABEL_INT_MAP=LABEL_INT_MAP,
        logger = project.logger,
    )