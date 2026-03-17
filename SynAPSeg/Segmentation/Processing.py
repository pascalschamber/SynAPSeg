#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this file handles the other modules needed to run the segmentation pipeline

"""
from typing import Dict, List, Optional, Any
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import copy
import re
import sys

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.Segmentation.Pipeline import SegmentationPipeline
from SynAPSeg.IO.image_parser import ImageParser
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.IO.project import Project


aicsimageio = ug.try_import('aicsimageio')

def get_reader(img_path):
    # from aicsimageio.readers import czi_reader
    if img_path.endswith('.czi'):
        return aicsimageio.AICSImage(
            img_path, 
            reader=aicsimageio.readers.czi_reader.CziReader, 
            reconstruct_mosaic=False
        )
    else:
        pass

# class Peeker:
#     def __init__(self, img_path):
#         self.img_path = img_path
#         self.get_reader(img_path)



    

class DispatcherCollection:
    """
    to handle scenes in czi images need to parse input files ahead of pipeline, 
    then if multiple scenes will pass same czi filepath but different scene id in load kwargs
    #TODO this will break compare rerun image files, but that's probably fine
    """
    def __init__(self, config, start_disp_i=0):
        self.config = config
        self.edited_examples = [] # "this attribute stores indices of dispatchers that successfully complete and write to disk"

        self.image_filepaths = config.image_filepaths_to_process 
        self.dispatchers = []
        self.n_disp = start_disp_i
        self.t0 = None
        self.get_dispatchers(config)

    def __iter__(self):
        """Return an iterator over the dispatchers."""
        return iter(self.dispatchers)

    def __getitem__(self, index):
        """Get the dispatcher at the specified index."""
        return self.dispatchers[index]

    def __setitem__(self, index, value):
        """Set the dispatcher at the specified index."""
        self.dispatchers[index] = value

    def __delitem__(self, index):
        """Delete the dispatcher at the specified index."""
        del self.dispatchers[index]

    def __len__(self):
        """Return the number of dispatchers."""
        return len(self.dispatchers)

    def get_n_disp(self):
        return self.n_disp
    
    def set_start_run_time(self, dispatcher):
        """ set disp start time and run start time if not already set """
        dispatcher.init_t0()
        if self.t0 is None:
            self.t0 = dispatcher.t0
    
    def get_progress(self, dispatcher):
        """returns string representation of progress through dispatchers"""
        image_i = dispatcher.image_i
        scene_name = dispatcher.scene_name
        self.set_start_run_time(dispatcher)
        return f"\n{'_'*60}\ni={image_i} | {image_i+1} of {self.n_disp} ({round((image_i/self.n_disp)*100, 1)}%) | scene_name:{scene_name}\n{dispatcher.image_path} | {ug.get_datetime()}"
    
    def filter(self, condition):
        """
        Returns a list of dispatchers that meet the given condition.
        
        Args:
        condition: A callable that takes a Dispatcher and returns True/False.
        
        Returns:
        List of Dispatcher objects.
        """
        return [d for d in self.dispatchers if condition(d)]
    
    # specifc functions
    #########################################################################################
        
    def get_dispatchers(self, SEG_CONFIG):
        """init dispatchers"""
        for i, image_path in enumerate(self.image_filepaths):
            
            # if processing an image file with multiple scenes
            if image_path.endswith('.czi'):
                czi = get_reader(image_path)
                scene_ids = czi.scenes
                for scene_id, scene_name in enumerate(scene_ids):
                    self.add_dispatcher(image_path, SEG_CONFIG, scene_id, scene_name)
            else:
                self.add_dispatcher(image_path, SEG_CONFIG, None, None)
        
        # TODO peek at shapes - validate they are compatible with config
                
        return self.dispatchers
    
    def add_dispatcher(self, image_path, SEG_CONFIG, scene_id, scene_name, **kwargs):
        image_i = self.get_n_disp()
        ex_md = init_ex_md(image_i, image_path, SEG_CONFIG, scene_id=scene_id, scene_name=scene_name)
        image_parser = ImageParser.create_parser(image_path, params=ex_md, load_kwargs={'scene_id': scene_id})
        self.dispatchers.append(
            Dispatcher(image_i=image_i, image_path=image_path, scene_id=scene_id, scene_name=scene_name, ex_md=ex_md, image_parser=image_parser)
        )
        self.n_disp += 1
        
        
class Dispatcher:
    def __init__(self, **kwargs):
        self.image_i = None
        self.image_path = None
        self.scene_id = None # used for czi images only
        self.scene_name = None # used for czi images only
        self.ex_md = None
        self.image_parser = None
        self.t0 = None # start time set when DispatcherCollection.get_progress() is called
        self.t1 = None # end time set at end of loop
        
        for k,v in kwargs.items():
            setattr(self, k, v)
            
    def get_image_parser(self):
        return self.image_parser
    
    def get_ex_md(self):
        return self.ex_md

    def init_t0(self):
        self.t0 = ug.dt()
    
    def set_t1(self):
        self.t1 = ug.dt()
        
    def get_elapsed_time(self):
        if not self.t1:
            self.set_t1()
        return self.t1 - self.t0
        
        



def check_skip_input_file(SEG_CONFIG, ex_md):
    continue_flag = False
    if SEG_CONFIG.get('FORCE_NO_SKIP', False):
        continue_flag = False
    elif SEG_CONFIG.get('SKIP_COMPLETED', True):
        if os.path.exists(ex_md['output_dir']):
            contentts = os.listdir(ex_md['output_dir'])
            if 'complete.txt' in contentts:
                continue_flag = True
    if continue_flag: 
        print(f"skiping {Path(ex_md['image_path']).stem} because it is marked complete.")
    return continue_flag

def update_example_metadata(ex_md, SEG_CONFIG):
    """ Adjustments and additions to metadata for this specific example"""
    pipeline_names = list(ex_md['segmentation_pipeline_config'].keys())
    for pdn in pipeline_names:
        ex_md['segmentation_pipeline_config'][pdn]['model_params']['output_dir'] = os.path.join(ex_md['output_dir'], pdn)
        if pdn=='n2v':
            if not ex_md['segmentation_pipeline_config']['n2v']['model_params'].get('multi_image', False):
                ex_md['segmentation_pipeline_config']['n2v']['model_params']['N2V_MODEL_NAME_BASE'] += f"_{ex_md['example_i']}"
        if not SEG_CONFIG.get('SHOW_FIGURES', True) and 'plotter' in ex_md['segmentation_pipeline_config'][pdn]['model_params']:
            if ex_md['segmentation_pipeline_config'][pdn]['model_params']['plotter'] is not None:
                ex_md['segmentation_pipeline_config'][pdn]['model_params']['plotter'].turn_off()
        


def maybe_log_run(SEG_CONFIG, dispatchers):
    
    if SEG_CONFIG.get('LOG_RUN', True):
        try:
            from SynAPSeg.utils.utils_logger import log_parameters

            # init project object and log run's segmentation config there
            proj = Project(SEG_CONFIG["OUTPUT_DIR_PROJ"])
            SEG_CONFIG._write_config(os.path.join(proj.configdir, f"{ug.get_datetime()}_SEG_CONFIG.yaml"))
            
            # save run in project's __logs__ folder
            log_file_path = os.path.join(proj.logdir, 'Segmentation_log.json')
            log_parameters(
                {
                    'PROJECT_NAME':SEG_CONFIG.PROJECT_NAME, 
                    'MODELS_BASE_DIR':SEG_CONFIG.MODELS_BASE_DIR,
                    'ROOT_DIR':SEG_CONFIG.ROOT_DIR,
                    'edited_examples': dispatchers.edited_examples,
                    'SEG_CONFIG': SEG_CONFIG if isinstance(SEG_CONFIG, dict) else ug.objdir(SEG_CONFIG, return_as_string=True)
                }, 
                log_file_path = log_file_path,
            )
            proj.update_example_info()
            
        except Exception as e:
            print(f'failed to write run config log. {e}')
            

def maybe_n2v_multi(SEG_CONFIG, disps: DispatcherCollection, pipeline: SegmentationPipeline):
    """
    # TODO: add support for training across multiple images
    default behavior is to train a new model only if it doesn't exist
    so if they do exist already, but you want to train a new one, need to specify use_existing = False in config
    """
    if 'n2v' not in SEG_CONFIG['MD_TEMPLATE']['segmentation_pipeline_config'] or pipeline is None:
        return None
    
    _model_config = SEG_CONFIG['MD_TEMPLATE']['segmentation_pipeline_config']['n2v']['model_params']
    config_ch_info = SEG_CONFIG['DATA_METADATA'].get('channel_info')
    n2v_model = pipeline.get_model('n2v')
    
    if _model_config.get('multi_image', False) is True:
        # check if models already exist - must be at least one model for each ch 
        _has_models = len(n2v_model.found_ch_models) >= len(config_ch_info)
        _dont_use_existing = (_model_config.get('use_existing') is False)
        
        # train new models for each channel
        if not _has_models or _dont_use_existing:
            ch_patches = n2v_model.ingest_images(disps)
            n2v_model.train(ch_patches)
    else:
        raise ValueError('single model mode not configured yet.')

    





# individual example related functions
#############################################################################################################
def init_ex_md(image_i, image_path, SEG_CONFIG, scene_id=None, scene_name=None) -> dict:
    """fetch/generate parameters relevant to this input image (aka 'example')"""
    
    # generate example id (aka image_i)
    print(Path(image_path).stem)
    image_i = Path(image_path).stem if image_i is None else image_i # if img index is not provided default to using file stem
    assert isinstance(image_i, int) or isinstance(image_i, str)
    _example_i = str(image_i).zfill(4) if isinstance(image_i, int) else image_i
    
    # init params for processing this image
    ex_md = copy.deepcopy(SEG_CONFIG['MD_TEMPLATE'])
    ex_md['COLOCALIZE_PARAMS'] = SEG_CONFIG.get('COLOCALIZE_PARAMS', None)
    ex_md['image_path'] = image_path
    ex_md['scene_id'] = scene_id
    ex_md['scene_name'] = scene_name
    ex_md['example_i'] = _example_i
    ex_md['output_dir'] = os.path.join(SEG_CONFIG['OUTPUT_DIR_EXAMPLES'], _example_i)
    ex_md['channel_colormaps'] = SEG_CONFIG.get("THUMBNAIL_CMAPS_DEFAULT") or ['blue', 'green', 'red', 'magenta']

    # add these attrs to ex_md
    getattrs = ['take_dims', 'project_dims']
    for a in getattrs:
        ex_md[a] = SEG_CONFIG[a]
    
    
    # save annoatation notes from previous run if they exist
    if not SEG_CONFIG['IS_NEW_RUN']:
        _ex_md = MetadataParser.try_get_metadata(ex_md['output_dir'], {'metadata':['metadata.yml']}, silent=True)
        if len(_ex_md)>0:
            ex_md['annotation_metadata'] = _ex_md['annotation_metadata']            
    return ex_md

def verify_input_parsed(img_parser, ex_md, SEG_CONFIG):
    """ log errors and configure output for this example"""
    # log and display any errors
    
    if img_parser.error_msg is not None:
        SEG_CONFIG['example_metadata']['errors']['image_i'] = img_parser.error_msg
        print(img_parser.error_msg)
        return False
    
    # setup example output
    if SEG_CONFIG.get('WRITE_OUTPUT', False):
        ug.verify_outputdir(SEG_CONFIG['OUTPUT_DIR_PROJ'])
        ug.verify_outputdir(SEG_CONFIG['OUTPUT_DIR_EXAMPLES'])
        ug.verify_outputdir(ex_md['output_dir'])
    update_example_metadata(ex_md, SEG_CONFIG)

    return True


def format_prediction_input(image_parser, image_obj, arr, ex_md, SEG_CONFIG):
    """slice array for expected input and arrange channels in wavelength order"""
    
    ex_md['take_dims'] = ex_md.get('take_dims', '')
    ex_md['project_dims'] = ex_md.get('project_dims', '')

    arr, arr_mip = image_parser.try_format_prediction_input(image_obj, arr, ex_md, SEG_CONFIG)
    
    # standardize the input to segmentation pipeline
    arr = uip.transform_axes(arr, ex_md['current_format'], "STCZYX")
    ex_md['current_format'] = "STCZYX" # update ex_md with the new format
    
    return arr, arr_mip



# Optimization functions
##############################################################################################################################################
def tune_stardist_params(stardist_pipeline, img, CH=0, slice_obj=slice(None), scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2]):
    """try differennt scale  factors  with/without using n2v model output for the predictions"""
    assert any([isinstance(CH, t) for t in (int, list)]), f"CH must be int or list, got: {type(CH)}"
    
    # prepare input
    sliced_img = img[slice_obj]
    sd_input_img = np.moveaxis(sliced_img, -1,1)[:,:,np.newaxis,:,:] # input should be S,C,Z,Y,X
    
    # iterate prediction channels
    for _CH in (CH if isinstance(CH, list) else [CH]):
        imgs, tts  = [], []
        pred_input_img = sd_input_img[:,_CH:_CH+1,:,:,:]

        for sf in scale_factors:
            sd_predictions = stardist_pipeline.run(# input should be S,C,Z,Y,X
                pred_input_img,
                pred_kwargs= dict(scale=sf)
                                                )[0][:,0] # output is shape (1, Y, X)
            imgs.append(
                up.overlay(
                    np.clip(pred_input_img[0, 0, 0, ...], 0, 1), 
                    uip.mask_to_outlines(sd_predictions[0]), 'red', (255,0,0)))
            tts.append(f"scale:  {sf}, n:{len(np.unique(sd_predictions[0])-1)}")
        up.plot_image_grid(imgs, n_cols=3,  titles=tts)

def tune_neurseg_prediction_threshold():
    # TODO - need to reimplement this
    pass

def analyze_histogram(label_seg_input, plot=False):
    import scipy.stats as stats
    hist_peaks = []
    nch = label_seg_input.shape[-1]
    for i in range(nch):
        data = label_seg_input[0,...,i].ravel()
        # Generate the histogram
        counts, bin_edges = np.histogram(data, bins=30)

        # Find the index of the peak (bin with maximum count)
        peak_index = np.argmax(counts)

        # Find the corresponding bin center and peak value
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_value = bin_centers[peak_index]
        peak_height = counts[peak_index]
        peak_bin_width = bin_edges[peak_index + 1] - bin_edges[peak_index]

        # Calculate skewness and kurtosis of the entire data
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Calculate Full Width at Half Maximum (FWHM)
        half_max = peak_height / 2
        left_index = np.where(counts >= half_max)[0][0]  # First bin above half maximum
        right_index = np.where(counts >= half_max)[0][-1]  # Last bin above half maximum
        FWHM = bin_edges[right_index + 1] - bin_edges[left_index]
                
        if plot:
            plot_histogram_stats(data, bin_edges, peak_value, FWHM, left_index, right_index)
            # compile stats info
            stats_summary = (
                f"Peak value: {peak_value}\n"
                f"Peak height: {peak_height}\n"
                f"Peak bin width: {peak_bin_width}\n"
                f"Skewness: {skewness}\n"
                f"Kurtosis: {kurtosis}\n"
                f"Full Width at Half Maximum (FWHM): {FWHM}"
            )
            print(stats_summary)
        hist_peaks.append(peak_value+(FWHM/2))
    return hist_peaks
    
def plot_histogram_stats(data, bin_edges, peak_value, FWHM, left_index, right_index):
    # Plot the histogram
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(peak_value, color='r', linestyle='--', label=f'Peak: {peak_value:.2f}')
    plt.axvline(bin_edges[left_index], color='g', linestyle=':', label=f'FWHM: {FWHM:.2f}')
    plt.axvline(bin_edges[right_index + 1], color='g', linestyle=':')
    plt.legend()
    plt.show()

    
