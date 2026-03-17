#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import skimage.measure

from SynAPSeg.Quantification.BasePipelineStage import BasePipelineStage
from SynAPSeg.Quantification.validation import DataRequirement, ConfigRequirement, DataType, DataSource
from SynAPSeg.Quantification.plugins.roi_handling import ROIAssigner
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config.constants import SPATIAL_AXES

__plugin_group__ = 'quantification'
__plugin__ = 'ObjectDetectionStage'
__parameters__ = 'object_detection.yaml'
__stage_key__ = 'object_detection'

class ObjectDetectionStage(BasePipelineStage):
    """
    Plugin stage: extracts region properties, intensities, and assigns ROIs.
    """
    __runOrderPreferences__ = {'before': ['colocalization'], 'after': ['roi_handling']}
    __compileOrderPreferences__ = {'before': ['roi_handling'], 'after': []}
    __blocksCompile__ = []

    def init_outputs(self):
        # name: container, key in data: name
        return [{
            'container_name': 'all_rpdfs',
            'container': [],
            'data_key': 'rpdf'
        }]
    
    @property
    def output_specifications(self):
        return ["rpdf", "roi_assignment_column"]

    @property
    def config_requirements(self):
        
        # TODO build these from param spec .yaml 
        return [
            ConfigRequirement(
                key="img_fmt", data_type=DataType.STRING, required=False, # TODO should be true but can't be set until loaded
                description="Format string (e.g., TCZYX) for intensity image axes"
            ),
            ConfigRequirement(
                key="OBJECTS_IMAGE_NAME", data_type=DataType.STRING, required=True,
                default_value="pred_stardist",
                description="Key in data dict for objects label image"
            ),
            ConfigRequirement(
                key="INTENSITY_IMAGE_NAME", data_type=DataType.STRING, required=False,
                default_value="mip_raw",
                description="Key in data dict for intensity image"
            ),
            ConfigRequirement(
                key="OBJECTS_IMAGE_SIZE_RANGE", data_type=DataType.LIST, required=False,
                default_value=None,
                description="List [min,max] for filtering by region area"
            ),
            ConfigRequirement(
                key="GET_CLC_ID_CH_INTENSITIES", data_type=DataType.DICT, required=False,
                default_value=None,
                description="Map colocalization id to channel indices for intensity extraction"
            ),
            ConfigRequirement(
                key="ROI_ASSIGNMENT_METHODS", data_type=DataType.LIST, required=False,
                default_value=["Coords"],
                description="Methods for ROI assignment"
            ),
            ConfigRequirement(
                key="GET_OBJECT_COORDS", data_type=DataType.BOOLEAN, required=False,
                default_value=True,
                description="Whether to extract object coordinates"
            ),
        ]


    def _execute(self, data: dict, config: dict) -> dict:
        # Get configuration parameters.
        #########################################################################
        stage_config = self.get_stage_config(config, __stage_key__)

        objects_img_key = config.get("OBJECTS_IMAGE_NAME")
        intensity_img_key = config.get("INTENSITY_IMAGE_NAME")
        size_range = stage_config.get("OBJECTS_IMAGE_SIZE_RANGE")
        rps_to_get = stage_config.get('RPS_TO_GET', None)
        GET_OBJECT_COORDS = stage_config.get("GET_OBJECT_COORDS", True)
        ADDITIONAL_PROPS = stage_config.get("ADDITIONAL_PROPS", None) # ['perimeter', 'num_pixels', 'intensity_std', 'solidity']
        EXTRA_PROPERTIES = stage_config.get("EXTRA_PROPERTIES", None) # [uc.skewedness, uc.kurtosis, uc.circularity])
        get_clc_intensities = stage_config.get("GET_CLC_ID_CH_INTENSITIES")

        ROI_ASSIGNMENT_METHODS = self.get_stage_config(config, 'roi_handling').get("ROI_ASSIGNMENT_METHODS")
        if isinstance(ROI_ASSIGNMENT_METHODS, str):
            ROI_ASSIGNMENT_METHODS = [ROI_ASSIGNMENT_METHODS]
        
        # these are not specified in config, unless user needs to overwrite. 
        # Normally will be infered from reading example metadata
        # should always present due to inital setup validation that occurs before exc here
        img_fmt = config["img_fmt"] 
        obj_fmt = config["obj_fmt"]
        PX_SIZES = config.get("PX_SIZES") # dict like {'X':1, 'Y':1, 'Z':1} # to avoid conflicting source, in seg config, should create this no matter what and allow user to input value
        voxel_size = [PX_SIZES[d] for d in 'ZYX' if d in PX_SIZES.keys()] if isinstance(PX_SIZES, dict) else None 
            
        # init outputs
        data["rpdf"] = pd.DataFrame()
        # first method is used to assign roi_i
        data["roi_assignment_column"] = None if not ROI_ASSIGNMENT_METHODS else f"roi_i_by{ROI_ASSIGNMENT_METHODS[0]}"
        
        # Retrieve necessary data.
        objects_img = data[objects_img_key]
        intensity_img = data.get(intensity_img_key) 
        if intensity_img is None:
            intensity_img = np.zeros_like(objects_img, dtype=np.uint8)
        ch_axis = img_fmt.index('C') if 'C' in img_fmt else None

        # handle channel info
        GET_CHS = config.get('GET_CHS') # represent channel indicies
        ch_clcid_map = None
        clcid_name_map = {0:'channel_0'}

        if ch_axis is not None:
            if not GET_CHS:
                nch = intensity_img.shape[ch_axis]
                GET_CHS = list(range(nch))
            GET_CHS = sorted(GET_CHS)
        
            # parse channel metadata
            chinfo = data['metadata']['data_metadata'].get('channel_info') or {}
            if len(chinfo) == 0:
                chinfo = {i:f"channel_{i}" for i in GET_CHS}
            chinfo = dict(sorted(chinfo.items()))

            # configure clc_id to channel indicies and names mappings - done since slicing can alter order
            ch_clcid_map = {i: GET_CHS[i] for i in range(len(GET_CHS))} # maps channel indices (in sliced data) to clc ids (represent true channel index if sliced)
            ch_names = list(chinfo.values())
            clcid_name_map = {i: ch_names[GET_CHS[i]] for i in range(len(GET_CHS))}


        

        # Extract region properties.
        #########################################################################
        self.logger.info(
            f"starting region props - objects_img:{objects_img.shape}, intensity_img:{intensity_img.shape} img_fmt:{img_fmt}, GET_OBJECT_COORDS:{GET_OBJECT_COORDS}\n \
            ADDITIONAL_PROPS: {ADDITIONAL_PROPS}, EXTRA_PROPERTIES: {EXTRA_PROPERTIES}."
        ) 
        data["rpdf"], rp_table_infostr = uc.get_rp_table(
            objects_img, 
            intensity_img, 
            ch_colocal_id=ch_clcid_map,
            ch_axis=ch_axis, 
            prt_str='uc.get_rp_table execution info:\n',
            rps_to_get = rps_to_get,
            get_object_coords=GET_OBJECT_COORDS,
            additional_props=ADDITIONAL_PROPS,
            extra_properties=EXTRA_PROPERTIES,
        )
        self.logger.info(f"{rp_table_infostr}\n")
        self.logger.info(f"Extract regionprops complete. rpdf shape: {data['rpdf'].shape}")

        # assign channel names
        if not data['rpdf'].empty:
            data["rpdf"]['ch_name'] = data['rpdf']['colocal_id'].map(clcid_name_map)

        # Optionally filter by size.
        #########################################################################
        from SynAPSeg.utils.utils_image_processing import _sanitize_size_range
        size_range = _sanitize_size_range(size_range)
        if size_range is not None:
            min_size, max_size = size_range
            self.logger.info(f"Filtering by size using range ({size_range})... ")
            data["rpdf"] = data["rpdf"][(data["rpdf"]['area'] > min_size) & (data["rpdf"]['area'] < max_size)]
            
        # Optionally extract object intensities in different channels.
        # TODO incorporate into get_rp_table - huge speedup if you just extract multiple channels at once 
        #########################################################################
        if get_clc_intensities:
            self.logger.info(f"Extract object intensities in different channels using (get_clc_intensities:{get_clc_intensities})... ")
            data["rpdf"] = ObjectIntensityExtractor.extract_obj_intensities(
                get_clc_intensities, intensity_img, data["rpdf"], img_fmt
            )

        # Optionally assign ROIs.
        #########################################################################
        

        if (
            # first check if roi handling stage is in pipeline
            ('roi_handling' in self.pipeline.get_stage_names()) & 

            # then check data is valid 
            (data.get('rois') is not None) & 

            ( (data.get("polygons_per_label") is not None) | 
              (data.get("labeled_mask") is not None) )
        ):         

            data["rpdf"] = ROIAssigner.assign_rois_to_rpdf(
                data["rpdf"], 
                data["polygons_per_label"], 
                data["labeled_mask"], 
                roi_assignment_methods=ROI_ASSIGNMENT_METHODS,
                lbls = objects_img,
                voxel_size=voxel_size,
                lbls_fmt = obj_fmt,
                roi_fmt = config["ROIS_FORMATS"][0],
                logger=self.logger,
            )
        
        self.logger.debug(f"current colocal_id counts\n{uc.get_colocal_id_counts(data['rpdf'])}")
        
        return data
    
    def _compile(self, data: dict, config: dict) -> dict:
        
        if 'rpdf' in data:

            # add roi and grouping var to dfs
            #################################################################################
            
            # check if roi assignment exists
            _missing_roi_assignment = (
                data['roi_assignment_column'] is None or 
                data['roi_assignment_column'] not in data['rpdf'].columns
            )
            if _missing_roi_assignment:
                self.logger.warning(f"ROI assignment is missing. Setting roi_i to 0 for all objects.")

            data['rpdf'] = data['rpdf'].assign(
                roi_i = 0 if _missing_roi_assignment else data['rpdf'][data['roi_assignment_column']], 
                **data['assign_md_attrs'],
                **data['extracted_fn_groups']
            )
            self.logger.debug(f"roi_assignment_column: {data['roi_assignment_column']}")
            self.logger.debug(f"assign_md_attrs: {data['assign_md_attrs']}")
            self.logger.debug(f"extracted_fn_groups: {data['extracted_fn_groups']}")
            self.logger.debug(f"grouping_cols: {data['grouping_cols']}")
                        
            # create mean df, any categorical variables defined in EXTRACT_GROUP_MAPS are automatically added
            grouping_cols = [c for c in data['grouping_cols'] if (c in data['rpdf'].columns and not all(pd.isnull(data['rpdf'][c])))]
            mean_rpdf = data['rpdf'].groupby(grouping_cols).mean(numeric_only=True).reset_index()
            sum_rpdf = data['rpdf'].groupby(grouping_cols).size().reset_index().rename(columns={0: 'count'})
            summary_df = pd.merge(sum_rpdf, mean_rpdf, on=grouping_cols, how='left')
            data['summary_df'] = summary_df

        return data



class ObjectIntensityExtractor:
    @staticmethod
    def extract_obj_intensities(GET_CLC_ID_CH_INTENSITIES, intensity_image, whole_rpdf, img_fmt):
        # ... (same as before)
        assert set(['colocal_id', 'label', 'coords']).issubset(whole_rpdf.columns.to_list())
        assert any([clcid in whole_rpdf['colocal_id'].unique() for clcid in GET_CLC_ID_CH_INTENSITIES.keys()]), \
            f"no colocal_ids defined in GET_CLC_ID_CH_INTENSITIES are in rpdf, got:{GET_CLC_ID_CH_INTENSITIES.keys()}, but rpdf colocal_ids: {whole_rpdf['colocal_id'].unique()}"
        ch_axis = img_fmt.index('C')
        spatial_axes = [i for i, c in enumerate(img_fmt) if c in SPATIAL_AXES]
        object_coords = (whole_rpdf
            .groupby('colocal_id')[['label', 'coords']]
            .apply(lambda x: dict(zip(x['label'], x['coords'])))
            .to_dict()
        )
        obj_ints_df_row_list = []
        for clc_id, ch_list in GET_CLC_ID_CH_INTENSITIES.items():
            ch_img = np.take(intensity_image, ch_list, axis=ch_axis)
            for obj_label, obj_coords in object_coords.get(clc_id, {}).items():
                obj_coords_tuple = tuple(obj_coords[:, i] for i in range(len(spatial_axes)))
                idx = [slice(None)] * intensity_image.ndim
                for spatial_idx, coord_vals in zip(spatial_axes, obj_coords_tuple):
                    idx[spatial_idx] = coord_vals
                idx[ch_axis] = slice(None)
                intensities = ch_img[tuple(idx)]
                if intensities.ndim != 2:
                    raise ValueError(f"Expected 2D intensities, got ndim={intensities.ndim}")
                mean_int = np.mean(intensities, axis=-1)
                intensity_dict = {f"intensity_mean_ch{ch}": val for ch, val in zip(ch_list, mean_int)}
                obj_ints_df_row_list.append({
                    'colocal_id': clc_id,
                    'label': obj_label,
                    **intensity_dict
                })
        obj_ints_df = pd.DataFrame(obj_ints_df_row_list)
        merged_rpdf = pd.merge(whole_rpdf, obj_ints_df, on=['colocal_id', 'label'], how='outer')
        return merged_rpdf