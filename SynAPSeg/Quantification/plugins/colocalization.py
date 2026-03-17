#!/usr/bin/env python3
"""
ColocalizationStage

This module defines a pipeline stage that performs colocalization analysis on segmented objects.
The process involves mapping detected objects across image channels to identify overlapping or
co-expressed populations based on user-defined rules.

This ColocalizationStage performs the following steps:

• Checks if colocalization parameters are defined in the metadata.
• Infers channel and colocalization information using `get_imgdb_colocal_nuclei_info`
  and instantiates an `ImgDB`.
• Logs per-channel information using `unpack_array_axis`.
• Performs colocalization with `uc.colocalize` using intersection thresholds.
• Adds a "ch_name" column to the region properties DataFrame (`rpdf_coloc`) based on channel IDs.
• Separates colocalized populations using `uc.separate_colocal_populations`.
• Updates the region properties (`whole_rpdf`) with the colocalization results.
• Appends new entries to the ROI DataFrame (`roi_df`) to reflect additional colocalized populations.
"""

import os
import sys
import numpy as np
import pandas as pd
import pprint

from SynAPSeg.Quantification import BasePipelineStage
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.utils.utils_ImgDB import ImgDB
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.utils import utils_general as ug



__plugin_group__ = 'quantification'
__plugin__ = 'ColocalizationStage'
__parameters__ = 'colocalization.yaml'
__stage_key__ = 'colocalization'

class ColocalizationStage(BasePipelineStage):
    """
    ColocalizationStage performs colocalization on detected objects and updates both the
    region properties DataFrame (whole_rpdf) and the ROI DataFrame.

    Expects:
        - metadata["COLOCALIZE_PARAMS"]["colocalizations"]
        - An object image under config["OBJECTS_IMAGE_NAME"] (e.g., "pred_stardist")
        - "whole_rpdf": Region properties DataFrame
        - "img_fmt": Format string for the image (e.g., "YXC")
        - "roi": Integer-labeled ROI mask
        - "roi_df": ROI property DataFrame

    Config parameters:
        - OBJECTS_IMAGE_NAME
        - COLOC_INTERSECTION_THRESHOLD
        - PLOT_RPDF_EXTRACTION (optional)
    """
    __runOrderPreferences__ = {'before': [], 'after': ['object_detection']}
    __compileOrderPreferences__ = {'before': [], 'after': []}


    def init_outputs(self):
        return [{
            'container_name': 'all_rpdfs',
            'container': [],
            'data_key': 'rpdf'
        }]
    
    
    def _execute(self, data, config) -> dict:
        stage_config = self.get_stage_config(config, __stage_key__)

        SEPARATE_COLOCALIZATIONS = stage_config.get("SEPARATE_COLOCALIZATIONS", True)
        coloc_thresh =      stage_config.get("COLOC_INTERSECTION_THRESHOLD", 0.01)
        coloc_params =      stage_config.get("COLOCALIZATIONS", None)
        
        objects_img_key =   config.get("OBJECTS_IMAGE_NAME", None)
        obj_fmt =           config.get("obj_fmt", None)
        
        md =                data.get("metadata", {})
        md['COLOCALIZE_PARAMS'] = {"colocalizations": coloc_params}
        # coloc_params =      md.get("COLOCALIZE_PARAMS", {}).get("colocalizations", None)
        
        coloc_rpdf_key =    config.get("COLOC_RPDF_KEY", "rpdf")
        roi_df_key =        config.get('ROI_DF_KEY', 'roi_df')
        

        if coloc_params is None:
            self.logger.info("No colocalization parameters provided; skipping colocalization stage.")
            return data
        
        if objects_img_key not in data:
            raise ValueError(f"Objects image not found under key '{objects_img_key}'")
        
        assert obj_fmt is not None

        if not (data[objects_img_key].shape[obj_fmt.index('C')] > 1): 
            raise ValueError(f"colocalization requires > 1 channel. {data[objects_img_key].shape} (fmt={obj_fmt})")
        

        # Infer image channel mapping
        image_channels, clc_nuc_info = MetadataParser.get_imgdb_colocal_nuclei_info(md)
        imgdb = ImgDB(image_channels=image_channels, colocal_nuclei_info=clc_nuc_info)

        self.logger.info(
            f"Performing colocalization with the following channel mapping:\n{pprint.pformat(imgdb.colocal_ids, indent=4)}")
        
        # Perform colocalization
        rpdf_coloc, prt_str = uc.colocalize(
            colocalization_params=imgdb.colocalizations,
            rpdf=data[coloc_rpdf_key],
            label_arr=data[objects_img_key],
            current_format=obj_fmt,
            clc_axes_fmt='C',
            intersection_threshold=coloc_thresh,
        )
        self.logger.info(prt_str)
        
        # Map colocal_id to human-readable names
        rpdf_coloc['ch_name'] = rpdf_coloc['colocal_id'].map({
            k: d['name'] for k, d in imgdb.colocal_ids.items()
        })

        # Separate colocalized populations
        if SEPARATE_COLOCALIZATIONS:
            rpdf_coloc = uc.separate_colocal_populations(rpdf_coloc, imgdb, image_id_column=None, logger=self.logger)
            self.logger.info(f"clc counts post SEPARATE_COLOCALIZATIONS:\n {uc.get_colocal_id_counts(rpdf_coloc)}")

        # Update main data
        data[coloc_rpdf_key] = rpdf_coloc


        # Update ROI DataFrame
        # Duplicate rows of base clc_id and assign them to new colocalizations
        if roi_df_key not in data:
            self.logger.info(f"ROI dataframe ({roi_df_key}) not in data, can not update it with colocal info.")
        else:
            roi_df = data.pop(roi_df_key)
            for clcProps in clc_nuc_info:
                clc_id = clcProps['colocal_id']
                base_clc_id = clcProps['co_ids'][-1]

                roi_df_rows = roi_df[roi_df['colocal_id'] == base_clc_id].copy()
                roi_df_rows = roi_df_rows.assign(colocal_id=clc_id)
                roi_df = pd.concat([roi_df, roi_df_rows], ignore_index=True)

            data[roi_df_key] = roi_df
        
        return data


# ──────────────────────────────────────────────────────────────────────
# Example usage (independent of quantification pipeline)
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Dummy demonstration 
    # TODO may be out of date
    dummy_md = {
        "COLOCALIZE_PARAMS": {"colocalizations": [[0, 1], [1, 0]]},
        "example_i": 0,
        "data_metadata": {
            "channel_info": {0: "Homer", 1: "TRAF3"}
        }
    }

    dummy_objects = np.random.randint(0, 3, size=(256, 256, 1)).astype(np.int32)
    dummy_intensity = np.random.rand(256, 256, 3)
    dummy_rpdf = pd.DataFrame({
        "colocal_id": np.random.randint(0, 2, size=50),
        "label": np.arange(50),
        "area": np.random.randint(20, 100, size=50),
        "coords": [np.array([[10, 20], [30, 40]]) for _ in range(50)],
        "centroid": [(15, 25) for _ in range(50)]
    })
    dummy_roi = np.random.randint(0, 2, size=(256, 256)).astype(np.int32)
    dummy_roi_df = pd.DataFrame({
        "colocal_id": [0],
        "label": [1],
        "area_px": [500]
    })

    dummy_data = {
        "metadata": dummy_md,
        "pred_stardist": dummy_objects,
        "whole_rpdf": dummy_rpdf,
        "img_fmt": "YXC",
        "roi": dummy_roi,
        "roi_df": dummy_roi_df,
        "mip_raw": dummy_intensity
    }

    dummy_config = {
        "OBJECTS_IMAGE_NAME": "pred_stardist",
        "INTENSITY_IMAGE_NAME": "mip_raw",
        "COLOC_INTERSECTION_THRESHOLD": 0.7,
        "PLOT_RPDF_EXTRACTION": False
    }

    stage = ColocalizationStage()
    updated_data = stage.run(dummy_data, dummy_config)

    print("Colocalization Log:")
    print(updated_data["colocalization_log"])
    print("Updated whole_rpdf shape:", updated_data["whole_rpdf"].shape)
    print("Updated roi_df:")
    print(updated_data["roi_df"])
