from __future__ import annotations
import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List, Iterable
import numpy as np
import pandas as pd
import yaml
import ast
from copy import deepcopy

from SynAPSeg.Quantification.BasePipelineStage import BasePipelineStage

    
__plugin_group__ = 'quantification'
__plugin__ = 'ABBAQuantificationPlugin'
__parameters__ = 'ABBA_quantification_plugin.yaml'
__stage_key__ = 'ABBA_quantification_plugin'

class ABBAQuantificationPlugin(BasePipelineStage):
    """
    Plugin stage for handling quantification of objects within regions defined by ABBA registered images
    """
    __runOrderPreferences__ = {'before': [], 'after': ['object_detection', 'colocalization']}
    __compileOrderPreferences__ = {'before': ['object_detection'], 'after': []}
    __blocksCompile__ = ['object_detection']

    def init_outputs(self):
        # name: container, key in data: name
        return [
            {'container_name': 'all_region_dfs',
            'container': [],
            'data_key': 'region_df'}, 
        ]
    
    @property
    def output_specifications(self):
        return ["region_df"]
    
    def _execute(self, data: dict, config: dict) -> dict:
        
        # local imports 
        # ----------------------------------------------------------------------------
        from SynAPSeg.utils import utils_general as ug
        from SynAPSeg.Plugins.ABBA import core_regionPoly as rp
        from SynAPSeg.Plugins.ABBA import utils_atlas_region_helper_functions as arhfs
        from SynAPSeg.utils.utils_geometry import (
            bbox_to_geojson_feature,
            bbox_hole_to_geojson_feature,
        )

        # config parsing
        ####################################################################################
        stage_config = self.get_stage_config(config, __stage_key__)
        
        STRUCTS_PATH = stage_config["STRUCTS_PATH"]
        ROIS_NAME = stage_config['ROIS_NAME']
        HANDLE_ROI_TYPE = stage_config.get('HANDLE_ROI_TYPE') or 'exclude'
        ROI_READER = stage_config.get('ROI_READER') or 'napari_csv'
        ROI_SPATIAL_COLS = stage_config['ROI_SPATIAL_COLS']
        X_COL = ROI_SPATIAL_COLS.get('X')
        Y_COL = ROI_SPATIAL_COLS.get('Y')
        UM_PER_PIXEL = config['PX_SIZES'].get('X') or 1.0

        path_to_example = config['path_to_example']


        # load ABBA ontology and registration .geojson -> convert to polyCollection
        ####################################################################################
        self.logger.debug('loading ontology...')
        
        # TODO make this a user-adjustable setting
        stage_config["GEOJSON_PATH"] = os.path.join(
            path_to_example, 
            "qupath", "qupath_export_geojson", 
            f"{Path(config['INTENSITY_IMAGE_NAME']).stem.replace('.ome','')}.geojson"
        )

        if not os.path.exists(stage_config["GEOJSON_PATH"]):
            raise ValueError(f"path to registration does not exist at: {stage_config['GEOJSON_PATH']}")
        
        if not os.path.exists(stage_config["STRUCTS_PATH"]):
            raise ValueError(f"path to ontology structures does not exist at: {stage_config['STRUCTS_PATH']}")
        
        # load atlas ontology and registration
        ont = arhfs.Ontology(pd.read_csv(STRUCTS_PATH))

        regionPolys = rp.polyCollection(
            stage_config["GEOJSON_PATH"], 
            ont=ont
        )
                
        # ROI handling
        ######################################################################################################
        # TODO handle arbitrary rois <-- basically implemented in regions to exclude 
        # but current implementation only supports exclusion of these regions

        self.logger.info('processing ROIs...')        
        HAS_ROIS = ROIS_NAME in os.listdir(path_to_example)

        img_shape = data[config['INTENSITY_IMAGE_NAME']].shape
        img_fmt = config['img_fmt']

        image_bounds_geojson = bbox_to_geojson_feature(
            (0.0, float(img_shape[img_fmt.index('Y')]), 
            0.0, float(img_shape[img_fmt.index('X')]))
        )

        # if regions to exclude, subtract these polys (exported from Napari) from the image's boundary
        if ROI_READER == "napari_csv" and HANDLE_ROI_TYPE == "exclude" and HAS_ROIS:
            self.logger.info('areas to exclude detected')
            # 1) Parse CSV → exclusions GeoJSON + Shapely polygons
            excl_df = pd.read_csv(os.path.join(path_to_example, ROIS_NAME))
            exclusions_fc, exclusion_polys = rp.parse_Napari_shapes_to_polygons(excl_df, x_col=X_COL, y_col=Y_COL)  

            # 3) Subtract from image bounds ROI poly
            image_bounds_shapely = rp.roi_feature_to_polygon(image_bounds_geojson)
            roi_minus = rp.subtract_exclusions_from_roi(image_bounds_shapely, exclusion_polys)
            
            roi_regionPolys = rp.polyCollection(
                geojsonPolyObjs = [
                    rp.geojsonPoly(
                        rp.shapely_to_geojson(roi_minus), 
                        reg_id=1
                    )
                ]
            )
        elif HANDLE_ROI_TYPE == "include" and HAS_ROIS:
            # need to update rpdf roi's based on location of regpoly within roi context 
            # todo this I think roi0 needs to be automaticalyy set to image bounds then roi 1 can exist within this region
            raise NotImplementedError
        
        else:  # whole‑image ROI
            roi_regionPolys = rp.polyCollection(
                geojsonPolyObjs = [
                    rp.geojsonPoly(
                        image_bounds_geojson, 
                        reg_id=1
                    )
                ]
            )

        # constrain region polys to ROIs - essentially this subdivides the region polys into smaller polygons
        region_df, constrained_regionPolys = rp.constrain_region_area_by_rois(
            regionPolys.polygons, 
            roi_regionPolys.polygons, 
            um_per_pixel=UM_PER_PIXEL
        )    
        
        constrained_regionPolys = constrained_regionPolys[1] # TODO handle multiple rois - here we take 1 b/c 0 is background
        
        # ROI processing sanity check 
        #----------------------------------------------------------------
        if bool(0): # plot rois overlayed on image (downscaled)
            import skimage.transform
            import matplotlib.pyplot as plt
            from SynAPSeg.utils import utils_plotting as up
            
            constrained_regionPolys.plot()
            
            assert intensity_img.ndim==3, intensity_img.shape

            # downscale img and polycollection 
            dsimg = skimage.transform.rescale(intensity_img[(config.get("GET_CHS") or [0])[0]], 1/25)
            scaled_rpc = rp.scale_polyCollection(constrained_regionPolys, 1/25)
            
            
            fig,ax = plt.subplots(figsize=(10,8))
            up.show(dsimg, ax=ax)
            bx1,bx2,by1,by2 = scaled_rpc.plot(ax=ax)
            plt.show()
            
        # ------------------------------------------------------------------
        # ROI assignment - Map detections to polygons using numba fast impl
        # ------------------------------------------------------------------        
        rpdf = data['rpdf']
        
        # convert centroids from image to polygon coordinates
        centroids = np.array([c[::-1] for c in rpdf["centroid"].to_list()])
        
        # iter detections, assign to roi_i's
        roi_nb_singles, roi_nb_multis, roi_infos = rp.separate_polytypes(constrained_regionPolys)
        roi_pp_result = rp.nb_process_polygons(roi_nb_singles, roi_nb_multis, centroids)
        
        # extract info from assigned poly - indexing into info and using poly index
        rpdf = (
            pd.DataFrame(list(
                np.array(
                    roi_infos + [{k: np.nan for k in roi_infos[0]}]  # empty info used for mapping unassigned objects (poly_i == -1)
                )[roi_pp_result]
            ))
            .assign(
                centroid_i=np.arange(len(centroids)),
                roi_i=1, # TODO handle multiple rois 
                # **(stage_config["assign_rpdf_attributes"] or {}),
            )
            .merge(rpdf, left_on="centroid_i", right_index=True, how="left")
        )
        
        # drop rows where poly_index = nan - these ones were not assigned to any region 
        rpdf = rpdf.dropna(subset=['poly_index'])
        rpdf['region_sides'] = rpdf['reg_side'] # for compatibility with propogation


        data['rpdf'] = rpdf
        data['region_df'] = region_df
        data['ont'] = ont
        data['constrained_regionPolys'] = constrained_regionPolys

        return data



    
    def _compile(self, data: dict, config: dict) -> dict:
        
        from SynAPSeg.Analysis.df_utils import filter_present_cols
        from SynAPSeg.Plugins.ABBA import core_compileCounts as Compile
        
        # ------------------------------------------------------------------
        # Compile detections by region - Propogate detections to full region counts 
        # ------------------------------------------------------------------
                       
        # extract 
        get_region_df_cols = [
            'region_sides',
            'st_level',
            'reg_id',
            'region_name',
            'acronym',
            'og_region_area_px',
            'region_area_px',
            'region_area_um',
            'region_area_mm',
        ]
        ignore_cols = [
            'centroid_i',
            'label',
            'bbox',
            'centroid',
        ]
        extract_mean_columns = [
            'area',
            'num_pixels',
            'axis_major_length',
            'perimeter',
            'axis_minor_length',
            'intensity_mean',
            'intensity_max',
            'intensity_min',
            'intensity_std',
            'solidity',
            'eccentricity',    
            'skewedness', 
            'kurtosis', 
            'circularity'
        ]

        rpdf_final = data['rpdf']
        region_df = data['region_df']
        ont = data['ont']
        
        stage_config = self.get_stage_config(config, __stage_key__)

        get_region_df_cols = filter_present_cols(region_df, get_region_df_cols)
        extract_mean_columns = filter_present_cols(rpdf_final, extract_mean_columns)
        ignore_cols = filter_present_cols(rpdf_final, ignore_cols)
      
        
        # structural indexers (rpdf cols) which define unique sets of region polys (have different mapping of poly_i to regions)
        REGION_INDEXERS = filter_present_cols(region_df,
            stage_config.get('REGION_INDEXERS') or ['roi_i', 'region_sides']
        )
        # indexers which define populations which may have unique region counts within a structural context    
        POPULATION_INDEXERS = filter_present_cols(rpdf_final,
            stage_config.get('POPULATION_INDEXERS') or ['roi_i', 'colocal_id', 'ch_name']
        )             
        REGION_AREA_COL = stage_config.get('REGION_AREA_COL') or 'region_area_mm'
        LARGEST_HEMISPHERE_ONLY = stage_config.get('LARGEST_HEMISPHERE_ONLY') or False
        
        # optionally, only compile counts for a single side 
        if LARGEST_HEMISPHERE_ONLY:
            side_counts = rpdf_final.groupby(['region_sides']).size()
            largest_side = side_counts.idxmax()
            self.logger.info("\n".join([
                'filtered data to include side with largest number of detections only:'
                f" kept {largest_side} (detection counts: {side_counts.to_dict()})"
            ]))
            rpdf_final = rpdf_final[rpdf_final['region_sides'] == largest_side]
            region_df = region_df[region_df['region_sides'] == largest_side]


        EXP_LUT = Compile.map_poly_hierarchy(
            region_df,
            ont,
            REGION_INDEXERS
        )
        
        PRP_LUT = Compile.make_PRP_LUT_merged(
            rpdf_final, 
            EXP_LUT, 
            POPULATION_INDEXERS, 
            REGION_INDEXERS
        )

        final_counts = Compile.extract_counts(
            rpdf_final, region_df, 
            PRP_LUT, POPULATION_INDEXERS, REGION_INDEXERS,
            extract_mean_columns, get_region_df_cols
        ).assign(
                density = lambda df: df['count']/df[REGION_AREA_COL],
                **data['assign_md_attrs'],
                **data['extracted_fn_groups'],
                **{'density_area_col': REGION_AREA_COL},
        )

        region_df = region_df.assign(
            **data['assign_md_attrs'],
            **data['extracted_fn_groups'],
        )

        rpdf_final = rpdf_final.assign(
            **data['assign_md_attrs'],
            **data['extracted_fn_groups'],
        )

        data['summary_df'] = final_counts
        data['region_df'] = region_df
        data['rpdf'] = rpdf_final

        return data

