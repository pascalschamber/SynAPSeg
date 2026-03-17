#!/usr/bin/env python3
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

from SynAPSeg.Quantification import BasePipelineStage
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.utils import utils_colocalization_3D
from SynAPSeg.config.constants import STANDARD_FORMAT, DISPLAY_FORMAT


__plugin_group__ = 'quantification'
__plugin__ = 'ROIHandlingStage'
__parameters__ = 'roi_handling.yaml'
__stage_key__ = 'roi_handling'        # this is name of plugin

class ROIHandlingStage(BasePipelineStage):
    """
    ROIHandlingStage summarizes the morphological/intesnsity properties of an roi and prepares 
        it for use to localize objects within roi subregions

    Expects the input data dictionary to contain:
        "rois": a list of ROI arrays (only 1 supported as of now) or a geojson feature collection.
        A normalized intensity image under the key specified by INTENSITY_IMAGE_NAME.

    Uses configuration parameters:
        REMAP_ROIS: dictionary for remapping ROI values (or null).
        ROI_REMOVE_SMALL_OBJ_SIZE: threshold for removing small ROI objects.
        REID_ROIS: boolean; if True, reassign ROI labels.
        PLOT_RPDF_EXTRACTION: boolean; if True, plot overlays for debugging.
        INTENSITY_IMAGE_NAME: key name in data for intensity image.
        img_fmt: format string (e.g., "YXC") for the intensity image.

    Updates the data dictionary with:
        "labeled_mask": a mask obtained from polygon conversion of the ROI.
        "polygons_per_label": the polygon representation of the ROI.
        "roi_df": a dataframe summarizing ROI properties.
    """
    __runOrderPreferences__ = {'before': ['object_detection'], 'after': []}
    __compileOrderPreferences__ = {'before': [], 'after': ['object_detection']}

    def init_outputs(self):
        # name: container, key in data: name
        return [{
            'container_name': 'all_roi_dfs',
            'container': [],
            'data_key': 'roi_df'
        }]

    def __call__(self, roi, intensity_image, img_fmt, INTENSITY_IMAGE_NAME=None, REID_ROIS=False, ALLOW_ROI_IMG_SHAPE_MISMATCH=True, **config_kwargs):
        """ 
        OUTDATED - NOT FUNCTIONAL
        user friendly wrapper function for useing outside pipeline's automatic context 
            pulled out commonly used kwargs, but more are described in execute or default config parameters
        
        # TODO this needs to be updated since adding of 3d roi assignment methods
        
        Returns
            dict with keys: ['rois', 'mip_raw', 'labeled_mask', 'polygons_per_label', 'roi_df']
        """

        INTENSITY_IMAGE_NAME = INTENSITY_IMAGE_NAME or 'mip_raw'

        return self.run(
            data={
                'rois':[roi],
                INTENSITY_IMAGE_NAME:intensity_image,
            },
            config = {
                'INTENSITY_IMAGE_NAME':INTENSITY_IMAGE_NAME,
                'img_fmt': img_fmt,
                'REID_ROIS':REID_ROIS,
                'ALLOW_ROI_IMG_SHAPE_MISMATCH': ALLOW_ROI_IMG_SHAPE_MISMATCH,
                **config_kwargs
            }
        )

    def _execute(self, data: dict, config: dict) -> dict:
        stage_config = self.get_stage_config(config, __stage_key__)

        # skip 
        if ("rois" not in data):
            self.logger.warning(f" skipping... \n\trois not in data.keys() but roi handling stage in quant pipeline.\n\tensure this behavior is desired, remove roi handling from pipeline.stages, or declare rois in config filemap ")
            return data
        
        # get rois from data
        #############################################################################################
        if (not isinstance(data['rois'], list)) or (len(data['rois'])==0):
            return self.raise_exit_flag(f"Input data does not contain 'rois'\ngot:{data.get('rois')}\nensure quant_config FILE_MAP['ROIS'] is set correctly\n", data)

        rois = data["rois"] # either list[np.ndarray] or geojsonPolyCollection
        
        # Parameters from config - all can be handled safely if not provided?
        #############################################################################################
        intensity_image_key =           config.get("INTENSITY_IMAGE_NAME", None)
        img_fmt =                       config.get("img_fmt", "ZYX") 
        rois_formats =                  config.get("ROIS_FORMATS", ["YX"])
        roi_types =                     config.get("ROI_TYPES", ["mask"]) # must be mask or polygon
        
        remap_rois =                    stage_config.get("REMAP_ROIS", None)
        size_range =                    uip._sanitize_size_range(stage_config.get("ROI_OBJECTS_SIZE_RANGE"))
        reid_rois =                     stage_config.get("REID_ROIS", False)
        ALLOW_ROI_IMG_SHAPE_MISMATCH =  stage_config.get("ALLOW_ROI_IMG_SHAPE_MISMATCH", False) # whether to raise error if roi format != img format 
        HANDLE_SHAPE_MISMATCH_KEY =     stage_config.get("HANDLE_ROI_IMG_SHAPE_MISMATCH", "img") # transform format to this object, must be either ["img" or "roi"]

        plot_extraction =               stage_config.get("PLOT_RPDF_EXTRACTION", False) 

        # get region prop args 
        PX_SIZES =                      config.get("PX_SIZES") # PX_SIZE_XY = config.get("PX_SIZE_XY", 1)

        rps_to_get =                    stage_config.get('RPS_TO_GET')
        ADDITIONAL_PROPS =              stage_config.get("ROI_ADDITIONAL_PROPS")
        GET_OBJECT_COORDS =             stage_config.get("ROI_GET_OBJECT_COORDS") or False 
        EXTRA_PROPERTIES =              stage_config.get("ROI_EXTRA_PROPERTIES")

        
        self.logger.debug((
            f"roi_handing stage config:\n"
            f" img_fmt: {img_fmt} | rois_formats: {rois_formats} | roi_types: {roi_types}\n"
            f" PX_SIZES: {PX_SIZES}\n PROPERTIES: {rps_to_get}\n Extra:{EXTRA_PROPERTIES}\n ADDITIONAL_PROPS: {ADDITIONAL_PROPS}\n"
        ))

        # take first roi for now - TODO build support for multiple ROIs
        #############################################################################################
        roi_array = rois[0]          # can be np.array or polyCollection
        roi_type = roi_types[0]
        roi_fmt = rois_formats[0]

        IS_3D = 'Z' in roi_fmt

        # ROI preprocessing
        #############################################################################################
        NEED_ROI_AS_ARRAY = False # TODO make proper knob - if False, saves alot of time if we don't need roi pixel intensities 

        # handle roi polygons
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if roi_type == 'polygon': 

            # for standard processing - functions require polygons as shapely objects as dict mapped it's respective label
            from SynAPSeg.Plugins.ABBA.core_regionPoly import polyCollection
            polys: polyCollection = roi_array
            polygons_per_label = {
                i+1:[p.to_shapely()] for i,p in enumerate(polys.polygons)
            }

            # Convert polygons to labeled mask
            if NEED_ROI_AS_ARRAY:
                baseshape = data[intensity_image_key].shape
                _shape = tuple([baseshape[img_fmt.index(dim)] for dim in 'YX'])
                self.logger.debug(f"Converting polygons to array using intensity img shape: {_shape}.")
                labeled_mask = uc.create_labeled_mask(polygons_per_label, _shape)
                self.logger.debug("Converted polygons to labeled ROI mask.")
            else:
                labeled_mask = None
        else:
            labeled_mask = roi_array


        # handle roi array
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if roi_type == 'mask':
            if remap_rois is not None:
                roi_array = uip.map_arr_values(roi_array, remap_rois)
                self.logger.debug("Applied remapping of ROI IDs.")

            # filter by size
            from SynAPSeg.utils.utils_image_processing import _sanitize_size_range, filter_area_objects
            size_range = _sanitize_size_range(size_range)
            if size_range is not None:
                nlbls = len(uip.unique_nonzero(roi_array))
                roi_array = filter_area_objects(roi_array, size_range, preserve_labels=True)
                self.logger.debug(f"Removed {nlbls - len(uip.unique_nonzero(roi_array))} ROI objects using size range: {size_range}.")

            # Reassign ROI labels
            if reid_rois:
                roi_array = uip.relabel(roi_array)
                self.logger.debug("Relabeled ROI objects.")
    
            # Convert to polygons - only 2d input (or CYX) supported here
            if roi_fmt=='YX':  
                polygons_per_label = uc.semantic_to_polygons_rasterio(roi_array)
                self.logger.debug("Converted ROI array to polygon representation.")
            
            elif roi_fmt == 'CYX': # if has ch axis, YX p.p.l are nested inside dict with ch indicies as keys
                polygons_per_label = {
                    ch_i: uc.semantic_to_polygons_rasterio(a) 
                    for ch_i, a in 
                    enumerate(uip.unpack_array_axis(roi_array, roi_fmt.index('C')))
                }
                self.logger.debug("Converted ROI array to polygon representation - channels are keys.")
            else:
                polygons_per_label = None
                self.logger.debug(f"Polygon representation of ROI array is not supported for ROI with format: {roi_fmt}. Passing with null value.")



        # get intensity img if provided
        #############################################################################################
        if intensity_image_key is None: # if not provided make an empty array with same shape as roi
            self.logger.info(f"intensity_image_key not provided, making an empty array with same shape as roi")
            intensity_img = np.zeros_like(labeled_mask, dtype='uint8')
            img_fmt = roi_fmt 

        elif intensity_image_key not in data:
            raise ValueError(f"Intensity image key '{intensity_image_key}' was provided but not found in data.")

        else:
            intensity_img = data[intensity_image_key]
            assert intensity_img.ndim == len(img_fmt), f"intensity_img.ndim != len(img_fmt), got: {intensity_img.ndim} != {len(img_fmt)}"
        


        # handle intensity image formating 
        if roi_type == 'mask' or NEED_ROI_AS_ARRAY:
            
            # handle intensity image <-> ROI format matching
            #############################################################################################
            # only transform if spaital axes mismatch
            spatial_axes_img = [c for i, c in enumerate(img_fmt) if c in "ZYX"]
            spatial_axes_mask = [c for i, c in enumerate(roi_fmt) if c in "ZYX"]
            _spaital_axes_mismatch = (set(spatial_axes_img) != set(spatial_axes_mask))
            self.logger.debug(
                f"intensity_img shape: {intensity_img.shape}, labeled_mask shape: {labeled_mask.shape}\n"
                f"img-roi spaital axes match: {not _spaital_axes_mismatch}\n\tspatial_axes img:{set(spatial_axes_img)}, roi:{set(spatial_axes_mask)}\n"
                f"\timg_fmt: {img_fmt}, roi_fmt: {roi_fmt}"
            )

            # handle img-ROI shape mismatch - e.g. insert z dim to mask if in intensity_img but not in mask - think ROI areas might be off then
            # while will also handle case where image is transformed to match roi format the data will not be updated with the new format
            if (labeled_mask.shape != intensity_img.shape): # and _spaital_axes_mismatch: <-- this prevents reshaping along e.g. C axis
                                
                if not ALLOW_ROI_IMG_SHAPE_MISMATCH:
                    raise ValueError(f"ROI and intensity_img shape mismatch, got: ({labeled_mask.ndim} != {intensity_img.ndim}) & ALLOW_ROI_IMG_SHAPE_MISMATCH={ALLOW_ROI_IMG_SHAPE_MISMATCH}")

                self.logger.warning(f"shape mismatch between labeled_mask and intensity_img (key={HANDLE_SHAPE_MISMATCH_KEY}) - attempting to handle")

                target_fmt = img_fmt if HANDLE_SHAPE_MISMATCH_KEY == "img" else roi_fmt
                transformed = uip.morph_to_target_shape(
                    # get current and target formats and shapes
                    arr = labeled_mask if HANDLE_SHAPE_MISMATCH_KEY == "img" else intensity_img,
                    current_fmt = roi_fmt if HANDLE_SHAPE_MISMATCH_KEY == "img" else img_fmt,
                    target_shape = intensity_img.shape if HANDLE_SHAPE_MISMATCH_KEY == "img" else labeled_mask.shape,
                    target_fmt = target_fmt
                )

                # update current object to the transformed object
                if HANDLE_SHAPE_MISMATCH_KEY == "img":
                    labeled_mask = transformed
                    roi_fmt = target_fmt

                else:
                    intensity_img = transformed
                    img_fmt = target_fmt

                self.logger.debug(f"handled roi intensity_img shape mismatch (key={HANDLE_SHAPE_MISMATCH_KEY}): labeled_mask shape: {labeled_mask.shape}, intensity_img shape: {intensity_img.shape}")

            # Optional debug plotting
            #############################################################################################
            if plot_extraction:
                if polygons_per_label is None:
                    self.logger.error(f"cannot plot_extraction because polygons_per_label is None.")
                else:
                    show_ch = 0
                    show_ppl = polygons_per_label[show_ch] if roi_fmt=='CYX' else polygons_per_label
                    uc.plot_polygons_over_image(roi_array, show_ppl)
                    if intensity_image_key in data:
                        intensity_img = data[intensity_image_key]
                        composite_img = uip.transform_axes(intensity_img, img_fmt, STANDARD_FORMAT)
                        composite_img = uip.reduce_dimensions(composite_img, STANDARD_FORMAT, project_dims=uip.subtract_dimstr(STANDARD_FORMAT, DISPLAY_FORMAT))

                        composite_img = up.create_composite_image_with_colormaps(
                            composite_img, ['blue', 'green', 'red', 'magenta']
                        )
                        uc.plot_polygons_over_image(composite_img, show_ppl)

        # Summarize ROI properties
        #############################################################################################
        self.logger.info('running ROI feature extraction...')
        if roi_type == 'mask':
            roi_df = uc.summarize_roi_array_properties(
                intensity_img,
                labeled_mask,
                img_fmt,
                roi_fmt,
                coerce_roi_fmt=False, # this should already be handled more sophisticatedly above
                PX_SIZES=PX_SIZES,
                rps_to_get = rps_to_get,
                get_object_coords=GET_OBJECT_COORDS,
                additional_props=ADDITIONAL_PROPS,
                extra_properties=EXTRA_PROPERTIES,
            )
        else:
            roi_df = uc.summarize_roi_properties(
                polygons_per_label, 
                intensity_img, 
                labeled_mask, 
                img_fmt, 
                PX_SIZES=PX_SIZES,
            )
        self.logger.debug("Summarized ROI properties into a dataframe.")

        # Update data
        data["labeled_mask"] = labeled_mask
        data["polygons_per_label"] = polygons_per_label
        data["roi_df"] = roi_df

        return data

    def _compile(self, data: dict, config) -> dict:
        """
        Compile the data into a summary dataframe.
        """
        if 'roi_df' in data:
            data['roi_df'] = (data['roi_df'].assign(
                **data['assign_md_attrs'],
                **data['extracted_fn_groups'],
            ))
            
            # append roi info to summary
            if data['summary_df'] is not None:
                grouping_cols = [c for c in data['grouping_cols'] if (c in data['roi_df'].columns and not all(pd.isnull(data['roi_df'][c])))]
                summary_df = pd.merge(left=data['summary_df'], right=data['roi_df'], on=grouping_cols, how='left')
                summary_df['count_per_um'] = summary_df['count'] / summary_df['roi_area_um']
                data['summary_df'] = summary_df
                
        return data



class ROIAssigner:
    """Handles assignment of objects to ROIs using various methods."""
    
    SUPPORTED_METHODS = ['Centroid', 'Coords', 'Masking', 'Distance'] + ['Overlap3D', 'Distance3D']
    
    
    @staticmethod
    def assign_rois_to_rpdf(
        rpdf: pd.DataFrame, 
        polygons_per_label: Optional[Dict[Any, List] | Dict[int, Dict[Any, List]]] = None, 
        roi: Optional[np.ndarray]=None,
        roi_assignment_methods: Optional[List[str]] = None,
        lbls: Optional[np.ndarray] = None,
        voxel_size: Optional[list[float]] = None,
        lbls_fmt: Optional[str] = None,
        roi_fmt: Optional[str] = None, 
        logger: Optional[Any] = None, 
    ) -> pd.DataFrame:
        """
        Assign ROIs to objects in the dataframe using specified methods.
            Note: optionality of input args depends on method used.
        
        Args:
            rpdf: DataFrame containing object data with 'coords' and 'centroid' columns
            polygons_per_label: Dictionary mapping labels to polygon lists, not used for 3D
                if roi array is CYX fmt, YX p.p.l are nested inside dict with ch indicies as keys
            roi: ROI mask array. Required for masking and 3D methods
            roi_assignment_methods: List of methods to apply. Defaults to ['Coords']
            lbls: labeled objects array. Required for 3D
            voxel_size: image voxel size (spacing). Used in distance 3d. default [1,1,1]
        
        Returns:
            DataFrame with added ROI assignment columns
            
        Raises:
            ValueError: If unsupported methods are specified or method application fails
        """

        # TODO: add support for multiple ROIs - in config need to represent as list of dicts 
        # with keys for data_key to get roi_array, roi_type (mask or polygon), and roi_assignment_methods
        
        method_handlers = {
            'Coords': ROIAssigner._apply_coords_method,
            'Distance': ROIAssigner._apply_distance_method,
            'Centroid': ROIAssigner._apply_centroid_method,
            'Masking': ROIAssigner._apply_masking_method,
            'Overlap3D': ROIAssigner._apply_overlap3d_method,
            'Distance3D': ROIAssigner._apply_distance3d_method,
        }

        if roi_assignment_methods is None:
            roi_assignment_methods = ['Coords']
        elif isinstance(roi_assignment_methods, str):
            roi_assignment_methods = [roi_assignment_methods]
        
        if logger is not None:
            logger.info(f"Assigning objects to rois using (ROI_ASSIGNMENT_METHODS: {roi_assignment_methods})... ")

        # Validate methods
        ROIAssigner._validate_methods(roi_assignment_methods)
        
        # Create a copy to avoid modifying the original
        result_df = rpdf.copy()

        # handle roi reshaping 
        if all([el is not None for el in [roi, lbls, roi_fmt, lbls_fmt]]):
            if roi.shape != lbls.shape:
                roi = uip.morph_to_target_shape(roi, roi_fmt, lbls.shape, lbls_fmt)
                roi_fmt = lbls_fmt
        
        # Pre-compute commonly used data. TODO also using to pass method-specific params (e.g. voxel_size) but would be better to handle separately
        computed_data = ROIAssigner._precompute_data(
            result_df, roi_assignment_methods, lbls, roi, voxel_size, lbls_fmt, roi_fmt
        )

        
        # Apply each method
        applied_methods = []
        for method in roi_assignment_methods:
            try:
                method_handlers[method](result_df, polygons_per_label, roi, computed_data)
                applied_methods.append(method)
                if logger is not None:
                    logger.info(f"Successfully applied ROI_ASSIGNMENT_METHOD: {method}")
            except Exception as e:
                raise ValueError(f"Failed to apply {method} method: {str(e)}\n") from e
        
        # Verify all methods were applied
        if len(roi_assignment_methods) != len(applied_methods):
            raise ValueError(
                f"Not all requested ROI assignment methods were applied. "
                f"Requested: {roi_assignment_methods}, applied: {applied_methods}"
            )
        
        computed_data_info = ''
        for k,v in computed_data.items():
            fval = f"shape:{v.shape}" if isinstance(v, np.ndarray) else f"len({len(v)})" if isinstance(v, list) else str(v)
            computed_data_info += f"{k}: {fval}\n"
        logger.debug(f"Computed data: \n{computed_data_info}")
        
        return result_df
    
    @staticmethod
    def _validate_methods(roi_assignment_methods: List[str]) -> None:
        """Validate that all requested methods are supported."""
        unsupported = set(roi_assignment_methods) - set(ROIAssigner.SUPPORTED_METHODS)
        if unsupported:
            raise ValueError(
                f"Unsupported ROI assignment methods: {unsupported}. "
                f"Supported: {ROIAssigner.SUPPORTED_METHODS}"
            )
    
    @staticmethod
    def _precompute_data(
        df: pd.DataFrame, 
        methods: List[str], 
        lbls: Optional[np.ndarray] = None,
        roi: Optional[np.ndarray]=None,
        voxel_size: Optional[list[float]] = None,
        lbls_fmt: Optional[str] = None,
        roi_fmt: Optional[str] = None, 
    ) -> Dict[str, Any]:
        """Pre-compute data needed by multiple methods to avoid redundant calculations."""
        
        computed: dict = {}

        # add fmt info
        computed['lbls_fmt'] = lbls_fmt
        computed['roi_fmt'] = roi_fmt
        computed['roi_num_channels'] = None
        if isinstance(roi,np.ndarray) and isinstance(roi_fmt, str) and ('C' in roi_fmt):
            computed['roi_num_channels'] = roi.shape[roi_fmt.index('C')]

        # Centroid data needed by Distance, Centroid, and Masking methods
        centroid_methods = {'Distance', 'Centroid', 'Masking'}
        if any(method in centroid_methods for method in methods):
            centroids_list = df['centroid'].to_list()
            computed['centroids_list'] = centroids_list
            
            # For Distance and Centroid methods (need coordinate reversal)
            if any(method in {'Distance', 'Centroid'} for method in methods):
                computed['object_centroids'] = np.array([
                    np.array(c[::-1]) for c in centroids_list
                ])
            
            # For Masking method (need integer coordinates)
            if 'Masking' in methods:
                computed['masking_coords'] = np.rint(np.array(centroids_list)).astype(int)
        
        # Coords data (only needed by Coords method)
        if 'Coords' in methods:
            coords_list = df['coords'].to_list()
            computed['coords_geometric'] = uc.convert_coordinates_image_to_geometric(coords_list)
            computed['sorted_coords'] = [
                uc.sort_coordinates_by_distance(coords)[0] 
                for coords in computed['coords_geometric']
            ]
        
        # 3D methods - TODO: add accepting inputs for distance radius/voxel_size
        methods_3d = ['Distance3D', 'Overlap3D']
        computed['voxel_size'] = voxel_size # currently only used by 3d distance 
               
        if any(method in methods_3d for method in methods):
            if lbls is None: 
                raise ValueError('3D methods require original labels array used to construct rpdf')
                    # basic validation checks
            if lbls is None or roi is None:
                raise ValueError("3D label arrays not found in computed_data")
            if lbls.shape != roi.shape:
                raise ValueError(f"lbls.shape != roi.shape, {lbls.shape, roi.shape}")
            if 'CZYX' != lbls_fmt or lbls.ndim != 4: # cleaner if just enforce standard format
                raise ValueError(f"{lbls_fmt}, {lbls.shape}")
            if 'label' not in df.columns:
                raise ValueError('`label` column required')
            
            if 'colocal_id' not in df.columns:
                df['colocal_id'] = 0
            
            # we assume colocal_ids map directly to image channel indicies
            # but this could be more explicitly handled, if have access to imgdb object
            ch_axis = lbls_fmt.index('C')
            if set(df['colocal_id'].unique()) != set(range(lbls.shape[ch_axis])):
                raise ValueError(f"{set(df['colocal_id'].unique())} != {set(range(lbls.shape[ch_axis]))}")
            
            computed['lbls'] = lbls
            computed['ch_axis'] = ch_axis
        
        return computed
    
    @staticmethod
    def _apply_coords_method(
        df: pd.DataFrame, 
        polygons_per_label: Dict, 
        roi: np.ndarray, 
        computed_data: Dict
    ) -> None:
        """
        Apply the Coords method for ROI assignment.
        
        Creates cols in df:
            roi_i_byCoords, roi_polyi_byCoords
        """
        
        assert polygons_per_label is not None, "polygons_per_label is None.\n  Note if using a 3D ROI this is expected, so instead use a 3D assignment method like `Overlap3D`"
        
        # initialize the output columns in the df
        df['roi_i_byCoords'] = np.nan
        df['roi_polyi_byCoords'] = np.nan
        
        if computed_data['roi_num_channels'] is None: 
            df['roi_i_byCoords'], df['roi_polyi_byCoords'] = \
                uc.assign_labels_to_object_indices(
                    computed_data['sorted_coords'], polygons_per_label
                )
            
        # if roi array has multiple channels 
        # slice subset of detections in this channel (colocal_id) and localize to roi_array for each channel 
        else: 
            # create temp col in df with sorted_coords, facilitates mapping df row's coords to colocal_ids
            df['sorted_coords'] = computed_data['sorted_coords']
            
            for ch_i, ppl in polygons_per_label.items():
                _mask = df['colocal_id']==ch_i
                if len(df.loc[_mask]) == 0:
                    print(f"No detections found for colocal_id {ch_i}")
                    continue
                
                df.loc[_mask, 'roi_i_byCoords'], df.loc[_mask, 'roi_polyi_byCoords'] = \
                    uc.assign_labels_to_object_indices(
                        df.loc[_mask, 'sorted_coords'].to_list(), ppl
                    )
            
            df.drop(['sorted_coords'], axis=1, inplace=True)


        

    
    @staticmethod
    def _apply_distance_method(
        df: pd.DataFrame, 
        polygons_per_label: Dict, 
        roi: np.ndarray, 
        computed_data: Dict
    ) -> None:
        """Apply the Distance method for ROI assignment."""
        nearest_labels, nearest_poly_indices, distances = uc.compute_distances_to_rois(
            polygons_per_label, computed_data['object_centroids']
        )
        df['roi_i_byDistance'] = nearest_labels
        df['roi_polyi_byDistance'] = nearest_poly_indices
        df['roi_distance'] = distances
    
    @staticmethod
    def _apply_centroid_method(
        df: pd.DataFrame, 
        polygons_per_label: Dict, 
        roi: np.ndarray, 
        computed_data: Dict
    ) -> None:
        """Apply the Centroid method for ROI assignment."""
        assigned_ids_centroid, poly_subindices_centroid = uc.assign_labels(
            computed_data['object_centroids'], polygons_per_label
        )
        df['roi_i_byCentroid'] = assigned_ids_centroid
        df['roi_polyi_byCentroid'] = poly_subindices_centroid
    
    @staticmethod
    def _apply_masking_method(
        df: pd.DataFrame, 
        polygons_per_label: Dict, 
        roi: np.ndarray, 
        computed_data: Dict
    ) -> None:
        """ returns label for each centroid by indexing into the roi array """
        coords = computed_data['masking_coords']
        assigned_ids_masking = roi[coords[:, 0], coords[:, 1]]
        df['roi_i_byMasking'] = assigned_ids_masking
        df['roi_polyi_byMasking'] = np.nan

    @staticmethod
    def _apply_atlas_method(
        df: pd.DataFrame, 
        polygons_per_label: Dict, 
        roi: np.ndarray, 
        computed_data: Dict
    ) -> None:
        """Apply ray casting for centroid in polygon hierarchy for atlas regions. """
        from SynAPSeg.Plugins.ABBA import core_regionPoly as rp
        centroids = computed_data['object_centroids']
        constrained_regionPolys = polygons_per_label
        rpdf = df

        roi_nb_singles, roi_nb_multis, roi_infos = rp.separate_polytypes(constrained_regionPolys)
        roi_pp_result = rp.nb_process_polygons(roi_nb_singles, roi_nb_multis, centroids)

        # extract roi_i from assigned poly - indexing into info and using reg_id as roi_i
        roi_reg_ids = [roi_infos[roi_poly_i]['reg_id'] for roi_poly_i in roi_pp_result]
        assert len(roi_reg_ids) == len(rpdf)

        # TODO don't think this will update df in place
        rpdf = (
            pd.DataFrame(list(np.array(roi_infos + [{k: np.nan for k in roi_infos[0]}])[roi_pp_result]))
            .assign(
                centroid_i=np.arange(len(centroids)),
                roi_i=1, # TODO handle multiple rois 
                # **(cfg["assign_rpdf_attributes"] or {}),  
            )
            .merge(rpdf, left_on="centroid_i", right_index=True, how="left")
        )
        
        # but this might, not sure if all required info is assigned since roi_infos isn't set
        df['roi_i_byAtlas'] = 1
        df['roi_polyi_byAtlas'] = roi_pp_result
        df['reg_id'] = roi_reg_ids
        

    @staticmethod
    def _apply_overlap3d_method(df, polygons_per_label, roi, computed_data):
        """Apply 3D overlap-based assignment."""
        # Extract 3D arrays from computed_data
        lbls = computed_data['lbls']
        lbls_fmt = computed_data['lbls_fmt']
        ch_axis = computed_data['ch_axis']

        # prep outputdf 
        df['roi_i_byOverlap3D'] = np.nan
        df['roi_polyi_byOverlap3D'] = np.nan
        
        # apply method over channels 
        for ch_i in range(lbls.shape[ch_axis]):
            # slice channel
            indexer = uip.nd_slice(lbls, ch_axis, ch_i)
            _lbls = uip.safe_squeeze(lbls[indexer], 3)
            _roi = uip.safe_squeeze(roi[indexer], 3)

            # Compute mapping
            mapping, stats = utils_colocalization_3D.map_synapses_to_dendrites_overlap(_lbls, _roi)
            
            # insert output at this colocal id - assumes channels map to colocal id
            dfInds = df['colocal_id']==ch_i
            df.loc[dfInds, 'roi_i_byOverlap3D'] = df.loc[dfInds, 'label'].map(mapping)
        
            
    @staticmethod
    def _apply_distance3d_method(df, polygons_per_label, roi, computed_data):
        """Apply 3D distance-based assignment."""
        # Extract 3D arrays and parameters
        lbls = computed_data['lbls']
        lbls_fmt = computed_data['lbls_fmt']
        ch_axis = computed_data['ch_axis']
        radius = computed_data.get('distance_radius', np.inf)
        voxel_size = computed_data.get('voxel_size', [1,1,1])
        
        # prep outputdf 
        df['roi_i_byDistance3D'] = np.nan
        df['roi_polyi_byDistance3D'] = np.nan
        df['roi_distance3D'] = np.nan

        # apply method over channels 
        for ch_i in range(lbls.shape[ch_axis]):
            # slice channel
            indexer = uip.nd_slice(lbls, ch_axis, ch_i)
            _lbls = uip.safe_squeeze(lbls[indexer], 3)
            _roi = uip.safe_squeeze(roi[indexer], 3)

            # Compute mapping
            mapping, stats = utils_colocalization_3D.map_synapses_to_dendrites_distance(
                _lbls, _roi, radius=radius, voxel_size=voxel_size
            )
            # extract distances (for all objects, even if dist > thresh)
            distances = {lbl: _stats['min_dist'] for lbl, _stats in stats['mindist_mapping'].items()}

            # insert output at this colocal id - assumes channels map to colocal id
            dfInds = df['colocal_id']==ch_i
            df.loc[dfInds, 'roi_i_byDistance3D'] = df.loc[dfInds, 'label'].map(mapping)
            df.loc[dfInds, 'roi_distance3D'] = df.loc[dfInds, 'label'].map(distances)
        


if __name__ == '__main__':
    # Demonstration of ROIHandlingStage with dummy inputs.
    dummy_roi = np.random.randint(0, 3, size=(256, 256))
    dummy_rois = [dummy_roi]
    dummy_intensity = np.random.rand(256, 256, 3)

    data = {
        "rois": dummy_rois,
        "mip_raw": dummy_intensity,
        "img_fmt": "YXC"
    }

    dummy_config = {
        "REMAP_ROIS": None,
        "ROI_REMOVE_SMALL_OBJ_SIZE": 100,
        "REID_ROIS": False,
        "PLOT_RPDF_EXTRACTION": False,
        "INTENSITY_IMAGE_NAME": "mip_raw",
        "img_fmt": "YXC"
    }

    stage = ROIHandlingStage()
    updated_data = stage.run(data, dummy_config)

    print("ROI Handling Log:")
    print(updated_data["roi_handling_log"])
    print("ROI DataFrame head:")
    print(updated_data["roi_df"].head())
