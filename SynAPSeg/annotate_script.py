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
main annotation script for validation of model predictions and defining custom ROIs
"""

"""
######################################################################################################################################################################
MAIN
######################################################################################################################################################################
"""

def get_model_pred(arr, FMT = 'ZYX', pred_kwargs=None, kwargs:dict=None):
    from SynAPSeg.models.factory import ModelPluginFactory
    modelbasedir = os.environ['MODELS_BASE_DIR']
    default_models = {
        '3d':'stardist3d_2025_1125_v3.10_aug_minimal_32x128x128', 
        '2d':'synapsedist2D_v3.6.0.8_augAffine_smallerGrid_long',
    }
    model_path = os.path.join(modelbasedir, default_models['3d' if len(FMT) == 3 else '2d'])

    par = {
        'model_path': model_path,
        "in_dims_model": FMT,
        "in_dims_pipe": FMT,
        "out_dims_pipe": FMT,
        'load_model_kwargs': {'weights_filename':'weights_last.h5'},
        "preprocessing_kwargs": {"norm": [1, (99.99 if len(FMT) == 3 else 99.9)]},
        'predict_kwargs':pred_kwargs or {},
    }
    par.update(kwargs or {})

    model = ModelPluginFactory.get_plugin('Stardist', **par)
    return model(arr)[0]


if __name__ == '__main__':

    verify_and_set_env_dirs(
        # Set base project directory
        ###############################################################################################
        dict(
            PROJECTS_ROOT_DIR = r"J:\SEGMENTATION_DATASETS"
            # PROJECTS_ROOT_DIR = r"R:\Confocal data archive\Lina\SEGMENTATION_DATASETS"
        )
    )

    logger = get_logger("Annotator_logger", log_filename="Annotator.log")
    logger.info("segmentation_example_annotator started.")

    # normal
    if bool(1):

        ######################################################################################################################################################################
        # PARAMS
        ######################################################################################################################################################################
        # load specific project example  with this var
        # EXAMPLE_PROJ = "2025_0702_bassoonStainingOptimization" #"2025_0928_hpc_psd95andrbPV_zstacks"

        # EXAMPLE_PROJ = "2026_0108_synapseg_supFig1" #"2025_0714_cohort_3_camkiiXpsd95GephPVgp"
        # EXAMPLE_PROJ = "TripleHM_BLA_test"
        # EXAMPLE_PROJ = "VHL_VglutHomer"
        
        # EXAMPLE_PROJ = "2025_0402_AGP_tilescans"
        # EXAMPLE_PROJ = "2025_0605_AGP_hpc_psd95_tilescans_round2"
        # EXAMPLE_PROJ = "2025_0828_hpc_psd95_tilescans_round3"
        

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
        
        # dend filtering 
        ##############################################################
        print(image_dict.keys())
        ex2roi = pd.read_excel(r"J:\SEGMENTATION_DATASETS\2025_0928_hpc_psd95andrbPV_zstacks\outputs\2025_1212_044118_Quantification_2025_0928_hpc_psd95andrbPV_zstacks\good_dends_260311.xlsx")
        import ast
        ex_int = int(ex.name)
        dffilt = ex2roi.query(f"ex_i == {ex_int}")
        
        get_roi_is = dffilt['dend_lbls'].values[0]
        if pd.isnull(get_roi_is): raise ValueError()
        get_roi_is = ast.literal_eval(get_roi_is)
            
        # _ = uip.filter_label_img(
        #     image_dict['pred_stardist_n2v_3d_v310.tiff'], list(get_roi_is)
        # )
        


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
        
        if bool(0):
            size_filtered = uip.filter_area_objects(viewer.layers['annotated_dends_filt_final_ch0'].data, (6000, None))
            viewer.add_labels(
                size_filtered, 
                name='annotated_dends_filt_final'
            )
            print(list(uip.unique_nonzero(viewer.layers['annotated_dends_filt_final_ch0'].data)))
            print(list(uip.unique_nonzero(viewer.layers['annotated_dends_filt_final'].data)))
            
            print(list(uip.unique_nonzero(uip.relabel(viewer.layers['annotated_dends_filt_final_ch0'].data, connectivity=3) )))












        if bool(0): # extract patches defined by a shapes layer

            shapes_layer_name = 'Shapes'
            patch_coords = [coords.astype('int32') for coords in viewer.layers[shapes_layer_name].data]

            patch_coords = [np.array([[10937, 24627],
                    [10784, 25700],
                    [16584, 26530],
                    [16738, 25458]]),
            np.array([[ 2398, 23319],
                    [ 2245, 24392],
                    [ 8046, 25223],
                    [ 8199, 24150]]),
            np.array([[5391,  694],
                    [4314,  815],
                    [4968, 6638],
                    [6045, 6517]])]

            img_layer_name = 'raw_img_ch1'
            base_img = viewer.layers[img_layer_name].data
            display_img = uip.norm_percentile(base_img, [1, 99.8], ch_axis=None, clip=True)
            display_img = (display_img * 255).astype('uint8')

            # get ch 2
            ch2_img = viewer.layers['raw_img_ch0'].data
            display_ch2_img = uip.norm_percentile(ch2_img, [1, 99.99], ch_axis=None, clip=True)
            display_ch2_img = (display_ch2_img * 255).astype('uint8')

            composite_img = np.stack([display_img, display_ch2_img],-1).astype('uint8')
            composite_img = up.create_composite_image_with_colormaps(composite_img, ['green', 'cyan'])

            # load the atlas regions
            from SynAPSeg.Plugins.ABBA import utils_atlas_region_helper_functions as arhfs
            from SynAPSeg.Plugins.ABBA import core_regionPoly as rp
            cfg = {
                "structs_path": r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\SynAPSeg\SynAPSeg\Plugins\ABBA\kim_mouse_10um_v1.1_structures.csv",
                "geojson_path": os.path.join(ex.path_to_example, 'qupath', 'qupath_export_geojson', 'raw_img.geojson'),
            }
            ont = arhfs.Ontology(pd.read_csv(cfg["structs_path"]))
            regionPolys = rp.polyCollection(cfg["geojson_path"], ont=ont)

            # 3. Extract the patches            
            for i,pc in enumerate(patch_coords):
                # tl, tr, br, bl
                rect = pc[:, 0].min(), pc[:, 1].min(), pc[:, 0].max(), pc[:, 1].max()
                
                print(rect)
                patch = display_img[rect[0]:rect[2], rect[1]:rect[3], ...]
                patch_ch2 = display_ch2_img[rect[0]:rect[2], rect[1]:rect[3], ...]
                composite_patch = np.stack([patch, patch_ch2],-1).astype('uint8')
                composite_patch = up.create_composite_image_with_colormaps(composite_patch, ['green', 'blue'])
                composite_patch = (composite_patch * 255).astype('uint8')
                uip.pai(composite_patch)


                fig,ax = plt.subplots()
                ax.origin = 'upper'
                extent = [rect[1], rect[3], rect[2], rect[0]] #if i == 2 else [rect[1], rect[3], rect[0], rect[2]]
                up.show(composite_patch, ax=ax, extent=extent)
                bx1,bx2,by1,by2 = regionPolys.plot(ax=ax, polypatch_kwargs={'edgecolor':'white'})
                ax.axis('off')
                up.add_scalebar(ax, 0.071, length_units=50, title='')
                up.save_fig(os.path.join(ex.path_to_example, f'patch_{i}.svg'), dpi=500)
                plt.show()
            


            figsize = (7.5, 2.5) if i == 2 else (2.5, 7.5)
            fig,ax = plt.subplots(figsize=figsize)
            for i, pc in enumerate([src_ordered]):
                for ii in range(pc.shape[0]):
                    ax.plot(pc[ii,1], pc[ii,0], marker=f'${ii}$', label=f"patch {i}")
            ax.legend()
            plt.xlim(0, display_img.shape[1])
            plt.ylim(0, display_img.shape[0])
            ax.invert_yaxis()
            plt.show()
                
                
        if bool(0):
            predlayer = 'raw_img_ch1'
            pred = get_model_pred(
                dat(predlayer),#[1648:, 970:1300],
                FMT = 'ZYX',
                pred_kwargs={
                    "prob_thresh": 0.3, 
                    "prediction_padding": 8,
                },
                kwargs={
                    "preprocessing_kwargs": {"norm": [1, 99.8]}
                }
            )
            viewer.add_labels(pred, name=f'pred_{predlayer}')
            # viewer.add_image(dat(predlayer)[1648:, 970:1300], name=f'{predlayer}_crop')

            # TO TEST
            ###################
            # 3. Inspect the "Probability Map"
            # Before StarDist performs the "Star-convex polygon" fitting, it generates a probability map.

            # Run the prediction but look at the raw probability output: prob, dists = model.predict_instances_big(img).

            # If the puncta are missing from the prob map on the full image but present on the crop, it is definitely a normalization/preprocessing issue.

            # If they are in the prob map but not segmented, it is a post-processing issue (like Non-Maximum Suppression thresholds being too aggressive).

        if bool(0):
            from skimage.transform import rescale
            roimask = np.repeat(dat('ROI_1')[np.newaxis], 9, axis=0)
            extent = uip.find_extent(roimask)
            slices = [slice(extent[i*2], extent[(i*2)+1]) for i in range(len(extent)//2)]
            scalefactors = [4, 1, 1]
            for i in [1,3]:
                ch = dat('raw_img_ch'+str(i))
                masked = np.where(roimask>0, ch, 0)
                cropped = masked[tuple(slices)]
                viewer.add_image(rescale(cropped, scalefactors), name='croppedch'+str(i)+'_roi')

        if bool(0):
            from SynAPSeg.models.factory import ModelPluginFactory
            FMT = 'ZYX'
            par = {
                # 'model_path': r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\BygraveCode\models\synapsedist2D_v3.6.0.3_dilate_3DdataSlices", 
                'model_path': os.environ['MODELS_BASE_DIR'] + r"\stardist3d_2025_1125_v3.10_aug_minimal_32x128x128",
                "in_dims_model": FMT,
                "in_dims_pipe": FMT,
                "out_dims_pipe": FMT,
                'load_model_kwargs': {'weights_filename':'weights_last.h5'},
                "predict_kwargs":{"prob_thresh": 0.3, "prediction_padding": 8},
                "preprocessing_kwargs": {"norm": [1, 99.99]},
            }
            model = ModelPluginFactory.get_plugin('stardist', **par)

            pred_on_layer = 'raw_img_ch0'
            inputimg = viewer.layers[pred_on_layer].data

            pred = model(inputimg)[0]
            viewer.add_labels(pred, name=f'pred_stardist3d_v3.10_{pred_on_layer}')
            print(len(uip.unique_nonzero(pred)))

        # pred = viewer.layers['pred_stardist_v3.6.0.3'].data.copy()
        # viewer.add_labels(pred, name='pred_stardist_v3.6.0.3')

        # crops = uip.find_extent_and_crop([dat('roi_0'), dat('mip_raw_ch1'), dat('pred_stardist_v3.6.0.3')])

        # for c, fn in zip(crops, ['roi', 'raw_image', 'labels']):
        #     outpath = os.path.join(r"J:\__compiled_training_data__\2d_synapses\AGP_tilescans", f"0000_AGP_tilescans_{fn}.tiff")
        #     # tifffile.imwrite(outpath, c)
        #     up.show(c)

        # sp = r"D:\BygraveLab\Confocal data archive\Pascal\SEGMENTATION_DATASETS\2025_0909_AMI_cohort3_NeuN\examples\0000\hpc_regions.csv"
        # shapes_ = pd.read_csv(sp)

        # from napari_builtins.io._read import read_csv, csv_to_layer_data
        # data, column_names, layer_type = read_csv(sp)

        # shpdata, types, layer_type = csv_to_layer_data(sp)
        # viewer.add_shapes(shpdata, **types)

        #######################################################################
        # ANALYSES
        #######################################################################
        if bool(0): # filter detections - using good/bad examples (object labels) to create filter params
            # baddetects = [4258, 4207, 3831, 4076, 4040, 4078, 4261]
            # gooddetects= [4010, 3974, 2451, 1259, 4984, 3247, 3227]

            baddetects = [8135, 8434, 6790, 3095, 8516, 8494]
            gooddetects = [8921, 9218, 9169, 9028, 3610, 8731, 8791]

            # create rpdf
            seg_layer = 'pred_neurpose_ch1'
            int_layer = 'raw_img_ch1'

            rpdf = uc.get_rp_table(dat(seg_layer), uip.norm_percentile(dat(int_layer), (1,99.9)), ch_axis=None)
            get_detect_val = lambda v: -1 if v in baddetects else 1 if v in gooddetects else 0
            map_detects = {l:get_detect_val(l) for l in rpdf['label']}

            rpdf['GTlabel'] = rpdf['label'].map(map_detects).astype('str')

            if bool(0):
                sns.scatterplot(data=rpdf, x='intensity_mean', y='area', hue='GTlabel', palette={'-1':'red', '0':(0.5,0.5,0.5,0.1), '1':'cyan'})
                sns.rugplot(data=rpdf, x='intensity_mean', y='area', hue='GTlabel', palette={'-1':'red', '0':(0.5,0.5,0.5,0.0), '1':'cyan'})
                plt.show()
                sns.displot(data=rpdf, x='intensity_mean', kind='hist', hue='GTlabel', palette={'-1':'red', '0':(0.5,0.5,0.5,0.1), '1':'cyan'}, rug=True)

            print(rpdf.groupby(['GTlabel']).describe())

            # filter
            #########################################
            threshold_dict = dict(
                area = (1700, np.inf),
                # intensity_mean = (5400, np.inf),
                # intensity_mean = (3600, np.inf),
                intensity_mean = (0.11, np.inf)
            )
            frpdf, filter_counts_post_thresh = uc.region_prop_table_filter(rpdf, threshold_dict)
            keep_labels = frpdf['label'].to_list()
            viewer.add_labels(uip.filter_label_img(dat(seg_layer), keep_labels), name=f"{seg_layer}_filtered")

        if bool(0): # convert region polys (napari shapes) to geojson polygon collection
            for ex in project.examples:
                shape_annots = pd.read_csv(os.path.join(ex.path_to_example, "hpc_regions.csv"))
                # map shape indicies to names, assumes index 0 is whole hpc outline
                index2name = {0:'hpc'}
                index2name = ug.merge_dicts(index2name, {i:f'{i}' for i in shape_annots['index'].unique() if i not in index2name.keys()})

                # convert napari shape to regionpoly
                import shapely
                from Plugins.ABBA import core_regionPoly as rp
                import geojson
                geoFeatCollection, shapely_shapes = rp.parse_Napari_shapes_to_polygons(shape_annots, x_col='axis-1', y_col='axis-0')

                # clip the polygons so they are constrained to the image dimensions
                bbox = [0,0,*(image_dict['raw_img'].shape[-2:])[::-1]] # (min0, min1, max0, max1), + convert from image coords
                bbox_poly = shapely.geometry.box(*bbox)
                shapely_shapes = [poly.intersection(bbox_poly) for poly in shapely_shapes]

                # merge polys that are not index 0
                to_merge = shapely_shapes[1:]
                merged = shapely.unary_union(to_merge)

                # remove regions defined in polys where ind != 0 from whole hpc outline
                # this makes it so whole_hpc region doesn't include these other regions
                subFrom = shapely_shapes[0]
                subbed = subFrom.difference(merged)

                # convert to geojson feature collection and write to example
                properties = [
                    dict(reg_id=1,  region_name='whole hpc minus pyr', reg_side='left', st_level=5, acronym='hpc', roi_i=0),
                    dict(reg_id=2,  region_name='pyramidal layers', reg_side='left', st_level=5, acronym='pyr', roi_i=0),
                ]
                properties = [ug.merge_dicts(
                    d, 
                    {'obj_i':i, 
                    'measurements':{'ID':str(np.random.random(1)[0])},
                    'classification': {'names':[str(d['reg_id']), d['reg_side']]}
                    }
                ) for i,d in enumerate(properties)]

                feats = []
                for s, props in zip([subbed, merged], properties):
                    feats.append(rp.shapely_to_geojson(s, props))
                fc = geojson.FeatureCollection(feats)
                rp.write_geojson_featureCollection(os.path.join(ex.path_to_example, 'rois.geojson'), fc)

                regionPolys = rp.polyCollection(os.path.join(ex.path_to_example, 'rois.geojson'))
                regionPolys.plot()  # looks perfect !
                plt.show()

        # testing dend length calculation
        ######################################################
        import numpy as np
        from skimage.morphology import skeletonize
        from scipy.ndimage import label
        import networkx as nx

        def dendrite_length_3d(mask, spacing=(1.0, 1.0, 1.0)):
            """
            Compute dendrite length from a 3D binary mask by skeletonization.
            
            Parameters
            ----------
            mask : np.ndarray
                3D binary array (Z, Y, X) where dendrite = 1.
            spacing : tuple of float
                (dz, dy, dx) voxel size in physical units.
            
            Returns
            -------
            total_length : float
                Sum of all skeleton segment lengths (physical units).
            longest_path_length : float
                Length of the longest path between two skeleton endpoints.
            """
            dz, dy, dx = spacing

            # 1. 3D skeleton
            skel = skeletonize(mask.astype(bool))
            px_in_skel = np.sum(skel)

            # 2. Get coordinates of skeleton voxels
            coords = np.argwhere(skel)  # shape (N, 3) with (z, y, x)
            if coords.size == 0:
                return px_in_skel, None, 0.0, 0.0

            # Map voxel index -> integer node id
            coord_to_id = {tuple(c): i for i, c in enumerate(coords)}

            # 3. Build graph with edges between 26-connected neighbors
            G = nx.Graph()
            for i, (z, y, x) in enumerate(coords):
                # physical coords in whatever units spacing is in (e.g. microns)
                Z = z * dz
                Y = y * dy
                X = x * dx

                # choose a 2D projection; here we use X–Y
                G.add_node(i, pos=(X, Y), z=Z)

                # optional: also keep original voxel indices if useful
                G.nodes[i]["voxel_coord"] = (z, y, x)

            # Neighborhood offsets for 26-connectivity
            neighbor_offsets = [
                (dz_, dy_, dx_)
                for dz_ in (-1, 0, 1)
                for dy_ in (-1, 0, 1)
                for dx_ in (-1, 0, 1)
                if not (dz_ == 0 and dy_ == 0 and dx_ == 0)
            ]

            for i, (z, y, x) in enumerate(coords):
                for dz_i, dy_i, dx_i in neighbor_offsets:
                    nz, ny, nx_ = z + dz_i, y + dy_i, x + dx_i
                    neighbor = (nz, ny, nx_)
                    if neighbor in coord_to_id:
                        j = coord_to_id[neighbor]
                        if i < j:  # avoid double-adding edges
                            # physical distance between these voxels
                            dist = np.sqrt((dz * dz_i) ** 2 +
                                        (dy * dy_i) ** 2 +
                                        (dx * dx_i) ** 2)
                            G.add_edge(i, j, weight=dist)

            # 4a. Total length of skeleton: sum of all edge weights
            total_length = sum(d["weight"] for *_ , d in G.edges(data=True))

            # 4b. Longest path length between endpoints
            # endpoints = nodes with degree 1
            endpoints = [n for n in G.nodes if G.degree[n] == 1]

            longest_path_length = 0.0
            if len(endpoints) >= 2:
                # Compute all-pairs shortest path lengths only among endpoints
                lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
                for u in endpoints:
                    for v in endpoints:
                        if u < v and v in lengths[u]:
                            longest_path_length = max(longest_path_length, lengths[u][v])

            return px_in_skel, G, total_length, longest_path_length       

        if bool(0):
            import ast
            dz, dy, dx = 0.46, 0.085, 0.085
            gdends = pd.read_excel(r"J:\SEGMENTATION_DATASETS\2025_0928_hpc_psd95andrbPV_zstacks\good_dends.xlsx")
            get_roi_is = gdends[gdends['ex_i']==int(EXAMPLE_I)]['dend_lbls'].values[0]
            get_roi_is = ast.literal_eval(get_roi_is)

            dendlens = []
            graphs = []
            for roi_i in get_roi_is:
                dend_mask = (dat('annotated_dends_filt_ch0')==roi_i)

                px_in_skel, G, total_len, longest_len = dendrite_length_3d(dend_mask, spacing=(dz, dy, dx))

                graphs.append(G)
                dendlens.append({'roi_i':roi_i, 'px_in_skel':px_in_skel,  'length_total':total_len, 'length_longest': longest_len})
                print("Total skeleton length:", total_len)
                print("Longest branch-like path:", longest_len)

            dendlens = pd.DataFrame(dendlens)

        if bool(0):
            # for G in graphs[:1]:
            for i, roi_i in list(enumerate(get_roi_is))[-2:-1]:
                fig,axs = plt.subplots(1,2)
                G = graphs[i]
                pos = nx.get_node_attributes(G, "voxel_coord")
                pos = {k:cc[1:] for k,cc in pos.items()}
                nx.draw(G, pos=pos, node_size=5, with_labels=False, ax=axs[0])

                # plot dend mask
                dend_mask = (dat('annotated_dends_filt_ch0')==roi_i)
                dend_mask_crop = uip.find_extent_and_crop(dend_mask)
                up.show(dend_mask_crop, ax=axs[1])
                plt.suptitle(f"ex={EXAMPLE_I}, roi_i:{roi_i}\n{dendlens.iloc[i,:].to_dict()}")
                plt.tight_layout()
                plt.show()

                dist_counter = 0
                # spacing = np.array((dz, dy, dx)) # don't include z
                spacing = np.array((1, 1, 1)) # don't include z
                n_coords = nx.get_node_attributes(G, "voxel_coord")
                st = np.array(n_coords[0]).astype('float64')
                largest_delta = 0 
                for n, ccc in n_coords.items():
                    if n==0: continue

                    curr = np.array(ccc).astype('float64')
                    delta = np.abs(curr-st)
                    delta *= spacing
                    delta = np.sum(delta)
                    if delta>largest_delta:
                        largest_delta = delta
                    dist_counter+= np.sum(delta)
                    st = curr
                print(dist_counter)

        # the graph looked weird, but may be b/c node lacked fixed coord positions
        # alt would be to just count num of pixels in the skeletonized image and multiply by voxel size to get total dend length

        # so it seems the issue is the order of the coords list, b/c it is sorted by ax0 it
        # introduces large jumps when the actual order should be next closest point,
        # but this could be tricky to figure out due to branches in the skeleton

        # therefore clostest approxiamtion of linear distance was the num px in the 2d skel,
        # this maps pretty close to real distance and I can compute a custom region property
        # to do this..

        if bool(0):
            remove_lbls = []
            for i in remove_lbls:
                viewer.layers['dends_filt_ch0'].data = np.where(dat('dends_filt_ch0') == i, 0, dat('dends_filt_ch0'))

        # merge lbls
        if bool(0):
            merge_from, merge_into = 116,136
            viewer.layers['dends_filt_ch0'].data = np.where(dat('dends_filt_ch0') == merge_from, merge_into, dat('dends_filt_ch0'))

        # # testing roi assignment
        ##########################
        # import skimage
        # from scipy import ndimage as ndi
        # from config.constants import STANDARD_FORMAT

        # # testing roi assignment
        # roi = image_dict['ROI_dends_filt']#[:1]
        # img = image_dict['raw_img'][:,:,:]#, :1]
        # obj = image_dict['pred_stardist_3d'][:,:,:]#, :1]
        # obj[0, 0, 0] = dat('pred_stardist_3d_ch0')

        # roi_fmt = exmd['data_metadata']['data_formats']['ROI_dends_filt']
        # img_fmt = exmd['data_metadata']['data_formats']['raw_img']
        # obj_fmt = exmd['data_metadata']['data_formats']['pred_stardist_3d']

        # roi, roi_fmt = uip.standardize_collapse(roi, roi_fmt, STANDARD_FORMAT)
        # img, img_fmt = uip.standardize_collapse(img, img_fmt, STANDARD_FORMAT)
        # obj, obj_fmt = uip.standardize_collapse(obj, obj_fmt, STANDARD_FORMAT)

        # uip.pai([roi, img, obj])
        # img_ch_axis = img_fmt.index('C') if 'C' in img_fmt else None
        # rpdf_objs = uc.get_rp_table(obj, img, ch_axis=img_ch_axis, get_object_coords=True)

        # if 'C' not in roi_fmt and 'C' in img_fmt:
        #     _roi = uip.morph_to_target_shape(roi, roi_fmt, img.shape, img_fmt)

        # roi_max_val = _roi.max()+1

        # rpdf_rois = uc.get_rp_table(
        #     np.where(_roi==0, roi_max_val, _roi),
        #     img,
        #     ch_axis=img_ch_axis,
        #     get_object_coords=False
        # )
        # rpdf_rois.loc[rpdf_rois['label']==roi_max_val, 'label'] = 0

        # from skimage.measure import regionprops_table
        # # move channels last
        # roiimg = np.moveaxis(img, img_ch_axis, -1)
        # _df = pd.DataFrame(regionprops_table(roi, roiimg, properties=uc._DEFAULT_REGION_PROPERTIES_TO_EXTRACT))

        # # testing coordinate space conversions
        # orow = rpdf_objs.loc[6, :]
        # ocentroid = orow['centroid']
        # ocoords = orow['coords']
        # print(obj[tuple(np.rint(ocentroid).astype('int32'))])
        # obj_inds = [tuple(el) for el in np.rint(ocoords).astype('int32')]
        # print([obj[idx] for idx in obj_inds])

        # print(roi[tuple(np.rint(ocentroid).astype('int32'))])
        # print([roi[idx] for idx in obj_inds])

        # # parse spacing like in pipeline
        # import ast
        # config = {}
        # config["PX_SIZES"] = {
        #     dim: size_in_m * 1e6
        #     for dim, size_in_m in
        #     ast.literal_eval(ex.exmd['image_metadata']['scaling']).items()
        # }
        # voxel_size = None  # list-like, default: [1,1,1]
        # try:
        #     if config["PX_SIZES"] is not None: # e.g. {'X':1, 'Y':1, 'Z':1}
        #         voxel_size = [config["PX_SIZES"][d] for d in 'ZYX']
        # except:
        #     pass

        # import importlib

        # from Quantification.plugins.roi_handling import ROIAssigner
        # from Quantification.plugins import roi_handling

        # importlib.reload(roi_handling)
        # ROIAssigner = roi_handling.ROIAssigner
        # # polygons_per_label = uc.semantic_to_polygons_rasterio(roi)

        # data = dict(
        #     rpdf = rpdf_objs,
        #     polygons_per_label = None,
        #     rois = [roi],
        # )
        # ROI_ASSIGNMENT_METHODS = ['Overlap3D', 'Distance3D'][:] #['Centroid', 'Coords', 'Masking', 'Distance'] + ['Overlap3D', 'Distance3D']
        # t0 = ug.dt()
        # data["rpdf"] = ROIAssigner.assign_rois_to_rpdf(
        #     data["rpdf"],
        #     data["polygons_per_label"],
        #     data["rois"][0],
        #     roi_assignment_methods=ROI_ASSIGNMENT_METHODS,
        #     lbls=obj,
        #     voxel_size=voxel_size,
        #     lbls_fmt = obj_fmt,
        #     roi_fmt = roi_fmt
        # )
        # print(f"{ug.dt() - t0}")
        # rpdf2 = data["rpdf"]
        # rpdf2['roi_i_byOverlap3D'].value_counts()

        # # inspect known values
        # rpdf2['label'] = rpdf2.pop('label')
        # getlbls = [2771, 2347, 2274, 2921] + [99_999]
        # resdf = rpdf2[rpdf2['label'].isin(getlbls)]

        # # based on review of individual synapses using 0.46 radius thresh may include some FPs
        # # but this val could be nice if a synapse is one above in the zplane

        # # testing roi summary
        # importlib.reload(uc)
        # roi_df = uc.summarize_roi_array_properties(img,  roi, img_fmt, roi_fmt, coerce_roi_fmt=True,
        #                                            PX_SIZES=config["PX_SIZES"], additional_props=['perimeter'], extra_properties=['uc.circularity', 'uc.kurtosis'])

        # import scipy
        # res0 = scipy.stats.kurtosis(img[0, 0, :25, :25])
        # resNone = scipy.stats.kurtosis(img[0, 0, :25, :25], axis=None)

        # from skimage.measure import _regionprops_utils
        # m = np.zeros((10,10,10))
        # m[2:8, 2:8, 2:8] = 1
        # _regionprops_utils.perimeter(m, 4)

        # ch_axis = obj_fmt.index('C')
        # for ch_i in range(obj.shape[ch_axis]):
        #     indexer = uip.nd_slice(obj, ch_axis, ch_i)
        #     _lbls = uip.safe_squeeze(obj[indexer], 3)
        #     print(obj.shape, _lbls.shape)
        #     # _roi = uip.safe_squeeze(roi[indexer], 3)

        # """Apply the Masking method for ROI assignment."""
        # coords = computed_data['masking_coords']
        # assigned_ids_masking = roi[coords[:, 0], coords[:, 1]]
        # df['roi_i_byMasking'] = assigned_ids_masking
        # df['roi_polyi_byMasking'] = np.nan

        if bool(0):       
            volume = image_dict['raw_img'][0,0,1]
            seg = image_dict['pred_neurseg3d'][0,0,1]

            # from utils.VolumeViewer import VolumeViewer as vv
            # VV = vv(volume, seg)

            clean = uip.zero_border_nd(seg, pad=82, fmt="zyx", axes=("y","x"))
            # viewer.update(new_volume=..., new_seg=...)
            up.show(clean)

            # extract somas
            ##################################################################################################
            median_connectivity=1
            median_iter=5
            sigma=1
            labels_binary_opening = 12
            labels_binary_erode = 20 # usually 20 - 28
            labels_binary_dilate = 10
            get_n_largest = 6 # in order: 6, 3, 4, 4, 5, 5, 6, 5, 7, 8, 5, 8, 4, 4, 4, 4, 4, 3

            inputimg = uip.mip(uip.norm_percentile(clean, (1,99), ch_axis=None))
            pred_input_preproc=inputimg.copy()
            # up.show(pred_input_preproc)

            # pred_input_preproc = skimage.filters.gaussian(pred_input_preproc, sigma=(sigma, sigma), truncate=3.5)
            # up.show(pred_input_preproc)
            pred_input_preproc = skimage.morphology.opening(pred_input_preproc, footprint=np.ones((labels_binary_opening, labels_binary_opening)))
            thresh = skimage.filters.threshold_triangle(pred_input_preproc)
            mipt = pred_input_preproc>thresh
            miptt = skimage.morphology.binary_erosion(mipt, footprint=np.ones((labels_binary_erode,labels_binary_erode)))
            # up.show(pred_input_preproc)
            # up.show(mipt)
            up.show(miptt)

            ll = uip.extract_largest_objects(miptt, get_n_largest)
            up.show(ll)

            # finalize soma detections
            ####################################################################################################################################
            llo = skimage.morphology.closing(ll, footprint=np.ones((12, 12)))
            llo = skimage.morphology.dilation(llo, footprint=np.ones((labels_binary_dilate*3, labels_binary_dilate*3)))
            up.show(llo)

            up.show(np.where(llo==0, inputimg, 0))

            # extract soma from dend seg, and preprocess dentrite segmentations
            ##################################################################
            N = clean.shape[0]
            somas = np.where(np.repeat(llo[np.newaxis, ...], repeats=N, axis=0), 2, 0)
            somas = uip.relabel(somas,connectivity=3)
            # somas = np.where(somas==3, 0, somas)
            dends = np.where(somas, 0, clean)
            up.show(somas)
            up.show(dends)

            # separate dends best as possible
            ####################################################################################################################################
            dends_opening_r = 4 #  2 was best for old ones, 3+ for more recent batches 
            dends_pp = skimage.morphology.opening(dends, footprint=np.ones((dends_opening_r, dends_opening_r, dends_opening_r)))
            dends_lbld = uip.relabel(dends_pp, connectivity=3)

            _rps = ['label', 'area', 'bbox', 'centroid', 'intensity_mean', 'axis_major_length']
            rpdf = uc.get_rp_table(dends_lbld, volume, ch_axis=None, rps_to_get=_rps)
            # sns.scatterplot(rpdf, x='area', y='axis_major_length'); plt.xscale('log');plt.yscale('log')

            keep_labels = rpdf.query("(axis_major_length>150)")['label'].to_list()
            # keep_labels = rpdf.query("(axis_major_length<250)")['label'].to_list()
            dends_filt = uip.filter_label_img(dends_lbld, keep_labels)
            up.plot_image_grid([uip.mip(a) for a in [uip.relabel(dends, connectivity=3), dends_lbld, dends_filt]], n_rows=1)

            # skeleton = skimage.morphology.skeletonize(dends_lbld)
            # up.show(skeleton)

            #
            viewer.add_labels(dends_lbld)
            viewer.add_labels(dends_filt)
            # viewer.add_labels(skeleton)
            viewer.add_labels(somas)

            # SAVE DENDS
            ############
            if bool(0):
                ex.write_data(dat("dends_filt"), 'dends_filt.tiff', fmt='ZYX')
                ex.write_data(dat("somas"), 'somas.tiff', fmt='ZYX')

        if bool(0):
            #######################################################################
            # testing colocal issue
            #######################################################################
            ex.exmd['COLOCALIZE_PARAMS']={'colocalizations': [[3, 2], [2, 3]]}
            from IO.metadata_handler import MetadataParser
            from utils.utils_ImgDB import ImgDB
            image_channels, clc_nuc_info = MetadataParser.get_imgdb_colocal_nuclei_info(ex.exmd)
            imgdb = ImgDB(image_channels=image_channels, colocal_nuclei_info=clc_nuc_info)

            colocal_ids = {
                0: {'name': 'GAD67', 'ch_idx': 0},
                1: {'name': 'virus', 'ch_idx': 1},
                2: {'name': 'TARPy2', 'ch_idx': 2},  # non-synaptic tarp (not coloc with vglut)
                3: {'name': 'VGlut1', 'ch_idx': 3},
                4: {'name': 'VGlut1+TARPy2', 'ch_idx': [3, 2], 'co_ids': [3, 2]}, # synaptic tarp
                5: {'name': 'TARPy2+VGlut1', 'ch_idx': [2, 3], 'co_ids': [2, 3]},}
            BCI = 2 # base clcid of interest

            rpdf = pd.read_csv(r"\\?\R:\Confocal data archive\Molly\Btbd11 WT_KO ICC\2.9.25 ICC PFA Btbd11 KO TARP and PSD95\2025_0209_Molly_btbd11_ICC_TARP\outputs\attempted analysis to separate coloc and dendritic signal but didn't seem to work\2025_1002_154537_Quantification_2025_0209_Molly_btbd11_ICC_TARP\temp\all_rpdfs\all_rpdfs_rpdf_0000.csv")
            rpdf = pd.read_csv(r"\\?\R:\Confocal data archive\Molly\Btbd11 WT_KO ICC\2.9.25 ICC PFA Btbd11 KO TARP and PSD95\2025_0209_Molly_btbd11_ICC_TARP\outputs\attempted analysis to separate coloc and dendritic signal but didn't seem to work\2025_1002_154537_Quantification_2025_0209_Molly_btbd11_ICC_TARP\all_rpdfs.csv")
            rpdf = rpdf[rpdf['ex_i']==0]
            uc.get_colocal_id_counts(rpdf, all_colocal_ids=list(colocal_ids.keys()))

            # rpdf_coloc = uc.separate_colocal_populations(ex_rpdf, imgdb, image_id_column=None, logger=None)

        if bool(0): # fix manual annots

            ln = 'Labels'
            int_ln = 'raw_img_ch1'
            shape_ln, shape_i = 'Shapes', 0
            out_alias_suffix = 'number0'

            lbls = dat(ln)        
            int_img = dat(int_ln)

            # get boundary defined by shapes layer
            shape_data = dat(shape_ln)
            shape_data = np.rint(shape_data[shape_i]).astype('int32')
            slc = bounds2slice(shape_data, fmt='bbox')

            lbls = uip.relabel(lbls)
            lbls = fill_outline_labels(lbls, connectivity=1)
            lbl_crop = lbls[slc]
            # up.show(filled)

            img_crop = int_img[slc]

            up.mask_to_outline_contours(img_crop, lbl_crop)

            from IO.writers import write_array
            out_dir = os.path.join(ex.path_to_example, 'neun_soma_annotations')
            for fn, im in zip([f'img_ch1_{out_alias_suffix}.tiff', f'labels_ch1_{out_alias_suffix}.tiff'], [img_crop, lbl_crop]):
                write_array(im, out_dir, fn, fmt='YX', 
                            tiff_metadata={'og_img': ex.get_image_path(), 'slice': str([(slice(1, 2, None))] + list(slc))}
                )

        if bool(0): # compare segmethod against benchmark
            from Train.benchmark import get_benchmark_dataset
            get_benchmark_dataset
            benchmark_dir = os.path.join(ex.path_to_example, 'neun_soma_annotations')
            benchmark_data = {}
            for n in [0,1]:
                imgp = ug.get_contents(benchmark_dir, filter_str=f'img.*number{n}.*', pattern=True)[0]
                lblp = ug.get_contents(benchmark_dir, filter_str=f'labels.*number{n}.*', pattern=True)[0]
                img, lbl = uip.imread(imgp), uip.imread(lblp)
                benchmark_data[n] = {'img':img, 'lbl':lbl}

        if bool(0): # segment neun soma

            from SynAPSeg.models import ModelPluginFactory
            from IO.env import verify_and_set_env_dirs
            from scipy.ndimage import median_filter
            from scipy.ndimage import gaussian_filter
            import geojson
            import Plugins.ABBA.core_regionPoly as rp
            from shapely.geometry import shape as shapelyShape
            from rasterio import Affine
            from rasterio.features import shapes as rf_shapes # needs to be explicitly imported
            from shapely.ops import unary_union

            verify_and_set_env_dirs()

            model = ModelPluginFactory.get_plugin(
                    'stardist',
                    **dict(
                        model_path = os.path.join(os.environ['MODELS_BASE_DIR'], "2D_versatile_fluo"),
                        in_dims_model = 'YX',
                        out_dims_model = 'YX',
                        out_dims_pipe = "YX",
                        in_dims_pipe = 'YX',
                        preprocessing_kwargs = {'norm':(1,99.9)},
                        predict_kwargs = {
                            'prob_thresh':0.6,
                            'nms_thresh':0.9,
                        },
                    )
                )

            # testing seg params on crop region
            size = 1024
            x,y = 9000, 11000
            GET_CH = 1
            img = image_dict['raw_img'][GET_CH, x:x+size, y:y+size]

            input_img = uip.subtract_rolling_ball(img, radius=18)
            input_img2 = median_filter(input_img, 7)
            input_img3 = gaussian_filter(img, 7)
            # up.show_gif([img, input_img, input_img2, input_img3], duration=500)

            res = model(input_img3)[0]
            print(len(uip.unique_nonzero(res)))

            up.show_gif([img, input_img3, res], duration=500)

            _rpdf = uc.get_rp_table(res, img[GET_CH], ch_axis=None)
            keeplbls = _rpdf.query("area>1000")['label'].to_list()
            resf = uip.filter_label_img(res, keeplbls)
            up.show(resf)

            fimg = np.zeros_like(dat('pred_stardist_ch0'))
            fimg[9000:13000, 11000:17000] = resf
            viewer.add_labels(fimg)

            # proc whole image
            # img = dat('raw_img_ch1')[9000:13000, 11000:17000]
            # img = img[np.newaxis]
            t0 = ug.dt()
            imf = uip.subtract_rolling_ball(image_dict['raw_img'][1], radius=18)
            print(f"{ug.dt()-t0}")

        if bool(0): # ROI assignment 
            ########################################################################
            from Quantification.plugins.roi_handling import ROIAssigner
            data={
                'rois':[viewer.layers['PV_DENDS'].data],
                'mip_raw':image_dict['mip_raw'],
            }
            config = {
                'INTENSITY_IMAGE_NAME':'mip_raw',
                'img_fmt': 'CYX',
                'REID_ROIS':False,
                'ALLOW_ROI_IMG_SHAPE_MISMATCH': True,
            }

            from Quantification.plugins.roi_handling import ROIHandlingStage

            stage = ROIHandlingStage()

            data = stage.run(data,config)

            ROI_ASSIGNMENT_METHODS = ['Coords']
            objects_img = image_dict['pred_stardist_2d'][0,0,:,0]

            data["rpdf"] = uc.get_rp_table(objects_img, image_dict['mip_raw'], ch_axis=0, get_object_coords=True)
            data["rpdf"] = ROIAssigner.assign_rois_to_rpdf(
                data["rpdf"], 
                data["polygons_per_label"], 
                data["rois"][0], 
                roi_assignment_methods=ROI_ASSIGNMENT_METHODS,
            )
            for roi_i in [1,3]:
                count = len(data['rpdf'][(data["rpdf"]["colocal_id"]==1) & (data["rpdf"]["roi_i_byCoords"]==roi_i)])
                dend_area = ((0.071**2) * data['roi_df'].query(f"(colocal_id == 1) & (roi_i=={roi_i})")['roi_area_px'].values[0])
                density = count/dend_area
                print(roi_i, '-->', density)          

            # calculated mean over individual dendrites
            #########################################################
            results = []
            for dfn, adf in data["rpdf"].groupby(["colocal_id", "roi_i_byCoords", "roi_polyi_byCoords"]):
                print(dfn)
                clcid, roi_i, poly_i = dfn
                if roi_i == 0:
                    continue
                count = len(adf)
                area = data['polygons_per_label'][roi_i][poly_i].area
                area = area * (0.071**2)

                density = count/area
                print(density)
                print()

                results.append(dict(zip(['clcid', 'roi_i', 'poly_i', 'count', 'area', 'density'],[clcid, roi_i, poly_i, count, area, density])))
            res = pd.DataFrame(results)
            res.groupby(['clcid', 'roi_i']).mean().reset_index()

        if bool(0): # remove layers
            from Annotation.napari_utils import delete_layers
            layers_to_remove = ['pred_neurseg_ch0', 'pred_neurseg_ch2', 'pred_stardist_2d_ch0', 'pred_stardist_2d_ch3']
            delete_layers(viewer, layers_to_remove)

        # uniform color for labels
        ########################################################
        if bool(0):
            layername = 'annotated_pred_stardist_ch1'
            from Annotation.napari_utils import get_single_color_cmap
            viewer.layers[layername].colormap = get_single_color_cmap()

        # !!! not implemented !!! clean up - remove tiny objects and fill label holes
        if bool(0):
            layername = 'annotated_pred_stardist_ch2'
            int_layer = 'raw_img_ch2'

            # !!! current implementation needs to be adapted to not exceed current object boundary
            def clean_labeled_image(label_image, min_size=64, area_threshold=64):
                """
                Remove small objects and fill internal holes in a labeled image.
                """
                from skimage.morphology import remove_small_objects, remove_small_holes
                from scipy import ndimage as ndi

                out = remove_small_objects(label_image, min_size=min_size)

                # One bounding box per original label (None for labels that are absent)
                bboxes = ndi.find_objects(out)

                for lbl, slc in enumerate(bboxes, start=1):
                    if slc is None:          # this label is not present
                        continue

                    view = out[slc]
                    mask = (view == lbl)     # pixels belonging to this original label
                    if not mask.any():
                        continue
                    # change this to not exceed objects current outer boundary !!!
                    filled = remove_small_holes(mask, area_threshold=area_threshold) * lbl
                    # bg = np.where(view!=lbl, view, 0)
                    # out[slc] = bg + filled
                    interior_fill = filled & ~mask
                    view[interior_fill] = lbl
                    out[slc] = view

                return out

            viewer.layers[layername].data = uip.relabel(viewer.layers[layername].data)
            rpdf = uc.get_rp_table(viewer.layers[layername].data, viewer.layers[int_layer].data)
            uc.plot_fancy_2d_hist(rpdf, 'area', 'intensity_mean')
            viewer.layers[layername].data = clean_labeled_image(viewer.layers[layername].data, area_threshold=180)

        # label size filtering
        ########################################################
        if bool(0):

            obj_layer_name = 'annotated_pred_neurseg_ch3'
            obj_layer_name = obj_layer_name if obj_layer_name in viewer.layers else obj_layer_name.replace('annotated_','')
            int_layer_name = 'mip_raw_ch3'

            obj_arr = viewer.layers[obj_layer_name].data
            int_arr = viewer.layers[int_layer_name].data

            # plot props
            labeled_array = scipy.ndimage.label(obj_arr)[0]
            rpdf = uc.get_rp_table(labeled_array, int_arr)

            uc.plot_fancy_2d_hist(rpdf, 'area', 'intensity_mean', ax_scales=['linear', 'linear'])

            if bool(1):
                sm_removed = uip.remove_small_objs(labeled_array, 600)
                viewer.add_labels(sm_removed)

                # assign roi_ids based on roi layer
                region_id_layer = 'ROI_0'
                out_layer_name = 'PV_DENDS'
                roi_arr = viewer.layers[region_id_layer].data
                roi_assigned = np.zeros_like(sm_removed)
                for val in np.unique(roi_arr):
                    union = np.where(roi_arr==val, sm_removed, 0)
                    roi_assigned = np.where(union>0, val, roi_assigned) 
                viewer.add_labels(roi_assigned, name=out_layer_name)

        # get rps - cfos
        if bool(0):

            if bool(0): # cfos
                obj_layer = 'pred_stardist_ch1'
                int_layer = 'raw_img_ch1'
                query_str = "(intensity_mean > 25) & (area > 1000)"
                relabel = False
            else: # PV - neurseg
                obj_layer = 'pred_stardist_ch2'
                int_layer = 'raw_img_ch2'
                query_str = "(intensity_mean > 10) & (area > 4000)"
                relabel = True

            objimg = (viewer.layers[obj_layer].data)
            objimg = uip.relabel(objimg) if relabel else objimg
            intimg = viewer.layers[int_layer].data
            uip.pai([objimg, intimg])

            rpdf = uc.get_rp_table(objimg, intimg)
            uc.plot_fancy_2d_hist(rpdf, 'area', 'intensity_mean', ax_scales=['linear', 'linear'])

            # objimg_filtered = uip.remove_small_objs(objimg, 4000) # 4000 area for PV soma
            rpdf_filt = rpdf.query(query_str) 
            objimg_filtered = uip.filter_label_img(objimg, rpdf_filt['label'].to_list())
            viewer.add_labels(objimg_filtered, name=f"annotated_{obj_layer}")

        # detect camkii nuc
        if bool(0):
            # objlayer = raw_img_ch0
            intlayer = 'raw_img_ch0'
            inputdata = viewer.layers[intlayer].data
            # inputdata = inputdata[8000:10000, 1600:3600]
            # up.show(inputdata)
            roilayer = 'PYR_layers' 
            inputrois = viewer.layers[roilayer].data

            _roi = np.where(inputrois==1, inputdata, 0)
            y0, y1, x0, x1 = uip.find_extent(_roi)
            input_crop = _roi[y0:y1+1, x0:x1+1]
            # bg = np.random.randint(0, input_crop.max()//2, size=input_crop.shape).astype('uint8')
            # input_crop = np.where(input_crop>0, input_crop, bg)

            from skimage import io, filters, morphology, measure, segmentation, util
            from scipy import ndimage as ndi

            def segment_nuclei(img,
                   sigma=1.5,
                   min_size=80,
                   h=0.3,
                   thresh_method="otsu"
                ):
                # 1) Smooth
                img_s = filters.gaussian(img, sigma=sigma, preserve_range=True)

                # 2) Threshold
                if thresh_method == "otsu":
                    thr = filters.threshold_otsu(img_s)
                    mask = img_s > thr
                elif thresh_method == "yen":
                    thr = filters.threshold_yen(img_s)
                    mask = img_s > thr
                elif thresh_method == "local":
                    # Sauvola for uneven background
                    thr = filters.threshold_sauvola(img_s, window_size=51)
                    mask = img_s > thr
                else:
                    raise ValueError("Unknown thresh_method")

                # 3) Remove tiny specks & fill small holes
                mask = morphology.remove_small_objects(mask, min_size=min_size)
                mask = morphology.remove_small_holes(mask, area_threshold=min_size)

                if not np.any(mask):
                    return np.zeros_like(img, dtype=np.int32)

                # 4) Distance transform on the binary mask
                dist = ndi.distance_transform_edt(mask)

                # 5) Find seeds using h-maxima (suppresses shallow peaks -> fewer over-segs)
                # h is relative to max(dist); tune it
                h_abs = h * dist.max()
                peaks = morphology.h_maxima(dist, h_abs)
                markers, _ = ndi.label(peaks)

                # 6) Watershed
                labels = segmentation.watershed(-dist, markers, mask=mask)

                return labels       

            labels = segment_nuclei(input_crop, sigma=1, thresh_method='otsu', h=0.3)
            labels = uip.remove_small_objs(labels, 1000).astype('int32')
            # up.show(labels)
            viewer.add_labels(labels, name=f"{intlayer}_labels")

        if bool(0): # predict with model
            from SynAPSeg.models import ModelPluginFactory
            model = ModelPluginFactory.get_plugin(
                    'stardist',
                    **dict(
                        model_path = r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\BygraveCode\models\2D_versatile_fluo",
                        in_dims_model = 'YX',
                        out_dims_pipe = "YX",
                        in_dims_pipe = 'YX',
                        preprocessing_kwargs = {'norm':(1, 99.8), 'clip':False},
                        predict_kwargs = {'prob_thresh': 0.1},
                    )
                )

            intlayer = 'raw_img_ch0'
            outlayername = 'annotated_pred_stardist_ch0'
            roilayer = 'PYR_layers' 
            # roilayer = 'annotated_PYR_layers_ch0'
            inputdata = viewer.layers[intlayer].data
            inputrois = viewer.layers[roilayer].data
            if bool(1):
                # to solve local threshold issues, run pred on cropped regions (pyr layers) then add back to og img
                pred_result = np.zeros_like(inputrois, dtype='int32')
                current_max_label = 0
                for ul in uip.unique_nonzero(inputrois):
                    _roi = np.where(inputrois==ul, inputdata, 0)
                    y0, y1, x0, x1 = uip.find_extent(_roi)
                    input_crop = inputdata[y0:y1+1, x0:x1+1]

                    _pred = model(input_crop)[0]
                    # model.preprocessing_kwargs = {'norm':(1, 99.8), 'clip':False}
                    # model.predict_kwargs['prob_thresh'] = 0.1
                    # pp = model.preprocess(input_crop, **model.preprocessing_kwargs)
                    # _pred = model.predict(pp, **model.predict_kwargs)
                    _pred += np.where(_pred>0, current_max_label, 0)
                    current_max_label = _pred.max()
                    pred_result[y0:y1+1, x0:x1+1] = _pred
                pred_result = uip.relabel(pred_result)
            else:
                # inputdata = inputdata[8000:10000, 1600:3600]
                pred_result = model(inputdata)[0]

            pred_result = uip.remove_small_objs(pred_result, 1500).astype('int32')
            uip.pai(pred_result); print(len(uip.unique_nonzero(pred_result)))
            viewer.add_labels(pred_result, name = (outlayername or f"{intlayer}_pred"))

    # spawn annotator instance for img directory
    if bool(0):
        from Annotation import napari_utils as nu

        viewer, widget_objects = nu.simple_viewer(
            EXAMPLE_I = 2,
            EXAMPLES_BASE_DIR = r"D:\BygraveLab\Confocal data archive\Pascal\SEGMENTATION_DATASETS\__compiled_training_data__\3d_dendrites\AMI_PV_dends_3d",
            _FILE_MAP = {
                'images':['.*as8bit_img.tiff'], # pattern match filenames
                'labels':['.*neursegPred_mask.tiff'],
            },
            IMG_FORMAT = 'ZYX',
        )

        if bool(0):
            from skimage import filters, morphology
            volume = viewer.layers['ex0010_PV_crop_r1r2c1c2-500-1524-500-1524_as8bit_img_ch0'].data
            enhanced = filters.meijering(volume.astype(np.float64))
            up.show(enhanced)
            uip.pai(enhanced)
            up.show(uip.to_binary(enhanced, 0.275390625))
            viewer.add_image(uip.to_8bit(enhanced))

        if bool(0):
            l = viewer.layers['ex0010_PV_crop_r1r2c1c2-500-1524-500-1524_as8bit_img_ch0']
            masked = np.where(l.data>14, 1, 0).astype('int32')
            up.show(masked)

            from skimage import measure
            lbld = measure.label(masked).astype('int32')
            rpdf = uc.get_rp_table(lbld[..., np.newaxis], l.data[..., np.newaxis], rps_to_get=['label', 'area', 'bbox', 'centroid', 'intensity_mean'])
            keep_lbls = rpdf[rpdf['area'] > 6000]['label'].to_list()
            for ii in [13,15,41,130, 18]:
                keep_lbls.remove(ii)
            filt = uip.filter_label_img(lbld, keep_lbls)
            up.show(filt)
            viewer.add_labels(uip.to_binary(filt), name='labels')

        if bool(0):
            ly = viewer.layers['labels']
            t = np.where(l.data>13, ly.data, 0).astype('int32')
            up.show(t)
            ly.data = t

    if bool(0): # loading individual images and masks from a h5 file for deepd3 annotations

        import flammkuchen as fl
        import scipy.ndimage as ndimage

        def remove_small_objects(arr, min_size, relabel = True):
            # Label connected components
            labeled_arr = ndimage.label(arr)[0] if relabel else arr

            # Remove small objects
            sizes = np.bincount(labeled_arr.ravel())
            mask = sizes >= min_size  # Keep objects with min_size or more pixels
            mask[0] = 0  # Background remains zero
            filtered = mask[labeled_arr]
            return filtered

        # 2. fill small holes in binary masks
        def fill_holes(binary_mask, hole_size_threshold=50):
            """
            Fill small holes in a 3D binary mask slice by slice.
            
            Args:
                binary_mask (numpy.ndarray): 3D binary mask with shape (Z, Y, X).
                hole_size_threshold (int): Minimum size of a hole to fill.

            Returns:
                numpy.ndarray: Mask with holes filled.
            """
            filled_mask = np.zeros_like(binary_mask, dtype=bool)
            for z in range(binary_mask.shape[0]):  # Iterate over Z-slices
                slice_mask = binary_mask[z]
                filled_slice = skimage.morphology.remove_small_holes(slice_mask, area_threshold=hole_size_threshold)
                filled_mask[z] = filled_slice
            return filled_mask

        def close_gaps(binary_mask, closing_radius=2):
            """
            Close small gaps in a 3D binary mask slice by slice using morphological closing.
            
            Args:
                binary_mask (numpy.ndarray): 3D binary mask with shape (Z, Y, X).
                closing_radius (int): Radius for the morphological closing operation.

            Returns:
                numpy.ndarray: Mask with small gaps closed.
            """
            closed_mask = np.zeros_like(binary_mask, dtype=bool)
            selem = skimage.morphology.disk(closing_radius)

            for z in range(binary_mask.shape[0]):  # Iterate over Z-slices
                slice_mask = binary_mask[z]
                closed_slice = skimage.morphology.binary_closing(slice_mask, selem)
                closed_mask[z] = closed_slice
            return closed_mask

        # 1/4/24
        # mask filling seems to work well with these params, except for on x30 will need smaller thresholds since resolution is lower than the rest.
        file_path = r"D:\BygraveLab\segmentation\synapse_datasets\online_datasets\deepd3\raw\DeepD3_Training.d3set"
        # file_path = r"R:\Confocal data archive\Pascal\SEGMENTATION_DATASETS\DeepD3_dataset\downloaded_data\DeepD3_Training.d3set"

        d = fl.load(file_path)

        get_ex = 'x12'
        stack, dends, spines = [d['data'][ds_name][get_ex] for ds_name in ['stacks', 'dendrites', 'spines']]
        uip.pai([stack, dends, spines])
        og_dends_spines = np.clip(dends+spines, 0, 1)

        stack_norm = (uip.normalize_01(stack)*255).astype('uint8')

        # cleaning up and making labels less convervative
        spines_filtered = remove_small_objects(spines, 6)
        spines_filled = fill_holes(spines_filtered, 3)

        dends_closed = close_gaps(dends)

        annots_merged = np.clip(dends_closed+spines_filled, 0, 1)
        annots_merged_filled = close_gaps(annots_merged)

        spines_lbld, n = ndimage.label(spines_filled)

        # make the annotations more expansive
        int_filt = ndimage.median_filter(np.where(stack_norm<20, 0, stack_norm), size=3)

        # def fill_mask_gaps_3d(intensity_image, binary_mask):
        thresh = skimage.filters.threshold_otsu(int_filt)
        bright_regions = int_filt > thresh

        # Step 2: Refine bright regions using 3D morphological operations
        refined_regions = skimage.morphology.binary_closing(bright_regions, skimage.morphology.ball(2))  # 3D closing with radius=3

        # Step 3: Merge the refined regions with the original mask
        filled_mask = np.logical_or(annots_merged_filled, refined_regions)
        filled_mask = remove_small_objects(filled_mask, 1000)    

        image_dict = {
            "raw_img":stack_norm, 
            "spines_raw": spines,
            "spines_labeled": spines_lbld,
            "filled_mask": filled_mask,
        }
        get_image_list = list(image_dict.keys())
        FILE_MAP = {
            "images": ["raw_img"], 
            "labels": ["spines_raw","spines_labeled","filled_mask"]
        }
        exmd = {
            'data_metadata':
                {'data_shapes':{},
                'data_formats': {"raw_img":"ZYX", "spines_raw":"ZYX", "spines_labeled":"ZYX", "filled_mask":"ZYX"}},
            'annotation_metadata':{'notes':'', 'status':''},
        }
        path_to_example = os.path.join(r"D:\BygraveLab\segmentation\synapse_datasets\online_datasets\deepd3\extracted_processed", 
                           f"{get_ex}")

        # init view
        ######################################################
        viewer, widget_objects = create_napari_viewer(
            exmd, LABEL_INT_MAP, path_to_example, FILE_MAP, image_dict, get_image_list
        )

        if bool(0): # export 
            # update dend mask with any added spines
            dends = np.where(viewer.layers['spines_labeled'].data != 0, 1, viewer.layers['filled_mask_ch0'].data)
            viewer.add_labels(dends)

    if bool(0): # loading and viewing czi file directly
        ###################################################################################
        EXAMPLES_BASE_DIR = r"D:\BygraveLab\ConfocalImages\SEGMENTATION_DATASETS\3d_annotations\pascal\raw_images"
        BASEDIR_CONTENTS = ug.get_contents(EXAMPLES_BASE_DIR, filter_str='.czi', endswith=True)
        EXAMPLE_I = 3
        CZI_METADATA = uczi.compile_czi_metadata(BASEDIR_CONTENTS)
        Z_MULTIPLE = 4
        path_to_example = BASEDIR_CONTENTS[EXAMPLE_I]
        example_str = Path(path_to_example).name
        exmd = CZI_METADATA[CZI_METADATA['fn'] == example_str].to_dict(orient='records')[0]
        exmd['annotation_metadata'] = {'notes':'', 'status':''}

        # load image data
        print(path_to_example)
        czi, scene_ids = uczi.read_czi(path_to_example)
        fmt_str = 'CZYX'
        img = uczi.czi_scene_to_array(czi, scene_ids[0], fmt_str, 0, None)[...,0]

        # set z multiple
        if Z_MULTIPLE is not None:
            img = uip.slice_by_multiple(img, fmt_str.index('Z'), Z_MULTIPLE)
        print(img.shape)

        # normalize image
        img_norm = uip.norm_percentile(img, (1,99.99), ch_axis=0)

        # create image dict
        image_dict = {'raw_img': img, 'norm_img':img_norm}
        get_image_list = list(image_dict.keys())
        FILE_MAP['3D'] = get_image_list
        viewer, widget_objects = create_napari_viewer(FILE_MAP, image_dict, get_image_list)

        # init labels
        viewer.add_labels(np.zeros_like(img[0].astype('int32')), name='pred_stardist_ch')

        lbls = tifffile.imread(r"D:\BygraveLab\ConfocalImages\SEGMENTATION_DATASETS\3d_annotations\pascal\annotations\2024_0913_virusTesting_PSD95eGFP-HM_DIO-geph-fingr-mScarlet_mark--40x-4z-dHPC--\coord0-128-256-384_pred_stardist_ch0.tiff")
        uip.pai(lbls)
        viewer.add_labels(lbls)

        if bool(0):
            outdir = ug.verify_outputdir(os.path.join(Path(EXAMPLES_BASE_DIR).parent,"annotations"))
            outdir = ug.verify_outputdir(os.path.join(outdir, f"{Path(example_str).stem}"))
            y1,x1,c = 192,192, 1
            sizey, sizex = 64*2, 64*2
            y2, x2 = y1+sizey, x1+sizex
            print(f"{y1}:{y2}, {x1}:{x2}")

            ch_str = f"_ch{c}"
            get_imgs = ['raw_img', 'pred_stardist']
            get_layers = [f"{el}{ch_str}" for el in get_imgs]
            layer_slices, outpaths = [], []
            for layer in get_layers:
                data = viewer.layers[layer].data
                dslice = data[:, y1:y2, x1:x2]
                uip.pai(dslice)
                outpath = os.path.join(outdir, f"coord{y1}-{y2}-{x1}-{x2}_{layer}.tiff")
                up.show(dslice)
                layer_slices.append(dslice)
                outpaths.append(outpath)

            # create overlay image
            overlay = up.overlay(uip.mip(layer_slices[0]), uip.mask_to_outlines(uip.mip(layer_slices[1])), 'red', (255,0,0))
            up.show(overlay)

            if bool(0):
                for i in range(len(layer_slices)):
                    dslice, outpath = layer_slices[i], outpaths[i]
                    tifffile.imsave(outpath, dslice)
