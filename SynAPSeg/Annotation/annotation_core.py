import os
import sys
import pandas as pd
from pathlib import Path
import napari
from magicgui import magic_factory, magicgui
import scipy.ndimage as ndi
import yaml
import numpy as np
from typing import Optional, Any

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.Annotation import napari_widgets as nw
from SynAPSeg.Annotation.annotation_IO import preproc_images_for_display
from SynAPSeg.Annotation.napari_custom_key_bindings import add_custom_keybindings
from SynAPSeg.Annotation.napari_utils import get_napari_cmaps, make_standard_vispy_cmaps


def build_display_layer_name(img_name, c_i, layer_type=None, skip_layer_types = ['annotations', 'ROIS']):
    """ convert img name to layer name by auto-appending a ch_x based on 
        index if img name doesn't end with ch_x
        else parsing the index from the ch_x string
            e.g. an annotated single ch 
        checks layer type if provided for exceptions to naming with suffix _chX
    """
    if layer_type and (layer_type in skip_layer_types):
        return img_name

    _ch = "" if MetadataParser.get_ch_from_str(img_name) else f"_ch{c_i}"
    layer_name = f"{img_name}{_ch}"          #+ (f"_ch{c_i}" if not img_name.endswith(f"_ch{c_i}") else "")
    return layer_name

def parse_label_region_props(layer:napari.layers.Labels, int_image_layer:Optional[napari.layers.Image]=None):
    
    if not isinstance(layer, napari.layers.Labels):
        print("Layer must be a napari.layers.Labels object")
        return None
    
    # determine the intensity image from layer chanel name
    int_img = None if int_image_layer is None else int_image_layer.data
        
    return get_region_props(layer.data, int_img)

def parse_viewer_layer_types(viewer) -> tuple[list[str], list[str], list[str]]:
    """ parse layers in anapari viewer and return a tuple of lists of layer names 
        sorted into Image, Labels, and Shapes """
    all_layer_names = get_napari_viewer_layer_names(viewer)
    image_layers, labels_layers, shapes_layers = [], [], []
    for n in all_layer_names:
        layer = viewer.layers[n]
        if isinstance(layer, napari.layers.Image):
            image_layers.append(n)
        elif isinstance(layer, napari.layers.Labels):
            labels_layers.append(n)
        elif isinstance(layer, napari.layers.Shapes):
            shapes_layers.append(n)
        else:
            print(f"Unknown layer type: {layer}")
    return image_layers, labels_layers, shapes_layers

def map_labels_layer_to_int_image(viewer) -> dict[str,str]:
    image_layers, labels_layers, shapes_layers = parse_viewer_layer_types(viewer)
    idx2img_names = {} # map channel index to list of image layers that have that channel index
    
    for img_name in image_layers:
        ch_int = MetadataParser.get_ch_from_str(img_name, as_int=True)
        if ch_int is None:
            continue
        if ch_int not in idx2img_names.keys():
            idx2img_names[ch_int] = []
        idx2img_names[ch_int].append(img_name)
    
    labels2int_img_map = {}
    for lbl_layer_name in labels_layers:
        ch_int = MetadataParser.get_ch_from_str(lbl_layer_name, as_int=True)
        if ch_int is None:
            continue
        
        map_order = ['raw_img', 'mip_raw'] # map preferentially to raw_img over mip
        for mo in map_order:
            int_ln = f"{mo}_ch{ch_int}"
            if int_ln in idx2img_names.get(ch_int, []):
                labels2int_img_map[lbl_layer_name] = int_ln
                break
        else:
            print(f"WARNING --> Could not map labels layer {lbl_layer_name} to an image layer")
    print(f"inferred labels layers to intensity image mapping:\n{labels2int_img_map}\n")
    return labels2int_img_map


def get_region_props(labels_img:np.ndarray, int_img:Optional[np.ndarray]=None, properties=("label", "area", "intensity_mean")) -> pd.DataFrame:
    from skimage.measure import regionprops_table
    int_img = np.full(labels_img.shape, np.nan) if int_img is None else int_img

    props = regionprops_table(
        label_image=labels_img,
        intensity_image=int_img,
        properties=properties,
    )

    # add background obj
    rpdf = pd.concat(
        [
            pd.DataFrame([{k: (0 if k == "label" else np.nan) for k in properties}]),
            pd.DataFrame(props),
        ],
        ignore_index=True,
    ).set_index("label", drop=False).rename(columns={"index": "label"})

    return rpdf


def display_images(
    viewer,
    images_to_display,
    apply_colormaps=True,
    set_lbl_contours=1,
    channel_colormaps: Optional[list] = None,
    parse_regionprops = True,
):
    """add to viewer and apply options based on "load_as" arg (either 'images', 'labels', 'annotations', or 'ROIS')"""

    # setup colormaps
    cmaps = get_napari_cmaps()

    # iterate over images to display, adding to viewer based on load_as key
    label_layers_to_parse_feats = []
    for imgd in images_to_display:
        t0 = ug.dt()

        # non-image-array handling
        ########################
        display_type = imgd.get('display_type')

        if display_type == 'shapes':

            layer = viewer.add_shapes(
                data = imgd['poly_coords'],
                name=imgd['img_name'],
                shape_type=imgd['shape_type'],
                ndim=imgd['ndim'],
                features=imgd['features'],
                feature_defaults={"name": "unnamed", "type": "exclusive", "custom": ""},
                text={
                    "string": "{name}",
                    "anchor": "center",
                    "color": "yellow",
                    "size": 9,
                },
                face_color='transparent',
                edge_width=10,
                visible=False,
            )
            layer.parent_layer_name = imgd.get('parent_layer_name') or None

            # Set drawing tool to polygon by default
            layer.mode = "add_polygon"

        # image array handling
        ######################
        else:
            n_channels = len(imgd['arr_ch_list'])       
            for c_i, channel_arr in enumerate(imgd['arr_ch_list']):
                layer_name = build_display_layer_name(imgd['img_name'], c_i, layer_type=imgd['load_as'])

                if imgd['load_as'] == "images":
                    layer = viewer.add_image(channel_arr, name=layer_name, blending="additive", visible=False)
                    if apply_colormaps:
                        if channel_colormaps and isinstance(channel_colormaps, list):
                            _cmaps = make_standard_vispy_cmaps()
                            if c_i < len(channel_colormaps):
                                cmap = channel_colormaps[c_i]
                                if cmap in _cmaps.keys():
                                    layer.colormap = _cmaps[cmap]
                            # TODO can add support for actual colormap objects

                        elif c_i in cmaps:
                            layer.colormap = cmaps[c_i] if n_channels > 1 else cmaps[4] # if only 1 channel default to gray, else bgrm

                elif imgd['load_as'] in ["labels", "annotations", "ROIS"]:
                    # TODO implement priority loading of annotated_ versions of images, probably upstream
                    # Example: Different handling for annotations or ROIs if needed
                    layer = viewer.add_labels(channel_arr, name=layer_name, blending="additive", visible=False)
                    layer.opacity = 1.0
                    layer.contour = set_lbl_contours
                    layer.brush_size = 1
                    label_layers_to_parse_feats.append(layer_name)

                else:
                    raise ValueError(imgd['load_as'])

        print(f"display_image for {imgd['img_name']} took {ug.dt()-t0}.")
    
    if parse_regionprops and len(label_layers_to_parse_feats) > 0:
        labels2int_img_map = map_labels_layer_to_int_image(viewer)
        for n in label_layers_to_parse_feats:
            ll = viewer.layers[n]
            int_img_name = labels2int_img_map.get(n)
            if int_img_name and int_img_name in viewer.layers:
                int_img = viewer.layers[int_img_name]
            else:
                int_img = None
            rpdf = parse_label_region_props(ll,  int_image_layer=int_img)
            print(rpdf.head())
            ll.features = rpdf


def create_napari_viewer(
        exmd,
        path_to_example,
        FILE_MAP, 
        image_dict,
        get_image_list: Optional[list[str]] = None,
        apply_colormaps=True,
        shapes_data = None,
        set_lbl_contours = 1,
        LABEL_INT_MAP = None,
        channel_colormaps: Optional[list[str]] = None,
        logger: Optional[Any] = None,
    ): 
    """ 
    main function to create a napari viewer handling data display and widget setup

    Args:
        exmd : dict
            Example metadata dict or an empty dict
        path_to_example : str
            Path to example directory
        FILE_MAP : dict[str, list[str]]
            Dict mapping display types to lists of filenames
            valid display types: images, labels, annotations, ROIs, 'ignore', 'shapeLayer_ROIs', 'metadata'
        image_dict : dict
            Dict mapping file stems to arrays
        get_image_list : Optional[list[str]]
            Optional list of image filenames to display (if None, all images in FILE_MAP will be displayed)
        apply_colormaps : bool
            Whether to apply colormaps
        shapes_data : Optional[dict]
            deprecated - shapes data should be passed through filemap
        set_lbl_contours : int
            Set all lbl layer contours. may want to disable this default if dealing with really large images
        LABEL_INT_MAP : Optional[dict]
             (not currently in working order, so unused) - optional dict mapping labels to int images
        channel_colormaps : Optional[list[str]]
            Optional list of strings which indicate a colormap to use for displaying image channels
            accepted colormaps must be one of ('blue', 'green', 'red', 'magenta', 'gray')
        logger : Optional[Any]
            Logger
    """
    
    # preprocess images for display
    t0 = ug.dt()
    images_to_display, viewer_kwargs = preproc_images_for_display(exmd, FILE_MAP, image_dict, get_image_list)
    
    if logger:
        logger.info(f"viewer_kwargs: {viewer_kwargs}")
        
    print(f"preproc_images_for_display completed in {ug.dt()-t0}.")
    
    # init viewer
    global viewer
    viewer = napari.Viewer(**viewer_kwargs)
    
    # parse display colormaps, if present
    if not channel_colormaps and exmd.get('channel_colormaps'):
        channel_colormaps = exmd.get('channel_colormaps')

    # add images to viewer
    t0 = ug.dt()
    display_images(viewer, images_to_display, apply_colormaps=apply_colormaps, set_lbl_contours=set_lbl_contours, channel_colormaps=channel_colormaps)
    print(f"display_images completed in {ug.dt()-t0}.")
    
    # build widgets
    t0 = ug.dt()
    widget_dict = build_widgets(viewer, path_to_example, exmd, logger)
    print(f"build_widgets completed in {ug.dt()-t0}.")

    return viewer, widget_dict


###### WIDGETS ####################################################################################################
def build_widgets(viewer, path_to_example, exmd, logger):
    """ 
    Build widgets for the annotation GUI - adds widgets to viewer and returns references to widgets (for debugging)
    Args:
        viewer: napari viewer
        path_to_example: path to example directory
        exmd: example metadata dict
        logger: logger
    Returns:
        widget_dict: dict of widgets - maintains references to widgets
        keys: 
            exwidg (nw.ExportWidget)
            tab_spec (dict[str, list[widget]])
    """
    # other widgets to add
    # nw.LabelMorphologyWidget(viewer) # maybe doesn't work
    # enable_exit_warning  # think there was some bug so i diabled but was working for me
    # FilterCollection widget was broken and probs has much simplier impl.
    # there are also a few in napari utils to copy over
    # update_segmentation widget
    # viewer.window.add_dock_widget(update_segmentation, area='right')
    
    # --- create core objects / widgets that depend on exwidg, viewer, etc. ---
    exwidg = nw.ExportWidget(path_to_example, exmd, logger=logger, viewer=viewer)
    
    # individual tools (keep references instead of adding directly to docks)
    widgetPatchNavigator = nw.PatchNavigator(viewer)
    autofill_plugin = nw.AutofillPlugin(viewer)

    image_filter_widget = nw.create_image_filter_widget(viewer)      # filtering tools
    threshold_widget = nw.BinaryThresholdWidget(viewer)
    object_filter_widget = nw.FilterLabelsWidget(viewer)
    morph_closing_widget = nw.MorphologicalClosingWidget(viewer)

    layer_export_widget = nw.layer_export(viewer, exwidg)            # export tools
    shape_layer_annotator = nw.ShapeFeatureAnnotator(viewer, exwidg)

    add_roi_widget = nw.AddROIWidget(viewer, exwidg)                 # labeling tools
    add_note_widget = nw.AddNoteWidget(exwidg)                       # utilities / notes
    metadata_widget = nw.MetadataWidget(viewer, exwidg)              # mark complete / metadata
    label_edit_widget = nw.LabelEditWidget(viewer)                   # label editing

    # -------------------------------------------------------------------------
    # Build tab spec for the control panel
    # -------------------------------------------------------------------------
    tab_spec = {
        "Core": [
            autofill_plugin,
            add_roi_widget,
            shape_layer_annotator,
            layer_export_widget,
            add_note_widget,
            metadata_widget,
        ],
        "Image filters": [
            image_filter_widget,
            threshold_widget,
            morph_closing_widget,
        ],
        "Label filters": [
            object_filter_widget,
            label_edit_widget,
            # nw.LabelErosionWidget(viewer).widget, # TODO fix allow max val to 65535, also refactor structure to be same as others
        ],
        "Utils": [
            widgetPatchNavigator,
            nw.CodeExecutionWidget(viewer, exwidg),
        ]
    }

    # -------------------------------------------------------------------------
    # Create and add the tabbed control panel dock
    # -------------------------------------------------------------------------
    control_panel = nw.make_tabbed_control_panel(tab_spec)
    viewer.window.add_dock_widget(control_panel, name="Control Panel", area="right")

    
    # add keybindings
    add_custom_keybindings(viewer)

    # return some widgets - maybe not req anymore
    widget_dict = {
        'export': exwidg,                  
        'tab_spec': tab_spec,
    }

    # add exit warning 
    # nw.enable_exit_warning(viewer)
    
    return widget_dict
    

def load_by_key(path_to_example, key):
    load_names = []
    for name in os.listdir(path_to_example):
        if key in name:
            load_names.append(name)
    return MetadataParser.read_example(path_to_example, load_files=load_names)


def get_napari_viewer_layer_names(viewer):
    return [el.name for el in viewer.layers]


def get_intimg_layer(layer_name, LABEL_INT_MAP):
    """map each layer name to corresponding intensity image layer
    tries to match keys in LABEL_INT_MAP to end of layer_name, return corresponding mapped value"""
    for k,v in LABEL_INT_MAP.items():
        if layer_name.endswith(k):
            return v
    return None


def get_label_layer_properties(rpdf, clc_id, annotate_props = ['intensity_mean', 'area', 'eccentricity']):
    clc_props = rpdf[rpdf['colocal_id']==clc_id].copy()
    bg_props = pd.DataFrame([{k:0 for k in clc_props.columns.to_list()}])
    clc_props = pd.concat([bg_props, clc_props], ignore_index=True).sort_values('label')
    properties = {prop:clc_props[prop].values for prop in annotate_props}
    return properties


def get_next_example(examples_dir):
    """ grabs the path to the next available example by checking folders for first instance that is not completed"""
    
    for ex_i in os.listdir(examples_dir):
        ex_dir_path = os.path.join(examples_dir, ex_i) 
        if not is_completed_example(ex_dir_path):
            return ex_i
    print('no examples to process were found.')
    return None 

def is_completed_example(example_dir):
    if Path(example_dir).stem.startswith("."):
        return True
    if "complete.txt" in os.listdir(example_dir):
        return True
    if "__skip_this_example.txt" in os.listdir(example_dir):
        return True
    
    return False


def get_export_layers(x=None):
    return [l.name for l in viewer.layers]

def current_layer_selection():
    if viewer.layers.selection.active:
        return viewer.layers.selection.active.name
    return ''


def get_example_path(EXAMPLE_I, dir_examples, FILE_MAP):
    """implements example indexing, paths, and loading metadata 
        Args:
            EXAMPLE_I: Any = name of example dir, or if 'auto' fetches next non-complete annotation
    """
    if EXAMPLE_I.lower() == 'auto': 
        EXAMPLE_I = get_next_example(dir_examples)
    example_str = str(EXAMPLE_I).zfill(4)

    path_to_example = os.path.join(dir_examples, example_str)
    ex_contents = os.listdir(path_to_example)
    exmd = MetadataParser.try_get_metadata(path_to_example, FILE_MAP, silent=True)
    
    path_key = 'image_path' if 'image_path' in exmd else 'czi_path'
    print(f'\n{example_str} --> {Path(exmd[path_key]).stem}')
    return example_str, path_to_example, ex_contents, exmd

def build_image_list(FILE_MAP, ex_contents, annot_prefixes=['annotated_', 'ROI_']):
    """Searches the example directory for files to load."""
    image_list = [item for sublist in FILE_MAP.values() for item in sublist]
    annotated_files = [el for el in ex_contents if any(el.startswith(prefix) for prefix in annot_prefixes)]
    image_list = list(set(image_list+annotated_files))
    
    
    # check if any have same filename (even if different suffix) as this would lead to error when images are added to viewer
    if len(set([ug.get_prefix(el) for el in image_list])) < len(image_list):
        raise ValueError(f"if any have same filename (even if different suffix) as this would lead to error when images are added to viewer.\ngot:{image_list}")

    return image_list


if __name__ == '__main__':

    viewer = napari.Viewer()
    afw = nw.AutofillPlugin(viewer)
    print(afw.name)
