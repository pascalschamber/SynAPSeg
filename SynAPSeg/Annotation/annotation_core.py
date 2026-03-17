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

def display_images(viewer, images_to_display, apply_colormaps=False, set_lbl_contours=1, channel_colormaps: Optional[list]=None):
    """ add to viewer and apply options based on "load_as" arg (either 'images', 'labels', 'annotations', or 'ROIS')"""
    
    # setup colormaps
    cmaps = get_napari_cmaps()
    
    # iterate over images to display, adding to viewer based on load_as key
    for imgd in images_to_display:
        t0 = ug.dt()
        n_channels = len(imgd['arr_ch_list'])       

        for c_i, channel_arr in enumerate(imgd['arr_ch_list']):
            layer_name = build_display_layer_name(imgd['img_name'], c_i)
            
            
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

            else:
                raise ValueError(imgd['load_as'])

                        
        print(f"display_image for {imgd['img_name']} took {ug.dt()-t0}.")
                


def create_napari_viewer(
        exmd, # should be examples_metadata dict, but think can be empty
        path_to_example, # path to dir containing images
        FILE_MAP, # dict mapping file names to display types 
        image_dict, # dict mapping file stems to arrays
        get_image_list: Optional[list[str]] = None, # optional list of image filenames to display (if None, all images in FILE_MAP will be displayed)
        apply_colormaps=True, 
        shapes_data = None, # optional dict of shapes data to display
        set_lbl_contours = 1, # set all lbl layer contours, may want to disable this default if dealing with really large images
        LABEL_INT_MAP = None,  # not currently in working order, so unused
        channel_colormaps: Optional[list[str]] = None, # optional list of strings which indicate a colormap to use for displaying image channels. accepted values are one of ['blue', 'green', 'red', 'magenta', 'gray']
        logger: Optional[Any] = None,
    ): # more to add
    
    # preprocess images for display
    t0 = ug.dt()
    images_to_display, viewer_kwargs = preproc_images_for_display(exmd, FILE_MAP, image_dict, get_image_list)
    
    if logger:
        logger.info(f"viewer_kwargs: {viewer_kwargs}")
        
    print(f"preproc_images_for_display completed in {ug.dt()-t0}.")
    
    # init viewer
    global viewer
    viewer = napari.Viewer(**viewer_kwargs)
    
    # add images to viewer
    t0 = ug.dt()

    # parse display colormaps, if present
    if not channel_colormaps and exmd.get('channel_colormaps'):
        channel_colormaps = exmd.get('channel_colormaps')

    display_images(viewer, images_to_display, apply_colormaps=apply_colormaps, set_lbl_contours=set_lbl_contours, channel_colormaps=channel_colormaps)
    print(f"display_images completed in {ug.dt()-t0}.")
    
    if shapes_data is not None:
        for k,v in shapes_data.items():
            if isinstance(v, str): # data not yet loaded, expected to be path to data 
                if v.endswith('.csv'):
                    df = pd.read_csv(v)
            sdata = ''
            viewer.add_shapes(v, name=k)

        
    ###### WIDGETS ####################################################################################################
    ###################################################################################################################
   
    # other widgets to add
    # nw.LabelMorphologyWidget(viewer) # maybe doesn't work
    # enable_exit_warning  # think there was some bug so i diabled but was working for me
    # FilterCollection widget was broken and probs has much simplier impl.
    # there are also a few in napari utils to cpy over
    # update_segmentation widget
    # viewer.window.add_dock_widget(update_segmentation, area='right')
        
    # --- create core objects / widgets that depend on exwidg, viewer, etc. ---
    exwidg = nw.ExportWidget(path_to_example, exmd, logger=logger)
    widgetPatchNavigator = nw.PatchNavigator(viewer)
    autofill_plugin = nw.AutofillPlugin(viewer)

    # individual tools (keep references instead of adding directly to docks)
    image_filter_widget = nw.create_image_filter_widget(viewer)      # filtering tools
    threshold_widget = nw.BinaryThresholdWidget(viewer)

    layer_export_widget = nw.layer_export(viewer, exwidg)            # export tools

    add_roi_widget = nw.AddROIWidget(viewer, exwidg)                 # labeling tools
    add_note_widget = nw.AddNoteWidget(exwidg)                       # utilities / notes
    metadata_widget = nw.MetadataWidget(viewer, exwidg)              # mark complete / metadata
    label_edit_widget = nw.LabelEditWidget(viewer)                   # label editing

    # -------------------------------------------------------------------------
    # Build tab spec for the control panel
    # -------------------------------------------------------------------------
    tab_spec = {
        "Utilities": [
            autofill_plugin,
            add_roi_widget,
            layer_export_widget,
            add_note_widget,
            metadata_widget,
        ],
        "Filtering": [
            image_filter_widget,
            threshold_widget,
        ],
        "Label Editing": [
            label_edit_widget,
            # nw.LabelErosionWidget(viewer).widget, # TODO fix allow max val to 65535, also refactor structure to be same as others
        ],
        "Navigation": [
            widgetPatchNavigator,
        ],
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
        # 'filter': nw.get_filter_object(viewer, LABEL_INT_MAP), # slows down loading 
        'export': exwidg,                  
        # 'label_morph': nw.add_label_morphology_widget(viewer),
        # 'label_delete': nw.add_label_deletion_widget(viewer),
    }

    # add exit warning 
    # nw.enable_exit_warning(viewer)
    print(f"widget setup completed in {ug.dt()-t0}.")
    return viewer, widget_dict






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


