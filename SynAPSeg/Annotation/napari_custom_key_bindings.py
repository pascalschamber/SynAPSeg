import napari
from napari.utils.notifications import show_info
import numpy as np
import os
import sys

from SynAPSeg.Annotation.napari_utils import get_napari_cmaps
from SynAPSeg.IO.metadata_handler import MetadataParser


def custom3d_camera_controls(
    viewer: napari.Viewer,
    pan_speed: float = 1.0,
    rot_speed: float = 0.35,
    zoom_speed: float = 0.004,
):
    """
    Swap 3D camera controls:

    - LMB drag  ............ PAN
    - Shift + LMB drag ..... TILT + ZOOM

    This tries to disable VisPy's default mouse handling and then
    implements camera motion entirely in a custom mouse-drag callback.
    """

    # ------------------------------------------------------------------
    # 1. Try to shut down the built-in mouse camera behavior
    # ------------------------------------------------------------------
    qt_viewer = viewer.window.qt_viewer

    # napari camera model: stop it from enabling mouse pan/zoom
    cam_model = viewer.camera
    # (these exist in 0.5.x+ and 0.6.x) :contentReference[oaicite:1]{index=1}
    cam_model.mouse_pan = False
    cam_model.mouse_zoom = False

    # VisPy camera wrapper used by napari in the canvas
    vispy_cam_wrapper = getattr(qt_viewer.canvas, "camera", None)

    if vispy_cam_wrapper is not None:
        # These are the actual vispy cameras used for 2D/3D.
        # They are subclasses of Arcball / PanZoom with an `interactive` flag.
        for vcam in (
            getattr(vispy_cam_wrapper, "_2D_camera", None),
            getattr(vispy_cam_wrapper, "_3D_camera", None),
        ):
            if vcam is None:
                continue
            # Best-effort: tell VisPy not to react to mouse
            try:
                vcam.interactive = False
            except Exception:
                # Older/newer versions might differ; ignore if missing.
                pass

    # ------------------------------------------------------------------
    # 2. Custom drag handler that owns the 3D controls
    # ------------------------------------------------------------------
    @viewer.mouse_drag_callbacks.append
    def swapped_camera(viewer: napari.Viewer, event):
        # Only touch 3D view
        if viewer.dims.ndisplay != 3:
            return

        # Only left button
        if event.button != 1:
            return

        cam = viewer.camera

        # Snapshot camera state at drag start
        start_center = np.asarray(cam.center, dtype=float)
        start_angles = np.asarray(cam.angles, dtype=float)
        start_zoom = float(cam.zoom)
        start_pos = np.asarray(event.pos, dtype=float)

        # Decide which mode we’re in based on modifiers
        shift_down = "Shift" in event.modifiers

        # ------------------------------------------------------------------
        # LMB drag => PAN
        # ------------------------------------------------------------------
        if not shift_down:
            # We’ll implement a simple “screen-space pan”: dragging moves the
            # camera center in X/Y in world space, scaled by zoom.
            while event.type == "mouse_move":
                cur_pos = np.asarray(event.pos, dtype=float)
                dx, dy = cur_pos - start_pos  # in screen pixels

                # Basic mapping: screen X -> world X, screen Y -> world Y
                # Scale by zoom so panning feels similar at different zooms
                factor = pan_speed / max(start_zoom, 1e-6)
                new_center = start_center.copy()
                # Note: sign choices tuned so drag direction matches intuition
                new_center[0] -= dx * factor
                new_center[1] += dy * factor

                cam.center = tuple(new_center)
                yield

        # ------------------------------------------------------------------
        # Shift + LMB drag => TILT + ZOOM
        # ------------------------------------------------------------------
        else:
            # Vertical drag = tilt + zoom, horizontal drag = yaw/roll (here yaw)
            while event.type == "mouse_move":
                cur_pos = np.asarray(event.pos, dtype=float)
                dx, dy = cur_pos - start_pos

                rx, ry, rz = start_angles

                # Map dy to pitch (ry) and dx to yaw (rz)
                ry_new = ry + dy * rot_speed
                rz_new = rz + dx * rot_speed
                cam.angles = (rx, ry_new, rz_new)

                # Zoom based on vertical drag (scroll-like)
                zoom_delta = 1.0 + (-dy) * zoom_speed
                zoom_delta = max(0.1, zoom_delta)  # clamp extreme values
                cam.zoom = max(1e-6, start_zoom * zoom_delta)

                yield

        # Let napari know we handled this event; this *helps* avoid other
        # callbacks acting on it, though VisPy has already seen it earlier.
        event.handled = True



def add_custom_keybindings(viewer):
    # add custom key bindings
    ####################################

    # custom3d_camera_controls(viewer)

    def isin(elem, alist):
        for el in alist:
            if el == elem:
                return True
        return False
    def show_pred_ch(viewer, chs):
        for layer in viewer.layers:
            if isin(layer.name, chs):
                layer.visible=True
            else:
                layer.visible=False
    @viewer.bind_key('Ctrl-0')
    def show_all_mip_channels(viewer):
        show_els = ['mip_n2v_ch', 'mip_raw_ch']
        show_pred_ch(viewer, [f"{el}0" for el in show_els] + [f"{el}1" for el in show_els] + [f"{el}2" for el in show_els] + [f"{el}3" for el in show_els])

    @viewer.bind_key('Ctrl-1')
    def show_ch1(viewer):
        show_els = ['pred_stardist_ch', 'annotated_pred_stardist_ch', 'mip_n2v_ch', 'mip_raw_ch']
        show_pred_ch(viewer, [f"{el}0" for el in show_els])
    @viewer.bind_key('Ctrl-2')
    def show_ch2(viewer):
        show_els = ['pred_stardist_ch', 'annotated_pred_stardist_ch', 'mip_n2v_ch', 'mip_raw_ch']
        show_pred_ch(viewer, [f"{el}1" for el in show_els])
    @viewer.bind_key('Ctrl-3')
    def show_ch3(viewer):
        show_els = ['pred_stardist_ch', 'annotated_pred_stardist_ch', 'mip_n2v_ch', 'mip_raw_ch']
        show_pred_ch(viewer, [f"{el}2" for el in show_els])
    @viewer.bind_key('Ctrl-4')
    def show_ch4(viewer):
        show_els = ['pred_stardist_ch', 'annotated_pred_stardist_ch', 'mip_n2v_ch', 'mip_raw_ch']
        show_pred_ch(viewer, [f"{el}3" for el in show_els])
        
    @viewer.bind_key('Ctrl--')
    def hide_all_layers(viewer):
        for layer in viewer.layers:
            layer.visible=False
        
    # key binding for toggleing layer visibility
    @viewer.bind_key('Ctrl-Alt-H')
    def toggle_visibility(viewer):
        """Toggle the visibility of the selected layer"""
        
        current_layer = viewer.layers.selection.active
        if current_layer is not None:
            # Toggle the visibility of the selected layer
            current_layer.visible = not current_layer.visible
        else:
            show_info("No layer is currently selected.")
    
    # key binding for facilitating comparing mip and stack for current image layer
    @viewer.bind_key('Ctrl-Alt-G')
    def toggle_compare_stack(viewer):
        # get ch_x from current layer
        current_layer = viewer.layers.selection.active
        ch_x = MetadataParser.get_ch_from_str(current_layer.name)
        if ch_x is None: 
            return
        valid_layers = [l for l in viewer.layers if (ch_x in l.name and isinstance(l, napari.layers.image.image.Image))]
        ch_map = {'mip_':get_napari_cmaps()[3], 'raw_':get_napari_cmaps()[1]}
        for l in valid_layers:
            l.visible=True
            l_cmap = ch_map['mip_'] if 'mip_' in l.name else ch_map['raw_'] if 'raw_' in l.name else None
            l.colormap=l_cmap
            # print(l.name, l.visible)

    # reset all image layers
    @viewer.bind_key('Ctrl-Alt-F')
    def reset_image_layers(viewer):
        valid_layers = [l for l in viewer.layers if (isinstance(l, napari.layers.image.image.Image))]
        for l in valid_layers:
            l.colormap='gray'
    
    def str_contains_val(astr, vals, match_all=False):
        matchFxn = all if match_all is True else any
        return matchFxn([v in astr for v in vals])
    
    def get_matching_layer_name(viewer, match_vals):
        for l in viewer.layers:
            if str_contains_val(l.name, match_vals, match_all=True):
                return l
        print(f"could not find matching layer for vals: {match_vals}")
        return None
        
    def get_current_layer(viewer):
        return viewer.layers.selection.active
    def get_visible_layers(viewer):
        return [l for l in viewer.layers if l.visible is True]
    
    
    # switch between 2d/3d images
    @viewer.bind_key('Ctrl-Alt-D')
    def switch_2d3d(viewer):
        _DEBUG = False
        switch_map = ['mip_raw','raw_img']
        assert len(switch_map) == 2, f"switch_map must contain exactly 2 values, got: {switch_map}"

        current_layer = get_current_layer(viewer)
        ch_x = MetadataParser.get_ch_from_str(current_layer.name)
        if _DEBUG: print(f"in switch 2d3d selected current layer: {current_layer.name}, ch: {ch_x}")

        # check if layer name contains mapped phrase
        if ch_x is None or not str_contains_val(current_layer.name, switch_map):
            # if not selected check all visible layers, getting first one with a map phrase
            vis_map_layer = [l for l in get_visible_layers(viewer) if str_contains_val(l.name, switch_map)]
            if len(vis_map_layer) < 1:
                print(f'in switch 2d3d --> not valid on this layer: {current_layer.name}')
                return
            else:
                current_layer = vis_map_layer[0]
        
        # so if layer name contains first map val, then hide it and get layer name of second val
        mapVals = [switch_map[1] if str_contains_val(current_layer.name, switch_map[0], match_all=True) else switch_map[0]] + [ch_x]
        switchToLayer = get_matching_layer_name(viewer, mapVals)
        if _DEBUG: print(f"in switch 2d3d switchToLayer: {switchToLayer.name}, mapVals: {mapVals}")
        if switchToLayer is not None:
            current_layer.visible = False
            switchToLayer.visible = True
            
    
    def toggle_layer_visibility(viewer):
        """
        Toggles visibility between 'pred_n2v_ch0' and 'raw_img_ch0' layers.
        If one is visible, it hides it and shows the other.
        """
        name1, name2 = "raw_img_ch1", "pred_n2v_ch1"
        layer1 = viewer.layers[name1] if name1 in viewer.layers else None
        layer2 = viewer.layers[name2] if name2 in viewer.layers else None

        if layer1 is None or layer2 is None:
            print("Error: One or both layers not found")
            return

        if layer1.visible:
            layer1.visible = False
            layer2.visible = True
        else:
            layer1.visible = True
            layer2.visible = False


    # Register the keybinding
    @viewer.bind_key("Ctrl-Alt-B")
    def toggle_layers(viewer):
        toggle_layer_visibility(viewer)

    return viewer