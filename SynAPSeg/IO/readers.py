"""
this script implements custom readers for czi and ome tiff image pyramids
    not sure if aicsimgio sufficiently handle all situations
    
pylibCZIrw lib has several benefits over ascilibczi 
    - way faster, zeiss developed, handles all types of images, metadata easy to acess, simple

Think it should replace ascilibczi, esp. for doing region roi loading from rpdf properties (e.g. the roi of a synapse)
    however need to complete wrapper class so it can be a drop in replacement


"""
from __future__ import annotations

import sys
import os
from typing import Iterable, Tuple, List, Dict, Optional, Callable, Any, Union, Sequence
import numpy as np
from pathlib import Path
from itertools import product
from functools import wraps
from dataclasses import dataclass
import tifffile




try:
    from pylibCZIrw import czi as pyczi
    pyCziReader = pyczi.CziReader # see ref: https://github.com/ZEISS/pylibczirw/blob/main/doc/jupyter_notebooks/pylibCZIrw_4_1_0.ipynb
except:
    pyczi = None
    pyCziReader = None
    

IndexLike = Union[int, slice, str, None]

def accept_czi_or_path(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: allow `func` param to be a Reader or a path; open/close if path."""
    @wraps(func)
    def wrapper(czidoc: Union[pyCziReader, str, Path], *args, **kwargs):
        if isinstance(czidoc, (str, Path)):
            with pyczi.open_czi(str(czidoc)) as doc:
                return func(doc, *args, **kwargs)
        return func(czidoc, *args, **kwargs)    
    return wrapper
        
@accept_czi_or_path
def get_czi_shape(czidoc: pyCziReader) -> str:
    """ returns the shape of a czi object or if passed a path to czi file will read it first """

    tbb = czidoc.total_bounding_box
    parts = [f"{k}: {v[1]-v[0]}" for k, v in tbb.items()]
    parts.append(f" origin=(x0: {tbb['X'][0]}, y0: {tbb['Y'][0]})")
    parts.append(get_scene_info(czidoc))  # safe: already a Reader here
    
    return ", ".join(parts)
    
@accept_czi_or_path
def get_scene_info(czidoc: pyCziReader) -> str:
    """ returns basic info about number of scenes and thier indicies. note: if multiscene czidoc.scenes_bounding_rectangle returns the shapes of each """
    try:
        scenes = list(czidoc.scenes_bounding_rectangle.keys())
        return f"scenes: (N={len(scenes)}) --> {scenes}"
    except Exception as e:
        return f"failed to fetch scene info (error message): {e}"
    

@accept_czi_or_path
def print_czi_obj_attrs(czidoc: pyCziReader) -> None:
    """ helper function to see all attributes, note: metadata is large """
    print_params_list = [
        'scenes_bounding_rectangle',
        'scenes_bounding_rectangle_no_pyramid',
        'total_bounding_box',
        'total_bounding_box_no_pyramid',
        'total_bounding_rectangle',
        'total_bounding_rectangle_no_pyramid',
        'get_channel_pixel_type',
        'pixel_types',
        'get_cache_info',
        'custom_attributes_metadata',
    ] 
    for param in print_params_list:
        value = getattr(czidoc, param)
        appendstr = ''
        try:
            if callable(value):
                value = value()
                param += '()'
        except Exception as e:
            appendstr += f"\n\t[error calling: {e}]"
        print(f"{param}: {value}{appendstr}")



def _parse_slice_like(x) -> Union[int, slice, None]:
    """Parse an index-like value into int, slice, or None.

    Accepts int, slice, None, or strings "start:stop[:step]" (whitespace allowed).
    Empty fields map to None (e.g., ":1024" -> slice(None, 1024, None)).

    Args:
    x: Index-like value.

    Returns:
    Union[int, slice, None]: Normalized index.

    Raises:
    ValueError: Malformed slice string.
    TypeError: Unsupported type.
    """

    if x is None or isinstance(x, (int, slice)):
        return x
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad slice string: {x!r}")
        a = int(parts[0]) if parts[0] else None
        b = int(parts[1]) if parts[1] else None
        c = int(parts[2]) if len(parts) == 3 and parts[2] else None
        return slice(a, b, c)
    raise TypeError(f"Unsupported index type: {type(x)}")

def _expand_sel(sel: IndexLike, size: Optional[int]) -> List[int]:
    """Expand an index-like selection into an explicit list of indices.

    Behavior:
    int -> [int]
    None -> range(size)
    slice/string -> list(range(start, stop, step)) using size to fill opens.

    Args:
    sel: Index-like selection.
    size: Axis length used to expand None or open-ended slices.

    Returns:
    List[int]: Concrete indices.

    Raises:
    ValueError: size required but not provided.
    """

    s = _parse_slice_like(sel)
    if isinstance(s, int):
        return [s]
    if s is None:
        if size is None:
            raise ValueError("Size required to expand None selection.")
        return list(range(size))
    # slice
    start = 0 if s.start is None else s.start
    stop  = size if s.stop  is None else s.stop
    step  = 1 if s.step  is None else s.step
    return list(range(start, stop, step))

def _slice_to_bounds(s: IndexLike, total_len: int):
    """Convert a selection into contiguous ROI bounds and optional stride.

    Returns a tuple suitable for reader ROIs:
    int  -> (idx, 1, None)
    None -> (0, total_len, None)
    slice -> (start, stop-start, step or None)

    Args:
    s: Index-like selection.
    total_len: Full axis length to resolve open bounds.

    Returns:
    Tuple[int, int, Optional[int]]: (start, length, post_read_step).
    """

    s_parsed = _parse_slice_like(s)
    if isinstance(s_parsed, int):
        return s_parsed, 1, None
    if s_parsed is None:
        return 0, total_len, None
    start = 0 if s_parsed.start is None else s_parsed.start
    stop  = total_len if s_parsed.stop  is None else s_parsed.stop
    step  = s_parsed.step
    return start, max(0, stop - start), (None if step in (None, 1) else step)


def read_czi_subsection(
    filepath: str | Path,
    dims: Optional[str] = None,
    coords: Optional[Tuple[IndexLike, ...]] = None,
    scene: Optional[int] = None,
    scene_idx: int = 0,
    zoom: float = 1.0,
    roi_relative: bool = True,
):
    """Read a subsection (or the whole image) from a single-scene CZI.

    If `dims` and `coords` are both omitted, the function loads the **entire image** for a single scene (default is scene 0):
        all available non-spatial axes and full XY --> "HTCZYX"

    Args:
      path: Path to the `.czi` file.
      dims: Axis string (e.g., "CZYX"). If None, infer present axes and use "TZCHSYX"
        subset in that order (only axes present in the file).
      coords: Tuple of indices/slices/None matching `dims`. If None while `dims` is given,
        defaults to full-range (None) for every axis. If both `dims` and `coords` are None,
        read the whole image.
      scene: Scene identifier. If None, use `scene_idx`.
      scene_idx: Scene index if `scene` is None.
      zoom: Read zoom (1.0 = native).
      roi_relative: Whether XY slices are relative to image origin (recommended). *not relative is not supported*

    Returns:
      (ndarray, dict): Stacked array and metadata dict.
    
    Example usage:
        # load a huge image with multi-file scenes
        filepath = "image.czi"
        
        # get shape
        print(get_czi_shape(filepath)) 
        
        arr, meta = read_czi_subsection(
                filepath,
                dims="TCZYX",
                scene_idx=1, # load scenes from a multi-scene file
                coords=(0, None, 0, None, None),
                # coords=(0, None, 0, "0:1024", "256:512"), # load with fancy indexing
                zoom = 0.1,  # load downscaled version
        )
        print(arr.shape)
    """
    

    with pyczi.open_czi(str(filepath)) as czidoc:

        # Scene handling
        scenes = list(czidoc.scenes_bounding_rectangle.keys())
        scene_used = None if not scenes else scenes[scene_idx] if scene is None else scene 

        # if mulitple scenes need to use current scenes bounding rectangle as roi origin, but still use total bounding dims
        # Bounding box and per-axis lengths from tbb (more robust than metadata here)
        tbb = czidoc.total_bounding_box  # {'X': (x0,x1), 'Y': (y0,y1), 'T': (t0,t1), ...}
        if scenes:
            sbb = czidoc.scenes_bounding_rectangle[scene_used]
            tbb['X'] = (sbb.x, sbb.x + sbb.w)
            tbb['Y'] = (sbb.y, sbb.y + sbb.h)
        
        axis_len_from_tbb = {ax: (rng[1] - rng[0]) for ax, rng in tbb.items()}

        # If dims/coords omitted => read whole image (all present axes, full XY)
        _dims = (dims or '').upper()
        if dims is None and coords is None:
            # Use a stable preferred order; include only axes actually present
            preferred = [ax for ax in "HTCZ" if ax in axis_len_from_tbb] + ["Y", "X"]
            _dims = "".join(preferred)
            _coords = tuple(None for _ in _dims)
        elif dims is not None and coords is None:
            # User provided dims only: default to full range for each axis
            _coords = tuple(None for _ in _dims)
        else:
            _coords = coords # user provided coords and dims
        
        if _dims is None:
            raise ValueError(f"if you provide coords, must also provide dims")
        if len(_dims) != len(_coords):
            raise ValueError("dims and coords lengths must match.")
        if "X" not in _dims or "Y" not in _dims:
            raise ValueError("Both X and Y must be present in dims.")


        # Build per-axis selections; use tbb lengths for expansion
        axis_to_sel: Dict[str, List[int]] = {}
        x_sel = y_sel = None

        for ax, sel in zip(_dims, _coords):
            if ax == "X":
                x_sel = sel if isinstance(sel, (int, slice)) else _parse_slice_like(sel) if isinstance(sel, str) else sel
            elif ax == "Y":
                y_sel = sel if isinstance(sel, (int, slice)) else _parse_slice_like(sel) if isinstance(sel, str) else sel
            else:
                axis_to_sel[ax] = _expand_sel(sel, axis_len_from_tbb.get(ax))

        # ROI from XY
        x0, x1 = tbb["X"]
        y0, y1 = tbb["Y"]
        width_full, height_full = x1 - x0, y1 - y0
        if roi_relative:
            x_origin, y_origin = x0, y0
        else:
            x_origin, y_origin = 0,0 # czidoc.total_bounding_box['X'][0], etc. maybe??
        
        xs, xlen, xstep = _slice_to_bounds(x_sel, width_full)
        ys, ylen, ystep = _slice_to_bounds(y_sel, height_full)
        roi = (x_origin + xs, y_origin + ys, xlen, ylen)

        # Iteration axes (non-spatial) in dims order
        iter_axes = [ax for ax in _dims if ax not in ("X", "Y")]
        for ax in iter_axes:
            if ax not in axis_to_sel or len(axis_to_sel[ax]) == 0:
                axis_to_sel[ax] = [0]
        combos = list(product(*[axis_to_sel[ax] for ax in iter_axes])) if iter_axes else [()]

        # Read and stack planes
        planes = []
        for combo in combos:
            plane = {}
            for ax, idx in zip(iter_axes, combo):
                if ax in ("T", "Z", "C", "H", "S"):
                    plane[ax] = int(idx)
            img = czidoc.read(plane=plane, scene=scene_used, zoom=zoom, roi=roi)
            # a = _normalize_plane_to_yx(np.asarray(img), y_len=ylen, x_len=xlen)
            if img.ndim == 3:
                assert img.shape[-1] == 1, f"{img.shape} -> RGB pixel type not handled "
                a = img[:, :, 0]
            else:
                a = img

            if ystep:
                a = a[::ystep, :]
            if xstep:
                a = a[:, ::xstep]
            planes.append(a)

        if len(planes) == 1:
            stacked = planes[0]
            y_out, x_out = stacked.shape
        else:
            y_out, x_out = planes[0].shape
            stacked = np.stack(planes, axis=0)
            shape_axes = [len(axis_to_sel[ax]) for ax in iter_axes]
            stacked = stacked.reshape(*shape_axes, y_out, x_out)

        meta = {
            "scene_used": scene_used,
            "zoom": zoom,
            "roi_abs_xywh": roi,
            "axis_order": list(iter_axes) + ["Y", "X"],
            "selections": {
                **{ax: (vals if len(vals) > 1 else vals[0]) for ax, vals in axis_to_sel.items()},
                "Y": _parse_slice_like(y_sel),
                "X": _parse_slice_like(x_sel),
            },
            "plane_shape_yx": (y_out, x_out),
        }
        return stacked, meta
 


    
try:
    # Optional: nicer OME-XML parsing if available
    from ome_types import from_xml as ome_from_xml  # type: ignore
    _HAVE_OME_TYPES = True
except Exception:
    _HAVE_OME_TYPES = False


@dataclass(frozen=True)
class LevelInfo:
    index: int
    shape: Tuple[int, ...]
    axes: str
    dtype: np.dtype
    downsample: float
    tile_shape: Optional[Tuple[int, ...]]
    compression: Optional[str]


class PyramidOMEReader:
    """
    Lazy wrapper around an OME-TIFF with multi-resolution pyramid. Uses tifffile lib.
        - Does NOT load pixel data at init.
        - Discovers series, levels, shapes, axes, dtype, downsampling.
        - Exposes OME-XML and a minimal structured metadata summary.
        - Loads arrays on demand via `get_array(level=..., ...)`.

    Notes:
        * Axis order follows tifffile's `Series.axes` ('TCZYX' or 'CZYX', etc.).
        * `get_array` returns a numpy array; may optionally pass `indexers`
        (a tuple of slices/ints/None per axis) to window or pick channels, timepoints, etc.
        * For large images, prefer slicing to avoid loading entire planes.
    
    Example usage:
        ometiffpath = "raw_img.ome.tiff"
        reader = PyramidOMEReader(ometiffpath)

        # See what’s inside (no pixels loaded yet)
        print(reader.info())
        # {'path': '...', 'axes': 'TCZYX', 'shape_level0': (...), 'num_levels': 4, 'levels': [...], ...}

        # List levels and their shapes / downsample factors
        for L in reader.levels:
            print(f"Level {L.index}: shape={L.shape} axes={L.axes} downsample≈{L.downsample:.1f}x tiles={L.tile_shape}")

        # Grab the full pyramid level 2 array (e.g., for a quick whole-slide view)
        arr_lv2 = reader.get_array(level=2)

        # Grab a window at level 0: first timepoint, first channel, all Z, Y[0:4096], X[0:4096]
        # Assuming axes 'TCZYX'
        win = reader.get_array(
            level=0,
            # indexers=(0, 0, slice(None), slice(0, 4096), slice(0, 4096)), # currently not working
            squeeze=True
        )
    """

    def __init__(self, path: str, series: int = 0) -> None:
        self._path = path
        self._series_index = series

        # Probe structural info without materializing pixels
        with tifffile.TiffFile(self._path) as tf:
            if series >= len(tf.series):
                raise IndexError(f"Requested series {series}, but file has only {len(tf.series)} series.")
            s = tf.series[series]

            # Basic series info
            self._axes: str = s.axes  # e.g. 'TCZYX'
            self._dtype: np.dtype = s.dtype
            self._shape: Tuple[int, ...] = tuple(s.shape)
            # tifffile reports levels for pyramids (SubIFDs / OME pyramid)
            levels = getattr(s, "levels", None) or [s]
            self._levels: List[LevelInfo] = []

            # Estimate downsample factor vs level 0 using last two spatial axes found in axes
            yx_axes = self._get_yx_indices(self._axes)

            base_y = self._shape[yx_axes[0]]
            base_x = self._shape[yx_axes[1]]

            for i, lev in enumerate(levels):
                shp = tuple(lev.shape)
                # Try extracting tile shape & compression from the first page of this level
                tile_shape = None
                compression = None
                try:
                    first_page = lev.pages[0]
                    if first_page.tilewidth and first_page.tilelength:
                        # Build a tile shape aligned to axes (only YX tiles are defined in TIFF) stores (tile_y, tile_x)
                        tile_shape = (first_page.tilelength, first_page.tilewidth)
                    if first_page.compression:
                        compression = str(first_page.compression)
                except Exception:
                    pass

                # Downsample (spatial) relative to base
                y_idx, x_idx = yx_axes
                y, x = shp[y_idx], shp[x_idx]

                ds = max(base_y / max(1, y), base_x / max(1, x)) # Avoid divide by zero

                self._levels.append(
                    LevelInfo(
                        index=i,
                        shape=shp,
                        axes=s.axes,
                        dtype=s.dtype,
                        downsample=ds,
                        tile_shape=tile_shape,
                        compression=compression,
                    )
                )

            # OME-XML (raw)
            self._ome_xml: Optional[str] = tf.ome_metadata if hasattr(tf, "ome_metadata") else None
            # Quick metadata peek (physical pixel sizes, channels, etc.)
            self._meta_summary: Dict[str, Any] = self._build_meta_summary(self._ome_xml)

    # ---------------------------
    # Public properties / helpers
    # ---------------------------
    @property
    def path(self) -> str:
        return self._path

    @property
    def axes(self) -> str:
        """Axis string, e.g. 'TCZYX'."""
        return self._axes

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of level 0 in axis order `axes`."""
        return tuple(self._shape)

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    @property
    def levels(self) -> List[LevelInfo]:
        """List of LevelInfo for each resolution level (0 = full res)."""
        return self._levels.copy()

    @property
    def ome_xml(self) -> Optional[str]:
        """Raw OME-XML string, if present."""
        return self._ome_xml

    @property
    def metadata_summary(self) -> Dict[str, Any]:
        """
        Minimal structured summary of OME metadata.
        Includes pixel sizes, units, channels (names), acquisition date/time, etc. when available.
        """
        return dict(self._meta_summary)

    def info(self) -> Dict[str, Any]:
        """ snapshot of key properties."""
        return {
            "path": self._path,
            "series_index": self._series_index,
            "axes": self._axes,
            "dtype": str(self._dtype),
            "shape_level0": self._shape,
            "num_levels": self.num_levels,
            "levels": [
                {
                    "index": L.index,
                    "shape": L.shape,
                    "axes": L.axes,
                    "dtype": str(L.dtype),
                    "downsample": L.downsample,
                    "tile_shape": L.tile_shape,
                    "compression": L.compression,
                }
                for L in self._levels
            ],
            "has_ome_xml": self._ome_xml is not None,
            "metadata_summary": self.metadata_summary,
        }

    # ---------------------------
    # Pixel access
    # ---------------------------
    def get_array(
        self,
        level: int = 0,
        indexers: Optional[Tuple[Union[int, slice, None], ...]] = None,
        dtype: Optional[np.dtype] = None,
        squeeze: bool = False,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Load pixels from the specified pyramid `level` into a NumPy array.

        Parameters
        ----------
        level : int
            Pyramid level to read (0 = full resolution).
        indexers : tuple[int|slice|None, ...], optional # NOT IMPLEMENTED
            Per-axis indexers (in the same order as `self.axes`), used to window
            or pick specific T/C/Z/XY ranges without loading the entire array.
            For example:
              - pick channel 0 from 'TCZYX': indexers=(0, 0, slice(None), slice(None), slice(None))
              - window spatial: indexers=(..., slice(y0,y1), slice(x0,x1))
            If None, the entire level is read.
        dtype : np.dtype, optional
            If provided, cast the result to this dtype.
        squeeze : bool
            If True, squeeze singleton axes.
        copy : bool
            If True, ensure the returned array owns its memory

        Returns
        -------
        np.ndarray
        """
        if level < 0: # handle negative indicies e.g. -1 returns last level 
            level = self.num_levels + level
            
        if not (0 <= level < self.num_levels):
            raise IndexError(f"Level {level} out of range (0..{self.num_levels-1}).")

        with tifffile.TiffFile(self._path) as tf:
            s = tf.series[self._series_index]
            
            # testing using zarr; supports slicing efficiently
            # (tifffile builds a VirtualStack as needed).
            # zarr_arr = s.aszarr(level=level)
            # print(dir(zarr_arr))
            # print(help(zarr_arr))

            if indexers is None:
                arr = s.asarray(level=level)
            else:
                raise NotImplementedError
                # Normalize indexers to full rank
                indexers = self._normalize_indexers(indexers, ndim=zarr_arr.ndim)
                arr = np.asarray(zarr_arr[indexers])

        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if squeeze:
            arr = np.squeeze(arr)
        if copy:
            arr = np.array(arr, copy=True)
        return arr

    # ---------------------------
    # Internals
    # ---------------------------
    @staticmethod
    def _get_yx_indices(axes: str) -> Tuple[int, int]:
        try:
            y_idx = axes.index("Y")
            x_idx = axes.index("X")
        except ValueError as e:
            raise ValueError(f"Could not locate Y/X axes in axis string '{axes}'.") from e
        return y_idx, x_idx

    @staticmethod
    def _normalize_indexers(indexers: Tuple[Union[int, slice, None], ...], ndim: int):
        if len(indexers) > ndim:
            raise IndexError(f"Too many indexers: got {len(indexers)}, array ndim={ndim}.")
        # pad with slice(None) for unspecified trailing axes
        padded = list(indexers) + [slice(None)] * (ndim - len(indexers))
        return tuple(padded)

    @staticmethod
    def _build_meta_summary(ome_xml: Optional[str]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if ome_xml is None:
            return summary
        if _HAVE_OME_TYPES:
            try:
                ome = ome_from_xml(ome_xml)
                summary["images"] = len(ome.images)
                if ome.images:
                    img = ome.images[0]
                    px = img.pixels
                    summary["axes"] = "".join(px.get_dimension_order().value) if px.dimension_order else None
                    summary["sizeT"] = px.size_t
                    summary["sizeC"] = px.size_c
                    summary["sizeZ"] = px.size_z
                    summary["sizeY"] = px.size_y
                    summary["sizeX"] = px.size_x
                    summary["type"] = px.type.value if px.type else None
                    # physical sizes (may be None)
                    summary["PhysicalSizeX"] = getattr(px, "physical_size_x", None)
                    summary["PhysicalSizeY"] = getattr(px, "physical_size_y", None)
                    summary["PhysicalSizeZ"] = getattr(px, "physical_size_z", None)
                    summary["PhysicalSizeXUnit"] = getattr(px, "physical_size_x_unit", None)
                    summary["PhysicalSizeYUnit"] = getattr(px, "physical_size_y_unit", None)
                    summary["PhysicalSizeZUnit"] = getattr(px, "physical_size_z_unit", None)
                    # channel names
                    summary["channels"] = [ch.name for ch in px.channels if ch is not None]
                    # acquisition timestamp
                    summary["acquisition_date"] = getattr(img, "acquisition_date", None)
            except Exception:
                # TODO: Fall through to raw XML if parsing fails
                # summary["ome_xml_raw_present"] = False # TODO: shouldn't this be invoked?
                pass
        # Even if parsed, keep the raw for reference
        summary["ome_xml_raw_present"] = True
        return summary
