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
    import zarr
    _HAVE_ZARR = True
except ImportError:
    zarr = None
    _HAVE_ZARR = False




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
        * `get_array` returns a numpy array; supports ultra-fast windowed slicing
          via named axis keyword arguments (`C=0, Z=slice(0, 16), Y=slice(100, 200), X=slice(100, 200)`),
          dictionary indexers, or positional tuples matching `self.axes`.
        * For large images, prefer windowed slicing to avoid loading unneeded tiles/planes.
    
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

        # 1. Grab a window at level 0 using named axis keyword arguments:
        win = reader.get_array(
            level=0,
            C=0,
            Z=slice(0, 16),
            Y=slice(100, 200),
            X=slice(100, 200),
        )

        # 2. Or using a dictionary with string slice notation:
        win = reader.get_array(
            level=0,
            indexers={'C': 0, 'Z': '0:16', 'Y': '100:200', 'X': '100:200'}
        )

        # 3. Or using positional indexers (matching axes, e.g. 'TCZYX'):
        win = reader.get_array(
            level=0,
            indexers=(0, 0, slice(0, 16), slice(100, 200), slice(100, 200)),
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
        indexers: Optional[Union[Tuple[Any, ...], Dict[str, Any], Sequence[Any], slice, int, str]] = None,
        dtype: Optional[np.dtype] = None,
        squeeze: bool = False,
        copy: bool = True,
        **axis_kwargs: Any,
    ) -> np.ndarray:
        """
        Load pixels from the specified pyramid `level` into a NumPy array.

        Supports ultra-fast windowed subregion reads via Zarr/TiffPageSeries without
        loading the entire image or unrequested tiles/planes into memory.

        Parameters
        ----------
        level : int
            Pyramid level to read (0 = full resolution). Supports negative indexing (e.g. -1 for lowest res).
        indexers : tuple, list, dict, slice, int, or str, optional
            Per-axis indexers used to window or pick specific T/C/Z/Y/X ranges.
            Can be provided in multiple formats:
              - By axis keyword arguments:
                  get_array(level=0, C=0, Z=slice(0, 16), Y=slice(100, 200), X=slice(100, 200))
                  get_array(level=0, C=0, Y="100:200", X="100:200")
              - By dictionary:
                  get_array(level=0, indexers={'C': 0, 'Z': slice(0, 16), 'Y': slice(100, 200), 'X': slice(100, 200)})
              - By positional tuple / sequence matching `self.axes`:
                  get_array(level=0, indexers=(0, slice(0, 16), slice(100, 200), slice(100, 200)))
                  get_array(level=0, indexers=(0, ..., slice(100, 200), slice(100, 200)))
                  get_array(level=0, indexers=(0, None, slice(100, 200), slice(100, 200)))
            If None (and no axis kwargs are provided), the entire level is read.
        dtype : np.dtype, optional
            If provided, cast the result to this dtype.
        squeeze : bool
            If True, squeeze singleton axes.
        copy : bool
            If True, ensure the returned array owns its memory.
        **axis_kwargs : int, slice, str, None
            Convenient axis-named slicing arguments (e.g. C=0, Z=slice(0, 16), Y=slice(100, 200), X=slice(100, 200)).
            Case-insensitive.

        Returns
        -------
        np.ndarray
            Loaded pixel data for the requested slice/window.
        """
        if level < 0: # handle negative indices e.g. -1 returns last level
            level = self.num_levels + level
            
        if not (0 <= level < self.num_levels):
            raise IndexError(f"Level {level} out of range (0..{self.num_levels-1}).")

        lev_info = self._levels[level]
        norm_indexers = self._normalize_indexers(indexers, axes=lev_info.axes, axis_kwargs=axis_kwargs)

        with tifffile.TiffFile(self._path) as tf:
            s = tf.series[self._series_index]

            # Fast path 1: Entire array requested -> s.asarray is fastest for bulk uncompressed/full plane reads
            is_full_slice = norm_indexers is None or all(
                isinstance(idx, slice) and idx.start is None and idx.stop is None and (idx.step is None or idx.step == 1)
                for idx in norm_indexers
            )

            if is_full_slice:
                arr = s.asarray(level=level)
            else:
                # Fast path 2: Windowed subregion read via Zarr chunk store
                if _HAVE_ZARR:
                    try:
                        z_store = s.aszarr(level=level)
                        z_arr = zarr.Array(z_store, read_only=True)
                        arr = np.asarray(z_arr[norm_indexers])
                    except Exception:
                        # Fallback if zarr extraction fails on unusual TIFF format
                        full_arr = s.asarray(level=level)
                        arr = np.asarray(full_arr[norm_indexers])
                else:
                    full_arr = s.asarray(level=level)
                    arr = np.asarray(full_arr[norm_indexers])

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
    def _normalize_indexers(
        indexers: Optional[Union[Dict[str, Any], Sequence[Any], slice, int, str]],
        axes: str,
        axis_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Union[int, slice], ...]]:
        """
        Normalize various indexing formats (tuples, slices, dicts, kwargs, string slices)
        into a concrete tuple of (int | slice) matching the length and order of `axes`.
        """
        ndim = len(axes)

        # 1. Handle dictionary or keyword-based named axes indexing
        combined_dict: Dict[str, Any] = {}
        if isinstance(indexers, dict):
            combined_dict.update(indexers)
        if axis_kwargs:
            if indexers is not None and not isinstance(indexers, dict):
                raise ValueError("Cannot specify both positional indexers and axis keyword arguments.")
            combined_dict.update(axis_kwargs)

        if combined_dict:
            axis_map = {ax.upper(): i for i, ax in enumerate(axes)}
            idx_list: List[Union[int, slice]] = [slice(None)] * ndim
            for k, v in combined_dict.items():
                k_up = k.upper()
                if k_up not in axis_map:
                    raise ValueError(f"Axis '{k}' not found in image axes '{axes}'.")
                parsed = _parse_slice_like(v)
                idx_list[axis_map[k_up]] = slice(None) if parsed is None else parsed
            return tuple(idx_list)

        if indexers is None:
            return None

        # 2. Handle scalar int, slice, or string slice passed directly
        if isinstance(indexers, (int, slice, str)):
            indexers = (indexers,)

        idx_list_raw = list(indexers)

        # 3. Handle Ellipsis (...)
        if Ellipsis in idx_list_raw:
            if idx_list_raw.count(Ellipsis) > 1:
                raise IndexError("An index can only have a single ellipsis ('...')")
            e_idx = idx_list_raw.index(Ellipsis)
            num_missing = ndim - (len(idx_list_raw) - 1)
            if num_missing < 0:
                raise IndexError(
                    f"Too many indexers: got {len(idx_list_raw) - 1} + ellipsis, expected at most {ndim}."
                )
            idx_list_raw = idx_list_raw[:e_idx] + [slice(None)] * num_missing + idx_list_raw[e_idx + 1:]

        if len(idx_list_raw) > ndim:
            raise IndexError(f"Too many indexers: got {len(idx_list_raw)}, expected at most {ndim} for axes '{axes}'.")

        # 4. Pad trailing unmentioned dimensions with slice(None)
        if len(idx_list_raw) < ndim:
            idx_list_raw.extend([slice(None)] * (ndim - len(idx_list_raw)))

        # 5. Parse strings like "0:100" and normalize None -> slice(None)
        normalized: List[Union[int, slice]] = []
        for elem in idx_list_raw:
            if elem is None:
                normalized.append(slice(None))
            else:
                parsed = _parse_slice_like(elem)
                normalized.append(slice(None) if parsed is None else parsed)

        return tuple(normalized)

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

def test_reader():
    # from SynAPSeg.IO.readers import PyramidOMEReader

    reader = PyramidOMEReader(r"E:\TEL_PROJECTS\examples\TEL15_s000_1-3\raw_image.ome.tiff")

    # tiny region (cell)
    t0=ug.dt()
    subregion = reader.get_array(
        level=0,
        C=slice(0, 1),
        Y=slice(2560, 2580),
        X=slice(2560, 2580),
    )
    print("\nShape of requested subregion:", subregion.shape, f"\ntook: {ug.dt()-t0}")

    t0=ug.dt()
    subregion = reader.get_array(
        level=0,
        C=slice(None),
        Y=slice(2500, 3000),
        X=slice(2500, 3000),
    )
    print("\nShape of requested subregion:", subregion.shape, f"\ntook: {ug.dt()-t0}")

    t0=ug.dt()
    subregion = reader.get_array(level=0)
    print("\nShape of requested subregion:", subregion.shape, f"\ntook: {ug.dt()-t0}")


import struct

def get_png_shape(filepath):
    """
    Reads only the header of a PNG file to determine its shape.
    Returns: (height, width, channels)
    """
    with open(filepath, 'rb') as f:
        # 1. Read and verify the 8-byte PNG signature
        if f.read(8) != b'\x89PNG\r\n\x1a\n':
            raise ValueError("Not a valid PNG file.")

        # 2. Skip the 4-byte chunk length (IHDR is always 13 bytes)
        f.read(4)

        # 3. Verify the chunk type is IHDR
        if f.read(4) != b'IHDR':
            raise ValueError("Missing or misplaced IHDR chunk.")

        # 4. Read the 13-byte IHDR chunk data
        ihdr = f.read(13)

        # 5. Unpack Width (4 bytes) and Height (4 bytes)
        # PNG uses Big-Endian byte order, which is why we use ">II"
        width, height = struct.unpack(">II", ihdr[0:8])
        
        # Byte index 9 in IHDR contains the Color Type
        color_type = ihdr[9]
        print(color_type)

        # 6. Determine the number of channels based on the PNG specification
        color_to_channels = {
            0: 1,  # Grayscale
            2: 3,  # RGB (Truecolor)
            3: 1,  # Indexed-color (Palette) - Stores 1 channel of indices
            4: 2,  # Grayscale + Alpha
            6: 4   # RGBA (Truecolor + Alpha)
        }
        
        channels = color_to_channels.get(color_type)
        if channels is None:
            raise ValueError(f"Unknown PNG color type: {color_type}")

        # Return in the standard machine-learning / math axis order
        return (height, width, channels)


