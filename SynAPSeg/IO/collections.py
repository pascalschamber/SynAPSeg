from __future__ import annotations
"""
Currently not implemented, but would provide a solution for ensuring consistent shaped array formats

ArraySet: A lightweight container for named numpy arrays with known axis formats.

This module provides two primary components:

    * ArrayEntry -- a dataclass holding an array along with:
        - `format`: a string describing axis order (e.g., "STCZYX" or "ZYX")
        - `shape`: automatically inferred from the array data

    * ArraySet -- a read-only mapping (dict-like) from string keys to arrays,
      with convenient access to metadata such as shapes and axis format strings.

The ArraySet is designed for workflows where multiple arrays representing
related data (e.g., raw images, masks, label maps, feature volumes) need to be
handled together while preserving structured knowledge about their axis layout.

Typical construction is via `ArraySet.from_dict()`, which accepts a dictionary
where each entry has the form:

    {
      "<name>": {
         "format": "<axis string>",
         "data":   <numpy ndarray>
      },
      ...
    }

Example
-------
>>> import numpy as np
>>> img_dict = {
...     "raw_img": {"format": "STCZYX", "data": np.random.rand(1,1,3,10,256,256)},
...     "mask":    {"format": "ZYX",    "data": np.random.randint(0,2,(10,256,256))}
... }
>>> img_dict = ArraySet.from_dict(img_dict)

# Access data directly like a dictionary:
>>> arrays["raw_img"].shape
(1, 1, 3, 10, 256, 256)

# Query metadata:
>>> arrays.format_of("raw_img")
'STCZYX'
>>> arrays.shape_of("mask")
(10, 256, 256)

# Get a helpful overview:
>>> print(arrays.summary())
raw_img: shape=(1, 1, 3, 10, 256, 256) format='STCZYX' dtype=float64
mask:    shape=(10, 256, 256)          format='ZYX'    dtype=int64

Notes
-----
- The ArraySet implements the Mapping interface and behaves like a read-only
  dictionary. Iteration yields keys; indexing returns the underlying array.
- The `format` string is not enforced beyond an optional axis-length check,
  but can be used downstream to guide reshaping, dimension labeling, or
  visualization logic.
"""

from dataclasses import dataclass, field, InitVar
from typing import Callable, Dict, Iterator, Mapping, Optional, Tuple, Union, Any
import numpy as np
import tifffile
# TODO connect to image parser class for lazy loading of different image types


# --------------------------- per-entry metadata ------------------------------

ReaderFn = Callable[[str], np.ndarray]

def _default_tiff_reader(path: str) -> np.ndarray:
    if tifffile is None:
        raise ImportError("tifffile is required to read .tif files lazily. `pip install tifffile`.")
    return tifffile.imread(path)

def _probe_tiff_shape(path: str) -> Optional[Tuple[int, ...]]:
    """Try to read array shape from TIFF header without loading image data."""
    if tifffile is None:
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            # Prefer first series; works for OME-TIFF and many multi-page TIFFs
            s = tf.series[0]
            return tuple(s.shape)
    except Exception:
        return None

from SynAPSeg.utils import utils_general as ug

@dataclass
class ArrayEntry:
    """
    Holds one array's metadata and a lazy data loader.

    Construct with either:
      - eager: data=<np.ndarray>, format='STCZYX'
      - lazy:  path=<str>,        format='STCZYX', reader=<callable?>
    
    Example usage:
        image_path = r"J:\SEGMENTATION_DATASETS\2026_1007_AMI_cfos_timecourse_syt2\examples\0002\raw_img.tiff"
        fmt = 'CZYX'
        arr = ArrayEntry(fmt, path=image_path)
        arr(c=0) -> returns channel 0 of array
    """
    format: str
    # Accept user-facing `data` during init, but store privately as `_data`
    data: InitVar[Optional[np.ndarray]] = None
    path: Optional[str] = None
    reader: Optional[ReaderFn] = None

    # Internal storage/caches
    _data: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _shape: Optional[Tuple[int, ...]] = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
    
    
    def __post_init__(self, data: Optional[np.ndarray]):
        
        if isinstance(data, property): # i.e. not passed with data
            pass
        
        elif data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("'data' must be a numpy ndarray when provided.")
            self._data = data
            self._shape = data.shape
            self._loaded = True

    # ---- public API ----
    @property
    def data(self) -> np.ndarray:
        """Return the ndarray, loading it on first access if created from a path."""
        if self._data is not None:
            return self._data
        if self.path is None:
            raise ValueError("No data or path set for this ArrayEntry.")
        rdr = self.reader or _default_tiff_reader
        arr = rdr(self.path)
        self._data = arr
        self._shape = arr.shape
        self._loaded = True
        return arr

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array; tries cheap TIFF probe before loading."""
        if self._shape is not None:
            return self._shape
        if self.path:
            probed = _probe_tiff_shape(self.path)
            if probed is not None:
                self._shape = probed
                return probed
            # fallback: load to know shape
            return self.data.shape
        raise ValueError("ArrayEntry has neither data nor path.")
    
    def morph(self, target_format:str, target_shape:Optional[tuple]=None):
        """ call uip morph_to_target_format or shape on self.data """
        from SynAPSeg.utils import utils_image_processing as uip
        
        if target_shape is None:
            return uip.morph_to_target_format(self.data, self.format, target_format)
        else:
            return uip.morph_to_target_shape(self.data, self.format, target_shape, target_format)
    
    # ---- Indexing & Subscripting API ----

    def _slice_by_dims(self, dim_map: Dict[str, Any]) -> np.ndarray:
        """Build a tuple of slices matching self.format and index self.data."""
        fmt_upper = self.format.upper()
        index = [slice(None)] * len(fmt_upper)

        for dim, val in dim_map.items():
            dim_upper = dim.strip().upper()
            if dim_upper not in fmt_upper:
                raise KeyError(
                    f"Dimension '{dim}' not found in format '{self.format}'."
                )
            axis = fmt_upper.index(dim_upper)
            index[axis] = val

        return self.data[tuple(index)]

    @staticmethod
    def _parse_slice_str(val_str: str) -> Union[int, slice]:
        """Parse strings like '0', '1:5', or '::2' into int or slice objects."""
        val_str = val_str.strip()
        if ":" in val_str:
            parts = [int(p.strip()) if p.strip() else None for p in val_str.split(":")]
            return slice(*parts)
        return int(val_str)

    def __getitem__(self, key: Union[str, Dict[str, Any], Any]) -> np.ndarray:
        """
        Supports:
          1. String expressions: arr['c=0'] or arr['c=0, t=1:5']
          2. Dict mapping:       arr[{'c': 0, 't': slice(1, 5)}]
          3. Standard numpy:     arr[0, :, 1]
        """
        if isinstance(key, str) and "=" in key:
            dim_map = {}
            for part in key.split(","):
                k, v = part.split("=", 1)
                dim_map[k.strip()] = self._parse_slice_str(v)
            return self._slice_by_dims(dim_map)

        if isinstance(key, dict):
            return self._slice_by_dims(key)

        # Fallback: standard positional NumPy indexing (e.g., arr[0, :, 2])
        return self.data[key]

    def __call__(self, **kwargs: Union[int, slice]) -> np.ndarray:
        """
        Supports direct keyword syntax via parentheses:
          arr(c=0) or arr(c=0, z=slice(2, 8))
        """
        if not kwargs:
            return self.data
        return self._slice_by_dims(kwargs)
    
        


# --- container ---------------------------------------------------------------
class ArraySet(Mapping[str, np.ndarray]):
    """
    Container for named arrays with known format strings.
    Behaves like a read-only dict[str, np.ndarray] when indexed.
    Use .shape_of(key) and .format_of(key) to query metadata.

    Expected input to `from_dict`:
        {
          "raw_img": {"format": "STCZYX", "data": np.ndarray(...)},
          "mask":    {"format": "ZYX",    "data": np.ndarray(...)},
          ...
        }
    """

    def __init__(self, entries: Dict[str, ArrayEntry]):
        self._entries: Dict[str, ArrayEntry] = dict(entries)

        # ---- construction ----
    @classmethod
    def from_dict(cls, d: Dict[str, dict], *, validate: bool = True) -> "ArraySet":
        """
        Build from a plain dictionary where each item is like:
            name -> {
                "format": "STCZYX",
                "data":   np.ndarray OR "path": "/path/to/file.tif",
                # optional:
                "reader": callable(path) -> np.ndarray
            }
        Accepts aliases: "fmt"/"axes" for format, and "array"/"ndarray" for data.
        """
        entries: Dict[str, ArrayEntry] = {}
        for name, item in d.items():
            fmt = item.get("format")
            arr: Optional[np.ndarray] = item.get("data")
            path: Optional[str] = item.get("path")
            reader: Optional[ReaderFn] = item.get("reader")

            if fmt is None:
                raise ValueError(f"Item '{name}' must include a 'format' (e.g., 'STCZYX').")
            if arr is None and path is None:
                raise ValueError(f"Item '{name}' must include either 'data' (ndarray) or 'path' (str).")
            if arr is not None and not isinstance(arr, np.ndarray):
                raise TypeError(f"Item '{name}': 'data' must be a numpy ndarray when provided.")
            if path is not None and not isinstance(path, str):
                raise TypeError(f"Item '{name}': 'path' must be a string when provided.")

            entry = ArrayEntry(format=str(fmt), data=arr, path=path, reader=reader)

            if validate:
                _maybe_validate_entry(entry, name)

            entries[name] = entry
        return cls(entries)

    # ---- Mapping interface (read-only) ----
    def __getitem__(self, key: str) -> np.ndarray:
        return self._entries[key].data

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
    
    def __getattr__(self, key: str) -> ArrayEntry:
        """ implements support for querying attrs using syntax like self.raw_img.format → 'STCZYX' """
        if key in self._entries:
            return self._entries[key]
        raise AttributeError(f"'ArraySet' object has no attribute '{key}'")

    # ---- convenience metadata queries ----
    def shape_of(self, key: str) -> Tuple[int, ...]:
        """Return the array shape for a given key."""
        return self._entries[key].shape

    def format_of(self, key: str) -> str:
        """Return the declared format string (e.g., 'STCZYX') for a given key."""
        return self._entries[key].format

    def entry(self, key: str) -> ArrayEntry:
        """Return the full ArrayEntry (array + metadata)."""
        return self._entries[key]

    # optional: quick overview
    def summary(self) -> str:
        """Human-readable one-line summary per item."""
        lines = []
        for name, e in self._entries.items():
            shp = e._shape if e._shape is not None else "<lazy>"
            dtype = getattr(e.__dict__.get("data"), "dtype", "<lazy>")
            lines.append(f"{name}: shape={shp} format='{e.format}' dtype={dtype}")
        return "\n".join(lines)
    
    def __str__(self):
        return self.summary()
    
    def __repr__(self):
        return str(self)
    
    # ---- autocompletion helpers ----
    def __dir__(self):
        """Include data keys in attribute completion (arrays.<TAB>)."""
        # start with the default attributes
        base = set(super().__dir__())
        # add dynamic keys (so e.g. arrays.raw_img appears in completion)
        return sorted(base | set(self._entries.keys()))

    def _ipython_key_completions_(self):
        """Enable key completions for arrays['<TAB>'] in IPython/Jupyter."""
        return list(self._entries.keys())
    

# --------------------------- optional validation -----------------------------

_ALLOWED_AXES = set("STCZYX")  # extend as needed

def _maybe_validate_entry(entry: ArrayEntry, name: str) -> None:
    axes = entry.format
    unknown = set(axes) - _ALLOWED_AXES
    if unknown:
        raise ValueError(f"Item '{name}': unknown axis letters in format '{axes}': {unknown}")
    # If we already know shape (either eager or probed), check ndim vs format length.
    try:
        ndim = len(entry.shape)  # triggers cheap probe or load if needed
    except Exception:
        # Could not determine shape now; skip check until first load.
        return
    if len(axes) != ndim:
        raise ValueError(
            f"Item '{name}': format '{axes}' (len={len(axes)}) does not match "
            f"ndim={ndim} for shape {entry.shape}"
        )


