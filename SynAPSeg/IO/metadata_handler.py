"""
This module provides an interface to project / example metadata

"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List, TYPE_CHECKING, MutableMapping
from copy import deepcopy
import os
import numpy as np
from pathlib import Path
import pandas as pd
import re
import logging
from collections.abc import Mapping
import sys
import tifffile
import yaml
from tabulate import tabulate

from SynAPSeg.utils import utils_general as ug

if TYPE_CHECKING:
    from SynAPSeg.IO.project import Example


def handle_example_index(GET_EXS, where_map, exclude_exi_strs=[]):
    """ parse user input to get examples that should be processed. GET_EXS must be one of ['complete', 'all', or a list of examples list(int or str)]"""
    if GET_EXS == 'complete':
        example_filepath_list = [Path(p).name for p in where_map['complete']]
    elif GET_EXS == 'all':
        example_filepath_list = ug.flatten_list(list(where_map.values()))
    elif isinstance(GET_EXS, list):
        example_filepath_list = [str(el).zfill(4) for el in GET_EXS]
    else:
        raise ValueError(GET_EXS)
    
    # remove specified exs
    example_filepath_list = [el for el in example_filepath_list if el not in exclude_exi_strs]
    
    # check found
    assert len(example_filepath_list) != 0, print(f"warning no exes found")
    return example_filepath_list


# new impl of class for this

class MetadataParser:
    """
    MetadataParser: Encapsulates all logic for parsing and validating metadata
    for segmentation examples, including YAML loading and image file shape validation.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def read_yaml(file_path):
        with open(file_path) as file:
            val = yaml.load(file, Loader=yaml.CLoader)
        return val

    @staticmethod
    def try_get_metadata(path_to_example, FILE_MAP=None, silent=False):
        """
        Attempt to load annotation metadata and initialize it if not present.

        Args:
            path_to_example (str): Path to example folder.
            FILE_MAP (dict): Mapping of metadata types to filenames.
            silent (bool): Suppress output.

        Returns:
            dict: The example metadata dictionary.
        """
        if FILE_MAP is None:
            FILE_MAP = {'metadata': ['metadata.yml']}

        exmd = {}
        metadata_path = os.path.join(path_to_example, FILE_MAP['metadata'][0])
        if 'metadata' in FILE_MAP and os.path.exists(metadata_path):
            exmd = MetadataParser.read_yaml(metadata_path)
        return exmd
    
    @staticmethod
    def read_example(
            example_dir,
            load_files=None,
            silent=False,
            validate_shapes_and_format=False,
            exmd=None,
            use_prefix_as_key=True,
        ):
        """
        Loads files from an example directory given list of file names.

        Args:
            example_dir (str): Path to the example folder.
            load_files (list or str): Files to load.
            silent (bool): If True, suppress prints.
            validate_shapes_and_format (bool): Whether to check shapes/format.
            exmd (dict): Example metadata dictionary, required for validation.
            use_prefix_as_key (bool): Whether to use the prefix of the file name as the key.
                # TODO: this currently exists for compatibility with old code, but should be removed
                # need to update all code that calls this function to use the full file name as the key

        Returns:
            dict: Dictionary of loaded data.
        """
        if not load_files: # default try to load these files
            load_files = [
                'metadata.yml',
            ]

        if not os.path.exists(example_dir):
            raise ValueError(f"{example_dir} does not exist")

        if not silent:
            print('Reading example...')


        out = {}
        return_one_flag = False
        if isinstance(load_files, str):
            load_files = [load_files]
            return_one_flag = True

        for fn in load_files:
            try:
                file_path = os.path.join(example_dir, fn)
                if not os.path.exists(file_path):
                    if not silent:
                        print(f"WARNING --> read_example did not find {fn} in {example_dir}")
                    continue
                elif fn.lower().endswith(('ome.tiff', 'ome.tif')):
                    # TODO handle loading image pyramids into napari
                    val = tifffile.imread(file_path)
                elif fn.lower().endswith(('.tiff', '.tif')):
                    val = tifffile.imread(file_path)
                elif fn.lower().endswith(('.yml', '.yaml')):
                    val = MetadataParser.read_yaml(file_path)
                elif fn.lower().endswith('.geojson'):
                    from SynAPSeg.Plugins.ABBA import core_regionPoly
                    val = core_regionPoly.polyCollection(file_path)
                else:
                    raise ValueError(f"Cannot parse file {fn}")
                outkey = ug.get_prefix(fn) if use_prefix_as_key else fn
                out[outkey] = val

            except Exception as e:
                raise IOError(f"error in read_example, fn:{fn}\nfull path:{file_path}\nerror:\n{e}")

        if not silent:
            print('Loaded:')
            for k, v in out.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: {v.shape}")

        if validate_shapes_and_format:
            if exmd is None:
                print("Need to provide exmd if validating shapes and format")
            else:
                MetadataParser.check_metadata_shapes_and_format(out, exmd)

        if return_one_flag:
            return out[fn[:fn.rindex(".")]]
        return out
    
    @staticmethod
    def check_metadata_shapes_and_format(image_dict, exmd):
        """
        Run checks to ensure data shapes and formats are specified in exmd.

        Args:
            image_dict (dict): Dictionary of loaded arrays/images.
            exmd (dict): Example metadata.
        """
        potential_errors = []
        for k, v in image_dict.items():
            if k == 'metadata':
                continue

            shape = list(v.shape)

            # Auto-set shape if not already defined
            if exmd['data_metadata'].get('data_shapes', {}).get(k) is None:
                exmd['data_metadata']['data_shapes'][k] = shape

            for check_key in ['data_formats', 'data_shapes']:
                dfd = exmd['data_metadata'].get(check_key, {})
                dff = dfd.get(k, '-_-')
                
                # TODO inserting check to look for prefix for backwards compatibility
                if dff == '-_-':
                    dff = dfd.get(ug.get_prefix(k), '-_-')

                print(f"{k} {v.shape} {dff}")

                pe = ''
                if dff == '-_-':
                    pe = (f"Potential error: {check_key} is not specified for {k}. "
                            "Add it to exmd['data_metadata'] to load into annotator.")

                if check_key == 'data_shapes':
                    if list(exmd['data_metadata']['data_shapes'][k]) != shape:
                        pe = (f"Potential error: {check_key} shape mismatch for {k}. "
                                f"Expected {exmd['data_metadata']['data_shapes'][k]}, got {shape}.")

                if pe:
                    potential_errors.append(f"{k}: {pe}")

        if potential_errors:
            print("\nIn check_metadata_shapes_and_format, potential errors:")
            for e in potential_errors:
                print(f"  - {e}")
    
    @staticmethod
    def get_layers_by_ch(get_chs, annotated_files, layer_base_name, ALLOW_RAW_PREDS=True):
        """get only specified get_chs and extract the raw pred"""
        get_label_names = []
        for i in get_chs:
            if f'annotated_{layer_base_name}_ch{i}.tiff' in annotated_files:
                get_label_names.append(f'annotated_{layer_base_name}_ch{i}.tiff')
            elif ALLOW_RAW_PREDS and f'{layer_base_name}_ch{i}.tiff' in annotated_files:
                get_label_names.append(f'{layer_base_name}_ch{i}.tiff')

        return get_label_names #+ [f'pred_stardist_ch{i}.tiff' for i in get_chs]

    @staticmethod
    def get_imgdb_colocal_nuclei_info(exmd):
        """use provided channel info and imgdb object to setup colocalization parameters (see utils_ImgDB.py) """
        # add all channels as clcs
        image_channels = [dict(name=v, ch_idx=i, colocal_id=i) for i, (k,v) in enumerate(exmd['data_metadata']['channel_info'].items())]
        # add colocalizations to perform based on inputed channel mapping 
        ch_info = list(exmd['data_metadata']['channel_info'].values())
        current_max_clc_id = max([d['colocal_id'] for d in image_channels])
        clc_nuc_info = []
        if exmd.get('COLOCALIZE_PARAMS', {}).get('colocalizations'):
            for cp_i, cp in enumerate(exmd['COLOCALIZE_PARAMS']['colocalizations']):
                clc_nuc_info.append({
                    'name': '+'.join([str(ch_info[i]) for i in cp]),
                    'ch_idx': [i for i in cp], 'co_ids': [i for i in cp], 
                    'colocal_id': current_max_clc_id+1
                })
                current_max_clc_id+=1
        return image_channels, clc_nuc_info


    @staticmethod
    def is_ex_dir(path_to_example):
        """check if path is an example directory by looking if it contains metadata.yml file"""
        return Example.is_ex_dir(path_to_example)

    @staticmethod
    def get_example_dirs(root_dir):
        """get paths to example directories given a root directory"""
        example_dirs = [el for el in ug.get_contents(root_dir) if MetadataParser.is_ex_dir(el)]
        return example_dirs
    
    @staticmethod
    def get_ch_from_str(layer_name, as_int=False):
        """
        Helper function to extract '_chX' from a string, only if it's at the end.
        
        Args:
            layer_name (str): The input string to search.
            as_int (bool): Whether to return the channel as an integer.

        Returns:
            str or int or None: The matched channel string or integer, or None if not found.
        """
        extract_pattern = r'_ch\d+$' if not as_int else r'_ch(\d+)$'
        match = re.search(extract_pattern, layer_name)
        if not match:
            return None
        ch_x = match.group(1) if as_int else match.group(0)
        return int(ch_x) if as_int else ch_x

    @staticmethod
    def write_metadata(path_to_example:str, exmd:dict, alt_path:Optional[str]=None) -> None:
        """
        Write/Overwrite this example’s metadata.yml with exmd

        args:
            alt_path (str): if provided write to this path instead of ex dir 
                must be full path; ../subdir/metadata.yml
                this exists to allow writing file with name other than metadata.yml

        """
        def _writer(path, exmd):
            # add datetime stamp
            exmd['__last_modified__'] = ug.get_datetime()
            with open(path, 'w') as f:
                yaml.dump(exmd, f, default_flow_style=False)

        
        try:
            if alt_path is not None:
                outpath = alt_path
            else:
                assert os.path.exists(path_to_example), f"path to example does not exist"
                outpath = os.path.join(path_to_example, "metadata.yml")
            
            _writer(outpath, exmd)
            
        except Exception as e:
            print(f'failed to write metadata. error:\n{e}')
    
    @staticmethod
    def maybe_merge_exmd(path_to_example, ex_md, uuids=None):
        """
            handle archival and merging if example metadata already exists 
                e.g. generated preds with a different model, but want to preserve all the existing data
                could also be re-run with more/different data so need to determine if safe to merge metadata
                    assume if source image path is same & any other unique identifiers (like scene_name), then it is safe
                    but to be safe, create a backup anyways
            args:
                path_to_example: str
                exmd: dict
                uuids: default ['image_path', 'scene_id', 'scene_name']
            returns:
                input ex_md if no current metadata.yml
                or modified exmd where keys,values from new have been non-destructivly merged 
                creates a backup of existing metadata if it exists
        """
        if not os.path.exists(os.path.join(path_to_example, "metadata.yml")):
            return ex_md
        
        from copy import deepcopy
        prev_md = MetadataParser.try_get_metadata(path_to_example)

        # by default archive prev version
        MetadataParser.archive_metadata(path_to_example)

        # if dicts are identical can skip below stuff
        merged_md = deepcopy(prev_md)
        if prev_md != ex_md: # first check we're in the right ex dir - all img data should have been overwritten already but old files not in this run would still be present.    
            
            # santity check exmds come from same source data
            mismatch = MetadataParser.check_key_mismatch(prev_md, ex_md, uuids=uuids)
            # if only new keys being added can update
            if mismatch: # not same_source:
                print(mismatch) # warn

            # # check if just adding # TODO 
            # if not_just_adding:
            #     print () # warn - destructive, but we have backup

            # update other vals 
            merged_md = deepcopy(prev_md)
            MetadataParser.deep_merge_metadata(merged_md, ex_md)

        # return merged 
        return merged_md
    
    @staticmethod
    def archive_metadata(path_to_example):
        """ copy current `metadata.yml` file to `previous_metadata` folder, creating if needed """
        
        exmd_hist_dir = ug.verify_outputdir(os.path.join(path_to_example, '__archived__'))
        prev_exmd = MetadataParser.try_get_metadata(path_to_example)
        ts = ug.get_datetime()
        prev_exmd = {"__archived__": ts, **prev_exmd}
        aoutpath = os.path.join(exmd_hist_dir, f"replaced_{ts}_metadata.yml")
        MetadataParser.write_metadata('', prev_exmd, aoutpath)
    
    @staticmethod
    def check_key_mismatch(prev_md, exmd, uuids:Optional[list[str]]=None):
        """check if specific keys match between two dicts
            used to confirm data source by checking things like image paths & other unique identifiers match between two exmd's
            args:
                uuids, optional. default uses ['image_path', 'scene_id', 'scene_name']
            returns None if all match, else returns error message
        """
        uuids = uuids or ['image_path', 'scene_id', 'scene_name'] # 'image_metadata' dict should also matchup
        old_vals, new_vals = [[d[uk] for uk in uuids] for d in [prev_md, exmd]]
        checks = [v1==v2 for v1, v2 in zip(old_vals, new_vals)]
        if not all(checks):
            emsg = (
                'metadata overwrite warning: origin-defining identifiers do not align. This could indicate' 
                'overwriting example data that came from a differnce source image (e.g. example ids are not aligned?)\n'
                f"uuids:{uuids}, checks: {checks}\n"
                "(old_vals, new_vals): {(old_vals, new_vals)}\n"
            )
            return emsg
        return None
    
    @staticmethod
    def deep_merge_metadata(
        base_d: MutableMapping[str, Any], # e.g. 
        new_d: Mapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """
        Deep-merge `incoming` into `base` *in place*.

        - Keeps all keys from `base`.
        - Adds any new keys from `incoming`.
        - If a key exists in both and both values are mappings, merge recursively.
        - Otherwise (leaf-level or type mismatch), `incoming` value overwrites `base`.

        Returns
        -------
        base : the same object, updated.
        """
        for key, new_val in new_d.items():
            old_val = base_d.get(key, None)

            if isinstance(old_val, Mapping) and isinstance(new_val, Mapping):
                # Recurse into nested dicts
                MetadataParser.deep_merge_metadata(old_val, new_val)
            else:
                # Leaf-level (or different types): new value wins
                base_d[key] = deepcopy(new_val)

        return base_d
        
    
    @staticmethod
    def write_data_to_example(ex:Example, data, filename, fmt, tiff_metadata=None, OUTPUT_IMAGE_PYRAMID=False) -> None:
        success=False
        if isinstance(data, np.ndarray):
            from IO.writers import write_array
            write_array(data, ex.path_to_example, filename, fmt, ex.exmd, tiff_metadata=tiff_metadata, OUTPUT_IMAGE_PYRAMID=OUTPUT_IMAGE_PYRAMID)
            success = True
        else:
            raise NotImplementedError
        if success:
            ex.write_metadata()
            

    @staticmethod
    def metadata_list_to_df(example_paths) -> pd.DataFrame:    
        """Given a list of paths to examples, load the metadata files and convert them to a df representation"""
        metadatas = MetadataParser.get_metadata_from_paths(example_paths)
        mdf = MetadataParser.list_of_dicts_to_dataframe(metadatas)
        
        # Check for duplicated filenames
        duplicate_fns = mdf.value_counts('image_fn').reset_index()
        dup_dict = dict(zip(duplicate_fns['image_fn'].to_list(), duplicate_fns['count'].to_list()))
        mdf['duplicates'] = mdf['image_fn'].map(dup_dict)
        return mdf

    @staticmethod
    def get_metadata_from_paths(example_paths) -> list[Dict]:
        """Load metadata from a list of example paths"""
        metadatas = []
        for p in example_paths:
            exmd = MetadataParser.try_get_metadata(p, {'metadata': ['metadata.yml']}, silent=True)
            exmd['image_fn'] = Path(exmd['image_path']).name
            sorted_dict = {key: exmd[key] for key in sorted(exmd)}
            metadatas.append(sorted_dict)
        return metadatas

    @staticmethod
    def list_of_dicts_to_dataframe(list_of_dicts) -> pd.DataFrame:
        """Convert a list of dictionaries to a pandas DataFrame"""
        flat_dicts = [ug.flatten_dict(d) for d in list_of_dicts]
        return pd.DataFrame(flat_dicts)

    @staticmethod
    def match_exmd_img_path(exmds, astr, sort_key='image_fn') -> list[Dict]:
        """Given an image filename, search a list of metadata dicts for the ones that contains it"""
        res = [md for md in exmds if re.match(astr, Path(md['image_path']).stem)]
        assert len(res) > 0, f'no matching paths found for str: {astr}'
        return sorted(res, key=lambda x: x[sort_key])
    
    @staticmethod
    def get_logger(
        name: str, 
        log_dir: Optional[str] = None, 
        log_filename: Optional[str] = None, 
        level='INFO'
    ) -> logging.Logger:
        """ setup a logger for a project """
        from common.Logging import get_logger
        return get_logger(
            name = name, 
            log_dir = log_dir, 
            log_filename = log_filename, 
            level=getattr(logging, level)
        )
    
    
# to incorporate into a example_metadata class with fixed querying attrs
# TODO
# also fix exmd key 'image_path' --> 'source_path'




# --- funcs for matching channel info ---

CH_INFO_OPTION_RE = re.compile(r"\s*(.*?)\s*\(\s*(.*?)\s*\)\s*")  # captures "display (pattern)"

def process_channel_info(channel_info: Dict[Any, str], filepath: str, strict=False, warn=True) -> dict:
    """
    The function searches `filepath` for the first matching pattern for each value in channel_info and returns that option's Display str. 
    If no patterns match and strict=False the last option is used. Otherwise, raises ValueError.
    Used in segmentation pipeline
    
    Args:
        channel_info:
            For each wavelength key, allow multiple choices in the form:
                "Display1 (pattern1) | Display2 (pattern2) | ..."
        filepath:
            a string representing a full path
        strict:
            if true raise value error if match fails, if false default to last option's display name 
        warn: 
            if true print warning message if fails 

    Example:
        channel_info = {
            405: "IgG3-GAD67 (IgG3-GAD67) | gp-VGlut1 (gp-VGlut1)",
            488: "PSD95 (IgG2A-PSD95) | IgG2A-GAD67 (IgG2A-GAD67)",
            568: "AAV-mCh-Cre (AAV-mCh-Cre) | AAV-mCh (AAV-mCh)",
            647: "gp-Vglut1 (gp-VGlut1-647) | SAP102 (rab-SAP102) | rab-TARPy2 (rab-TARPy2) | rab-GluA4 (rab-GluA4)",
        }
        filepath = r"D:\data\imgs\8.19.25 Glyoxal 8.4.div14 AAV-mCh rab-GluA4_647 IgG2A-GAD67_488 gp-VGlut1_405 1.1.czi"
        print(process_channel_info(channel_info, filepath))
        -> {405: 'gp-VGlut1', 488: 'IgG2A-GAD67', 568: 'AAV-mCh', 647: 'rab-GluA4'}
    """
    processed = {}

    for wl, name_spec in channel_info.items():
        if isinstance(name_spec, str) and _is_matching_expression(name_spec):
            options = _parse_options(name_spec)
            chosen = _match_option(options, filepath, strict=strict, warn=warn)
            processed[wl] = chosen if chosen is not None else name_spec
        else:
            # no special syntax, keep as-is
            processed[wl] = name_spec
    return processed

def _is_matching_expression(s: str) -> bool:
    """Return True if the string ends with a '| (...)' group."""
    return bool(re.search(r".*\|.*\(.+?\)", s))

def _parse_options(name_spec: str) -> List[Tuple[str, str]]:
    """
    Parse a spec like:
      "A (patA) | B (patB) | C (patC)"
    into a list of (display, pattern) in order.
    """
    parts = [p.strip() for p in name_spec.split("|")]
    options = []
    for p in parts:
        m = CH_INFO_OPTION_RE.fullmatch(p)
        if m:
            display, pattern = [el.strip() for el in m.groups()]
            options.append((display or pattern, pattern)) # e.g. if "(IgG3-GAD67) | (gp-VGlut1)" options would be [(IgG3-GAD67, IgG3-GAD67), (gp-VGlut1, gp-VGlut1)]
        else:
            # coerce plain name without (pattern) to use plain name as pattern
            options.append((p.strip(), p.strip()))  # empty pattern never matches
    return options

def _match_option(options: List[Tuple[str, str]], filename: str, strict=False, warn=True):
    """
    Return the display name of the first option whose pattern is found in filename (case-insensitive). 
        If none match, return the last option's display name.
    """
    # Search using literal substring semantics (escape pattern). Case-insensitive.
    for display, pattern in options:
        if pattern and re.search(re.escape(pattern), filename, flags=re.IGNORECASE):
            return display
    
    # handle no match found
    emsg = f"could not match options (display, pattern) to filename\n\toptions: {options}\n\tfilename: {filename}"
    if strict:
        raise ValueError(emsg)
    
    # Fallback: last option
    if warn: print(f'Warning! --> {emsg}')
    return options[-1][0] if options else None






class GroupExtractor:
    """
    handles a list of GroupParsers as input, they parse categorical variables to assign to each example's data 
        by matching regex pattern to metadata attributes such as image_path 
        (e.g. {'column_name':{'pattern1':'group1', 'pattern2':'group2'}})
    """
    def __init__(self, group_parsers):
        self.group_parsers = group_parsers
        
    def extract_groups(self, exmd):
        _extracted_groups = {}
        for gp in self.group_parsers:
            _extracted_groups[gp.alias] = gp.extract_group(exmd)
        return _extracted_groups
    
    def get_groupping_vars(self):
        return [gp.alias for gp in self.group_parsers]
    
    def __str__(self):
        return ", ".join([str(gp) for gp in self.group_parsers])
            
        
class GroupParser:
    """
    parse categorical variables from example's metadata by matching regex pattern to metadata attribute
        used to assign categories to example's data for analysis
        must be in form {key_in_exmd: {**pattern_group_mapping}}
        e.g. {'image_path':{'pattern1':'group1', 'pattern2':'group2'}}
    """
    def __init__(self, alias, exmd_attr, mapping, error_on_fail=True):
        self.alias = alias
        self.exmd_attr = exmd_attr
        self.mapping = mapping
        self.error_on_fail = error_on_fail
        
    def __str__(self):
        return f"(alias:{self.alias}, exmd_attr:{self.exmd_attr}, mapping:{self.mapping})"
    
    def extract_group(self, exmd, exmd_attr=None):
        exmd_attr = exmd_attr if exmd_attr is not None else self.exmd_attr
        
        if exmd_attr is None:
            raise ValueError(f"attribute cannot be None.")
        if exmd_attr not in exmd:
            raise ValueError(f"exmd_attr \'{exmd_attr}\' not in exmd.\n\texmd keys: {exmd.keys()}")
    
        return self.pattern_match(exmd[exmd_attr], self.mapping, error_on_fail=self.error_on_fail)
    
    def pattern_match(self, string, mapping, error_on_fail=True):
        """
        Extract the group from a string based on a regex search that maps text to group names.

        Parameters:
        - string: The filename from which to extract the group name.
        - mapping: A dictionary where keys are regex patterns and values are group names.

        Returns:
        - group_name: The name of the group if a match is found; otherwise, None.
        """
    
        for pattern, group_name in mapping.items():
            matched = re.search(pattern, string)
            if matched:
                if group_name is None:
                    # return the matched group instead of the group name
                    return matched.group().strip()
                return group_name
        if error_on_fail:
            raise ValueError(f"could not parse group from str: {string}\npatterns: {mapping.keys()}")
        return None


