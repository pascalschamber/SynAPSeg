from __future__ import annotations
import os
from pathlib import Path
import re
import importlib
from timeit import default_timer as timer
from datetime import datetime
from collections.abc import Iterable as IterableBaseClass
from typing import List, Pattern, Union, Dict, Optional, Iterable
import shutil
import fnmatch
from copy import deepcopy



def verify_outputdir(adir, makedirs=False):
    """make an outputdir if it doesn't exist"""
    if not os.path.exists(adir):
        os.mkdir(adir) if not makedirs else os.makedirs(adir)
    return adir

def verify_paths_exist(listOfPaths):
    """check if each path in a list exists"""
    doNotExist = []
    for p in listOfPaths: 
        if not os.path.exists(p):
            doNotExist.append(p)
    if len(doNotExist) > 0:
        indent_str = '\n\t-'
        raise IndexError(f"{len(doNotExist)} / {len(listOfPaths)} paths do not exist:{indent_str}{indent_str.join(doNotExist)}")
    else:
        return True

def get_prefix(
        s:str, 
        exceptions:Optional[List[str]] = ['.ome', '.OME']
    ) -> str:
    """
        Find the last occurrence of '.' returning string up to that point or returns input if not found
            used to get everything preceeding the last '.' as when reading a file name but don't want the extension

        Args:
            exceptions: special cases where suffix has mulitple .'s (e.g. 'image.ome.tiff'). If present will replace the exp str with empty str 
                e.g. get_prefix(s='raw_img.ome.tiff', exceptions=['.ome']) 
                    steps: raw_img.ome.tiff -> raw_img.ome -> returns 'raw_img'
    """
    idx = s.rfind('.')
    if idx == -1: # If there is no '.', return the whole string
        return s

    # If '.' is found, return the substring before it
    out = s[:idx]

    # check for expections - cases where suffix has mulitple .'s
    if exceptions:
        for exp in exceptions:
            if out.endswith(exp):
                out = out[:out.rfind(exp)]
                
    return out
    
    

def get_matches(iterable: Iterable, filter_str: list|str, pattern=True, startswith=False, endswith=False) -> List[str]:
    """
    Helper function for get contents
    Gets elements from iterable that match the filter_str, or if filter_str is a list, checks if any string in the list matches. 
    Wraps different ways to do matching.

    Args:
        iterable (list): list of strings to be filtered.
        filter_str (list, str): A list of filter strings or single string. Checks if any string in the list matches.
            if not a list, converts to a list
        startswith (bool, optional): Determines whether to match the filter strings only at the begining of the input strings.
            If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
        endswith (bool, optional): Determines whether to match the filter strings only at the end of the input strings.
            If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
            If both endswith and starts with are True, will default to startswith
            If both endswith and starts with are False, the filter strings will be matched anywhere within the input strings 
                using the `in` operator.
        pattern: (bool): if True, filter_str applied using re.search
    Returns:
        list: A new list containing the filtered strings.
    """
    filters = [filter_str] if not isinstance(filter_str, list) else filter_str # allow filters to be a single string or list of str
    
    # determine match function 
    if startswith:
        match_func = str.startswith
    elif endswith:
        match_func = str.endswith
    elif pattern: # use regex
        match_func = lambda s, _pattern: True if re.search(_pattern, s) else False
    else: # string methods
        match_func = str.__contains__
    
    # apply match function
    matches = sorted([s for s in iterable if any(match_func(s, f) for f in filters)])
    return matches
    
def get_contents(adir, filter_str='', startswith=False, endswith=True, filetype=None, fail_on_empty=True, pattern=False, warn=True) -> List[str]:
    """
    Gets elements from directory content that match the string, or if filter_str is a list, checks if any string in the list matches. 
        Wraps different ways to do matching.

    Args:
        adir (list): A directory to read contents as list of strings to be filtered.
        filter_str (list, str): A list of filter strings or single string. Checks if any string in the list matches.
        startswith (bool, optional): Determines whether to match the filter strings only at the begining of the input strings.
            If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
        endswith (bool, optional): Determines whether to match the filter strings only at the end of the input strings.
            If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
            If both endswith and starts with are True, will default to startswith
            If both endswith and starts with are False, the filter strings will be matched anywhere within the input strings 
                using the `in` operator.
        filetype (None, str): string to match to end of filename after other matching is applied
        fail_on_empty (bool): if True will raise error if no matches are found
        pattern: (bool): if True, filter_str applied using re.search
    Returns:
        list[str]: sorted list containing the filtered strings.
    """
    filters = [filter_str] if isinstance(filter_str, str) else filter_str # allow filters to be a single string or list of str
    content = [s for s in os.listdir(adir)]
    
    # apply matching
    if pattern: # need to sidestep own fxn's default behavior, since adding pattern matching with regex since lots of code relys on it
        matches = get_matches(content, filters, pattern=pattern)
    else:
        matches = get_matches(content, filters, pattern=pattern, startswith=startswith, endswith=endswith)
    
    # maybe check filetype 
    if filetype is not None:
        matches = [s for s in matches if s.endswith(filetype)]
    
    # re-build paths
    filtered_paths = [os.path.join(adir, el) for el in matches]
    
    # handle empty dirs
    if len(filtered_paths) == 0:
        msg = f'no content found for {adir} using filters {filter_str}\ncurrent content:\n{content}'
        if fail_on_empty: 
            raise ValueError(msg)
        else: 
            if warn: 
                print(msg)
        
    return filtered_paths

def get_contents_recursive(directory, filetype='', pattern=None, file_list=None):
    """
    get contents recursively, checking if path ends with 'filetype' str, of if a pattern is supplied check using re.match
    Args:
        filetype (None, str): str to match to end of filename
        pattern: (None, str): if True, filter_str applied using re.search
    """
    if file_list is None:
        file_list = []
    
    # Check all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # If the item is a directory, recurse into it
        if os.path.isdir(item_path):
            get_contents_recursive(item_path, filetype=filetype, pattern=pattern, file_list=file_list)
        
        # check if match, determine match func to use
        _use_re = True if pattern is not None else False
        _match = get_matches([item_path], (pattern if _use_re else filetype), pattern = _use_re, endswith=not _use_re)
        
        if len(_match) > 0:
            file_list.append(item_path)

    return file_list


def get_most_recent_file(
    target: Union[str, List[str]],
    sort_by: str = "created"
) -> Optional[str]:
    """
    Return the full path to the most recently created or modified file.

    Args:
        target (str | List[str]): A directory path or list of file paths.
        sort_by (str): Either 'created' or 'modified' — determines which timestamp to use.

    Returns:
        str | None: The path to the most recent file, or None if none found.
    """
    # Validate sort_by input
    sort_by = sort_by.lower()
    if sort_by not in ("created", "modified"):
        raise ValueError("sort_by must be either 'created' or 'modified'")

    # Choose appropriate time function
    get_time = os.path.getctime if sort_by == "created" else os.path.getmtime

    try:
        # Collect valid file paths
        if isinstance(target, str) and os.path.isdir(target):
            files = [
                os.path.join(target, f)
                for f in os.listdir(target)
                if os.path.isfile(os.path.join(target, f))
            ]
        elif isinstance(target, (list, tuple)):
            files = [f for f in target if os.path.isfile(f)]
        else:
            return None

        if not files:
            return None

        # Return the file with the most recent timestamp
        most_recent = max(files, key=get_time)
        return most_recent

    except Exception as e:
        print(f"Error: {e}")
        return None


def filter_by_regex(strings: List[str],
                    pattern: Union[str, Pattern],
                    match_type: str = 'search') -> List[str]:
    """
    Return all strings from the input list that match the given regex.

    Args:
        strings: List of input strings to filter.
        pattern: A regex pattern (string or compiled Pattern). 
                 If a string is given, it will be compiled with re.IGNORECASE off.
        match_type: One of 'search', 'match', or 'fullmatch':
            - 'search'    : find the pattern anywhere in the string (default)
            - 'match'     : only at the beginning of the string
            - 'fullmatch' : entire string must match the pattern

    Returns:
        A list of strings for which the regex returned a truthy match.
    """
    if isinstance(pattern, str):
        regex = re.compile(pattern)
    else:
        regex = pattern

    if match_type == 'search':
        test = regex.search
    elif match_type == 'match':
        test = regex.match
    elif match_type == 'fullmatch':
        test = regex.fullmatch
    else:
        raise ValueError("match_type must be one of 'search', 'match', or 'fullmatch'")

    return [s for s in strings if test(s)]




def insert_base_path(path_to_modify, base_path_to_insert):
    """
    Modifies a file path by replacing its base directory with a new base directory.

    This function is useful when begining of a file path differs across systems or drives,
    and you need to adjust the file path to match a different base directory.

    Args:
        path_to_modify (str): The original file path to be modified.
        base_path_to_insert (str): The new base directory to replace the original base.

    Returns:
        str: The modified file path with the new base directory.

    Example:
        path_to_modify = r"C:\\Users\\user\\OneDrive - Tufts\\Classes\\Rotation\\models\\file.h5"
        base_path_to_insert = r"D:\\OneDrive - Tufts\\Classes\\Rotation\\models"
        modify_path(path_to_modify, base_path_to_insert)
        
        returns:
            'D:\\OneDrive - Tufts\\Classes\\Rotation\\models\\file.h5'
    """
    
    # Break down the paths into components
    modify_parts = os.path.normpath(path_to_modify).split(os.sep)
    insert_parts = os.path.normpath(base_path_to_insert).split(os.sep)
    
    # find where end of insert_path starts matching with path_to_modify
    start_matching = int(modify_parts.index(insert_parts[-1])) 
    assert start_matching != -1, 'no match found'
    
    match_inds = [] # find matching parts between the paths 
    for ii, mi in enumerate(list(range(0, start_matching+1))[::-1]):
        ii = (ii+1)*-1
        if ii < (-1 * len(insert_parts)):
            break        
        if modify_parts[mi] == insert_parts[ii]:
            match_inds.append([mi, ii])
        # print(modify_parts[mi], insert_parts[ii])

    # use the last common matching part to combine the parts together 
    last_match_mi, last_match_ii = match_inds[-1] 
    modified_path_parts = insert_parts[:last_match_ii+1] + modify_parts[last_match_mi+1:]
    modified_path = '\\'.join(modified_path_parts)
    # print(modified_path)
    return modified_path


def looks_like_path(s, exists=False):
    """ 
    Heuristic (contains slashes) to check if a string looks like a path
    
    Args:
        exists (bool): if looks like a path, check if it exists
    """
    result =  ("/" in s or "\\" in s)
    if result and exists:
        return os.path.exists(Path(s))
    return result




def clean_path_name(filename, replacement="_"):
    import unicodedata
    
    # 1. Handle Windows reserved characters: < > : " / \ | ? *
    # Also include other problematic symbols like parentheses and plus signs
    illegal_chars = r'[<>:"/\\|?*()+]'
    
    # 2. Remove control characters (non-printable)
    # This cleans up invisible junk from copy-pastes
    s = "".join(ch for ch in filename if unicodedata.category(ch)[0] != "C")
    
    # 3. Replace illegal/problematic symbols with the replacement character
    cleaned = re.sub(illegal_chars, replacement, s)
    
    # 4. Strip leading/trailing whitespace and dots (Windows doesn't like trailing dots)
    cleaned = cleaned.strip(". ")
    
    # 5. Prevent empty strings or reserved Windows names (CON, PRN, etc.)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", 
        "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", 
        "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    if cleaned.upper() in reserved_names or not cleaned:
        cleaned = f"file_{cleaned}" if cleaned else "unnamed_file"

    return cleaned



def objdir(obj, skip_callable=True, return_as_string=False, return_as_dict=False, tabulated=False):
    """for each attribute of an object print or return its value"""
    outstr, outdict, tabulated_entries = '', {}, []
    for attr in dir(obj):
        is_callable = True
        try:
            value = getattr(obj, attr)
            is_callable = callable(getattr(obj, attr))
        except Exception as e:
            value = f'(NONE) raised exception:{e}'
        
        if skip_callable and is_callable:
            continue
        
        
        if return_as_dict:
            outdict[attr] = value
        elif tabulated:
            tabulated_entries.append({'attr':attr, 'type':type(value).__name__, 'value':value})
        else:
            outstr += (f'- {attr} ({type(value).__name__}) --> {value}.\n')
    
    if tabulated:
        from tabulate import tabulate
        outstr = tabulate(tabulated_entries, headers="keys")
            
    if return_as_string:
        return outstr
    elif return_as_dict:
        return outdict
    else:
        print(outstr)
    

def get_most_recent_folder(directory, sort_by="modified", return_all=False):
    """
    Returns the most recently created or modified folder in the given directory.

    Parameters:
        directory (str): The path to the directory containing folders.
        sort_by (str): Sorting method, either "modified" (default) or "created".
        return_all (bool): if False, only returns the most recent dir, otherwise returns the sorted dirs as a list

    Returns:
        str: The name of the most recent folder, or None if no folders are found.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory path: {directory}")

    # Get all items in the directory
    items = [os.path.join(directory, item) for item in os.listdir(directory)]

    # Filter only directories
    folders = [item for item in items if os.path.isdir(item)]

    if not folders:
        raise ValueError('directory contains no subdirectories')

    # Choose sorting method
    if sort_by == "created":
        key_func = os.path.getctime  # Creation time
    elif sort_by == "modified":
        key_func = os.path.getmtime  # Modification time
    else:
        raise ValueError("Invalid sort_by argument. Use 'modified' or 'created'.")

    # Sort folders by the chosen method (most recent first)
    folders.sort(key=key_func, reverse=True)

    if return_all: 
        return folders
    return os.path.basename(folders[0])  # Return the most recent folder's name




IncludeType = Union[str, re.Pattern, Iterable[str]]

def copy_tree_selected(
    src: str | Path,
    dst: str | Path,
    include: IncludeType,
    keep_empty_dirs: bool = True,
    overwrite: bool = False,
    follow_symlinks: bool = False,
    preserve_times: bool = True,
    case_insensitive: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Copy a directory tree from `src` to `dst`, preserving the nested structure,
    but include only files whose *names* match a single regex.

    Unified include interface:
      - Regex string:   include=r"^(?:metrics\.json|.*_summary\.csv)$"
      - Compiled regex: include=re.compile(r"...", re.IGNORECASE)
      - List of names/patterns/extensions:
          include=["metrics.json", "*.csv", ".tif"]

      The list form is converted to one regex internally:
        • plain strings → exact filename matches
        • items with glob meta (* ? [ ]) → glob patterns
        • items starting with '.' → extension filters (e.g., '.tif', '.csv')

    Args:
        src: Source directory.
        dst: Destination directory (created if needed).
        include: Regex or iterable of names/patterns/extensions (see above).
        keep_empty_dirs: Mirror empty directories even if nothing inside matches.
        overwrite: Overwrite existing files at destination.
        follow_symlinks: Follow directory symlinks when walking.
        preserve_times: Use shutil.copy2 to preserve timestamps (else shutil.copy).
        case_insensitive: Case-insensitive filename matching for the include filter.
        dry_run: Print actions instead of performing them.

    Returns:
        Number of files copied (or that would be copied in dry_run).
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if not src.exists() or not src.is_dir():
        raise NotADirectoryError(f"Source directory does not exist or is not a directory: {src}")

    include_rx = _build_regex_from_include(include, case_insensitive=case_insensitive)

    # Ensure destination root exists (or print plan)
    if dry_run:
        print(f"[dry-run] Would create destination root: {dst}")
    else:
        dst.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    copy_fn = shutil.copy2 if preserve_times else shutil.copy

    for dirpath, dirnames, filenames in os.walk(src, followlinks=follow_symlinks):
        dirpath_p = Path(dirpath)
        rel = dirpath_p.relative_to(src)
        dest_dir = dst / rel

        selected = [f for f in filenames if include_rx.match(f)]

        if selected or keep_empty_dirs:
            if dry_run:
                print(f"[dry-run] Would create directory: {dest_dir}")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)

        for fname in selected:
            src_file = dirpath_p / fname
            dst_file = dest_dir / fname

            if dst_file.exists() and not overwrite:
                if dry_run:
                    print(f"[dry-run] Skip (exists): {dst_file}")
                continue

            if dry_run:
                print(f"[dry-run] Would copy: {src_file} -> {dst_file}")
                files_copied += 1
            else:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                copy_fn(src_file, dst_file, follow_symlinks=follow_symlinks)
                files_copied += 1

    return files_copied

def _build_regex_from_include(
    include: IncludeType,
    case_insensitive: bool = True,
) -> re.Pattern:
    """
    Normalize the `include` spec to a compiled regex that matches file *names*.

    Accepts:
      - regex string (e.g., r"^(?:metrics\.json|.*_summary\.csv)$")
      - compiled regex (re.Pattern)
      - iterable of strings (mix of exact names, globs like '*.json', or extensions '.tif')

    Strategy for iterables:
      - Exact names: fully escaped and anchored (e.g., ^README\.md$)
      - Globs: translated to regex via fnmatch.translate (e.g., '*.json')
      - Extensions ('.tif'): converted to ^.*\.tif$ (case-insensitive if requested)

    Returns:
      compiled re.Pattern
    """
    if isinstance(include, re.Pattern):
        return include

    flags = re.IGNORECASE if case_insensitive else 0

    if isinstance(include, str):
        # Treat as a regex string as-is
        return re.compile(include, flags)

    # Iterable of strings: build a single OR-regex
    exact_names = []
    ext_items = []
    glob_items = []

    for item in include:
        s = str(item)
        if any(ch in s for ch in "*?[]"):  # glob-like
            glob_items.append(s)
        elif s.startswith(".") and os.sep not in s and "/" not in s:
            # extension-like ('.tif', '.csv'); no path separators
            ext_items.append(s)
        else:
            # treat as exact filename
            exact_names.append(s)

    parts = []

    # Exact names → ^(?:name1|name2)$ (escaped)
    if exact_names:
        escaped = [re.escape(n) for n in exact_names]
        parts.append(r"^(?:%s)$" % "|".join(escaped))

    # Extensions → ^.*\.(?:tif|csv)$
    if ext_items:
        exts = [re.escape(e.lstrip(".")) for e in ext_items]
        parts.append(r"^.*\.(?:%s)$" % "|".join(exts))

    # Globs → combine their translated regexes under one non-capturing group
    # fnmatch.translate returns a regex with \Z(?ms) at the end; we strip anchors.
    def _strip_anchors(rx: str) -> str:
        # remove leading ^ and trailing \Z (and flags) if present
        rx = rx.strip()
        if rx.startswith("(?s:"):
            # python 3.12 style from fnmatch sometimes wraps with (?s:...)
            # keep as-is; we'll just remove the trailing \Z
            pass
        # Guaranteed to end with \Z
        if rx.endswith(r"\Z"):
            rx = rx[:-2]
        # Remove leading ^ if present
        if rx.startswith("^"):
            rx = rx[1:]
        # Remove trailing $ if present
        if rx.endswith("$"):
            rx = rx[:-1]
        return rx

    if glob_items:
        glob_rx_parts = [_strip_anchors(fnmatch.translate(g)) for g in glob_items]
        parts.append(r"^(?:%s)$" % "|".join(glob_rx_parts))

    if not parts:
        # If the iterable was empty, match nothing.
        return re.compile(r"^(?!)$", flags)

    combined = r"(?:%s)" % "|".join(parts)
    return re.compile(combined, flags)




def get_datetime(fmt="%Y_%m%d_%H%M%S"):
    """returns current datetime as a str"""
    formatted_datetime = datetime.now().strftime(fmt)
    return formatted_datetime

def merge_dicts(*dicts):
    """combines an arbitrary number of dictionaries"""
    return {k: v for d in dicts for k, v in d.items()}

def popget(dict, key, return_if_not_found=None):
    """ allow dict.pop() to function like dict.get(), returning return_if_not_found if key is not found otherwise pops key """
    if key in dict:
        return dict.pop(key)
    else:
        return return_if_not_found

def flatten_dict(d, parent_key='', sep='_'):
    """ flatten a nested dictionary, preserving hierarchy by referencing parent key in flattened names """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def sort_dict_by_list(d: Dict, order: List):
    """ sort a dict by keys defined in order, if not in order added to end """
    return dict(sorted(d.items(), key=lambda item: (order.index(item[0]) if item[0] in order else float('inf'))))

def get_key_from_value(data_dict: dict, target_value: str) -> Optional[str]:
    """
    Performs a reverse lookup on a dictionary.

    It searches for the first key whose corresponding value matches the
    supplied target string.

    Args:
        data_dict: The dictionary to search through.
        target_value: The string value to find within the dictionary's values.

    Returns:
        The key corresponding to the matched value, or None if no match is found.
    """
    # Iterating through the key-value pairs of the dictionary
    for key, value in data_dict.items():
        # Check if the current value matches the target value
        if value == target_value:
            # If a match is found, immediately return the corresponding key
            return key
    
    # If the loop finishes without finding a match, return None
    return None

def any_isin(el_to_check, strs_to_look_for):
    """ returns true if any one of the strs_to_look_for are in el_to_check"""
    for astr in strs_to_look_for:
        if astr in el_to_check:
            return True
    return False

def allin(list1, list2):
    """checks if all elements in list1 are in list2"""
    return all([i in list2 for i in list1])


def find_closest_and_remove(input_list, input_set):
    """ 
    For each element in the input_list, finds the closest number in the set by 
        calculating the absolute difference and then removes the element from the set.
    Returns:
        output (list): closest element from input_set
        closest_matches (list): corresponding element from input_list
        unmatched (list): elements from input_list that were not matched
    """
    input_list, input_set = deepcopy(input_list), deepcopy(input_set)
    output, closest_matches, unmatched = [], [], []
    for number in input_list:
        if len(input_set) == 0:
            unmatched.append(number)
            continue
        closest = min(input_set, key=lambda x: abs(x - number))
        output.append(closest)
        closest_matches.append(number)
        input_set.remove(closest) # Remove the found closest number from the set
    return output, closest_matches, unmatched


def flatten_list(nested_list):
    """recursively flatten lists of sublists"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten if it's a list
        else:
            flat_list.append(item)
    return flat_list


def intersection_of_lists(*lists):
    """
    Computes the intersection of multiple lists.
    
    Parameters:
        *lists: Arbitrary number of lists.
    
    Returns:
        A set containing the intersection of all input lists.
    """
    if not lists:
        return set()  # Return empty set if no lists provided
    if len(lists) == 1:
        return set(lists[0])
    return set(lists[0]).intersection(*lists[1:])


def dt(f=None, *args, **kwargs):
    """
    function timer
    If no arguments are passed, return the current time using timeit.default_timer().
    If a function `f` is provided, time its execution with the given arguments and return the elapsed time and result.
    
    Parameters:
    - f (callable, optional): The function to be timed. Defaults to None.
    - *args: Variable length argument list for the function `f`.
    - **kwargs: Arbitrary keyword arguments for the function `f`.
    
    Returns:
    - float: Current time if `f` is None, else the elapsed time to execute `f` and the result.
    """    
    if f is None:
        return timer()
    else:
        start = timer()
        result = f(*args, **kwargs)
        end = timer()
        elapsed = end - start
        return elapsed, result
    
    

def get_function(function_string: str):
    """
    use importlib to import a function from its string representation
    """
    assert '.' in function_string, f'function_string: ({function_string}) could not be parsed because it does not contain module it comes from'
    # Split the module and function name
    module_name, function_name = function_string.rsplit('.', 1)

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Get the function from the module
    function = getattr(module, function_name)
    return function

def try_import(module_name: str, p=True):
    """ attempt import - like a try-except block """
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        if p: print(f"import of {module_name} failed. error: {e}")
        return None


def get_existant_path(paths: Iterable[str]|str, fail_on_empty=True):
    """ iter list of paths, returning first one that os.path.exists returns true for, else raise error 
        args:
            paths (list(str)): paths to check
            fail_on_empty (bool): if True will raise error if no existing paths are found
    """
    if isinstance(paths, str):
        paths = [paths]

    if not isinstance(paths, IterableBaseClass):
        raise ValueError(f"`paths` must be an iterable of strings, but got: `{paths}`.")


    for p in paths:
        if os.path.exists(p):
            return p
        
    if fail_on_empty:
        raise FileNotFoundError(f"no existing paths found in: {paths}")
    
    return None
    


def dict_to_tabulated_rows(d, depth=0, tableformat='fancy_grid'):
    """
    Recursively formats a nested dictionary into a tabulated string with headers for key, type, and value
        e.g. if a value of dict is a dict it gets represented as a table 
    """
    from tabulate import tabulate
    table = []
    for key, val in d.items():
        val_type = type(val).__name__
        if isinstance(val, dict) and len(val) > 0:
            val_str = dict_to_tabulated_rows(val, depth + 1, tableformat=tableformat)
        else:
            val_str = str(val)
        table.append([key, val_type, val_str])
    
    return tabulate(table, headers=["key", "type", "value"], tablefmt=tableformat)


def is_nested_dict(obj, levels: int, *, exact: bool = False) -> bool:
    """Return True iff the first `levels` layers are dicts.
    - exact=False: allow deeper nesting beyond `levels`.
    - exact=True: require that layer `levels+1` is NOT a dict (i.e., exactly `levels` dict layers)."""
    if not isinstance(obj, dict):
        return False
    if levels == 1:
        return True if not exact else all(not isinstance(v, dict) for v in obj.values())
    return all(is_nested_dict(v, levels - 1, exact=exact) for v in obj.values())


def log_or_print(message, logger=None):
    """Helper function to log or print based on logger availability."""
    if logger:
        logger.info(message)
    else:
        print(message)