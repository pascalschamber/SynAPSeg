
"""
yaml_text_normalizer
--------------------
Utilities for converting Python-esque nested text files into clean YAML.

Typical issues addressed:
- Real tabs or literal "\t" sequences in the source (YAML forbids tabs).
- Python-style literals like "{'a': 1}" or "None/True/False" that are not valid YAML as-is.
- Sectioned text where a top-level key is followed by indented items such as "{...}" or "idx: {...}".

Public API:
- normalize_tabs(text, spaces=2) -> str
- parse_text_to_data(text, spaces=2) -> dict
- convert_text_to_yaml(text, spaces=2, json_fallback=False) -> str
- convert_file_to_yaml(in_path, out_path=None, spaces=2, overwrite=False, json_fallback=False) -> str
"""

from __future__ import annotations

import re, ast, os, json
from collections import OrderedDict
from typing import Any, Dict

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


def normalize_tabs(s: str, spaces: int = 2) -> str:
    """
    Replace real tabs and the literal two-character sequence '\\t' with spaces.
    Parameters
    ----------
    s : str
        Input text.
    spaces : int
        Number of spaces to replace each tab with (default=2).

    Returns
    -------
    str
        Text with tabs replaced by spaces.
    """
    rep = " " * spaces
    s = s.replace("\t", rep)
    s = s.replace("\\t", rep)
    return s


def _parse_inline_value(val_str: str):
    """
    Try to parse a Python literal; otherwise return the raw string.
    Returns a tuple (value, has_value_flag).
    """
    val = val_str.strip()
    if val == "":
        return None, False
    try:
        parsed = ast.literal_eval(val)
        return parsed, True
    except Exception:
        return val, True


def parse_text_to_data(text: str, spaces: int = 2) -> Dict[str, Any]:
    """
    Parse sectioned text into a Python dict using simple rules:
      - Lines with no leading spaces that end in ':' start a top-level section.
      - Indented lines under a section can be:
          * '{...}' a single-line dict -> appended to a list under that section.
          * '<idx>: {...}' -> added to a dict under that section with parsed key.
          * '<name>: <value>' -> added as key/value under that section (dict).
          * anything else -> appended as raw string into a list.
      - Python literals (dict/list/tuple/None/True/False/nums/strings) are parsed.

    Parameters
    ----------
    text : str
        Input text to parse.
    spaces : int
        Spaces to substitute for tabs before parsing.

    Returns
    -------
    Dict[str, Any]
        Parsed structure (plain dict/lists/scalars).
    """
    data: Dict[str, Any] = OrderedDict()
    current_key = None
    lines = normalize_tabs(text, spaces=spaces).splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        # Top-level section header
        if re.match(r"^[^\s].*?:\s*$", line) or re.match(r"^[^\s].*?:\s+.+$", line):
            current_key = None
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()

            if rest:
                parsed, has_value = _parse_inline_value(rest)
                data[key] = parsed if has_value else None
            else:
                data[key] = None
                current_key = key
            continue

        # Indented content
        if current_key is None:
            # Orphan indented line: ignore gracefully
            continue

        child = line.lstrip()

        # Pattern: "<k>: { ... }"
        m = re.match(r"^([^\s:]+)\s*:\s*(\{.*\})\s*$", child)
        if m:
            k_raw, dict_str = m.groups()
            if data[current_key] is None or not isinstance(data[current_key], dict):
                data[current_key] = OrderedDict()
            # Parse key
            try:
                k = ast.literal_eval(k_raw)
            except Exception:
                try:
                    k = int(k_raw)
                except Exception:
                    k = k_raw
            # Parse value dict
            try:
                v = ast.literal_eval(dict_str)
            except Exception:
                v = dict_str
            data[current_key][k] = v
            continue

        # Pattern: "{ ... }" -> list item
        if re.match(r"^\{.*\}\s*$", child):
            if data[current_key] is None:
                data[current_key] = []
            elif isinstance(data[current_key], dict):
                # Mixed types are not ideal; start a new list
                data[current_key] = []
            try:
                v = ast.literal_eval(child)
            except Exception:
                v = child
            data[current_key].append(v)
            continue

        # Pattern: "name: value" inside a block
        m2 = re.match(r"^([^\s:]+)\s*:\s*(.*)$", child)
        if m2:
            subk, subv = m2.groups()
            parsed, has_value = _parse_inline_value(subv)
            if data[current_key] is None or not isinstance(data[current_key], dict):
                data[current_key] = OrderedDict()
            data[current_key][subk] = parsed if has_value else None
            continue

        # Fallback: raw line -> list
        if data[current_key] is None:
            data[current_key] = []
        if isinstance(data[current_key], list):
            data[current_key].append(child)
        else:
            data[current_key] = [child]

    # Convert OrderedDict to plain dict recursively
    def _sanitize(o):
        if isinstance(o, OrderedDict):
            return { _sanitize(k): _sanitize(v) for k, v in o.items() }
        if isinstance(o, dict):
            return { _sanitize(k): _sanitize(v) for k, v in o.items() }
        if isinstance(o, (list, tuple)):
            return [ _sanitize(x) for x in o ]
        return o

    return _sanitize(data)


def convert_text_to_yaml(text: str, spaces: int = 2, json_fallback: bool = False) -> str:
    """
    Convert sectioned Python-esque text into a YAML (or JSON) string.

    Parameters
    ----------
    text : str
        Input text to parse and serialize.
    spaces : int
        Spaces to substitute for tabs before parsing (default=2).
    json_fallback : bool
        If True and PyYAML is not installed, return JSON instead of raising.

    Returns
    -------
    str
        YAML (or JSON, if json_fallback) string.
    """
    data = parse_text_to_data(text, spaces=spaces)

    if not _HAVE_YAML and not json_fallback:
        raise RuntimeError("PyYAML not installed. Install 'pyyaml' or pass json_fallback=True.")

    if _HAVE_YAML:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)  # type: ignore
    else:
        return json.dumps(data, ensure_ascii=False, indent=2)


def convert_file_to_yaml(in_path: str, out_path: str | None = None, spaces: int = 2,
                         overwrite: bool = False, json_fallback: bool = False) -> str:
    """
    Convert a file to YAML (or JSON if PyYAML missing and json_fallback=True).

    Parameters
    ----------
    in_path : str
        Input file path.
    out_path : str | None
        Output path. If None, uses same stem with '.yaml' or '.json' extension in the same dir.
    spaces : int
        Spaces to substitute for tabs before parsing (default=2).
    overwrite : bool
        Overwrite existing output file.
    json_fallback : bool
        If True and PyYAML is not installed, write JSON instead of raising.

    Returns
    -------
    str
        The path to the written output file.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    data = parse_text_to_data(raw, spaces=spaces)

    stem = os.path.splitext(os.path.basename(in_path))[0]
    if out_path is None:
        ext = ".yaml" if _HAVE_YAML or not json_fallback else ".json"
        out_path = os.path.join(os.path.dirname(in_path), stem + ext)

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (set overwrite=True).")

    if not _HAVE_YAML and not json_fallback:
        raise RuntimeError("PyYAML not installed. Install 'pyyaml' or pass json_fallback=True.")

    if _HAVE_YAML:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)  # type: ignore
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path
