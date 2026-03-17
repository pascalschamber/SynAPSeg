from __future__ import annotations
from enum import Enum, auto
from typing import Any, Set, Iterable

class ParameterFlag(Enum):
    HIDDEN = auto()
    REQUIRED = auto()
    DATA_PARAM = auto()

class SocketType(Enum):
    UI = auto()
    DISPATCH = auto()


class FlagParser:
    
    @staticmethod
    def parse_flags(flags: Any) -> Set[ParameterFlag]:
        if not flags: return set()
        if isinstance(flags, str):
            s = FlagParser._parse_flag(flags)
            return set(s,) if s else set()
        if isinstance(flags, Iterable):
            flags = [FlagParser._parse_flag(f) for f in flags if f is not None]
            return set(f for f in flags if f is not None)
        return set()
        
    @staticmethod
    def _parse_flag(s: str|ParameterFlag) -> None | ParameterFlag:
        if isinstance(s, ParameterFlag): return s
        s_lower = (s or "").strip().lower()
        if s_lower == "hidden": return ParameterFlag.HIDDEN
        if s_lower == "required": return ParameterFlag.REQUIRED
        if s_lower == "data_param": return ParameterFlag.DATA_PARAM
        return None
