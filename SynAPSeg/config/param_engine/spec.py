from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Optional, List, Mapping
import re
from copy import deepcopy
from .flags import ParameterFlag, FlagParser

DEFAULT_ATTRIBUTES = {'default_value', 'current_value','widget_type','tooltip','heading','flags'}

def construct_paramspecs_from_scoped_dict(_schema: dict[str, Any]) -> dict[str, ParamSpec]:
    """ _schema is flat param spec dict """
    schema = deepcopy(_schema)
    for scope, v in schema.items():
        if not isinstance(v, dict):
            raise ValueError(f"Cannot initialize param specs with schema value: v:{v}. must be dict type")
        #TODO type checking on scope --> must contain '.' dotted notation
        
        schema[scope] = ParamSpec(
            name=scope.split('.')[-1], 
            scope=scope,
            current_value=v.get('current_value'),
            default_value=v.get('default_value'),
            widget_type=v.get('widget_type'),
            tooltip=v.get('tooltip'),
            heading=v.get('heading'),
            flags=FlagParser.parse_flags(v.get('flags')),
            extra={kk: vv for kk, vv in v.items() if kk not in DEFAULT_ATTRIBUTES},  # TODO: this check is a bit fragile
        )
    return schema


@dataclass(frozen=False)
class ParamSpec:
    name: Optional[str] = None
    current_value: Any = None            # TODO only this attr should be unfrozen
    default_value: Any = None
    type: Optional[str] = None           # semantic type, e.g., 'int', 'int|str("auto")'
    widget_type: Optional[str] = None    # presentation hint, e.g., 'slider', 'text'
    scope: Optional[str] = None
    flags: Set[ParameterFlag] = field(default_factory=set)
    extra: Dict[str, Any] = field(default_factory=dict)
    heading: Optional[str] = None
    group: Optional[str] = None         # back-compat for segmentation model params
    tooltip: Optional[str] = None

    @staticmethod
    def from_yaml_item(scope: str, raw: Dict[str, Any]) -> "ParamSpec":

        flags = FlagParser.parse_flags(raw.get('flags'))

        return ParamSpec(
            name=scope.split('.')[-1],
            scope=scope,
            default_value=raw.get("default_value"),
            current_value=raw.get("current_value"),
            type=raw.get("type") or raw.get("widget_type"),  # back-compat mapping
            widget_type=raw.get("widget") or raw.get("widget_type"), # widget element
            tooltip=raw.get("tooltip"),
            flags=flags,
            extra=raw.get("extra") or {},
            heading=raw.get("heading"),
            group=raw.get("group"),
        )

    def is_hidden(self) -> bool:
        return ParameterFlag.HIDDEN in self.flags

    def is_required(self) -> bool:
        # Either explicit REQUIRED flag or extra.required true
        return ParameterFlag.REQUIRED in self.flags or bool(self.extra.get("required", False))

    def is_data_param(self) -> bool:
        return ParameterFlag.DATA_PARAM in self.flags
    
    def set_value(self, value):
        self.current_value = value

    def get_value(self):
        if self.current_value is None:
            return self.default_value
        return self.current_value
        
    def ui_descriptor(self, filter_attrs: Optional[List[str]]=None) -> Dict[str, Any]: # prev included --> merged_values: Optional[Dict[str, Any]] = None, 
        """Build a UI-friendly descriptor for a single parameter.
                Hidden params are included.
            Args:
                filter_attrs: list of param attrs to return
            
            Returns:
                dict representing attributes needed to build a UI widget
        """
               
        # parse description
        widget_kwargs = self.extra or {}
        widget_kwargs = widget_kwargs['extra'] if 'extra' in widget_kwargs else widget_kwargs
        group = widget_kwargs.pop('group') if 'group' in widget_kwargs else self.group

        desc = {
            "name": self.name,
            'scope': self.scope,
            'current_value': self.get_value(), # self.curr_value(merged_values), # in current UI need to update to handle this, prev. ver. just updated default val with the new one
            
            "default_value": self.default_value, 
            "widget_type": self.widget_type or "text",
            "tooltip": self.tooltip or "",
            "heading": self.extra.get("heading", "General"),
            "category": self.scope.split('.')[0],
            
            "group": group, 
            "widget_kwargs": widget_kwargs,
            "flags": self.flags,
        }

        if desc["group"] == "root":
            desc["group"] = None
        elif desc["group"] is None: 
            desc["group"] = desc["heading"]
        # backwards compatibility with IO.BaseConfig.set_params and widget configuration
            # it basically allows another level of nesting
            # if not included it would just be with general stuff like model_class, name, output_dir, etc.

        # should only have the following attributes unless special attrs are required, e.g. value_options, etc
        # TODO name needs to be removed from here when grouping, before pass to segconfigwidget
        # TODO if passing direct to segconfig widget 
        #   group should be replaced by category
        #   widget_kwargs (extras) should be poped and merged with dict   # DONE this should be handled differently in the config and just passed as extra attrs
        
        if filter_attrs is not None and isinstance(filter_attrs, list):
            desc = {k: v for k, v in desc.items() if k in filter_attrs}
        
        return desc

    
