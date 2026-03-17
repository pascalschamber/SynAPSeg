from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterable, Tuple, List, Callable, Mapping, Set
import hashlib
import json

from .flags import ParameterFlag, FlagParser
from .spec import ParamSpec, construct_paramspecs_from_scoped_dict
from .socket import unflatten_dict, flatten_paramspec_schema, get_spec_value, prune_flatened_struct, unflatten_dict

def _hash_inputs(*objs: Any) -> str:
    m = hashlib.sha256()
    for o in objs:
        m.update(json.dumps(o, sort_keys=True, default=str).encode("utf-8"))
    return m.hexdigest()


@dataclass
class SchemaInterpreter:
    """
    SchemaInterpreter acts as an interface between param specs, user values, UI widget config, and run config generation
        1. converts nested param spec config into flat dict with keys relaying scope
        2. allows updating of default values with user defined values (from run config or UI)
        3. performs validation on parameter values using known data types (segregating data_params for run time checking)
        4. handles extracting values from ParamSpecs (uses current value over default value if not None)
            - values are returned with input struct hierarchy (meant as run config output)
            - BaseConfig still handles reading config default params and resolving environment variables
    Design choices:
        - internally flat schema should be used for all operations except outputing nested config for run or UI 
    Public attributes:
        - schema: flattened version of nested param spec heirarchy
    Public methods:
        - values: extracted values from paramspec nested schema
        - flat_values: same but with flat keys 
    """
    
    default_specs: Dict[str, Dict[str, Dict]] 
    plugin_headings: Optional[List[str]] = field(default_factory=lambda: ['Model', 'Stages'])
    
    # old 
    schema_version: Optional[str] = None
    strict: bool = False  # if True, unknown keys & heuristic list parsing raise

    _param_specs: Dict[str, ParamSpec] = field(init=False, default_factory=dict)
    _cache: Dict[str, Any] = field(init=False, default_factory=dict)  # input_hash -> cached outputs

    def __post_init__(self):

        schema = flatten_paramspec_schema(self.default_specs, value_type='spec', plugin_headings=self.plugin_headings)
        schema = construct_paramspecs_from_scoped_dict(schema)
        self.schema = schema       
        
    @classmethod
    def from_specs(
        cls, 
        default_specs: Dict[str, Dict[str, Dict]], 
        plugin_headings: Optional[List[str]] = ['Model', 'Stages'],
        ) -> 'SchemaInterpreter':

        return cls(default_specs, plugin_headings)

    @classmethod
    def from_default_params_path(
        cls,
        default_params_path: str,
        plugin_headings: Optional[List[str]] = ['Model', 'Stages'],
        raw: bool = False,
        ) -> 'SchemaInterpreter':
        
        from SynAPSeg.IO.BaseConfig import read_config, group_by_heading
        spec_dict = read_config(default_params_path, raw=raw) # default params
        spec_dict = group_by_heading(spec_dict)

        return SchemaInterpreter.from_specs(spec_dict, plugin_headings=plugin_headings)

    

    def update_schema(self, updated_schema, flatten_input=True, clear_old=False):
        """ 
        update schema with values or specs, mixed types allowed (e.g. model config specs)
            if scope not in current schema input dict (updated_schema) values must be (dict | ParamSpec)
        
        Args:
            flatten_input if True updated_schema is expected to be nested, if False expects updated schema to be scoped 
            clear_old if True clears old schema before updating (default False)
        """
        errors = []
        
        if clear_old:
            self.schema = {}

        if flatten_input:
            values_by_header = flatten_paramspec_schema(updated_schema, value_type='spec', plugin_headings=self.plugin_headings)
        else: 
            values_by_header = updated_schema
        
        for scope, value in values_by_header.items():
            
            if scope not in self.schema:
                self._update_spec(scope, value)

            emsg = self._update_value(scope, value)

            if emsg: 
                errors.append(emsg)
        
        emsgs = [emsg for emsg in errors if emsg is not None ]
        if len(emsgs) > 0:
            print('\n'.join(emsgs))
            raise KeyError(f"scopes not defined (n={len(emsgs)})")
        
    def _update_spec(self, scope, value):
        assert isinstance(value, (dict, ParamSpec)), f"value must be (dict | ParamSpec), got value: `{value}` of type: {type(value)}"
        ps = value if isinstance(value, ParamSpec) else ParamSpec.from_yaml_item(scope, value)
        self.schema[scope] = ps

    def _update_value(self, scope, value):
        emsg = None
        try:
            v = get_spec_value(value)
            self.schema[scope].set_value(v)
        except KeyError:
            emsg = f"scope ({scope}) not in self.schema"
        return emsg
    

    # === Public API ===
    def get_ui_specs(self, filter_attrs=None, unflatten=True):
        """ convert param specs to dict representation compatible with UI config widget """
        _specs = {}
        for scope, ps in self.schema.items():
            _specs[scope] = ps.ui_descriptor(filter_attrs=filter_attrs)
        
        return unflatten_dict(_specs) if unflatten else _specs
        
    def to_run_config(self):
        """ 
        coerce interpreter schema to run config dict by removing headings.
            internally calls get_run_config with default spec headings + headings inserted in plugin params
        """
        headings = list(self.default_specs.keys()) + ['current_value', 'root']
        
        return self.get_run_config(to_replace={f"{h}.": "" for h in headings})

    def get_run_config(self, to_replace=None):
        conf = prune_flatened_struct(self.schema, to_replace=to_replace)
        conf = {k:get_spec_value(v) for k,v in conf.items()}
        return unflatten_dict(conf)
            
    def __str__(self):
        # format ui_groups for printing
        fstr = 'UI PARAM SPEC\n' + '#'*13 + "\n"
        for scope, ps in self.schema.items():
            fstr += f"┗ {scope}\n"
            # print(d['name'])
            d = ps.ui_descriptor()
            fstr += f"  ┗ {d['name']} (type={d['widget_type']} kwargs=[{d['widget_kwargs'] or 'none'}])\n"
            fstr += f"    ┣ {d['tooltip']}\n"
            fstr += f"    ┗ group={d['group']}\n"
        return(fstr)


def is_nested_dict(obj, levels: int, *, exact: bool = False) -> bool:
    """Return True iff the first `levels` + 1 layers are dicts.
    - exact=False: allow deeper nesting beyond `levels`.
    - exact=True: require that layer `levels+1` are all dicts (i.e., exactly `levels` + 1 dict layers)."""
    if not isinstance(obj, dict):
        return False
    if levels == 1:
        return True if not exact else all(isinstance(v, dict) for v in obj.values())
    return all(is_nested_dict(v, levels - 1, exact=exact) for v in obj.values())