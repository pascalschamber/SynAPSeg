from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Literal
from copy import deepcopy
from .spec import ParamSpec

def run_config_socket(updated_param_specs:Dict, plugin_key:str):
    """ DEPRECATED see segmentation script implementation 
        converts param specs to values only, without UI top-level headers, and removes root key in plugin params """
    converted_vals_only = specs_to_values(updated_param_specs)
    run_params = remove_toplevel_headers(converted_vals_only)
    final_params = unroot_plugin_params(run_params, plugin_key=plugin_key)
    return final_params

def specs_to_values(
    node: Any,
    *,
    spec_key: str = "default_value",
    current_key: str = "current_value",
) -> Any:
    """
    Convert a nested param spec dictionary to a values-only dict.

    Rules:
    - A mapping with `spec_key` is treated as a param spec (leaf):
        - Use `current_value` if present and not None; else use `default_value`.
        - If that chosen value is a dict/list, recurse into it so nested specs are handled.
    - A mapping without `spec_key` is treated as a container/header: recurse on its items.
    - Lists/tuples are mapped element-wise (useful if a current/default value is a list of specs).
    - Primitive values are returned as-is.
    """
    # Param spec (leaf)
    if isinstance(node, Mapping) and spec_key in node:
        value = node.get(current_key)
        if value is None:
            value = node.get(spec_key)
        return _recurse_values(value, spec_key=spec_key, current_key=current_key)

    # Container/header
    if isinstance(node, Mapping):
        return {k: specs_to_values(v, spec_key=spec_key, current_key=current_key) for k, v in node.items()}

    # List/tuple current/default values (may contain nested specs)
    if isinstance(node, (list, tuple)):
        return [specs_to_values(v, spec_key=spec_key, current_key=current_key) for v in node]

    # Primitive
    return node

def _recurse_values(value: Any, *, spec_key: str, current_key: str) -> Any:
    if isinstance(value, Mapping):
        return {k: specs_to_values(v, spec_key=spec_key, current_key=current_key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [specs_to_values(v, spec_key=spec_key, current_key=current_key) for v in value]
    return value


def remove_toplevel_headers(params):
    """ remove top level keys from params dict - used on output of specs_to_values """
    _params = {}
    for header, d in params.items():
        _params.update(d)
    return _params 

def unroot_plugin_params(headerless_params:Dict, plugin_key: str = 'MODEL_PARAMS'):
    """ 
        adapter function that removes plugin's params root key if present (e.g. present after validation)
            this is meant to be input to e.g. BaseConfig
        args:
            params: expected to be run params without toplevel headers (i.e. after running remove_toplevel_headers)
                e.g. {'IMG_DIR': None, ..., MODEL_PARAMS: {stardist2d:{root:{...}, predict_kwargs:{...}}}}
    """
    
    params = deepcopy(headerless_params)
    plugin_params = params.pop(plugin_key)
    _plugin_params = {}
    for plugin_name, pp in plugin_params.items():
        if 'root' in pp:
            _params = {}
            for k,v in pp.items():
                if k == 'root':
                    _params.update(v)
                else:
                    _params[k] = v
            pp = _params
        _plugin_params[plugin_name] = pp
    params[plugin_key] = _plugin_params
    return params

def flatten_paramspec_schema(
        nested_schema, 
        value_type: Literal["spec", "value"] = 'spec', 
        plugin_headings: Optional[List[str]] = ['Model']
    ) -> dict[str, ParamSpec | Any]:
    """ 
    convert nested specs or values to flat structure with nesting preserved in string keys
        e.g. {'key':{'inner':{'current_value':0}}} --> {'key.inner.k': 0}
        note: specs within nested specs (plugin_headings) are not preserved (e.g. segconfigwidget)
    values_by_heading: can be nested dict of paramSpecs or dict specs 
    value_type: returns spec's value if value_type==value else spec is returned

    
    """
    plugin_headings = plugin_headings or []
    flat_schema = {}  
        
    for header, spec_dict in nested_schema.items():
        if header in plugin_headings:                    
            for header_header, param_spec in spec_dict.items():
                is_paramspec = isinstance(param_spec, ParamSpec) # if model_params current vals contains model specs
                is_paramspecdict = isinstance(param_spec, dict) and ('default_value' in param_spec)
                is_spec = is_paramspec or is_paramspecdict
                
                # handle nested specs within a paramspecs value, nested specs are not preserved
                if is_paramspec:
                    _param_spec = param_spec.get_value()
                elif is_paramspecdict:
                    _param_spec = param_spec['current_value']
                else:
                    _param_spec = param_spec 

                if _param_spec is None: # handle case where value of plugin spec is none 
                    _handle_get_value(flat_schema, param_spec, False, value_type, [header, header_header])

                else:
                    # print(header, header_header, param_spec) 
                    for k, v in _param_spec.items():
                        for kk, vv in v.items():
                            if isinstance(vv, dict):
                                for kkk, vvv in vv.items():
                                    _handle_get_value(flat_schema, vvv, is_spec, value_type, [header, header_header, k, kk, kkk])
                            else:
                                _handle_get_value(flat_schema, vv, is_spec, value_type, [header, header_header, k, kk])
        else:
            for header_header, param in spec_dict.items():
                _handle_get_value(flat_schema, param, False, value_type, [header, header_header])
                
    return flat_schema

def _handle_get_value(flat_schema, param, is_spec, value_type, headers) -> None:
    """ help determine whether to return spec's value or spec it's self """
    
    if is_spec: 
        headers.insert(2, 'current_value')
    
    SCOPE_DEMARCATOR = '.'
    scope = SCOPE_DEMARCATOR.join(headers)
    
    flat_schema[scope] = get_spec_value(param) if value_type == 'value' else param

def get_spec_value(param):
    """ helper function to extract a value from different inputs (spec_dict, ParamSpec, or a value it's self)"""
    if isinstance(param, dict):
        if param.get('current_value') is not None:
            return param.get('current_value')
        elif 'default_value' in param:
            return param['default_value']
    elif isinstance(param, ParamSpec):
        return param.get_value()
    
    return param # backwards compat so nested values can be processed aswell

def get_specs_by_scope(default_specs):
    """ deprecated in favor of flatten_paramspec_schema """
    specs_by_scope: Dict[str, Dict[str, ParamSpec]] = {}
    def walk(node: Mapping[str, Any], scope: str):
        for k, v in node.items():
            if isinstance(v, Mapping) and not any(h in v for h in ('default_value','widget_type','flags')):
                walk(v, scope + (k,))
            else:
                if isinstance(v, Mapping) and ('default_value' in v or 'widget_type' in v or 'flags' in v):
                    default_attrs = {'default_value','widget_type','tooltip','heading','flags'}
                    
                    ps = ParamSpec(
                        name=k, scope=scope,
                        default_value=v.get('default_value'),
                        widget_type=v.get('widget_type'),
                        tooltip=v.get('tooltip'),
                        heading=v.get('heading'),
                        flags=cls.parse_flags(v.get('flags')),
                        extra={kk: vv for kk, vv in v.items() if kk not in default_attrs},
                    )
                    specs_by_scope.setdefault(scope, {})[k] = ps
                else:
                    pass
    walk(default_specs, ())

def prune_flatened_struct(flat_param_values:dict[str, Any], to_replace: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """ to_replace: dict mapping strings to replace in flat_param_values.values """
    if to_replace is None or len(to_replace) == 0:
        return flat_param_values
    _out = {}
    for nested_key, value in flat_param_values.items():
        for replace, _with in to_replace.items():
            if replace in nested_key:
                nested_key = nested_key.replace(replace, _with)
        _out[nested_key] = value
    return _out 



def unflatten_dict(
    flat: Dict[str, Any],
    *,
    sep: str = ".",
    on_conflict: str = "error",  # "error" | "overwrite" | "skip"
) -> Dict[str, Any]:
    """
    Convert a flattened dict with dotted keys into a nested dict.

    - sep: path separator in keys
    - on_conflict:
        * "error"     -> raise if we need to create a dict under a non-dict, or if a leaf already exists
        * "overwrite" -> replace existing values with the new one
        * "skip"      -> keep the first value, ignore subsequent writes at the same path
    """
    root: Dict[str, Any] = {}

    for dotted_key, value in flat.items():
        if not isinstance(dotted_key, str) or dotted_key == "":
            continue
        parts = dotted_key.split(sep)

        curr = root
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)

            if is_last:
                if part in curr:
                    # conflict at the leaf
                    if isinstance(curr[part], dict) and isinstance(value, dict):
                        curr[part].update(value)
                    elif on_conflict == "overwrite":
                        curr[part] = value
                    elif on_conflict == "skip":
                        pass  # keep existing
                    else:  # "error"
                        raise ValueError(f"Conflict at '{dotted_key}': key already set.")
                else:
                    curr[part] = value
            else:
                nxt = curr.get(part)
                if nxt is None:
                    nxt = {}
                    curr[part] = nxt
                elif not isinstance(nxt, dict):
                    # we'd need to go deeper but found a non-dict
                    if on_conflict == "overwrite":
                        nxt = {}
                        curr[part] = nxt
                    elif on_conflict == "skip":
                        # stop descending; ignore the rest of this dotted key
                        break
                    else:  # "error"
                        raise ValueError(
                            f"Cannot create nested key below non-dict at '{sep.join(parts[:i+1])}'."
                        )
                curr = nxt
    return root

def prepend_scope(base_scope:str, flat_params: Dict[str, Dict | Any]):
    """ prepend a global scope prefix to a locally scoped flat params dict
        
        example:
            base_scope: 'Model.MODEL_PARAMS.current_value'
            flat_params: {'stardist2d.root.model_path' = {'default_value': '', ... }}
            prepend_scope(...)
            -> {'Model.MODEL_PARAMS.current_value.stardist2d.root.model_path' = {'default_value': '', ... }}
    """
    rescoped = {}
    for local_scope, param in flat_params.items():
        new_scope = f"{base_scope}.{local_scope}"
        rescoped[new_scope] = param
    return rescoped



# example test 
#####################################################
if bool(0):
    
    ex_spec_dict = {
        'Run Configuration': {
            'IMG_DIR': {
                'default_value': None,
                'widget_type': 'directory',
                'tooltip': 'Directory containing images.',
                'heading': 'Run Configuration',
                'current_value': None}},
        'Model': {
            'MODEL_PARAMS': {
                'default_value': None,
                'widget_type': 'SegConfigModelsField',
                'tooltip': 'model specific configuration attributes - see model for specifics',
                'heading': 'Model',
                'current_value': {
                    'stardist_3d': {
                        'root': {
                                'in_dims_model': {'default_value': None,
                                'widget_type': 'str',
                                'tooltip': "Expected input dimension format for model.predict (e.g., for a 2d model 'YX' or a 3d model 'ZYX')",
                                'flags': ['hidden'],
                                'group': 'root',
                                'current_value': 'ZYX'}},
                            'predict_kwargs': {
                                'scale': {'default_value': 1.0,
                                'widget_type': 'float',
                                'tooltip': 'Scale factor for prediction',
                                'group': 'predict_kwargs',
                                'current_value': None}}
                        },
                    'stardist_2d': {
                        'root': {
                                'in_dims_model': {'default_value': None,
                                'widget_type': 'str',
                                'tooltip': "Expected input dimension format for model.predict (e.g., for a 2d model 'YX' or a 3d model 'ZYX')",
                                'flags': ['hidden'],
                                'group': 'root',
                                'current_value': 'ZYX'}},
                            'predict_kwargs': {
                                'scale': {'default_value': 1.0,
                                'widget_type': 'float',
                                'tooltip': 'Scale factor for prediction',
                                'group': 'predict_kwargs',
                                'current_value': None}}
                            }
                        }}}
    }

    ex_values_only = {
        'Run Configuration': {'IMG_DIR': None},
        'Model': {'MODEL_PARAMS': {'stardist_3d': {'predict_kwargs': {'scale':1.0}, 'root': {'in_dims_model': 'ZYX'}}, 'stardist_2d': {'predict_kwargs': {'scale':1.0}, 'root': {'in_dims_model': 'ZYX'}}}}
    }

    rootless_plugin_values_only = {
        'IMG_DIR': None,
        'MODEL_PARAMS': {'stardist_3d': {'predict_kwargs': {'scale':1.0}, 'in_dims_model': 'ZYX'}, 'stardist_2d': {'predict_kwargs': {'scale':1.0}, 'in_dims_model': 'ZYX'}}
    }

    # test 
    converted_vals_only = specs_to_values(ex_spec_dict)
    run_params = remove_toplevel_headers(converted_vals_only)
    final_params = unroot_plugin_params(run_params, plugin_key='MODEL_PARAMS')

    assert converted_vals_only == ex_values_only
    assert rootless_plugin_values_only == final_params
    assert run_config_socket(ex_spec_dict, plugin_key='MODEL_PARAMS') == final_params


    # flat to nested
    flat_values = {
        'Model.MODEL_PARAMS.current_value.neurseg.predict_kwargs.BinPredThresh': 0.5, 
        'Model.MODEL_PARAMS.current_value.neurseg.predict_kwargs.patch_shape': [256,  256]
    }
    pruned_nested_dict = {"MODEL_PARAMS": {"neurseg":{"predict_kwargs":{"BinPredThresh":0.5, "patch_shape": [256,256]}}}}

    pruned_values = prune_flatened_struct(flat_values, to_replace={'Model.':'', 'current_value.':''})
    unflattened = unflatten_dict(pruned_values)
    assert unflattened == pruned_nested_dict




