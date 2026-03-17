import pprint

class DatasetNode:
    def __init__(self, data: dict):
        """ converts a nested dictionary to an object where arbitray nested keys are indexable via object attribute notation
            e.g. data.config.initializers.start_index = x
        """
        # store the backing dict
        super().__setattr__('_data', data)

        self.keys = list(self._data.keys())

    def __getattr__(self, name):
        if name in self._data:
            val = self._data[name]
            # if it's a dict, wrap it in another DatasetNode
            if isinstance(val, dict):
                return DatasetNode(val)
            return val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == '_data':
            # allow initialization of _data
            super().__setattr__(name, value)
        elif name in self._data:
            # set existing key in dict
            self._data[name] = value
        else:
            # fallback to normal setattr for other attributes
            super().__setattr__(name, value)

    def __str__(self):
        """ str representation, with nested values expanded """
        asstr = ''
        for k,v in self._data.items():
            asstr += (f"{k}")
            v = flatten_dict(v,sep='.') if isinstance(v,dict) else v
            asstr += (f"{pprint.pformat(v, indent=2, compact=True)}")
        return asstr

    def __repr__(self):
        return f"{type(self).__name__}({pprint.pformat(self._data, indent=2, compact=True, width=100)})"
    
    def to_dict(self):
        return self._data


    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def items(self):
        for k, v in self._data.items():
            yield (k, DatasetNode(v) if isinstance(v, dict) else v)

    def keys(self):
        return self._data.keys()

    def values(self):
        for v in self._data.values():
            yield DatasetNode(v) if isinstance(v, dict) else v    


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


# # --- example usage ---

# DATASET_DIR_STRUCTURE = {
#     "images": {
#         "data_type": "image",
#         "main": "images",
#         "dtype": "float32",
#         "input_format": "YX",
#         "output_format": "YX",
#         "project_dimensions":  "STZ",
#         "take_dimensions": None,
#     },
#     "masks": {
#         "data_type": "mask",
#         "main": "masks",
#         "dtype": "int32",
#         "input_format": "YX",
#         "output_format": "YX",
#         "ROI_CH": None,
#         "project_dimensions":  "STZ",
#         "take_dimensions": None,
#     },
# }

# # create the object
# myclass = Dataset(DATASET_DIR_STRUCTURE)

# # get a value
# print(myclass.images.input_format)   # -> "YX"

# # set a value
# myclass.images.input_format = "ZYX"
# print(myclass.images.input_format)   # -> "ZYX"

# # underlying dict is updated too
# assert DATASET_DIR_STRUCTURE['images']['input_format'] == "ZYX"
