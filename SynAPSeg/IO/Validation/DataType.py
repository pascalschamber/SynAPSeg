from enum import Enum

class DataType(Enum):
    """Enumeration of supported data types for validation."""
    DATAFRAME = "pandas.DataFrame"
    NUMPY_ARRAY = "numpy.ndarray"
    LIST = "list"
    DICT = "dict"
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    PATH = "pathlib.Path"