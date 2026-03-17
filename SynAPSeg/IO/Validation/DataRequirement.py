from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from . DataType import DataType




@dataclass
class DataRequirement:
    """Specification for a data dependency."""
    key: str
    data_type: DataType
    required: bool = True
    description: str = ""
    shape_constraint: Optional[Callable] = None
    value_constraint: Optional[Callable] = None
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate that the requirement is satisfied by the data."""
        if self.key not in data:
            if self.required:
                return False, f"Required key '{self.key}' missing from data"
            return True, ""
        
        value = data[self.key]
        
        # Type validation
        if self.data_type == DataType.DATAFRAME and not isinstance(value, pd.DataFrame):
            return False, f"Key '{self.key}' must be pandas DataFrame, got {type(value)}"
        elif self.data_type == DataType.NUMPY_ARRAY and not isinstance(value, np.ndarray):
            return False, f"Key '{self.key}' must be numpy array, got {type(value)}"
        elif self.data_type == DataType.LIST and not isinstance(value, list):
            return False, f"Key '{self.key}' must be list, got {type(value)}"
        elif self.data_type == DataType.DICT and not isinstance(value, dict):
            return False, f"Key '{self.key}' must be dict, got {type(value)}"
        elif self.data_type == DataType.STRING and not isinstance(value, str):
            return False, f"Key '{self.key}' must be string, got {type(value)}"
        elif self.data_type == DataType.INTEGER and not isinstance(value, int):
            return False, f"Key '{self.key}' must be integer, got {type(value)}"
        elif self.data_type == DataType.FLOAT and not isinstance(value, (int, float)):
            return False, f"Key '{self.key}' must be float, got {type(value)}"
        elif self.data_type == DataType.BOOLEAN and not isinstance(value, bool):
            return False, f"Key '{self.key}' must be boolean, got {type(value)}"
        elif self.data_type == DataType.PATH and not isinstance(value, (str, Path)):
            return False, f"Key '{self.key}' must be path-like, got {type(value)}"
        
        # Shape constraint validation
        if self.shape_constraint and hasattr(value, 'shape'):
            if not self.shape_constraint(value.shape):
                return False, f"Key '{self.key}' failed shape constraint"
        
        # Value constraint validation  
        if self.value_constraint and not self.value_constraint(value):
            return False, f"Key '{self.key}' failed value constraint"
        
        return True, ""


