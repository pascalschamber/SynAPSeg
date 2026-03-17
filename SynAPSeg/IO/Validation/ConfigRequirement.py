from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass
from .DataType import DataType

@dataclass 
class ConfigRequirement:
    """Specification for a configuration parameter."""
    key: str
    data_type: DataType
    required: bool = True
    default_value: Any = None
    description: str = ""
    validator: Optional[Callable] = None
    
    def extract(self, config: Dict[str, Any]) -> Any:
        """Extract and validate configuration value."""
        if self.key not in config:
            if self.required and self.default_value is None:
                raise ValueError(f"Required config key '{self.key}' missing")
            return self.default_value
        
        value = config[self.key]
        
        # Apply validator if provided
        if self.validator and not self.validator(value):
            raise ValueError(f"Config key '{self.key}' failed validation")
        
        return value