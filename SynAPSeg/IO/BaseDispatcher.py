"""
This module implements a base dispatcher class that is used by both ExampleDispatcher and DispatcherCollection.

# TODO: restructure segmentation dispatcher to inherit from this
"""
from typing import Dict, List, Any
import logging
from pydantic import TypeAdapter, ValidationError as pyValErr

from SynAPSeg.common.Logging import setup_default_logger
from SynAPSeg.Quantification.validation import ValidationError

class DispatcherBase:
    """Mixin-style base class that implements methods shared by ExampleDispatcher and DispatcherCollection."""

    def attach_logger(self, logger=None):
        """Logger setup using a shared logger or creates one if logger=None."""
        self._shared_logger = logger

        if self._shared_logger is not None:
            # Wrap the shared logger with a LoggerAdapter that injects stage="..."
            self.logger = logging.LoggerAdapter(
                self._shared_logger, {"stage": self.__class__.__name__}
            )
        else:
            self.logger = setup_default_logger(self.__class__.__name__)

        self.logger.setLevel(logging.DEBUG)

    def validate_reqs(self, conf:Dict, assert_keys:list[Any], base_msg:str=''):
        """ check required keys are in conf and are not None """
        err = ValidationError(base_msg=base_msg or "")
        for k in assert_keys:
            if k not in conf.keys():
                err.add(f'key `{k}` is not present and is required')
            elif conf.get(k) is None:
                err.add(f'key `{k}` is None')
        if err.errors:
            raise err
    
    def validate_FILEMAP(self, file_map:Dict[str, List[str]], base_msg:str):
        """ handle setup and validation of config FILE_MAP key """

        err = ValidationError(
            expected_type = dict[str, list[str]],
            possible_keys = ["images", "labels", "annotations", "ROIS"], # though can be empty lists and will be added if not present
            base_msg = base_msg,
        )
        

        # Define the adapter
        adapter = TypeAdapter(Dict[str, List[str]])

        try: 
            adapter.validate_python(file_map)
        except pyValErr as e:
            err.reraise_pydantic(e)
            
        
        # after validating input is sound, set any non-present keys to be empty lists 
        for key in err.possible_keys:
            if key not in file_map:
                file_map[key] = []
