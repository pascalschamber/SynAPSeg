"""
Base class for all pipeline components
Implements common methods
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional
from SynAPSeg.utils import utils_general as ug

class BasePipelineComponent(ABC):
    """
    Base class for all components in a pipeline
        e.g. quantification pipeline, stages, etc.

    """

    def attach_logger(self, logger: Optional[logging.Logger]=None):
        """ Logger setup using a shared logger or creates one if logger=None"""
        self._shared_logger = logger
        
        if self._shared_logger is not None: # Wrap the shared logger with a LoggerAdapter that injects stage="..."
            self.logger = (
                logging.LoggerAdapter(self._shared_logger, {"stage": self.__class__.__name__})
            )
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def set_start_time(self):
        self.start_time = ug.dt()
        self.completed = False

    def set_end_time(self):
        self.end_time = ug.dt()
        self.start_time = self.start_time if hasattr(self, 'start_time') else 0
        self.elapsed_time = self.end_time - self.start_time

    def set_complete(self):
        self.completed = True
        self.set_end_time()
        self.logger.info(f"{self.__class__.__name__} complete. Elapsed time: {round((self.elapsed_time), 2)} seconds.\n") 