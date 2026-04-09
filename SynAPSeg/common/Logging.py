import logging
import os
from pathlib import Path
from typing import Optional
import sys
from rich.logging import RichHandler

def setup_default_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            show_path=False,   # Keeps the logs clean
            markup=True,       # Allows you to use [bold red] etc. in strings
            rich_tracebacks=True
        )
        # handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)

def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    level=logging.INFO
) -> logging.Logger:
    
    if log_dir is None:
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        log_filename = f"{name}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Console handler
    # stream_handler = logging.StreamHandler()
    stream_handler = RichHandler(
            show_path=False,   # Keeps the logs clean
            markup=True,       # Allows you to use [bold red] etc. in strings
            rich_tracebacks=True
    )
    stream_fmt = SafeStageFormatter("%(levelname)s: %(name)s - [%(stage)s] - %(message)s")
    stream_handler.setFormatter(stream_fmt)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_fmt = SafeStageFormatter("%(asctime)s %(levelname)s: %(name)s - [%(stage)s] - %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger

class SafeStageFormatter(logging.Formatter):
    """Formatter that ensures %(stage)s always exists."""
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "stage"):
            record.stage = "-"
        return super().format(record)
    

class StageAwareFormatter(logging.Formatter):
    def __init__(self, default_fmt, stage_fmts=None, datefmt=None):
        """ Optional: stage-specific formatting """
        super().__init__(default_fmt, datefmt=datefmt)
        self.default_fmt = default_fmt
        self.stage_fmts = stage_fmts or {}
        self._formatters = {
            "__default__": logging.Formatter(default_fmt, datefmt=datefmt),
            **{k: logging.Formatter(v, datefmt=datefmt) for k, v in self.stage_fmts.items()},
        }

    def format(self, record):
        if not hasattr(record, "stage"):
            record.stage = "-"
        fmt = self._formatters.get(record.stage, self._formatters["__default__"])
        return fmt.format(record)

# Example of swapping in stage-specific formatting (optional):
def make_stage_aware(logger: logging.Logger):
    """
    Replace existing handler formatters with StageAwareFormatter.
    Modify `stage_fmts` to taste.
    """
    stage_fmts = {
        "Load":    "[%(levelname)s] %(name)s | 📥 %(stage)s | %(message)s",
        "Process": "[%(levelname)s] %(name)s | ⚙️  %(stage)s | %(message)s",
        "Analyze": "[%(levelname)s] %(name)s | 📊 %(stage)s | %(message)s",
    }
    default_fmt = "[%(levelname)s] %(name)s | %(stage)s | %(message)s"

    for h in logger.handlers:
        # preserve datefmt if present
        datefmt = None
        if isinstance(h.formatter, logging.Formatter):
            datefmt = h.formatter.datefmt
        h.setFormatter(StageAwareFormatter(default_fmt, stage_fmts, datefmt=datefmt))
        
def attach_logging_adapter(shared_logger, record_attrs: dict = {'stage': '-'}):
    """ 
    used when passing a shared logger between multiple objects to reflect origins of log msgs 
        this is mostly here for demonstration

    returns 
        a logger that reflects the child class as msg source

    example usage
        class BaseStage:
        def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
            self.stage_name = stage_name
            self._shared_logger = logger
            # Wrap the shared logger with a LoggerAdapter that injects stage="..."
            self.logger = (
                logging.LoggerAdapter(self._shared_logger, {"stage": self.stage_name})
                if self._shared_logger is not None else None
            )

    """
    return logging.LoggerAdapter(shared_logger, record_attrs)


    

def rich_str(text, title=None):
    """ return str rep of text as if from rich Panel"""
    from io import StringIO
    from rich.console import Console
    from rich.panel import Panel

    # Create a Console instance with a custom file for capturing output
    capture_console = Console(file=StringIO(), force_terminal=False)

    # Create your Panel
    panel = Panel(text, title=title, expand=False)
    buff = capture_console.render(panel)
    txt = capture_console._render_buffer(buff)
    return txt

class rich_console:
    """ print messages to console using rich library's display features 
        TODO: get this to work with logger, e.g. log rich display elements 
    """

    def __init__(self, console=None):
        if not console:
            from rich.console import Console
            console = Console()
        self.console = console

    def __call__(self, text, title=None):
        """ default behavior is to use rich box """
        self.rich_box(text, title=title)

    def rich_box(self, text, title=None):
        from rich.panel import Panel
        panel = Panel(text, title=title, expand=False)
        self.console.print(panel)
    



