from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

# this hasn't been implemented, leaving as framework for communicating errors clearly

@dataclass
class ParamContext:
    param: str                                      # name
    scopes: Optional[str] = None
    expected: Optional[str] = None
    received: Optional[Any] = None
    source: Optional[str] = None                    # default|user|data
    hint: Optional[str] = None
    docs_url: Optional[str] = None


class ParamEngineError(Exception):
    """Base class for param_engine exceptions."""
    pass

class MissingRequiredParamError(ParamEngineError):
    def __init__(self, ctx: ParamContext):
        super().__init__(f"Missing required parameter '{ctx.param}'. {('Hint: ' + ctx.hint) if ctx.hint else ''}")
        self.ctx = ctx

class ParamValidationError(ParamEngineError):
    def __init__(self, ctx: ParamContext):
        msg = [f"Validation failed for '{ctx.param}'."]
        if ctx.expected: msg.append(f"Expected: {ctx.expected}.")
        if ctx.received is not None: msg.append(f"Received: {ctx.received!r}.")
        if ctx.source: msg.append(f"Source: {ctx.source}.")
        if ctx.hint: msg.append(f"Hint: {ctx.hint}.")
        super().__init__(' '.join(msg))
        self.ctx = ctx

class DataResolutionError(ParamEngineError):
    def __init__(self, ctx: ParamContext):
        msg = [f"Data resolution failed for '{ctx.param}'."]
        if ctx.hint: msg.append(f"Hint: {ctx.hint}.")
        super().__init__(' '.join(msg))
        self.ctx = ctx
