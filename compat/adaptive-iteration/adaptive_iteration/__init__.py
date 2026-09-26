"""adaptive_iteration has been renamed to ordal. This shim keeps old imports working.

Every old module path points at its ordal counterpart, including the former
``core`` subpackage (``adaptive_iteration.core.decision`` → ``ordal.decision``).
"""
import importlib
import sys
import types
import warnings

warnings.warn(
    "adaptive_iteration has been renamed to ordal: pip install ordal, then "
    "`import ordal` (the `core` subpackage is gone: adaptive_iteration.core.X → ordal.X)",
    DeprecationWarning,
    stacklevel=2,
)

from ordal import *  # noqa: E402,F401,F403
from ordal import __all__, __version__  # noqa: E402,F401

_CORE = ["assignment", "config", "decision", "domain", "evaluator", "evidence",
         "experiment", "hypothesis", "ledger", "lifecycle", "metrics", "registry",
         "screening", "shrinkage"]
_TOP = ["cli", "loop", "mcp_server", "migrate", "replay", "service",
        "adapters", "adapters.base", "adapters.short_video"]

core = types.ModuleType(f"{__name__}.core", "former adaptive_iteration.core; see ordal")
core.__path__ = []
sys.modules[core.__name__] = core
for _name in _CORE:
    _mod = importlib.import_module(f"ordal.{_name}")
    sys.modules[f"{__name__}.core.{_name}"] = _mod
    setattr(core, _name, _mod)
for _name in _TOP:
    _mod = importlib.import_module(f"ordal.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    if "." not in _name:
        globals()[_name] = _mod
