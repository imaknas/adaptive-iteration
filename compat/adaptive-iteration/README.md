# adaptive-iteration → ordal

This project is now **Ordal**: `pip install ordal`, `import ordal`.
See https://github.com/imaknas/ordal.

This package only installs `ordal` and makes the old import paths
(`adaptive_iteration`, `adaptive_iteration.core.*`, …) point at the new modules,
with a `DeprecationWarning`. Nothing else changes; ledgers need no conversion.
