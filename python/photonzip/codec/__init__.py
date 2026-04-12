from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules


def discover_codecs() -> None:
    for module_info in iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")


discover_codecs()
__all__ = ["discover_codecs"]
