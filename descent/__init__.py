"""
descent

Optimize classical force field parameters against reference data
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("descent")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
