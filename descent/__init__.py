"""
descent

Optimize classical force field parameters against reference data
"""

from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["__version__"]
