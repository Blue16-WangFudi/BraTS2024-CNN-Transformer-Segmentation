from __future__ import annotations

import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__all__ = ["__version__"]
__version__ = "0.1.0"
