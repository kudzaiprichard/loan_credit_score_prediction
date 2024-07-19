"""Module-alias shim so models pickled under `src.*` load with ONLY this package.

A scikit-learn Pipeline stores references to the fully-qualified import paths of
its custom steps (e.g. `src.preprocessing.GroupStatsEncoder`). To unpickle such a
model without the `src/` tree present, we point those import paths at this
package's identical implementations. Standard, safe technique for portable models.
"""
from __future__ import annotations

import sys
import types


def register_aliases() -> None:
    from . import config, transforms

    if "src" not in sys.modules:
        pkg = types.ModuleType("src")
        pkg.__path__ = []  # mark as package
        sys.modules["src"] = pkg

    # All transform symbols live in `transforms`; map the src submodules to it.
    for name in ("src.cleaning", "src.feature_engineering", "src.preprocessing"):
        sys.modules[name] = transforms
    sys.modules["src.config"] = config
