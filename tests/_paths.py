"""Where the committed example data lives, anchored on this file.

`DATA` is derived from `__file__`, not from the working directory, so tests resolve it
the same way whether pytest was started from the repository root, from `tests/`, or
from an editor's run button.
"""

from pathlib import Path

# tests/_paths.py -> tests/ -> the repository root.
DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
