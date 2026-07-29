"""Thin launcher so the data-layer CLI runs without installing the package.

Running ``python src/prepare_baf_data.py`` puts ``src/`` on ``sys.path``
automatically, which makes ``baf_data`` importable.
"""

import sys

from baf_data.cli import main

if __name__ == "__main__":
    sys.exit(main())
