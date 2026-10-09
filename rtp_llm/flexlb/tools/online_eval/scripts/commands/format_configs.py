#!/usr/bin/env python3
"""Check or normalize the reading order of authored project YAML."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pipeline.config_order import main

if __name__ == '__main__':
    raise SystemExit(main())
