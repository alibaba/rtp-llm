#!/usr/bin/env python3
"""List and execute cases through the scenario child protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scenario.child_cli import (
    instance_directory as instance_directory,
)
from scenario.child_cli import (
    main as main,
)
from scenario.child_cli import (
    select as select,
)
from scenario.child_cli import (
    summarize as summarize,
)

if __name__ == "__main__":
    raise SystemExit(main())
