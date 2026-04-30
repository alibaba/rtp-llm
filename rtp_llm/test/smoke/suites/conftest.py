"""OSS suites conftest — mounts sibling smoke/ dir and pins data root."""

import os
import sys
from pathlib import Path

# Mount `rtp_llm/test/` so the `smoke.*` namespace package (PEP 420) resolves:
# case_runner, common_def, comparers, etc.
_test_dir = str(Path(__file__).resolve().parents[2])
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

# Pin REL_PATH to OSS data tree before any comparer captures it.
from smoke.common_def import set_default_data_root  # noqa: E402

set_default_data_root("oss")
