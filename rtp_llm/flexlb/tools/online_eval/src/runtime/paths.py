"""Repository paths and the selected Java artifacts."""

from __future__ import annotations

import os
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[2]  # rtp_llm/flexlb/tools/online_eval

FLEXLB_DIR = Path(__file__).resolve().parents[4]  # rtp_llm/flexlb (maven root)

REPO_ROOT = Path(__file__).resolve().parents[6]  # repo root

MOCK_JAR = (
    FLEXLB_DIR
    / "flexlb-mock-engine"
    / "target"
    / "flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar"
)

API_JAR = Path(os.environ.get(
    "FLEXLB_FT_MASTER_JAR",
    str(FLEXLB_DIR / "flexlb-api" / "target" / "flexlb-api-1.0.0-SNAPSHOT.jar"),
))
