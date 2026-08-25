"""Summary loading helper for FlexLB online evaluation.

Markdown report generation moved to the Java client
(`org.flexlb.mockengine.JavaLoadClient#writeMarkdownReport`); only the shared
summary loading helper is still consumed by `generate_stability_report.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_summary(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
