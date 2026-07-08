"""Plan contract: a mapping from capture bs bucket (int) to backend class name (str).

The dispatcher computes an in-memory Plan fresh on each startup and does not
persist it; this class is the selector's output and also the lookup table used by
prepare_fmha_impl during capture. Pure CPU (dict plus json), no GPU import.

The disk-cache conventions such as cache_path, load_or_none, and dump are kept
for a future optional offline cache; the dispatcher does not use them in this
round. If the micro-benchmark overhead grows later, they can be enabled, at which
point the key must include kv_cache_dtype, and a plan hit must still pass the
current gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Plan:
    """Mapping from bs bucket -> backend class name, serializable to json."""

    assignments: Dict[int, str] = field(default_factory=dict)
    note: str = ""

    def backend_for(self, bucket: int) -> Optional[str]:
        """Look up the backend for a capture bucket; returns None if uncovered (handled by capture itself, e.g. fall back to fixed priority)."""
        return self.assignments.get(int(bucket))

    def buckets(self) -> List[int]:
        return sorted(self.assignments)

    # ── Serialization (json keys must be str, converted on read/write) ──────────────────────────────
    def to_dict(self) -> dict:
        return {
            "assignments": {str(k): v for k, v in self.assignments.items()},
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(
            assignments={int(k): str(v) for k, v in d.get("assignments", {}).items()},
            note=str(d.get("note", "")),
        )

    def dump(self, path: str) -> None:
        """Overwrite-write; auto-creates the parent directory. Not persisted in this round, kept for a future offline cache."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Plan":
        with open(path) as f:
            return cls.from_dict(json.load(f))
