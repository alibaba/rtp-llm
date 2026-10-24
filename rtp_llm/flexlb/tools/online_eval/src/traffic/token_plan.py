"""Read token-plan JSONL into prefix identities for offline diagnostics."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct


@dataclass(frozen=True)
class TokenPlanRequest:
    arrival_ms: float
    input_tokens: int
    block_size: int
    prefix_keys: tuple[bytes, ...]
    family: str


def read_token_plan(path):
    """Preserve full-block prefix hashing for explicit and compact token plans."""
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            block = row["cache_key_block_size"]
            if 'input_token_blocks' in row:
                physical = row['input_token_blocks'][:row['il']//block]
                blocks = (v if isinstance(v, list) else [v]*block for v in physical)
            else:
                blocks = (row['input_ids'][offset:offset+block]
                          for offset in range(0, len(row['input_ids'])-block+1, block))
            key, keys = b"", []
            for values in blocks:
                key = hashlib.sha256(key + struct.pack("<" + "i" * block, *values)).digest()
                keys.append(key)
            yield TokenPlanRequest(row["ts"], row["il"], block, tuple(keys),
                                   row.get("family", "unknown"))
