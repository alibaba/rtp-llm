"""Pure policy helpers for NCCL communicator memory transitions."""

import re
from hashlib import sha1
from typing import Sequence, Tuple

NCCL_IN_PROGRESS = 7
_GROUP_INDEX_RE = re.compile(r"\d+$")


def canonical_key(key: str) -> str:
    """Remove the rank-dependent suffix from a collective group key."""
    return _GROUP_INDEX_RE.sub("", key)


def fingerprint(found: Sequence[Tuple]) -> str:
    """Digest the ordered, canonicalized communicator group keys."""
    keys = "|".join(canonical_key(row[0]) for row in found)
    return sha1(keys.encode("utf-8")).hexdigest()[:12]


def rc_detail(key: str, rc: int) -> str:
    """Explain NCCL return codes in lifecycle error messages."""
    if rc == NCCL_IN_PROGRESS:
        return (
            f"{key}(rc={rc} ncclInProgress -- this communicator is non-blocking, "
            "so the call SUCCEEDED and merely needs polling, which is not "
            "implemented. capability() only screens the environment, so this one "
            "was most likely configured via pg_options.config.blocking)"
        )
    return f"{key}(rc={rc})"
