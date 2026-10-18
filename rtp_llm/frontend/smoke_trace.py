"""Bounded request identifiers for optional serving-path evidence."""

import re
from typing import Mapping, Optional


def frontend_request_fields(
    headers: Mapping[str, str], request_id: int
) -> Optional[dict]:
    """Read only a bounded case label; never log the request body or headers."""
    case = headers.get("x-rtp-smoke-case")
    if case is None:
        case = headers.get("x-k3-smoke-case", "")
    if (
        not isinstance(case, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", case) is None
    ):
        return None
    return {"case": case, "request_id": request_id}
