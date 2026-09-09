from typing import Any, Dict, List, Optional

UNSUPPORTED_LIFECYCLE_CONTROL_FIELDS = ("phase", "prepare_only", "commit_only")


def unsupported_lifecycle_control_field(req: Dict[Any, Any]) -> Optional[str]:
    for field in UNSUPPORTED_LIFECYCLE_CONTROL_FIELDS:
        if field in req:
            return field
    return None


def dedupe_addresses(addresses: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set = set()
    for address in addresses:
        if address in seen:
            continue
        seen.add(address)
        deduped.append(address)
    return deduped
