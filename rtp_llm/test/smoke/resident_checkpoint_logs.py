"""Identify physical checkpoint imports, not resident attachments, in server logs."""

import re


_COLD_READ = re.compile(
    r"(?:\[checkpoint_owner=([^\]\s]+)\] )?"
    r"(?:Streamed checkpoint tensors with |NewLoader imported (?:Meta consolidated|rank-local FT) checkpoint)"
)


def cold_checkpoint_owners(logs: str) -> set[str]:
    # Treat untagged reads as unexpected rather than silently losing coverage.
    return {match.group(1) or "<untagged>" for match in _COLD_READ.finditer(logs)}
