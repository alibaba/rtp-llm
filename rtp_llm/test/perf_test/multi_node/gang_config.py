"""Gang config string construction for multi-host perf-test runs.

The builder lives here (not in ``local_server_runner``) so Bazel targets
can import it without pulling the server startup stack and the
internal-source environment machinery along.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.distribute.worker_info import WorkerInfo


def _shard_index_key(member: "WorkerInfo"):
    """Sort key: the shard index encoded in the legacy ``...partN`` name."""
    match = re.search(r"part(\d+)$", member.name)
    if match:
        return (0, int(match.group(1)), member.name)
    return (1, 0, member.name)


def build_gang_config_string(
    members: Sequence["WorkerInfo"], local_world_size: int
) -> str:
    """Build the ``GANG_CONFIG_STRING`` value for one multi-host gang.

    ``members`` are the legacy member records the server side already
    parses (the ``WorkerInfo`` records ``members_from_test_env`` returns).
    The master (the member whose name ends with ``part0``) is placed first
    and the remaining members follow their shard index, matching the
    encoding the legacy SSH launcher emits::

        name:test_part0,ip:host0,port:12333;name:test_part1,ip:host1,port:12333

    ``local_world_size`` is the per-host rank-group width; the caller owns
    the rank layout (world rank = shard index * local world size).
    """
    member_list = list(members)
    if not member_list:
        raise RuntimeError("no gang members, unexpected!")
    if not isinstance(local_world_size, int) or local_world_size < 1:
        raise RuntimeError(
            f"local_world_size must be a positive integer, got {local_world_size!r}"
        )
    masters = [member for member in member_list if member.name.endswith("part0")]
    if len(masters) != 1:
        raise RuntimeError(f"gang master should contains 1 but got {len(masters)}")
    ordered = sorted(member_list, key=_shard_index_key)
    if masters[0].name != ordered[0].name:
        raise RuntimeError(
            f"gang master should be the first one but got {ordered[0].name}"
        )
    return ";".join(
        f"name:{member.name},ip:{member.ip},port:{member.server_port}"
        for member in ordered
    )
