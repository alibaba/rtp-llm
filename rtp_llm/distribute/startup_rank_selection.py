"""Select complete GPU hosts after the original gang has finished TCP rendezvous."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence, cast

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs
    from rtp_llm.distribute.distributed_server import DistributedServer
    from rtp_llm.ops import ParallelismConfig

STARTUP_RANK_SELECTION_ERROR_QPS = "py_rtp_startup_rank_selection_error_qps"
STARTUP_RANK_SELECTION_FALLBACK_SUCCESS_QPS = (
    "py_rtp_startup_rank_selection_fallback_success_qps"
)


class RankSelectionError(RuntimeError):
    """The original gang cannot form a serving topology from its healthy hosts."""


def choose_hosts(healthy: Sequence[bool], pc: ParallelismConfig) -> list[int]:
    """Keep healthy hosts in their original order, subject to the parallel sizes."""
    local = pc.local_world_size
    if pc.pp_size * pc.tp_size > local:
        raise RankSelectionError(
            "Host selection requires PP_SIZE * TP_SIZE <= LOCAL_WORLD_SIZE "
            f"(PP_SIZE={pc.pp_size}, TP_SIZE={pc.tp_size}, LOCAL_WORLD_SIZE={local})"
        )
    live = [index for index, ok in enumerate(healthy) if ok]
    if not live:
        raise RankSelectionError("No fully healthy host")
    if len(live) == len(healthy):
        return live
    if local % pc.tp_size != 0:
        raise RankSelectionError(
            "Removing a host requires complete local TP groups "
            f"(LOCAL_WORLD_SIZE={local}, TP_SIZE={pc.tp_size})"
        )
    if pc.pp_size != 1 or pc.ep_size == 1:
        raise RankSelectionError(
            "Removing a host requires PP_SIZE=1 and EP_SIZE different from 1 "
            f"(PP_SIZE={pc.pp_size}, EP_SIZE={pc.ep_size})"
        )
    return live


def _log_plan(
    host_addresses: Sequence[str],
    selected: Sequence[int],
    pc: ParallelismConfig,
) -> None:
    local = min(pc.local_world_size, pc.world_size)
    new_indices = {old: new for new, old in enumerate(selected)}
    rows = ["OLD_RANKS    NEW_RANKS    STATE          HOST"]
    for index, address in enumerate(host_addresses):
        old = index * local
        old_range = f"{old}..{old + local - 1}"
        new_range = "-"
        state = "unhealthy"
        if index in new_indices:
            new = new_indices[index] * local
            new_range = f"{new}..{new + local - 1}"
            state = "active"
        rows.append(f"{old_range:<12} {new_range:<12} {state:<14} {address}")
    logging.info(
        "Startup rank reorder | WORLD=%d -> %d | TP=%d | LOCAL_WORLD_SIZE=%d "
        "| ORIGINAL_RANK=%d\n%s",
        pc.world_size,
        len(selected) * local,
        pc.tp_size,
        pc.local_world_size,
        pc.world_rank,
        "\n".join(rows),
    )


def exchange_host_health(
    configs: PyEnvConfigs, rendezvous: DistributedServer
) -> list[bool]:
    """Return the same health bitmap to every launcher, in original host order."""
    from torch.distributed.elastic.utils.store import get_all

    pc = configs.parallelism_config
    ids = configs.distribute_config.startup_healthy_gpu_ids
    local_slots = range(min(pc.local_world_size, pc.world_size))
    healthy = ids is None or set(local_slots).issubset(ids)
    store = rendezvous.store
    if store is None:
        return [healthy]

    rank = rendezvous.host_index
    prefix = "startup_host_selection/healthy/"
    store.set(f"{prefix}{rank}", "1" if healthy else "0")
    flags = cast(
        list[bytes], get_all(store, rank, prefix, len(rendezvous.host_addresses))
    )
    return [flag == b"1" for flag in flags]


def _report_qps(metric: str, **tags: str) -> None:
    from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import report_worker

    report_worker.report_once(metric, 1, {"is_serving": "false", **tags})


def select_startup_hosts(
    health: Sequence[bool], pc: ParallelismConfig, host_addresses: Sequence[str]
) -> list[int]:
    """Each launcher selects and logs its own plan from the shared health bitmap."""
    report_fallback = pc.world_rank == 0 and not all(health)
    if report_fallback:
        _report_qps(STARTUP_RANK_SELECTION_ERROR_QPS)
    selected = choose_hosts(health, pc)
    if report_fallback:
        _report_qps(
            STARTUP_RANK_SELECTION_FALLBACK_SUCCESS_QPS,
            host_count_change=f"[{len(health)}->{len(selected)}]",
        )
    _log_plan(host_addresses, selected, pc)
    return selected
