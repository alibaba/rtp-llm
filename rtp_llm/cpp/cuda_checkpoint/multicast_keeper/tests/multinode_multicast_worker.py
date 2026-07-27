#!/usr/bin/env python3

import datetime
import gc
import os
import signal
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

_PROTOCOL_MAGIC = 0x3250434D505452
_PROTOCOL_VERSION = 3
_OP_PING = 1
_STATUS_OK = 0
_REQUEST = struct.Struct("<QHHIQQQQIIQ")
_RESPONSE = struct.Struct("<QHHIiIQQQQQIIQ")


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
node_rank = int(os.environ["RTP_MC_TEST_NODE_RANK"])
base_port = int(os.environ["MASTER_PORT"])
success_rounds = int(os.environ.get("RTP_MC_TEST_SUCCESS_ROUNDS", "3"))
failure_node = int(os.environ.get("RTP_MC_TEST_FAIL_HOLDER_NODE", "1"))
holder_pid = int(os.environ["RTP_MC_TEST_HOLDER_PID"])
buffer_bytes = int(os.environ.get("RTP_MC_TEST_BUFFER_BYTES", str(64 * 1024 * 1024)))
timeout = datetime.timedelta(
    seconds=int(os.environ.get("RTP_MC_TEST_DIST_TIMEOUT_SECONDS", "180"))
)
device = torch.device(f"cuda:{local_rank}")

group = None
buffer = None
handle = None
baseline = None
initial_holder_instance: Optional[str] = None
initial_holders_by_node: Dict[int, str] = {}


def keeper_socket_path() -> str:
    explicit = os.environ.get("RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET", "").strip()
    if explicit:
        return explicit
    directory = os.environ["NEKYIA_KEEPER_DIR"]
    return os.path.join(directory, "mcsk.sock")


def ping_holder() -> Tuple[bool, Optional[str], Optional[int], str]:
    request = _REQUEST.pack(
        _PROTOCOL_MAGIC,
        _PROTOCOL_VERSION,
        _OP_PING,
        _REQUEST.size,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    path = keeper_socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.settimeout(2.0)
            client.connect(path)
            client.sendall(request)
            payload = client.recv(_RESPONSE.size + 1)
    except OSError as error:
        return False, None, None, f"{path}: {error}"
    if len(payload) != _RESPONSE.size:
        return False, None, None, f"invalid response size {len(payload)}"
    fields = _RESPONSE.unpack(payload)
    magic, version, opcode, struct_size, status = fields[:5]
    local_device_count = fields[5]
    instance_hi, instance_lo = fields[6:8]
    if (
        magic != _PROTOCOL_MAGIC
        or version != _PROTOCOL_VERSION
        or opcode != _OP_PING
        or struct_size != _RESPONSE.size
        or status != _STATUS_OK
        or (instance_hi == 0 and instance_lo == 0)
    ):
        return False, None, None, "incompatible holder response"
    return (
        True,
        f"{instance_hi:016x}{instance_lo:016x}",
        local_device_count,
        "",
    )


def shim_loaded() -> bool:
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return any("mc_shim_unified" in line for line in maps)


def runtime_metadata(holder_instance: str) -> Dict[str, Any]:
    uuid_list = os.environ["RTP_MC_TEST_LOCAL_GPU_UUIDS"].split(",")
    local_team = os.environ["RTP_MC_TEST_LOCAL_GPUS"].split(",")
    return {
        "rank": rank,
        "world_size": world_size,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "hostname": socket.gethostname(),
        "job_id": os.environ["RTP_MC_TEST_JOB_ID"],
        "role": os.environ["RTP_MC_TEST_ROLE"],
        "local_team": local_team,
        "keeper_local_team": os.environ["RTP_LLM_MC_LOCAL_GPUS"].split(","),
        "keeper_global_team_size": int(os.environ["RTP_LLM_MC_FABRIC_TEAM_SIZE"]),
        "gpu_uuid": uuid_list[local_rank],
        "holder_instance": holder_instance,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": str(torch.cuda.nccl.version()),
        "driver": os.environ["RTP_MC_TEST_DRIVER_VERSION"],
        "fabric": os.environ.get("RTP_MC_TEST_FABRIC_STATUS", ""),
        "visible_devices": torch.cuda.device_count(),
    }


def validate_topology(holder_instance: str) -> None:
    global initial_holders_by_node
    local = runtime_metadata(holder_instance)
    records: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(records, local, group=group)
    errors: List[str] = []
    if rank == 0:
        typed_records = [record for record in records if record is not None]
        if len(typed_records) != world_size:
            errors.append("did not collect metadata from every global rank")
        if sorted(record["rank"] for record in typed_records) != list(
            range(world_size)
        ):
            errors.append("global ranks are not exactly 0..GLOBAL_TEAM_SIZE-1")
        singleton_fields = (
            "world_size",
            "local_world_size",
            "keeper_global_team_size",
            "job_id",
            "role",
            "torch",
            "torch_cuda",
            "nccl",
            "driver",
            "fabric",
        )
        for field in singleton_fields:
            values = {str(record[field]) for record in typed_records}
            if len(values) != 1:
                errors.append(f"{field} differs across ranks: {sorted(values)}")
        if any(
            record["keeper_global_team_size"] != world_size for record in typed_records
        ):
            errors.append("keeper global team size does not equal WORLD_SIZE")
        gpu_uuids = [record["gpu_uuid"] for record in typed_records]
        if len(set(gpu_uuids)) != world_size:
            errors.append("GPU UUIDs are not globally unique")
        node_ranks = sorted({record["node_rank"] for record in typed_records})
        expected_nodes = list(range(int(os.environ["RTP_MC_TEST_NNODES"])))
        if node_ranks != expected_nodes:
            errors.append(f"node ranks {node_ranks} != expected {expected_nodes}")
        holder_instances = set()
        hostnames = set()
        holders_by_node: Dict[int, str] = {}
        for current_node in expected_nodes:
            members = [
                record
                for record in typed_records
                if record["node_rank"] == current_node
            ]
            local_ranks = sorted(record["local_rank"] for record in members)
            expected_local_ranks = list(range(local_world_size))
            if local_ranks != expected_local_ranks:
                errors.append(
                    f"node {current_node} local ranks {local_ranks} "
                    f"!= {expected_local_ranks}"
                )
            teams = {tuple(record["local_team"]) for record in members}
            keeper_teams = {tuple(record["keeper_local_team"]) for record in members}
            holders = {record["holder_instance"] for record in members}
            hosts = {record["hostname"] for record in members}
            visible_counts = {record["visible_devices"] for record in members}
            if (
                len(teams) != 1
                or teams != keeper_teams
                or len(holders) != 1
                or len(hosts) != 1
            ):
                errors.append(
                    f"node {current_node} does not agree on team/holder/hostname"
                )
            if visible_counts != {local_world_size}:
                errors.append(
                    f"node {current_node} visible device counts {visible_counts} "
                    f"!= {{{local_world_size}}}"
                )
            holder_instances.update(holders)
            hostnames.update(hosts)
            if len(holders) == 1:
                holders_by_node[current_node] = next(iter(holders))
        if len(holder_instances) != len(expected_nodes):
            errors.append("each node must use a distinct node-local holder instance")
        if len(hostnames) != len(expected_nodes):
            errors.append("each node rank must report a distinct hostname")
    shared_result: List[Optional[Tuple[List[str], Dict[int, str]]]] = [
        (errors, holders_by_node) if rank == 0 else None
    ]
    dist.broadcast_object_list(shared_result, src=0, group=group)
    assert shared_result[0] is not None
    shared_errors, initial_holders_by_node = shared_result[0]
    if shared_errors:
        raise RuntimeError("; ".join(shared_errors))
    if rank == 0:
        print(
            f"TOPOLOGY_OK world_size={world_size} nodes={os.environ['RTP_MC_TEST_NNODES']} "
            f"local_world_size={local_world_size}",
            flush=True,
        )


def init_generation(generation: int) -> None:
    global group, buffer, handle
    os.environ["MASTER_PORT"] = str(base_port + generation)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timeout,
    )
    group = dist.group.WORLD
    buffer = symm_mem.empty(
        buffer_bytes // torch.tensor([], dtype=torch.bfloat16).element_size(),
        dtype=torch.bfloat16,
        device=device,
    )
    handle = symm_mem.rendezvous(buffer, group=group.group_name)
    torch.cuda.synchronize(device)


def run_collective() -> torch.Tensor:
    value = torch.tensor([rank + 1.0], dtype=torch.float32, device=device)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize(device)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}")
    return value.cpu()


def multicast_pointer() -> int:
    pointer = int(getattr(handle, "multicast_ptr", 0))
    if pointer == 0:
        raise RuntimeError("symmetric-memory rendezvous returned multicast_ptr=0")
    return pointer


def verify_holder_identity() -> str:
    ready, instance, holder_local_size, error = ping_holder()
    if not ready or instance is None:
        raise RuntimeError(f"local holder is unavailable: {error}")
    if holder_local_size != local_world_size:
        raise RuntimeError(
            f"holder local team size {holder_local_size} != {local_world_size}"
        )
    if initial_holder_instance is not None and instance != initial_holder_instance:
        raise RuntimeError(
            f"local holder identity changed: {initial_holder_instance} -> {instance}"
        )
    instances: List[Optional[Tuple[int, str]]] = [None] * world_size
    dist.all_gather_object(instances, (node_rank, instance), group=group)
    by_node: Dict[int, set[str]] = {}
    for current_node, current_instance in instances:
        by_node.setdefault(current_node, set()).add(current_instance)
    if any(len(values) != 1 for values in by_node.values()):
        raise RuntimeError(f"ranks disagree on node-local holder identity: {by_node}")
    return instance


def clear_symmetric_memory_state() -> None:
    for cache_name in (
        "_group_name_to_workspace_tensor",
        "_backend_streams",
        "_symm_mem_pools",
    ):
        cache = getattr(symm_mem, cache_name, None)
        if isinstance(cache, dict):
            cache.clear()


def teardown_generation() -> None:
    global group, buffer, handle
    if group is None:
        return
    dist.barrier(group=group)
    handle = None
    buffer = None
    clear_symmetric_memory_state()
    gc.collect()
    torch.cuda.synchronize(device)
    dist.destroy_process_group(group)
    group = None
    gc.collect()
    torch.cuda.synchronize(device)


def verify_peer_holder_failure() -> None:
    failure_rank = failure_node * local_world_size
    dist.barrier(group=group)
    if rank == failure_rank:
        os.kill(holder_pid, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and ping_holder()[0]:
            time.sleep(0.05)
        if ping_holder()[0]:
            raise RuntimeError("selected holder did not stop after SIGTERM")
    dist.barrier(group=group)

    local_ready, local_instance, _, local_error = ping_holder()
    statuses: List[Optional[Tuple[int, bool, Optional[str], str]]] = [None] * world_size
    dist.all_gather_object(
        statuses,
        (node_rank, local_ready, local_instance, local_error),
        group=group,
    )
    errors = []
    for current_node, ready, instance, error in statuses:
        if current_node == failure_node and ready:
            errors.append(f"node {current_node} still reports its killed holder ready")
        if current_node != failure_node and not ready:
            errors.append(
                f"unrelated node {current_node} holder became unavailable: {error}"
            )
        if (
            current_node != failure_node
            and ready
            and instance != initial_holders_by_node[current_node]
        ):
            errors.append(
                f"unrelated node {current_node} holder identity changed: "
                f"{initial_holders_by_node[current_node]} -> {instance}"
            )
    readiness = torch.tensor(
        [1 if local_ready else 0], dtype=torch.int32, device=device
    )
    dist.all_reduce(readiness, op=dist.ReduceOp.MIN, group=group)
    if readiness.item() != 0:
        errors.append("global readiness did not fail after a peer holder exited")
    shared_errors: List[Optional[List[str]]] = [errors if rank == 0 else None]
    dist.broadcast_object_list(shared_errors, src=0, group=group)
    if shared_errors[0]:
        raise RuntimeError("; ".join(shared_errors[0]))
    print(
        f"FAIL_CLOSED rank={rank} failure_node={failure_node} "
        "phase=pre_teardown_holder_readiness",
        flush=True,
    )


def main() -> None:
    global baseline, initial_holder_instance
    if success_rounds < 2:
        raise RuntimeError("RTP_MC_TEST_SUCCESS_ROUNDS must be at least 2")
    if failure_node < 0 or failure_node >= int(os.environ["RTP_MC_TEST_NNODES"]):
        raise RuntimeError("RTP_MC_TEST_FAIL_HOLDER_NODE is outside the node team")
    if not shim_loaded():
        raise RuntimeError("mc_shim_unified is not present in /proc/self/maps")

    torch.cuda.set_device(device)
    init_generation(0)
    initial_holder_instance = verify_holder_identity()
    validate_topology(initial_holder_instance)
    baseline = run_collective()
    pointer = multicast_pointer()
    print(
        f"READY rank={rank} generation=0 holder={initial_holder_instance} "
        f"multicast_ptr=0x{pointer:x} value={baseline.tolist()}",
        flush=True,
    )

    for generation in range(1, success_rounds + 1):
        verify_holder_identity()
        teardown_generation()
        init_generation(generation)
        current_holder = verify_holder_identity()
        result = run_collective()
        pointer = multicast_pointer()
        if not torch.equal(baseline, result):
            raise RuntimeError(
                f"collective result changed: {baseline.tolist()} -> {result.tolist()}"
            )
        print(
            f"ROUND_OK rank={rank} generation={generation} holder={current_holder} "
            f"multicast_ptr=0x{pointer:x} value={result.tolist()}",
            flush=True,
        )

    verify_peer_holder_failure()
    teardown_generation()
    print(
        f"TEST_PASS rank={rank} successful_rebuilds={success_rounds} "
        f"failure_node={failure_node}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            f"TEST_FAIL rank={rank} type={type(error).__name__} error={error}",
            flush=True,
        )
        raise
