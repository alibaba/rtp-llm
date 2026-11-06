# SPDX-License-Identifier: Apache-2.0
"""Real cross-OS DEVICE/FABRIC lookup and Graph probe, one process per GPU.

Run once on each participating OS with one shared TCPStore endpoint and unique
ranks/output paths. The controller must reserve idle GPUs and hold their locks.
This probe imports the prebuilt native bridge; it never builds extensions,
starts an RTP engine, resets GPUs or terminates other processes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

DIM = 256
LAYER = 1
TIMEOUT_S = 300
_PHASES = ("devices", "checkpoint", "attached", "drained", "released")
_FAILED_RESOURCES = []


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


class _NotRun(RuntimeError):
    pass


class _LookupModel:
    def __init__(self, embedding):
        self.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=embedding))]


class _Group:
    """Bounded startup/test barriers; errors unblock every pending phase."""

    def __init__(self, store, rank, world):
        self.store, self.rank, self.world = store, rank, world
        self.deadline = time.monotonic() + TIMEOUT_S
        self.prefix = "engram/device_fabric_probe/"
        self.completed = set()

    def exchange(self, phase, value):
        self.store.set(
            self.prefix + f"{phase}/{self.rank}",
            json.dumps({"ok": True, "value": value}),
        )
        self.completed.add(phase)
        keys = [self.prefix + f"{phase}/{rank}" for rank in range(self.world)]
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("DEVICE_FABRIC_PROBE_TIMEOUT")
        self.store.wait(keys, timedelta(seconds=remaining))
        values = [json.loads(self.store.get(key)) for key in keys]
        if any(value.get("ok") is not True for value in values):
            raise RuntimeError("DEVICE_FABRIC_PROBE_PEER_FAILED")
        return [value["value"] for value in values]

    def fail(self, error):
        for phase in _PHASES:
            if phase not in self.completed:
                self.store.set(
                    self.prefix + f"{phase}/{self.rank}",
                    json.dumps({"ok": False, "error": type(error).__name__}),
                )

    def finish(self):
        # Keep rank zero's TCPStore server alive until all clients have consumed
        # the final barrier, not merely published their own final value.
        self.store.set(self.prefix + f"finished/{self.rank}", "1")
        if self.rank == 0:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DEVICE_FABRIC_PROBE_TIMEOUT")
            self.store.wait(
                [self.prefix + f"finished/{rank}" for rank in range(self.world)],
                timedelta(seconds=remaining),
            )


def _write_checkpoint(root, plan):
    import torch
    from safetensors.torch import save_file

    weight = torch.empty((plan.total_rows, DIM), dtype=torch.uint8)
    scales = torch.empty((plan.total_rows, DIM // 32), dtype=torch.uint8)
    row_mod = torch.arange(plan.total_rows, dtype=torch.int64) % 4
    fp8_bytes = torch.tensor([0x38, 0x40, 0x44, 0x48], dtype=torch.uint8)
    for block in range(DIM // 32):
        weight[:, block * 32 : (block + 1) * 32] = fp8_bytes[(row_mod + block) % 4][
            :, None
        ]
        for owner in range(plan.shard_count):
            begin, end = plan.row_range(owner)
            # The first three blocks encode all three base-4 owner digits;
            # the block offset keeps the eight scale blocks nonuniform.
            digit = owner // (4 ** (block % 3))
            scales[begin:end, block] = (127, 128, 126, 129)[(digit + block) % 4]
    names = [f"layers.{LAYER}.engram.embed.{kind}" for kind in ("weight", "scale")]
    save_file(
        {names[0]: weight.view(torch.float8_e4m3fn), names[1]: scales},
        str(root / "engram.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": weight.numel() + scales.numel()},
                "weight_map": {name: "engram.safetensors" for name in names},
            },
            sort_keys=True,
        )
    )


def _reference(host_ids, plan):
    """CPU arithmetic oracle for the fixture's FP8 values and E8M0 scales."""
    import torch

    ids = torch.tensor(host_ids, dtype=torch.int64)
    safe = ids.clamp_min(0)
    blocks = torch.arange(DIM, dtype=torch.int64) // 32
    owner = safe // plan.shard_rows
    values = ((safe[..., None] + blocks) % 4 + 1).float()
    factors = torch.tensor([1.0, 2.0, 0.5, 4.0])
    owner_digit = owner[..., None] >> (2 * (blocks % 3))
    values *= factors[(owner_digit + blocks) % 4]
    values[ids < 0] = 0
    return values.to(torch.bfloat16)


def _verify_bytes(storage, plan, device):
    import torch

    rows = []
    for owner in range(plan.shard_count):
        begin, end = plan.row_range(owner)
        rows.extend((begin, (begin + end) // 2, end - 1))
    indices = torch.tensor(rows, dtype=torch.int64, device=device)
    row_ids = torch.tensor(rows, dtype=torch.int64)
    blocks = torch.arange(DIM, dtype=torch.int64) // 32
    fp8 = torch.tensor([0x38, 0x40, 0x44, 0x48], dtype=torch.uint8)
    scale = torch.tensor([127, 128, 126, 129], dtype=torch.uint8)
    expected_weight = fp8[(row_ids[:, None] + blocks) % 4]
    scale_blocks = torch.arange(DIM // 32, dtype=torch.int64)
    owner_ids = row_ids[:, None] // plan.shard_rows
    expected_scales = scale[
        ((owner_ids // (4 ** (scale_blocks % 3))) + scale_blocks) % 4
    ]
    weight, scales = storage.table_views(LAYER)
    torch.testing.assert_close(
        weight.index_select(0, indices).cpu(), expected_weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        scales.index_select(0, indices).cpu(), expected_scales, rtol=0, atol=0
    )
    mapping = storage._mappings[LAYER]
    padded_weight, padded_scales = mapping.weight_view(), mapping.scale_view()
    _require(plan.padded_rows - plan.total_rows == 7, "expected seven padding rows")
    _require(
        not torch.count_nonzero(padded_weight[plan.total_rows :]).item()
        and not torch.count_nonzero(padded_scales[plan.total_rows :]).item(),
        "physical tail padding is not zero",
    )
    return rows


def _expect_busy(storage, busy_error):
    try:
        storage.close()
    except busy_error:
        return
    raise AssertionError("close accepted a live Graph pin or tensor alias")


def _probe(args, record):
    os.environ.update(
        RTP_ENGRAM_STORAGE="cuda_fabric",
        DSV41_CPU_ENGRAM_LOOKUP="0",
        DSV41_ASYNC_ENGRAM_LOOKUP="0",
        DSV41_DECODE_ENGRAM_PREFETCH="0",
        DSV41_ENGRAM_DEVICE_SCALES="1",
        DSV41_ENGRAM_DIRECT_FP8="1",
    )
    import torch
    from torch.distributed import TCPStore

    from rtp_llm.models_py.modules.dsv41._engram_decode_prefetch import (
        DecodeEngramPrefetch,
    )
    from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import inspect_checkpoint
    from rtp_llm.models_py.modules.dsv41.engram_local_cuda import (
        LocalCudaEngramStorage,
        _checkpoint_identity,
    )
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
    from rtp_llm.models_py.modules.dsv41.engram_storage import StorageBusyError

    storage = embedding = graph = model = metadata = alias = None
    group = None
    try:
        store = TCPStore(
            args.master_addr,
            args.master_port,
            args.world_size,
            args.rank == 0,
            timeout=timedelta(seconds=TIMEOUT_S),
            wait_for_workers=True,
        )
        group = _Group(store, args.rank, args.world_size)
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        native = _load_native()
        record.update(
            phase="native_capability",
            native_module=str(Path(native.__file__).resolve()),
            native_sha256=hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        _require(
            all(
                hasattr(native, name)
                for name in (
                    "device_fabric_supported",
                    "device_granularity",
                    "DeviceOwner",
                    "DeviceFabricImport",
                    "fabric_handle_bytes",
                )
            ),
            "current prebuilt DEVICE/FABRIC native bridge required",
        )
        if not native.device_fabric_supported(args.device):
            raise _NotRun("CUDA_FABRIC_DEVICE_UNSUPPORTED")
        record["device_fabric_supported"] = True
        granularity = native.device_granularity(args.device, "fabric")
        baseline = dict(native.resource_counters())
        peers = group.exchange(
            "devices",
            dict(
                hostname=socket.gethostname(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                uuid=str(torch.cuda.get_device_properties(device).uuid),
                granularity=granularity,
                native_sha256=record["native_sha256"],
            ),
        )
        record["peers"] = peers
        if (
            len({(peer["hostname"], peer["boot_id"]) for peer in peers}) < 2
            or len({peer["boot_id"] for peer in peers}) < 2
        ):
            raise _NotRun("CROSS_OS_EVIDENCE_REQUIRED: at least two distinct boot IDs")
        record["remote_os_owner_ranks"] = [
            rank
            for rank, peer in enumerate(peers)
            if peer["boot_id"] != peers[args.rank]["boot_id"]
        ]
        _require(
            len({peer["uuid"] for peer in peers}) == args.world_size, "duplicate GPUs"
        )
        _require(
            len({peer["granularity"] for peer in peers}) == 1, "granularity mismatch"
        )
        _require(
            len({peer["native_sha256"] for peer in peers}) == 1,
            "native binary mismatch",
        )
        plan = plan_shards(
            args.world_size * (granularity // (DIM // 32)) - 7,
            DIM,
            args.world_size,
            granularity=granularity,
        )
        record["geometry"] = plan.as_dict()
        if args.world_size * plan.shard_bytes > 512 << 20:
            raise _NotRun("SYNTHETIC_BACKING_EXCEEDS_512_MIB_TOTAL_BOUND")
        expected = ExpectedLayout(
            layers=((LAYER, plan.total_rows),),
            head_dim=DIM,
            model_revision="cross-os-device-fabric-fixture-v1",
            hash_abi="sha256:" + hashlib.sha256(b"device-fabric-fixture").hexdigest(),
        )
        with tempfile.TemporaryDirectory(prefix="rtp-engram-fabric-") as directory:
            record["phase"] = "independent_checkpoint"
            checkpoint = Path(directory)
            _write_checkpoint(checkpoint, plan)
            inspected = inspect_checkpoint(checkpoint, expected)
            checkpoints = group.exchange(
                "checkpoint",
                dict(
                    identity=_checkpoint_identity(inspected, expected),
                    tensor_file_sha256=hashlib.sha256(
                        (checkpoint / "engram.safetensors").read_bytes()
                    ).hexdigest(),
                    stat_token=list(inspected.files[0].stat_token),
                ),
            )
            _require(
                len({item["identity"] for item in checkpoints}) == 1,
                "checkpoint metadata differs",
            )
            _require(
                len({item["tensor_file_sha256"] for item in checkpoints}) == 1,
                "independent fixture files have different payloads",
            )
            _require(
                len({tuple(item["stat_token"]) for item in checkpoints}) > 1,
                "independent replicas must exercise different local stat tokens",
            )
            record["checkpoint_replicas"] = checkpoints
            record["phase"] = "device_fabric_create_export_import_map_read"
            storage = LocalCudaEngramStorage.from_checkpoint(
                checkpoint,
                expected,
                device=args.device,
                store=store,
                rank=args.rank,
                world_size=args.world_size,
                timeout_s=TIMEOUT_S,
                storage_mode="cuda_fabric",
            )
            info = storage.info()
            record["storage"] = info
            _require(
                info["state"] == "READY" and info["storage_mode"] == "cuda_fabric",
                "Fabric storage did not become READY",
            )
            _require(
                (info["device"], info["shard_count"], info["shard_index"])
                == (args.device, args.world_size, args.rank),
                "storage topology differs from actual probe ranks",
            )
            _require(
                len(info["owners"]) == 2
                and len(info["imports"]) == 2 * args.world_size,
                "unexpected DEVICE backing/import count",
            )
            _require(
                all(
                    value["location_type"] == "DEVICE"
                    and value["handle_type"] == "FABRIC"
                    for value in info["owners"] + info["imports"]
                ),
                "actual allocation properties are not DEVICE/FABRIC",
            )
            _require(
                all(
                    value["allocation_device"] == args.device
                    for value in info["owners"]
                ),
                "local owner was allocated on another GPU",
            )
            physical = sum(value["allocation_bytes"] for value in info["owners"])
            _require(
                physical == info["local_physical_bytes"] == plan.shard_bytes
                and physical * args.world_size == plan.weight_span + plan.scale_span,
                "physical backing differs from exactly one/world of padded table",
            )
            record["byte_sample_rows"] = _verify_bytes(storage, plan, device)
            group.exchange("attached", True)
            record["phase"] = "lookup_and_graph"
            embedding = HostEngramEmbedding.from_storage(storage, LAYER)
            weight, scales = storage.table_views(LAYER)
            _require(
                embedding._uva[0].data_ptr() == weight.data_ptr()
                and embedding._uva[1].data_ptr() == scales.data_ptr(),
                "lookup copied the mapped DEVICE tables",
            )
            weight = scales = None
            first = []
            for owner in range(args.world_size):
                begin, end = plan.row_range(owner)
                first.append([begin, (begin + end) // 2, end - 1, -1])
            patterns = (
                [list(reversed(row)) for row in first],
                list(reversed(first)),
                [list(reversed(row)) for row in reversed(first)],
                first,
            )
            for dtype in (torch.int32, torch.int64):
                for host_ids in patterns:
                    ids = torch.tensor(host_ids, dtype=dtype, device=device)
                    torch.testing.assert_close(
                        embedding(ids, device).cpu(),
                        _reference(host_ids, plan),
                        rtol=0,
                        atol=0,
                    )
            model = _LookupModel(embedding)
            metadata = SimpleNamespace()
            ids = torch.tensor(first, dtype=torch.int64, device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = embedding(ids, device)
            stream.synchronize()
            storage._pin_graph()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_tables = DecodeEngramPrefetch(model, metadata)
                try:
                    output = embedding(ids, device)
                finally:
                    captured_tables.close()
            _expect_busy(storage, StorageBusyError)
            _require(storage.info()["state"] == "READY", "pinned close damaged storage")
            for host_ids in patterns:
                ids.copy_(torch.tensor(host_ids, dtype=torch.int64, device=device))
                graph.replay()
                torch.testing.assert_close(
                    output.cpu(), _reference(host_ids, plan), rtol=0, atol=0
                )
            torch.cuda.synchronize(device)
            graph.reset()
            graph = None
            storage._unpin_graph()
            model = metadata = captured_tables = None
            alias = storage.table_views(LAYER)[0][1:2]
            embedding.release_external_views()
            embedding = None
            gc.collect()
            _expect_busy(storage, StorageBusyError)
            alias = None
            gc.collect()
            # No owner is released until every consumer has drained its reads,
            # reset its Graph and dropped all table aliases.
            group.exchange("drained", True)
            record["phase"] = "resource_release"
            storage.close()
            _require(storage.info()["state"] == "CLOSED", "storage failed to close")
            storage.close()
            _require(
                dict(native.resource_counters()) == baseline, "native resources leaked"
            )
            group.exchange("released", True)
            group.finish()
            record.update(
                status="PASS",
                phase="complete",
                local_physical_bytes=physical,
                lookup_int32_int64_exact=True,
                remote_boundary_middle_bytes_exact=True,
                physical_padding_zero=True,
                changed_id_graph_replays=len(patterns),
                graph_pin_alias_guards=True,
                native_resources_restored=True,
                native_module=str(Path(native.__file__).resolve()),
                native_sha256=peers[args.rank]["native_sha256"],
            )
    except BaseException as error:
        if group is not None:
            try:
                group.fail(error)
            except BaseException:
                record["failure_publication_error"] = traceback.format_exc()
        # Never force-unpin/unmap an uncertain Graph or remote reader. Process
        # exit, controlled by the external launcher, owns failed resources.
        _FAILED_RESOURCES.append((storage, embedding, graph, model, metadata, alias))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1")
    )
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500"))
    )
    parser.add_argument("--rank", type=int, default=os.environ.get("RANK"))
    parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "2"))
    )
    parser.add_argument(
        "--device", type=int, default=int(os.environ.get("LOCAL_RANK", "0"))
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.rank is None
        or not 2 <= args.world_size <= 64
        or not 0 <= args.rank < args.world_size
        or args.device < 0
        or not 0 < args.master_port < 65536
    ):
        parser.error(
            "require rank in [0,world), world 2..64, nonnegative device and valid port"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        status="STARTING",
        rank=args.rank,
        world_size=args.world_size,
        device=args.device,
        scope="Real cross-OS DEVICE/FABRIC lookup/Graph fixture; not model, 32DP or GB300 evidence",
        controller_gpu_reservation_required=True,
    )
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
    exit_code = 1
    try:
        _probe(args, record)
        exit_code = 0
    except _NotRun as error:
        record.update(status="NOT_RUN", reason=str(error))
        exit_code = 2
    except BaseException:
        record.update(status="FAIL", traceback=traceback.format_exc())
    finally:
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
