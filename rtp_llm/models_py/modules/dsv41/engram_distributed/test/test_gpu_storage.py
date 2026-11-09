"""Real GPU sharing, byte/BF16 lookup, graph and resource-lifetime tests.

The default test reserves four local GPUs and exercises POSIX_FD without Fabric.
--cross-host uses a shared TCPStore endpoint and real Fabric handles on distinct
hosts; its external launcher must reserve and lock each participating GPU.
Both paths import the packaged native library and create bounded fixtures.
"""

import argparse
import array
import base64
import datetime
import gc
import hashlib
import json
import multiprocessing
import os
import socket
import struct
import tempfile
import threading
import time
import traceback
import unittest
from dataclasses import asdict
from datetime import timedelta
from multiprocessing.connection import wait
from pathlib import Path
from types import SimpleNamespace

WORLD = 4
LAYER = 1
DIM = 256
TIMEOUT_SECONDS = 240
_FAILED_RESOURCES = []


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _write_checkpoint(directory, plan):
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
        str(directory / "engram.safetensors"),
    )
    (directory / "model.safetensors.index.json").write_text(
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


def _verify_padded_views(storage, plan):
    import torch

    weight, scales = storage.table_views(LAYER)
    _require(weight.dtype == scales.dtype == torch.uint8, "native bytes changed dtype")
    _require(
        tuple(weight.shape) == (plan.padded_rows, DIM)
        and tuple(scales.shape) == (plan.padded_rows, DIM // 32),
        "stitched physical views do not cover the padded allocation",
    )
    _require(
        plan.padded_rows - plan.total_rows == 7, "fixture must have seven padding rows"
    )
    _require(
        not torch.count_nonzero(weight[plan.total_rows :]).item()
        and not torch.count_nonzero(scales[plan.total_rows :]).item(),
        "physical tail padding contains nonzero bytes",
    )


class _LookupModel:
    def __init__(self, embedding):
        self.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=embedding))]


def _worker(rank, rendezvous, checkpoint, outputs):
    os.environ.update(
        RTP_ENGRAM_STORAGE="cuda_local",
        DSV41_CPU_ENGRAM_LOOKUP="0",
        DSV41_ASYNC_ENGRAM_LOOKUP="0",
        DSV41_DECODE_ENGRAM_PREFETCH="0",
        DSV41_ENGRAM_DEVICE_SCALES="1",
        DSV41_ENGRAM_DIRECT_FP8="1",
    )
    import torch
    import torch.distributed as dist

    result_path = Path(outputs) / f"rank{rank}.json"
    storage = embedding = graph = model = metadata = alias = None
    try:
        from rtp_llm.models_py.modules.dsv41._engram_decode_prefetch import (
            DecodeEngramPrefetch,
        )
        from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding
        from rtp_llm.models_py.modules.dsv41.engram_distributed.local import (
            LocalCudaStartup,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
            ExpectedLayout,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
            plan_shards,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            StorageBusyError,
            _load_native,
        )

        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            "nccl",
            init_method=rendezvous,
            rank=rank,
            world_size=WORLD,
            timeout=datetime.timedelta(seconds=180),
        )
        native = _load_native()
        _require(
            hasattr(native.FabricAllocation, "export_fd")
            and hasattr(native, "FabricMapping"),
            "Bazel must supply the current DEVICE/POSIX_FD native bridge",
        )
        baseline = dict(native.resource_counters())
        granularity = native.device_granularity(rank, handle_type="posix_fd")
        granularities = [None] * WORLD
        dist.all_gather_object(granularities, granularity)
        _require(
            len(set(granularities)) == 1, "fixture requires equal device granularities"
        )
        plan = plan_shards(
            WORLD * (granularity // (DIM // 32)) - 7,
            DIM,
            WORLD,
            granularity=granularity,
        )
        _require(
            WORLD * plan.shard_bytes <= 512 << 20,
            "synthetic checkpoint backing exceeds the 512 MiB total bound",
        )
        checkpoint = Path(checkpoint)
        if rank == 0:
            _write_checkpoint(checkpoint, plan)
        dist.barrier()
        expected = ExpectedLayout(
            layers=((LAYER, plan.total_rows),),
            head_dim=DIM,
            model_revision="local-cuda-four-process-fixture-v1",
            hash_abi="sha256:" + hashlib.sha256(b"local-cuda-fixture").hexdigest(),
            backend="cuda_ipc",
        )
        storage = LocalCudaStartup(
            dist.distributed_c10d._get_default_store(),
            checkpoint=checkpoint,
            device=rank,
            rank=rank,
            world_size=WORLD,
            environ={"RTP_ENGRAM_COHORT_TIMEOUT": "180"},
        ).prepare(expected)
        info = storage.info()
        _require(info["state"] == "READY", "local storage did not become READY")
        _require(
            (info["device"], info["shard_count"]) == (rank, 1),
            "each distinct worker GPU must own exactly its own shard",
        )
        owners, imports = info["backings"], list(info["allocations"].values())
        _require(
            len(owners) == 2
            and len(imports) == 2
            and all(m["imported_handles"] == WORLD for m in imports),
            "unexpected backing/import count",
        )
        _require(
            all(
                value["memory_kind"] == "gpu"
                and value["device"] == rank
                and value["handle_type"] == "POSIX_FD"
                for value in owners
            ),
            "owner backing is not local physical DEVICE/POSIX_FD memory",
        )
        _require(
            all(value["memory_kind"] == "gpu" for value in imports)
            and all(value["handle_type"] == "POSIX_FD" for value in imports),
            "queried imported allocation properties do not cover all four DEVICE owners",
        )
        physical_bytes = sum(value["allocation_bytes"] for value in owners)
        _require(
            physical_bytes == plan.shard_bytes
            and info["physical_bytes"] == WORLD * plan.shard_bytes,
            "physical allocation must equal one shard, not a private full table",
        )
        _verify_padded_views(storage, plan)
        embedding = HostEngramEmbedding.from_storage(storage, LAYER)
        weight, scales = storage.table_views(LAYER)
        _require(
            embedding._uva[0].data_ptr() == weight.data_ptr()
            and embedding._uva[1].data_ptr() == scales.data_ptr(),
            "lookup copied the mapped weight/scales instead of sharing their addresses",
        )
        _require(
            tuple(weight.shape) == (plan.padded_rows, DIM)
            and tuple(scales.shape) == (plan.padded_rows, DIM // 32),
            "table views must preserve the physical padding extent",
        )
        weight = scales = None

        boundaries = []
        for shard in range(WORLD):
            begin, end = plan.row_range(shard)
            boundaries.extend((begin, begin + 1, end - 1, -1))
        first = [boundaries[i : i + 4] for i in range(0, len(boundaries), 4)]
        second = [list(reversed(row)) for row in reversed(first)]
        patterns = (first, second, first, second)
        for dtype in (torch.int32, torch.int64):
            for host_ids in patterns[:2]:
                ids = torch.tensor(host_ids, dtype=dtype, device=device)
                torch.testing.assert_close(
                    embedding(ids, device).cpu(),
                    _reference(host_ids, plan),
                    rtol=0,
                    atol=0,
                )

        model, metadata = _LookupModel(embedding), SimpleNamespace()
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
        with unittest.TestCase().assertRaises(StorageBusyError):
            storage.close()
        _require(
            storage.info()["state"] == "READY", "pinned close damaged READY storage"
        )
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
        with unittest.TestCase().assertRaises(StorageBusyError):
            storage.close()
        alias = None
        gc.collect()
        dist.barrier()
        storage.close()
        _require(
            storage.info()["state"] == "CLOSED", "storage did not release its mappings"
        )
        storage.close()
        _require(
            dict(native.resource_counters()) == baseline,
            "native resources leaked after drain",
        )
        result_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank": rank,
                    "world_size": WORLD,
                    "native_module": str(Path(native.__file__).resolve()),
                    "native_sha256": hashlib.sha256(
                        Path(native.__file__).read_bytes()
                    ).hexdigest(),
                    "device_granularity": granularity,
                    "logical_rows": plan.total_rows,
                    "padding_rows": 7,
                    "local_physical_bytes": physical_bytes,
                    "imported_DEVICE_allocations": sum(
                        m["imported_handles"] for m in imports
                    ),
                    "lookup_exact": True,
                    "changed_ID_graph_replays": len(patterns),
                    "graph_pin_and_alias_close_guards": True,
                    "native_resources_restored": True,
                    "scope": "Local four-GPU FD/lookup/graph fixture; not model or cross-host evidence",
                },
                indent=2,
            )
        )
        dist.destroy_process_group()
    except BaseException:
        result_path.write_text(
            json.dumps(
                {"status": "FAIL", "rank": rank, "traceback": traceback.format_exc()},
                indent=2,
            )
        )
        # A failed fixture must never force-unmap an uncertain live graph/view.
        _FAILED_RESOURCES.append((storage, embedding, graph, model, metadata, alias))
        raise


class LocalCudaEngramCudaTest(unittest.TestCase):
    def test_four_process_device_shards_lookup_graph_and_lifetime(self):
        import torch

        self.assertTrue(torch.version.cuda and torch.version.cuda.split(".")[0] == "13")
        self.assertGreaterEqual(
            torch.cuda.device_count(), WORLD, "reserve four GPUs for this target"
        )
        ctx = multiprocessing.get_context("spawn")
        artifact_root = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir())
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("TEST_TMPDIR")
        ) as directory:
            work = Path(directory)
            checkpoint = work / "checkpoint"
            checkpoint.mkdir()
            results = work / "results"
            results.mkdir()
            rendezvous = (work / "nccl_store").as_uri()
            processes = [
                ctx.Process(
                    target=_worker,
                    args=(rank, rendezvous, str(checkpoint), str(results)),
                )
                for rank in range(WORLD)
            ]
            started = []
            deadline = time.monotonic() + TIMEOUT_SECONDS
            failure = None
            try:
                for process in processes:
                    process.start()
                    started.append(process)
                remaining = {p.sentinel: p for p in started}
                while remaining:
                    ready = wait(
                        list(remaining), timeout=max(0, deadline - time.monotonic())
                    )
                    if not ready:
                        failure = "four-process local CUDA fixture timed out"
                        break
                    for sentinel in ready:
                        process = remaining.pop(sentinel)
                        process.join()
                        if process.exitcode != 0:
                            failure = (
                                f"local CUDA worker exited with code {process.exitcode}"
                            )
                    if failure:
                        break
            finally:
                # Only children created by this exact test are ever terminated.
                for process in started:
                    if process.is_alive():
                        process.terminate()
                for process in started:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
                records = []
                for rank in range(WORLD):
                    path = results / f"rank{rank}.json"
                    try:
                        record = json.loads(path.read_text())
                    except (OSError, ValueError) as error:
                        record = {
                            "status": "NOT_COMPLETED",
                            "rank": rank,
                            "error": str(error),
                        }
                    records.append(record)
                (artifact_root / "engram_local_cuda_four_process.json").write_text(
                    json.dumps({"workers": records, "failure": failure}, indent=2)
                )
            self.assertIsNone(failure, json.dumps(records, indent=2))
            self.assertTrue(all(record["status"] == "PASS" for record in records))
            self.assertEqual(len({record["native_sha256"] for record in records}), 1)
            self.assertTrue(all(process.exitcode == 0 for process in processes))


class LocalCudaEngramConfigTest(unittest.TestCase):
    def test_invalid_group_is_rejected_before_checkpoint_or_device_access(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.local import (
            LocalCudaStartup,
        )

        with self.assertRaisesRegex(ValueError, "at least two"):
            LocalCudaStartup(
                object(),
                checkpoint="/unused",
                device=0,
                rank=0,
                world_size=1,
            )
        with self.assertRaisesRegex(ValueError, "cross-host"):
            LocalCudaStartup(
                object(),
                checkpoint="/unused",
                device=0,
                rank=0,
                world_size=WORLD,
                gang_hosts=["host-a", "host-b", "host-a", "host-a"],
            )

    def test_malformed_real_scm_rights_reply_releases_received_fds(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.local import (
            _receive_fds,
        )

        errors = []
        # AF_UNIX paths cannot fit a long Bazel TEST_TMPDIR. mkdtemp keeps this
        # short directory private to this test (mode0700).
        with tempfile.TemporaryDirectory(
            prefix="rtp-engram-fd-", dir="/tmp"
        ) as directory:
            path = str(Path(directory) / "fd.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
                listener.bind(path)
                listener.listen(1)
                listener.settimeout(5)
                fd = os.open(os.devnull, os.O_RDONLY)
                try:
                    before = len(os.listdir("/proc/self/fd"))

                    def reply():
                        try:
                            connection, _ = listener.accept()
                            with connection:
                                connection.settimeout(5)
                                self.assertEqual(
                                    connection.recv(4096), struct.pack("!I", 1)
                                )
                                payload = struct.pack("!II", 0, 2)
                                connection.sendmsg(
                                    [payload],
                                    [
                                        (
                                            socket.SOL_SOCKET,
                                            socket.SCM_RIGHTS,
                                            array.array("i", [fd]),
                                        )
                                    ],
                                )
                        except BaseException:
                            errors.append(traceback.format_exc())

                    thread = threading.Thread(target=reply)
                    thread.start()
                    try:
                        with self.assertRaisesRegex(RuntimeError, "INVALID_FD_REPLY"):
                            _receive_fds(path, 0, 1, os.getpid(), 2, 5)
                    finally:
                        thread.join(timeout=6)
                    self.assertFalse(
                        thread.is_alive(), "test-owned FD responder did not stop"
                    )
                    self.assertEqual(errors, [])
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                finally:
                    os.close(fd)


TIMEOUT_S = 300
_PHASES = ("devices", "checkpoint", "handles", "attached", "drained", "released")


def _fabric_storage(checkpoint, expected, plan, group, peers, device, native):
    """Exercise real cross-host native handles with independent checkpoint replicas."""
    import torch

    from rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint import (
        checkpoint_samples,
        inspect_checkpoint,
        load_rows,
    )
    from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
        AllocationSlice,
        Manifest,
        TensorSpec,
        ValidatedManifest,
    )
    from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import (
        _verify_samples,
    )
    from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
        FabricEngramStorage,
    )

    inspected = inspect_checkpoint(checkpoint, expected)
    snapshots = {f.relative_path: f for f in inspected.files}
    owners, encoded = [], []
    begin, end = plan.row_range(group.rank)
    for tensor in inspected.layers[0].tensors:
        allocation = native.FabricAllocation(
            plan.shard_rows * tensor.shape[1], -1, device, memory_kind="gpu"
        )
        owners.append(allocation)
        load_rows(
            allocation,
            checkpoint,
            tensor,
            snapshots[tensor.relative_path],
            begin,
            end,
            tensor.shape[1],
            deadline=group.deadline,
        )
        encoded.append(base64.b64encode(allocation.export_handle()).decode())
    bundles = group.exchange("handles", encoded)
    tensors, handles = [], {}
    for role, tensor in enumerate(inspected.layers[0].tensors):
        parts = []
        for rank, peer in enumerate(peers):
            start, stop = plan.row_range(rank)
            aid = f"{tensor.name}/{rank}"
            parts.append(
                AllocationSlice(
                    start,
                    stop,
                    (stop - start) * tensor.shape[1],
                    plan.shard_rows * tensor.shape[1],
                    plan.granularity,
                    peer["hostname"],
                    aid,
                )
            )
            handles[aid] = base64.b64decode(bundles[rank][role])
        tensors.append(TensorSpec(tensor.name, tensor.shape, tuple(parts)))
    manifest = Manifest(
        expected.model_revision,
        group.prefix,
        expected.layout_abi,
        expected.hash_abi,
        expected.backend,
        tuple(tensors),
    )
    contract = ValidatedManifest(
        manifest, plan.shard_bytes * group.world, 2 * group.world, expected
    )
    storage = FabricEngramStorage._attach_validated(
        contract, handles=handles, device=device, native=native, backings=owners
    )
    samples = checkpoint_samples(checkpoint, manifest)
    storage.verify(
        lambda views, validated: _verify_samples(
            views, validated, samples, torch.device("cuda", device)
        )
    )
    return storage


class _NotRun(RuntimeError):
    pass


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
    padded_weight, padded_scales = storage.table_views(LAYER)
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
    from rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint import (
        inspect_checkpoint,
    )
    from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
        ExpectedLayout,
    )
    from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import plan_shards
    from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
        StorageBusyError,
        _load_native,
    )

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
                    "check_fabric_devices",
                    "device_granularity",
                    "FabricAllocation",
                    "FabricMapping",
                    "fabric_handle_bytes",
                )
            ),
            "current prebuilt DEVICE/FABRIC native bridge required",
        )
        native.check_fabric_devices([args.device])
        record["device_fabric_supported"] = True
        granularity = native.device_granularity(args.device, "fabric")
        baseline = dict(native.resource_counters())
        peers = group.exchange(
            "devices",
            dict(
                hostname=socket.gethostname(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                device=args.device,
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
            len({(peer["hostname"], peer["device"]) for peer in peers})
            == args.world_size,
            "duplicate GPUs",
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
        record["geometry"] = asdict(plan)
        if args.world_size * plan.shard_bytes > 512 << 20:
            raise _NotRun("SYNTHETIC_BACKING_EXCEEDS_512_MIB_TOTAL_BOUND")
        expected = ExpectedLayout(
            layers=((LAYER, plan.total_rows),),
            head_dim=DIM,
            model_revision="cross-os-device-fabric-fixture-v1",
            hash_abi="sha256:" + hashlib.sha256(b"device-fabric-fixture").hexdigest(),
            backend="fabric_gpu",
        )
        with tempfile.TemporaryDirectory(prefix="rtp-engram-fabric-") as directory:
            record["phase"] = "independent_checkpoint"
            checkpoint = Path(directory)
            _write_checkpoint(checkpoint, plan)
            inspected = inspect_checkpoint(checkpoint, expected)
            checkpoints = group.exchange(
                "checkpoint",
                dict(
                    layout=asdict(expected),
                    tensor_file_sha256=hashlib.sha256(
                        (checkpoint / "engram.safetensors").read_bytes()
                    ).hexdigest(),
                    stat_token=list(inspected.files[0].stat_token),
                ),
            )
            _require(
                all(item["layout"] == checkpoints[0]["layout"] for item in checkpoints),
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
            storage = _fabric_storage(
                checkpoint, expected, plan, group, peers, args.device, native
            )
            info = storage.info()
            record["storage"] = info
            _require(
                info["state"] == "READY",
                "Fabric storage did not become READY",
            )
            _require(
                info["device"] == args.device,
                "storage topology differs from actual probe ranks",
            )
            _require(
                len(info["backings"]) == 2
                and sum(m["imported_handles"] for m in info["allocations"].values())
                == 2 * args.world_size,
                "unexpected DEVICE backing/import count",
            )
            _require(
                all(
                    value["memory_kind"] == "gpu" and value["handle_type"] == "FABRIC"
                    for value in info["backings"] + list(info["allocations"].values())
                ),
                "actual allocation properties are not DEVICE/FABRIC",
            )
            _require(
                all(value["device"] == args.device for value in info["backings"]),
                "local owner was allocated on another GPU",
            )
            physical = sum(value["allocation_bytes"] for value in info["backings"])
            _require(
                physical == plan.shard_bytes
                and physical * args.world_size == info["physical_bytes"],
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


def cross_host_main(argv=None):
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
    import sys

    if "--cross-host" in sys.argv:
        sys.argv.remove("--cross-host")
        raise SystemExit(cross_host_main())
    unittest.main()
