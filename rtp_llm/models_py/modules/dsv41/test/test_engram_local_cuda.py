"""Real four-process DEVICE/FD storage, lookup and CUDA graph regression.

The Bazel target must reserve four GPUs and supply the current native bridge.
No IMEX, Fabric handles, GPU mocks, service processes or extension JIT builds
are used. A small immutable checkpoint is created inside the test sandbox.
"""

import array
import datetime
import gc
import hashlib
import json
import multiprocessing
import os
import socket
import tempfile
import threading
import time
import traceback
import unittest
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

    rows = plan.total_rows
    weight = torch.empty((rows, DIM), dtype=torch.uint8)
    scales = torch.empty((rows, DIM // 32), dtype=torch.uint8)
    row_mod = torch.arange(rows, dtype=torch.int64).remainder(4)
    fp8_bytes = torch.tensor([0x38, 0x40, 0x44, 0x48], dtype=torch.uint8)
    scale_bytes = (127, 128, 126, 129)
    for block in range(DIM // 32):
        weight[:, block * 32 : (block + 1) * 32] = fp8_bytes[
            (row_mod + block).remainder(4)
        ][:, None]
        for shard in range(WORLD):
            begin, end = plan.row_range(shard)
            scales[begin:end, block] = scale_bytes[(shard + block) % 4]
    names = (
        f"layers.{LAYER}.engram.embed.weight",
        f"layers.{LAYER}.engram.embed.scale",
    )
    save_file(
        {names[0]: weight.view(torch.float8_e4m3fn), names[1]: scales},
        str(directory / "engram.safetensors"),
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": weight.numel() + scales.numel()},
                "weight_map": {name: "engram.safetensors" for name in names},
            }
        )
    )


def _reference(host_ids, plan):
    """Independent exact values represented by the fixture's FP8/E8M0 bytes."""
    import torch

    ids = torch.tensor(host_ids, dtype=torch.int64)
    columns = torch.arange(DIM, dtype=torch.int64) // 32
    safe_ids = ids.clamp_min(0)
    shard = safe_ids // plan.shard_rows
    factors = torch.tensor([1.0, 2.0, 0.5, 4.0])
    base = ((safe_ids[..., None] + columns) % 4 + 1).float()
    result = base * factors[(shard[..., None] + columns) % 4]
    result[ids < 0] = 0
    return result.to(torch.bfloat16)


def _verify_padded_views(storage, plan):
    import torch

    mapping = storage._mappings[LAYER]
    weight, scales = mapping.weight_view(), mapping.scale_view()
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
        from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native
        from rtp_llm.models_py.modules.dsv41.engram_local_cuda import (
            LocalCudaEngramStorage,
        )
        from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
        from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
        from rtp_llm.models_py.modules.dsv41.engram_storage import StorageBusyError

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
            hasattr(native, "DeviceOwner") and hasattr(native, "DeviceImport"),
            "Bazel must supply the current DEVICE/POSIX_FD native bridge",
        )
        baseline = dict(native.resource_counters())
        granularity = native.device_granularity(rank)
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
        )
        storage = LocalCudaEngramStorage.from_checkpoint(
            checkpoint, expected, device=rank, timeout_s=180
        )
        info = storage.info()
        _require(info["state"] == "READY", "local storage did not become READY")
        _require(
            (info["device"], info["shard_count"], info["shard_index"])
            == (rank, WORLD, rank),
            "each distinct worker GPU must own exactly its own shard",
        )
        owners, imports = info["owners"], info["imports"]
        _require(
            len(owners) == 2 and len(imports) == 2 * WORLD,
            "unexpected backing/import count",
        )
        _require(
            all(
                value["location_type"] == "DEVICE"
                and value["allocation_device"] == rank
                and value["handle_type"] == "POSIX_FD"
                for value in owners
            ),
            "owner backing is not local physical DEVICE/POSIX_FD memory",
        )
        _require(
            all(value["location_type"] == "DEVICE" for value in imports)
            and sorted(value["allocation_device"] for value in imports)
            == sorted(list(range(WORLD)) * 2),
            "queried imported allocation properties do not cover all four DEVICE owners",
        )
        physical_bytes = sum(value["allocation_bytes"] for value in owners)
        _require(
            physical_bytes == info["local_physical_bytes"] == plan.shard_bytes,
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
            tuple(weight.shape) == (plan.total_rows, DIM)
            and tuple(scales.shape) == (plan.total_rows, DIM // 32),
            "public table views must exclude physical padding rows",
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
                    "imported_DEVICE_allocations": len(imports),
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
        from rtp_llm.models_py.modules.dsv41.engram_local_cuda import (
            LocalCudaEngramStorage,
        )

        with self.assertRaisesRegex(ValueError, "at least two"):
            LocalCudaEngramStorage.from_checkpoint(
                "/unused", None, device=0, store=object(), rank=0, world_size=1
            )
        with self.assertRaisesRegex(ValueError, "cross-host"):
            LocalCudaEngramStorage.from_checkpoint(
                "/unused",
                None,
                device=0,
                store=object(),
                rank=0,
                world_size=WORLD,
                gang_hosts=["host-a", "host-b", "host-a", "host-a"],
            )

    def test_malformed_real_scm_rights_reply_releases_received_fds(self):
        from rtp_llm.models_py.modules.dsv41.engram_local_cuda import _receive_fds

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
                                connection.recv(4096)
                                payload = json.dumps(
                                    {"nonce": "test-nonce", "rank": 0, "count": 2}
                                ).encode()
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
                            _receive_fds(path, "test-nonce", 0, 1, os.getpid(), 2, 5)
                    finally:
                        thread.join(timeout=6)
                    self.assertFalse(
                        thread.is_alive(), "test-owned FD responder did not stop"
                    )
                    self.assertEqual(errors, [])
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                finally:
                    os.close(fd)


if __name__ == "__main__":
    unittest.main()
