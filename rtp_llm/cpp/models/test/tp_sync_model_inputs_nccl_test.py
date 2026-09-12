"""Two local GPU component regression for the production ModelTypes tpSync.

This is exact transport/state testing, not CP8 or disaggregated model acceptance.
All rounds use the production NCCL callbacks, including CPU shape metadata.
"""

import ctypes
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.test.utils.port_util import PortManager

_WORLD_SIZE = 2
_WORKER_TIMEOUT_SECONDS = 240
_ROUND_SPECS = (
    ("empty_receiver_two_images_bf16", 12, 0, (2, 3), 7, torch.bfloat16),
    ("shrink_one_image_fp32", 5, 101, (1,), 5, torch.float32),
    ("grow_three_images_fp16", 10, 211, (1, 2, 1), 9, torch.float16),
    ("image_to_text_with_positions", 4, 401, (), 0, torch.float32),
    ("text_without_positions", 6, None, (), 0, torch.float32),
    ("text_to_image_bf16", 8, 701, (1, 2), 3, torch.bfloat16),
    ("v41_prefill_two_requests", 6, None, (), 0, torch.float32),
    ("v41_decode_three_requests", 3, None, (), 0, torch.float32),
    ("v41_decode_one_request", 1, None, (), 0, torch.float32),
    ("v41_empty_batch", 0, None, (), 0, torch.float32),
    ("v41_canonical_only", 2, None, (), 0, torch.float32),
    ("text_after_v41_without_metadata", 3, None, (), 0, torch.float32),
)
_V41_FIELDS = (
    "v41_token_types",
    "v41_token_valid",
    "engram_history_ids",
    "engram_history_valid",
    "v41_request_id",
    "v41_state_ready",
    "v41_is_fake",
)
# Input lengths, prefix lengths, sequence lengths, restored state, fake requests.
_V41_EXECUTION_ROUNDS = {
    "v41_prefill_two_requests": ((2, 4), (0, 16), (), (False, True), (False, False)),
    "v41_decode_three_requests": (
        (4, 5, 6),
        (),
        (33, 1, 91),
        (True, False, True),
        (False, True, False),
    ),
    "v41_decode_one_request": ((4,), (), (34,), (True,), (False,)),
    "v41_empty_batch": ((), (), (), (), ()),
}
_DEVICE_MAP_FIELDS = (
    "combo_tokens",
    "input_lengths",
    "sequence_lengths",
    "prefix_lengths",
    "lm_output_indexes",
)


def _write_report(path, report):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    staging.replace(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def _expected_round(index):
    name, token_count, position_start, feature_rows, width, dtype = _ROUND_SPECS[index]
    expected = {
        "combo_tokens": torch.arange(token_count, dtype=torch.int32) + 1000 * index,
        "input_lengths": torch.tensor([token_count], dtype=torch.int32),
        "sequence_lengths": torch.empty(0, dtype=torch.int32),
        "prefix_lengths": torch.tensor([position_start or 0], dtype=torch.int32),
        "lm_output_indexes": torch.tensor([token_count - 1], dtype=torch.int32),
        "request_id": torch.tensor([(1 << 40) + index], dtype=torch.int64),
        "request_pd_separation": torch.tensor([bool(index % 2)], dtype=torch.bool),
        "combo_position_ids": (
            torch.arange(token_count, dtype=torch.int32) + position_start
            if position_start is not None
            else None
        ),
        "text_tokens_mask": None,
        "mm_features_locs": None,
        "mm_features_spans": None,
        "multimodal_features": None,
        "need_all_logits": bool(index % 2),
        "need_all_hidden_states": bool((index + 1) % 2),
        "is_fake_stream": bool(index == 2),
    }
    expected.update({field: None for field in _V41_FIELDS})
    devices = {key: "cpu" for key, value in expected.items() if torch.is_tensor(value)}
    for field_index, field in enumerate(_DEVICE_MAP_FIELDS):
        devices[field] = "cuda" if (field_index + index) % 2 else "cpu"
    if feature_rows:
        features, locations, spans = [], [], []
        mask = torch.ones(token_count, dtype=torch.int32)
        start = 1
        for feature_index, rows in enumerate(feature_rows):
            assert start + rows <= token_count
            locations.append(start)
            spans.append([0, start + position_start, start + position_start + rows])
            mask[start : start + rows] = 0
            values = (
                torch.arange(rows * width * 2, dtype=torch.float32) / 8
                + 20 * index
                + 3 * feature_index
            ).to(dtype)
            features.append(values.reshape(rows, width * 2)[:, ::2])
            start += rows + 1
        expected["multimodal_features"] = features
        expected["text_tokens_mask"] = mask
        expected["mm_features_locs"] = torch.tensor(locations, dtype=torch.int32)
        expected["mm_features_spans"] = torch.tensor(spans, dtype=torch.int64)
        devices.update(
            text_tokens_mask="cpu", mm_features_locs="cpu", mm_features_spans="cpu"
        )
    if name in _V41_EXECUTION_ROUNDS or name == "v41_canonical_only":
        valid = torch.ones(token_count, dtype=torch.bool)
        if name in _V41_EXECUTION_ROUNDS:
            lengths, prefixes, sequences, ready, fake = _V41_EXECUTION_ROUNDS[name]
            batch_size = len(lengths)
            expected["input_lengths"] = torch.tensor(lengths, dtype=torch.int32)
            expected["prefix_lengths"] = torch.tensor(prefixes, dtype=torch.int32)
            expected["sequence_lengths"] = torch.tensor(sequences, dtype=torch.int32)
            expected["request_id"] = torch.arange(len(prefixes), dtype=torch.int64) + (
                1 << 40
            )
            expected["request_pd_separation"] = torch.zeros(
                len(prefixes), dtype=torch.bool
            )
            expected["v41_request_id"] = (
                torch.arange(batch_size, dtype=torch.int64) + (1 << 48) + index * 10
            )
            expected["v41_state_ready"] = torch.tensor(ready, dtype=torch.bool)
            expected["v41_is_fake"] = torch.tensor(fake, dtype=torch.bool)
            expected["v41_request_id"][expected["v41_is_fake"]] = 0
            row_counts = torch.tensor(
                lengths if prefixes else (1,) * batch_size, dtype=torch.int64
            )
            expected["lm_output_indexes"] = (row_counts.cumsum(0) - 1).to(torch.int32)
            valid = torch.repeat_interleave(~expected["v41_is_fake"], row_counts)
        expected["v41_token_types"] = torch.full((token_count,), -1, dtype=torch.int32)
        expected["v41_token_valid"] = valid
        expected["engram_history_ids"] = (
            torch.arange(token_count * 3, dtype=torch.int32).reshape(token_count, 3)
            + index * 100
        )
        expected["engram_history_valid"] = valid[:, None].expand(-1, 3).contiguous()
        for field in _V41_FIELDS:
            if torch.is_tensor(expected[field]):
                devices[field] = "cpu"
    return name, expected, devices


def _root_payload(expected, devices, device):
    payload = {}
    for field, value in expected.items():
        if field == "multimodal_features" and value is not None:
            features = []
            for feature in value:
                rows, width = feature.shape
                # A real strided CUDA source exercises production pack fallback.
                strided = torch.empty(
                    (rows, width * 2), dtype=feature.dtype, device=device
                )[:, ::2]
                strided.copy_(feature)
                assert not strided.is_contiguous()
                features.append(strided)
            payload[field] = features
        elif torch.is_tensor(value):
            payload[field] = value.to(device if devices[field] == "cuda" else "cpu")
        else:
            payload[field] = value
    return payload


def _check_tensor(actual, expected, device_type, rank, field):
    assert torch.is_tensor(actual), f"rank {rank} {field}: missing tensor"
    assert actual.shape == expected.shape, f"rank {rank} {field}: shape mismatch"
    assert actual.dtype == expected.dtype, f"rank {rank} {field}: dtype mismatch"
    assert actual.device.type == device_type, f"rank {rank} {field}: wrong device"
    if device_type == "cuda":
        assert actual.device.index == rank, f"rank {rank} {field}: wrong GPU"
    assert torch.equal(
        actual.cpu(), expected
    ), f"rank {rank} {field}: exact value mismatch"
    actual_hash, expected_hash = _tensor_hash(actual), _tensor_hash(expected)
    assert actual_hash == expected_hash, f"rank {rank} {field}: byte mismatch"
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "device": str(actual.device),
        "stride": list(actual.stride()),
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
    }


def _check_snapshot(actual, expected, devices, rank):
    assert set(actual) == set(expected), f"rank {rank}: snapshot field mismatch"
    tensors = {}
    for field, value in expected.items():
        result = actual[field]
        if value is None:
            assert result is None, f"rank {rank} {field}: stale state was retained"
            tensors[field] = None
        elif field == "multimodal_features":
            assert result is not None and len(result) == len(
                value
            ), f"rank {rank}: feature count mismatch"
            tensors[field] = [
                _check_tensor(feature, reference, "cuda", rank, f"{field}[{index}]")
                for index, (feature, reference) in enumerate(zip(result, value))
            ]
        elif torch.is_tensor(value):
            tensors[field] = _check_tensor(result, value, devices[field], rank, field)
        else:
            assert result == value, f"rank {rank} {field}: flag mismatch"
            tensors[field] = result
    return tensors


def _worker(rank, port, uds_dir, output_dir):
    report_path = Path(output_dir) / f"rank{rank}.json"
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
        "rounds": [],
    }
    _write_report(report_path, report)
    initialized = False
    try:
        # Register config types before compute-op default arguments use them.
        from rtp_llm.ops import NcclCommConfig, ParallelismConfig

        librtp_compute_ops = importlib.import_module("librtp_compute_ops")

        # Resolve the test bridge against the production communication registry.
        runtime_library = ctypes.CDLL(
            librtp_compute_ops.__file__, mode=ctypes.RTLD_GLOBAL
        )
        assert runtime_library._handle
        from rtp_llm.cpp.models.test import libtp_sync_model_inputs_test_ops as ops
        from rtp_llm.models_py.distributed.collective_torch import (
            Group,
            _get_group,
            destroy_distributed_environment,
            init_distributed_environment,
        )

        torch.set_num_threads(1)
        torch.set_default_device("cpu")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        props = torch.cuda.get_device_properties(rank)
        gpu_uuid = str(props.uuid)
        assert gpu_uuid, "A distinct physical GPU identity is required"
        report.update(
            status="initializing_nccl",
            gpu_uuid=gpu_uuid,
            gpu_name=props.name,
            compute_capability=[props.major, props.minor],
            cuda_device=rank,
            compute_ops_sha256=_file_hash(librtp_compute_ops.__file__),
            test_ops_sha256=_file_hash(ops.__file__),
        )
        _write_report(report_path, report)
        os.environ["RTP_LLM_CPU_TP_BROADCASTER_DIR"] = uds_dir
        os.environ["RTP_LLM_CPU_TP_BROADCASTER_ID"] = f"tp_sync_{port}"
        config = ParallelismConfig()
        config.world_rank = config.local_rank = config.tp_rank = rank
        config.world_size = config.local_world_size = config.tp_size = _WORLD_SIZE
        config.dp_size = 1
        config.dp_rank = 0
        config.use_ub_comm = False
        communication = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=port + 9,
            dp_tp_nccl_port=port + 1,
            ffn_tp_nccl_port=port + 6,
        )
        init_distributed_environment(
            config, communication, port, backend="nccl", timeout=60
        )
        initialized = True
        assert dist.get_world_size() == _WORLD_SIZE
        assert dist.get_backend(_get_group(Group.TP)) == "nccl"
        assert _get_group(Group.TP).size() == _WORLD_SIZE
        assert (
            ops.cpu_broadcaster_initialized()
        ), "Expected the real local UDS bootstrap"

        # Disable only the production UDS optimization. Its production fallback
        # promotes CPU tensors to CUDA and calls torch.distributed NCCL broadcast.
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        dist.barrier(device_ids=[rank])
        assert not ops.cpu_broadcaster_initialized(), "Metadata must traverse NCCL"
        report.update(status="running", backend="nccl", cpu_broadcaster_active=False)
        _write_report(report_path, report)
        state = ops.ModelInputsTpSyncTestState()
        if rank == 1:
            state.set_features([])
            empty = state.snapshot()
            assert empty["multimodal_features"] == []
            assert empty["combo_tokens"] is None
        for index in range(len(_ROUND_SPECS)):
            name, expected, devices = _expected_round(index)
            if rank == 0:
                state.replace(_root_payload(expected, devices, device))
            dist.barrier(device_ids=[rank])
            state.sync(rank, _WORLD_SIZE)
            torch.cuda.synchronize(device)
            actual = state.snapshot()
            tensors = _check_snapshot(actual, expected, devices, rank)
            assert not ops.cpu_broadcaster_initialized()
            report["rounds"].append(
                {"name": name, "status": "passed", "tensors": tensors}
            )
            _write_report(report_path, report)
        dist.barrier(device_ids=[rank])
        destroy_distributed_environment()
        initialized = False
        report.update(status="passed", finished_unix=time.time())
        _write_report(report_path, report)
    except BaseException:
        report.update(
            status="failed", error=traceback.format_exc(), finished_unix=time.time()
        )
        _write_report(report_path, report)
        raise
    finally:
        if initialized:
            destroy_distributed_environment()


class ModelInputsTpSyncNcclTest(unittest.TestCase):
    def test_real_two_rank_consecutive_multimodal_broadcasts(self):
        self.assertTrue(__debug__, "Run with Python assertions enabled")
        self.assertNotEqual(
            os.geteuid(), 0, "Run in the approved non-root dev container"
        )
        self.assertTrue(
            torch.cuda.is_available(), "Two real GPUs are required; no CPU fallback"
        )
        self.assertGreaterEqual(
            torch.cuda.device_count(), _WORLD_SIZE, "Two GPUs are required"
        )
        self.assertTrue(str(torch.version.cuda).startswith("13."), "CUDA13 is required")
        self.assertTrue(dist.is_nccl_available(), "The real NCCL backend is required")
        for rank in range(_WORLD_SIZE):
            self.assertEqual(
                torch.cuda.get_device_capability(rank)[0], 10, "Blackwell is required"
            )
        output_base = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir())
        )
        output_base.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="tp-sync-nccl-", dir=output_base))
        run_path = output_dir / "run.json"
        report = {
            "scope": "two_local_gpu_component_exact_transport",
            "release_topology_acceptance": False,
            "started_unix": time.time(),
            "status": "starting",
            "parent_pid": os.getpid(),
            "host": socket.gethostname(),
            "architecture": platform.machine(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "command": sys.argv,
            "test_source_sha256": _file_hash(__file__),
            "world_size": _WORLD_SIZE,
            "worker_timeout_seconds": _WORKER_TIMEOUT_SECONDS,
            "rank_reports": [
                str(output_dir / f"rank{rank}.json") for rank in range(_WORLD_SIZE)
            ],
            "worker_pids": [],
        }
        _write_report(run_path, report)
        print(f"tpSync NCCL component artifacts: {output_dir}", flush=True)
        locks = []
        processes = []
        try:
            ports, locks = PortManager().get_consecutive_ports(10)
            with tempfile.TemporaryDirectory(
                prefix="tp-sync-uds-", dir="/tmp"
            ) as uds_dir:
                context = mp.get_context("spawn")
                deadline = time.monotonic() + _WORKER_TIMEOUT_SECONDS
                for rank in range(_WORLD_SIZE):
                    process = context.Process(
                        target=_worker,
                        args=(rank, ports[0], uds_dir, str(output_dir)),
                        name=f"tp-sync-nccl-rank{rank}",
                    )
                    process.start()
                    processes.append(process)
                    report["worker_pids"].append(process.pid)
                    _write_report(run_path, report)
                report["status"] = "running"
                _write_report(run_path, report)
                while any(process.is_alive() for process in processes):
                    for path in report["rank_reports"]:
                        if Path(path).exists():
                            state = json.loads(Path(path).read_text("utf-8"))
                            self.assertNotEqual(
                                state["status"], "failed", state.get("error")
                            )
                    self.assertFalse(
                        any(process.exitcode not in (None, 0) for process in processes),
                        f"Worker crashed: {[(p.pid, p.exitcode) for p in processes]}",
                    )
                    self.assertLess(
                        time.monotonic(), deadline, "NCCL component watchdog expired"
                    )
                    time.sleep(0.1)
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(
                        process.exitcode, 0, f"Worker {process.pid} failed"
                    )
                rank_reports = [
                    json.loads(Path(path).read_text("utf-8"))
                    for path in report["rank_reports"]
                ]
                self.assertEqual(
                    len({item["gpu_uuid"] for item in rank_reports}), _WORLD_SIZE
                )
                self.assertEqual(
                    len({item["pid"] for item in rank_reports}), _WORLD_SIZE
                )
                for item in rank_reports:
                    self.assertEqual(item["status"], "passed")
                    self.assertEqual(item["backend"], "nccl")
                    self.assertFalse(item["cpu_broadcaster_active"])
                    self.assertEqual(
                        [row["name"] for row in item["rounds"]],
                        [row[0] for row in _ROUND_SPECS],
                    )
                report.update(
                    status="passed",
                    finished_unix=time.time(),
                    gpu_uuids=[item["gpu_uuid"] for item in rank_reports],
                    rank_report_sha256={
                        path: _file_hash(path) for path in report["rank_reports"]
                    },
                )
                _write_report(run_path, report)
        except BaseException:
            report.update(
                status="failed", error=traceback.format_exc(), finished_unix=time.time()
            )
            _write_report(run_path, report)
            raise
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
