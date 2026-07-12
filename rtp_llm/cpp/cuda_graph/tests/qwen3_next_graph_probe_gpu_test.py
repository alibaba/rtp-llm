import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.qwen3_next import (
    _CudaGraphLayerProbe,
    _graph_probe_stats,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


class _GraphProbeModel:
    def __init__(self):
        self.probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(2, 0, 2, -1, 3),
            layer_num=3,
        )
        self.debug_status_calls = 0
        self.buffer_getter_calls = 0

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        inputs.attention_inputs.is_cuda_graph = is_cuda_graph
        return self

    def prepare_cuda_graph(self, attention_inputs):
        pass

    def forward(self, inputs, fmha_impl=None):
        graph_bs = int(inputs.attention_inputs.input_lengths.size(0))
        token_values = inputs.input_ids.to(torch.float32).reshape(-1, 1)
        hidden = inputs.input_hiddens + token_values
        residual = hidden + 5
        self.probe.record(
            2,
            hidden,
            residual,
            graph_bs=graph_bs,
            is_cuda_graph=bool(inputs.attention_inputs.is_cuda_graph),
        )
        self.probe.record(
            0,
            hidden + 1,
            residual + 1,
            graph_bs=graph_bs,
            is_cuda_graph=bool(inputs.attention_inputs.is_cuda_graph),
        )
        return PyModelOutputs(hidden)

    def get_cuda_graph_probe_buffer(self, graph_bs):
        self.buffer_getter_calls += 1
        return self.probe.get_capture(graph_bs)

    def get_cuda_graph_probe_enabled(self):
        return self.probe.enabled

    def set_cuda_graph_probe_enabled(self, enabled):
        previous = self.probe.enabled
        self.probe.enabled = bool(enabled)
        return previous

    def get_cuda_graph_probe_debug_status(self, graph_bs):
        self.debug_status_calls += 1
        return {
            "module_env_enabled": True,
            "probe_created": True,
            "buffer_available": self.probe.get_buffer(graph_bs) is not None,
            "layers": self.probe.layers,
            "buffer_bucket_bs": tuple(sorted(self.probe._buffers)),
            "record_debug": self.probe.get_debug_status(),
        }


class TestQwen3NextGraphProbeGpu(unittest.TestCase):
    actual_bs = 21
    graph_bs = 24
    ring_max_graph_bs = 32
    num_tokens_per_bs = 4
    hidden_size = 2
    max_seq_len = 16
    tokens_per_block = 16

    def setUp(self):
        torch.cuda.set_device(0)
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="qwen3_next_graph_probe_"
        )
        self.checksum_file = os.path.join(self.temp_dir.name, "checksum.jsonl")
        self.ring_trigger_file = os.path.join(self.temp_dir.name, "ring.trigger")
        os.environ["WORLD_RANK"] = "0"
        os.environ["RTPLLM_DECODE_CHECKSUM_DEBUG"] = "1"
        os.environ["RTPLLM_DECODE_CHECKSUM_SYNC_DEVICE"] = "1"
        os.environ["RTPLLM_DECODE_CHECKSUM_FILE"] = self.checksum_file
        os.environ["RTPLLM_DECODE_CHECKSUM_EVERY"] = "1"
        os.environ["RTPLLM_DECODE_CHECKSUM_MAX_RECORDS"] = "0"
        os.environ["RTPLLM_DECODE_CHECKSUM_MAX_LANES"] = "0"
        os.environ["RTPLLM_DECODE_CHECKSUM_MAX_OUTPUT_STEPS_PER_TRACE"] = "8"
        self.graph_probe_enabled = os.environ.get(
            "TEST_QWEN3_NEXT_GRAPH_PROBE_ENABLED", "1"
        ) == "1"
        self.ring_only = os.environ.get(
            "TEST_QWEN3_NEXT_GRAPH_PROBE_RING_ONLY", "0"
        ) == "1"
        self.ring_budget_reject = os.environ.get(
            "TEST_QWEN3_NEXT_GRAPH_PROBE_RING_BUDGET_REJECT", "0"
        ) == "1"
        self.retrospective = os.environ.get(
            "TEST_QWEN3_NEXT_GRAPH_PROBE_RETROSPECTIVE", "0"
        ) == "1"
        self.dual_graph = os.environ.get(
            "TEST_QWEN3_NEXT_GRAPH_PROBE_DUAL_GRAPH", "0"
        ) == "1"
        if (
            self.ring_only
            or self.ring_budget_reject
            or self.retrospective
            or self.dual_graph
        ):
            os.environ["RTPLLM_DECODE_CHECKSUM_DEBUG"] = "0"
        os.environ["RTPLLM_QWEN3_NEXT_GRAPH_PROBE"] = (
            "1" if self.graph_probe_enabled else "0"
        )
        os.environ["RTPLLM_QWEN3_NEXT_GRAPH_PROBE_LAYERS"] = "99,77"
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_DEBUG"] = (
            "0" if self.retrospective or self.dual_graph else "1"
        )
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_DIR"] = self.temp_dir.name
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_TRIGGER_FILE"] = (
            self.ring_trigger_file
        )
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_RECORDS"] = "2"
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_GRAPH_BS"] = str(
            self.ring_max_graph_bs
        )
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_BYTES"] = (
            "1" if self.ring_budget_reject else str(1 << 30)
        )
        os.environ["RTPLLM_CUDA_GRAPH_PROBE_RING_TRIGGER_CHECK_EVERY"] = "1"
        self.retrospective_shm_name = f"/rtpllm_graph_probe_{os.getpid()}"
        os.environ["RTPLLM_RETROSPECTIVE_PROBE_SHM_NAME"] = (
            self.retrospective_shm_name
        )
        os.environ["RTPLLM_RETROSPECTIVE_PROBE_DIR"] = self.temp_dir.name
        os.environ["WORLD_SIZE"] = "1"

        self.model = _GraphProbeModel()
        self.runner = CudaGraphRunner()
        self.runner.init_decode(
            self.model,
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.tokens_per_block,
            [8, self.graph_bs]
            if self.retrospective or self.dual_graph
            else [self.graph_bs],
            self.num_tokens_per_bs,
            True,
            True,
        )
        torch.cuda.synchronize()

    def tearDown(self):
        self.runner = None
        Path("/dev/shm", self.retrospective_shm_name.lstrip("/")).unlink(
            missing_ok=True
        )
        self.temp_dir.cleanup()

    def _build_inputs(self, actual_bs=None, trace_ids=None):
        actual_bs = self.actual_bs if actual_bs is None else actual_bs
        token_rows = actual_bs * self.num_tokens_per_bs
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(
            token_rows, dtype=torch.int32, device="cuda"
        )
        inputs.input_hiddens = torch.zeros(
            (token_rows, self.hidden_size), dtype=torch.float32, device="cuda"
        )
        inputs.input_hiddens[0, 0] = float("nan")
        inputs.input_hiddens[0, 1] = float("inf")
        inputs.input_hiddens[4:8] = 1.0e30
        inputs.trace_ids = (
            [f"lane-{lane}" for lane in range(actual_bs)]
            if trace_ids is None
            else trace_ids
        )

        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.is_target_verify = True
        attention.dtype = get_typemeta(torch.zeros(1, dtype=torch.float32))
        attention.input_lengths = torch.full(
            (actual_bs,),
            self.num_tokens_per_bs,
            dtype=torch.int32,
        ).pin_memory()
        attention.prefix_lengths = torch.full(
            (actual_bs,),
            self.max_seq_len - self.num_tokens_per_bs,
            dtype=torch.int32,
        ).pin_memory()
        attention.sequence_lengths = torch.empty(
            0, dtype=torch.int32
        ).pin_memory()
        attention.sequence_lengths_plus_1_d = torch.full(
            (actual_bs,),
            self.max_seq_len - self.num_tokens_per_bs,
            dtype=torch.int32,
            device="cuda",
        )
        attention.decode_cu_seqlens_d = torch.arange(
            0,
            token_rows + 1,
            self.num_tokens_per_bs,
            dtype=torch.int32,
            device="cuda",
        )
        attention.cu_seqlens = torch.arange(
            0,
            token_rows + 1,
            self.num_tokens_per_bs,
            dtype=torch.int32,
            device="cuda",
        )
        attention.cu_kv_seqlens = attention.cu_seqlens.clone()
        block_ids = torch.arange(
            1, actual_bs + 1, dtype=torch.int32, device="cuda"
        ).reshape(actual_bs, 1)
        attention.kv_cache_kernel_block_id_device = block_ids
        attention.kv_cache_kernel_block_id_host = block_ids.cpu().pin_memory()
        attention.kv_cache_block_id_device = block_ids
        attention.kv_cache_block_id_host = attention.kv_cache_kernel_block_id_host
        attention.padding_offset = torch.zeros(
            token_rows, dtype=torch.int32, device="cuda"
        )
        attention.context_total_kv_length = token_rows
        attention.total_tokens = token_rows
        inputs.attention_inputs = attention
        return inputs

    def test_capture_replay_updates_persistent_lane_buffer_and_json(self):
        if self.dual_graph:
            self._assert_dual_graph_post_trigger_dump()
            return
        if self.retrospective:
            self._assert_retrospective_previous_bucket_dump()
            return
        capture_buffer, capture_layers = self.model.get_cuda_graph_probe_buffer(
            self.graph_bs
        )
        capture_ptr = capture_buffer.data_ptr()
        capture_values = capture_buffer.clone()
        inputs = self._build_inputs()

        self.assertTrue(self.runner.canRun(inputs))
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(self.graph_bs, self.runner.getCurrentRealGraphSize())

        replay_buffer, replay_layers = self.model.get_cuda_graph_probe_buffer(
            self.graph_bs
        )
        self.assertEqual(capture_ptr, replay_buffer.data_ptr())
        self.assertEqual((2, 0), replay_layers)
        self.assertEqual(capture_layers, replay_layers)
        self.assertEqual((2, self.graph_bs, 12), tuple(replay_buffer.shape))
        self.assertIsNone(self.model.probe.get_buffer(self.graph_bs * 4))
        self.assertFalse(torch.equal(capture_values, replay_buffer))

        token_values = inputs.input_ids.to(torch.float32).reshape(-1, 1)
        expected_hidden = inputs.input_hiddens + token_values
        expected_residual = expected_hidden + 5
        torch.testing.assert_close(
            replay_buffer[0, : self.actual_bs, :6],
            _graph_probe_stats(expected_hidden.reshape(self.actual_bs, -1)),
            equal_nan=True,
        )
        torch.testing.assert_close(
            replay_buffer[0, : self.actual_bs, 6:],
            _graph_probe_stats(expected_residual.reshape(self.actual_bs, -1)),
            equal_nan=True,
        )
        torch.testing.assert_close(
            replay_buffer[1, : self.actual_bs, :6],
            _graph_probe_stats((expected_hidden + 1).reshape(self.actual_bs, -1)),
            equal_nan=True,
        )
        torch.testing.assert_close(
            replay_buffer[1, : self.actual_bs, 6:],
            _graph_probe_stats((expected_residual + 1).reshape(self.actual_bs, -1)),
            equal_nan=True,
        )
        if self.ring_only:
            self._assert_ring_only_capture(inputs, replay_buffer.cpu().clone())
            return
        if self.ring_budget_reject:
            self.runner.forward(inputs)
            torch.cuda.synchronize()
            self.assertFalse(list(Path(self.temp_dir.name).glob("*.bin")))
            self.assertFalse(list(Path(self.temp_dir.name).glob("*.complete")))
            return

        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            raw_records = [line for line in checksum_stream if line.strip()]

        def reject_nonstandard_constant(value):
            raise ValueError(f"non-standard JSON constant: {value}")

        records = [
            json.loads(line, parse_constant=reject_nonstandard_constant)
            for line in raw_records
        ]
        self.assertEqual(["before_replay", "after_replay"], [r["stage"] for r in records])
        self.assertEqual([0, 0], [r["record_id"] for r in records])
        after_replay = records[1]
        if not self.graph_probe_enabled:
            self.assertEqual(
                {
                    "reason": "cpp_disabled",
                    "cpp_enabled": False,
                    "has_buffer_getter": False,
                    "has_status_getter": False,
                    "python_type": "",
                    "error": None,
                    "python_status": None,
                },
                after_replay["graph_probe_status"],
            )
            self.assertTrue(
                all("graph_probe" not in lane for lane in after_replay["lanes"])
            )
            self.assertFalse(list(Path(self.temp_dir.name).glob("*.complete")))
            return
        self.assertEqual(
            {
                "reason": "ready",
                "cpp_enabled": True,
                "has_buffer_getter": True,
                "has_status_getter": True,
                "python_status": {
                    "module_env_enabled": True,
                    "probe_created": True,
                    "buffer_available": True,
                    "layers": [2, 0],
                    "buffer_bucket_bs": sorted(self.model.probe._buffers),
                    "record_debug": self.model.probe.get_debug_status(),
                },
            },
            {
                key: after_replay["graph_probe_status"][key]
                for key in (
                    "reason",
                    "cpp_enabled",
                    "has_buffer_getter",
                    "has_status_getter",
                    "python_status",
                )
            },
        )
        self.assertEqual(self.actual_bs, after_replay["lane_count"])
        self.assertEqual(
            list(range(self.actual_bs)),
            [lane["lane"] for lane in after_replay["lanes"]],
        )
        lane0 = after_replay["lanes"][0]
        lane1 = after_replay["lanes"][1]
        lane2 = after_replay["lanes"][2]
        self.assertEqual([2, 0], lane0["graph_probe"]["layers"])
        self.assertIsNone(lane0["hidden_state"]["sum"])
        self.assertEqual([None, None], lane0["hidden_state"]["sample"])
        self.assertIsNone(lane1["graph_probe"]["values"][0][2])
        torch.testing.assert_close(
            torch.tensor(
                lane2["graph_probe"]["values"][1], dtype=torch.float32
            ),
            replay_buffer[1, 2].cpu(),
        )

        status_calls = self.model.debug_status_calls
        base_prefix = self.max_seq_len - self.num_tokens_per_bs
        inputs.attention_inputs.prefix_lengths[: self.actual_bs].fill_(base_prefix)
        inputs.attention_inputs.prefix_lengths[1] = base_prefix + 8
        inputs.input_ids.add_(10)
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        replay_buffer, _ = self.model.get_cuda_graph_probe_buffer(self.graph_bs)
        expected_ring_record_1 = replay_buffer.cpu().clone()
        self.assertEqual(status_calls + 1, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            records = [json.loads(line) for line in checksum_stream if line.strip()]
        self.assertEqual(4, len(records))
        expected_lanes = [lane for lane in range(self.actual_bs) if lane != 1]
        self.assertEqual(
            [expected_lanes, expected_lanes],
            [[lane["lane"] for lane in record["lanes"]] for record in records[-2:]],
        )
        self.assertFalse(list(Path(self.temp_dir.name).glob("*.bin")))
        self.assertFalse(list(Path(self.temp_dir.name).glob("*.jsonl.tmp")))

        inputs.input_ids.add_(100)
        Path(self.ring_trigger_file).touch()
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        replay_buffer, _ = self.model.get_cuda_graph_probe_buffer(self.graph_bs)
        expected_ring_record_2 = replay_buffer.cpu().clone()
        ring_tensor_files = list(
            Path(self.temp_dir.name).glob(
                f"cuda_graph_probe_ring_rank0_pid{os.getpid()}_gen*_runner0.bin"
            )
        )
        ring_metadata_files = list(
            Path(self.temp_dir.name).glob(
                f"cuda_graph_probe_ring_rank0_pid{os.getpid()}_gen*_runner0.jsonl"
            )
        )
        ring_completion_files = list(
            Path(self.temp_dir.name).glob(
                f"cuda_graph_probe_ring_rank0_pid{os.getpid()}_gen*_runner0.complete"
            )
        )
        self.assertEqual(1, len(ring_tensor_files))
        self.assertEqual(1, len(ring_metadata_files))
        self.assertEqual(1, len(ring_completion_files))
        self.assertFalse(list(Path(self.temp_dir.name).glob("*.tmp")))
        ring = torch.from_file(
            str(ring_tensor_files[0]),
            shared=False,
            size=2 * 2 * self.ring_max_graph_bs * 12,
            dtype=torch.float32,
        ).reshape(2, 2, self.ring_max_graph_bs, 12)
        self.assertEqual((2, 2, self.ring_max_graph_bs, 12), tuple(ring.shape))
        with ring_metadata_files[0].open(encoding="utf-8") as metadata_stream:
            ring_metadata = [json.loads(line) for line in metadata_stream if line.strip()]
        self.assertEqual(2, len(ring_metadata))
        self.assertEqual([1, 2], [r["record_id"] for r in ring_metadata])
        self.assertEqual([0, 0], [r["runner_id"] for r in ring_metadata])
        self.assertEqual([2, 0], ring_metadata[0]["layers"])
        self.assertEqual([self.actual_bs, self.actual_bs], [r["current_bs"] for r in ring_metadata])
        self.assertEqual([self.graph_bs, self.graph_bs], [r["graph_bs"] for r in ring_metadata])
        self.assertEqual(
            [self.ring_max_graph_bs, self.ring_max_graph_bs],
            [r["ring_max_graph_bs"] for r in ring_metadata],
        )
        self.assertEqual([12, 12], [r["field_count"] for r in ring_metadata])
        self.assertEqual(["float32", "float32"], [r["dtype"] for r in ring_metadata])
        completion = json.loads(ring_completion_files[0].read_text(encoding="utf-8"))
        self.assertEqual(2, completion["records"])
        torch.testing.assert_close(ring[0, :, : self.graph_bs], expected_ring_record_1)
        torch.testing.assert_close(ring[1, :, : self.graph_bs], expected_ring_record_2)
        self.assertTrue(
            torch.equal(
                ring[:, :, self.graph_bs :],
                torch.zeros_like(ring[:, :, self.graph_bs :]),
            )
        )
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            checksum_count_after_dump = sum(1 for line in checksum_stream if line.strip())

        status_calls = self.model.debug_status_calls
        inputs.attention_inputs.prefix_lengths[: self.actual_bs].fill_(base_prefix + 8)
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(
                checksum_count_after_dump,
                sum(1 for line in checksum_stream if line.strip()),
            )

        inputs.attention_inputs.sequence_lengths = (
            inputs.attention_inputs.input_lengths[: self.actual_bs].clone().pin_memory()
        )
        inputs.attention_inputs.sequence_lengths.add_(7)
        status_calls = self.model.debug_status_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls + 1, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(
                checksum_count_after_dump + 2,
                sum(1 for line in checksum_stream if line.strip()),
            )

        inputs.attention_inputs.sequence_lengths.add_(1)
        status_calls = self.model.debug_status_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(
                checksum_count_after_dump + 2,
                sum(1 for line in checksum_stream if line.strip()),
            )

    def _assert_ring_only_capture(self, inputs, expected_record_0):
        self.assertFalse(Path(self.checksum_file).exists())
        self.assertFalse(list(Path(self.temp_dir.name).glob("*.bin")))

        inputs.input_ids.add_(10)
        getter_calls = self.model.buffer_getter_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(getter_calls, self.model.buffer_getter_calls)
        expected_record_1 = self.model.get_cuda_graph_probe_buffer(
            self.graph_bs
        )[0].cpu().clone()

        inputs.input_ids.add_(100)
        Path(self.ring_trigger_file).touch()
        getter_calls = self.model.buffer_getter_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(getter_calls, self.model.buffer_getter_calls)
        expected_record_2 = self.model.get_cuda_graph_probe_buffer(
            self.graph_bs
        )[0].cpu().clone()

        tensor_file = next(Path(self.temp_dir.name).glob("*.bin"))
        metadata_file = next(Path(self.temp_dir.name).glob("*.jsonl"))
        ring = torch.from_file(
            str(tensor_file),
            shared=False,
            size=2 * 2 * self.ring_max_graph_bs * 12,
            dtype=torch.float32,
        ).reshape(2, 2, self.ring_max_graph_bs, 12)
        metadata = [
            json.loads(line)
            for line in metadata_file.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([1, 2], [record["record_id"] for record in metadata])
        torch.testing.assert_close(ring[0, :, : self.graph_bs], expected_record_1)
        torch.testing.assert_close(ring[1, :, : self.graph_bs], expected_record_2)
        self.assertTrue(
            torch.equal(
                ring[:, :, self.graph_bs :],
                torch.zeros_like(ring[:, :, self.graph_bs :]),
            )
        )
        self.assertFalse(torch.equal(expected_record_0, expected_record_1))

    def _assert_retrospective_previous_bucket_dump(self):
        retained_sequence_length_a = 41
        retained_sequence_length_b = 73
        inputs_a = self._build_inputs(
            actual_bs=7,
            trace_ids=["bad-trace"] + [f"replay-a-{lane}" for lane in range(1, 7)],
        )
        inputs_a.attention_inputs.sequence_lengths = torch.full(
            (7,), retained_sequence_length_a, dtype=torch.int32
        ).pin_memory()
        inputs_b = self._build_inputs(
            actual_bs=self.actual_bs,
            trace_ids=["bad-trace"]
            + [f"replay-b-{lane}" for lane in range(1, self.actual_bs)],
        )
        inputs_b.attention_inputs.sequence_lengths = torch.full(
            (self.actual_bs,), retained_sequence_length_b, dtype=torch.int32
        ).pin_memory()
        inputs_b.input_ids.add_(1000)
        inputs_c = self._build_inputs(
            actual_bs=self.actual_bs,
            trace_ids=[f"replay-c-{lane}" for lane in range(self.actual_bs)],
        )
        inputs_c.input_ids.add_(2000)
        torch.cuda.synchronize()
        allocated_before_a = torch.cuda.memory_allocated()
        getter_calls_after_capture = self.model.buffer_getter_calls

        self.assertTrue(self.runner.canRun(inputs_a))
        self.runner.forward(inputs_a)
        torch.cuda.synchronize()
        self.assertEqual(getter_calls_after_capture, self.model.buffer_getter_calls)
        self.assertEqual(8, self.runner.getCurrentRealGraphSize())
        expected_a = self.model.get_cuda_graph_probe_buffer(8)[0].cpu().clone()

        ring_sized_bytes = 2 * 2 * self.ring_max_graph_bs * 12 * 4
        self.assertLess(
            torch.cuda.memory_allocated() - allocated_before_a,
            ring_sized_bytes,
        )
        output_dir = Path(self.temp_dir.name)
        self.assertFalse(list(output_dir.glob("*.bin")))
        self.assertFalse(list(output_dir.glob("*.jsonl")))
        self.assertFalse(list(output_dir.glob("*.complete")))

        getter_calls_before_b = self.model.buffer_getter_calls
        self.assertTrue(self.runner.canRun(inputs_b))
        self.runner.forward(inputs_b)
        torch.cuda.synchronize()
        self.assertEqual(getter_calls_before_b, self.model.buffer_getter_calls)
        self.assertEqual(24, self.runner.getCurrentRealGraphSize())
        expected_b = self.model.get_cuda_graph_probe_buffer(24)[0].cpu().clone()

        published = self.runner.retrospective_probe_event(
            "bad-trace",
            "test-retrospective",
            retained_sequence_length_b + 1,
        )
        self.assertTrue(published["published"])
        generation = published["generation"]

        getter_calls_before_c = self.model.buffer_getter_calls
        self.assertTrue(self.runner.canRun(inputs_c))
        self.runner.forward(inputs_c)
        torch.cuda.synchronize()
        self.assertEqual(getter_calls_before_c, self.model.buffer_getter_calls)
        self.assertEqual(24, self.runner.getCurrentRealGraphSize())

        tensor_file = next(output_dir.glob("*.bin"))
        metadata_file = next(output_dir.glob("*.jsonl"))
        completion_file = next(output_dir.glob("*.complete"))
        self.assertFalse(list(output_dir.glob("*.tmp")))
        self.assertGreaterEqual(
            completion_file.stat().st_mtime_ns,
            max(tensor_file.stat().st_mtime_ns, metadata_file.stat().st_mtime_ns),
        )
        dumped = torch.from_file(
            str(tensor_file),
            shared=False,
            size=expected_b.numel(),
            dtype=torch.float32,
        ).reshape(expected_b.shape)
        torch.testing.assert_close(
            dumped[:, : self.actual_bs], expected_b[:, : self.actual_bs]
        )
        self.assertTrue(
            torch.equal(
                dumped[:, self.actual_bs :],
                torch.zeros_like(dumped[:, self.actual_bs :]),
            )
        )
        self.assertFalse(torch.equal(dumped[:, :7], expected_a[:, :7]))

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.assertEqual(generation, metadata["event_generation"])
        self.assertEqual("bad-trace", metadata["event_trace_id"])
        self.assertEqual("test-retrospective", metadata["event_reason"])
        self.assertEqual(
            retained_sequence_length_b + 1,
            metadata["event_observed_sequence_length"],
        )
        self.assertEqual(1, metadata["replay_id"])
        self.assertEqual(24, metadata["graph_bs"])
        self.assertEqual(self.actual_bs, metadata["current_bs"])
        self.assertEqual(
            expected_b.numel() * expected_b.element_size(), metadata["nbytes"]
        )
        self.assertEqual([2, 0], metadata["layers"])
        self.assertEqual("bad-trace", metadata["lanes"][0]["trace_id"])
        self.assertEqual(0, metadata["lanes"][0]["lane"])
        self.assertEqual(self.num_tokens_per_bs, metadata["lanes"][0]["input_length"])
        self.assertEqual(
            retained_sequence_length_b,
            metadata["lanes"][0]["sequence_length"],
        )
        self.assertEqual(self.actual_bs, len(metadata["lanes"]))

        completion = json.loads(completion_file.read_text(encoding="utf-8"))
        self.assertEqual(generation, completion["event_generation"])
        self.assertEqual(str(tensor_file), completion["tensor"])
        self.assertEqual(str(metadata_file), completion["metadata"])
        observed = self.runner.retrospective_probe_event()
        self.assertEqual(generation, observed["generation"])
        self.assertEqual(1, observed["ack_rank_mask"] & 1)
        self.assertEqual(0, observed["failure_rank_mask"] & 1)

        unmatched = self.runner.retrospective_probe_event(
            "unmatched-trace", "test-unmatched", retained_sequence_length_b + 2
        )
        self.assertTrue(unmatched["published"])
        unmatched_generation = unmatched["generation"]
        self.assertTrue(self.runner.canRun(inputs_c))
        self.runner.forward(inputs_c)
        torch.cuda.synchronize()
        observed = self.runner.retrospective_probe_event()
        self.assertEqual(unmatched_generation, observed["generation"])
        self.assertEqual(0, observed["ack_rank_mask"] & 1)
        self.assertEqual(0, observed["failure_rank_mask"] & 1)
        self.assertFalse(list(output_dir.glob(f"*gen{unmatched_generation}.*")))

    def _assert_dual_graph_post_trigger_dump(self):
        self.assertFalse(self.model.get_cuda_graph_probe_enabled())
        probe_buffer, probe_layers = self.model.get_cuda_graph_probe_buffer(
            self.graph_bs
        )
        self.assertEqual((2, 0), probe_layers)
        capture_values = probe_buffer.cpu().clone()

        trace_ids = ["bad-trace"] + [
            f"dual-lane-{lane}" for lane in range(1, self.actual_bs)
        ]
        inputs = self._build_inputs(trace_ids=trace_ids)
        inputs.attention_inputs.sequence_lengths = torch.full(
            (self.actual_bs,), 81, dtype=torch.int32
        ).pin_memory()
        inputs.input_ids.add_(1000)

        self.assertTrue(self.runner.canRun(inputs))
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        after_normal = probe_buffer.cpu().clone()
        torch.testing.assert_close(after_normal, capture_values)
        self.assertFalse(list(Path(self.temp_dir.name).glob("*.complete")))

        published = self.runner.retrospective_probe_event(
            "bad-trace", "test-dual-graph", 82
        )
        self.assertTrue(published["published"])
        generation = published["generation"]

        inputs.input_ids.add_(1000)
        self.assertTrue(self.runner.canRun(inputs))
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        after_debug = probe_buffer.cpu().clone()
        self.assertFalse(torch.equal(after_debug, after_normal))

        output_dir = Path(self.temp_dir.name)
        tensor_file = next(output_dir.glob(f"*gen{generation}.bin"))
        metadata_file = next(output_dir.glob(f"*gen{generation}.jsonl"))
        completion_file = next(output_dir.glob(f"*gen{generation}.complete"))
        self.assertFalse(list(output_dir.glob("*.tmp")))
        dumped = torch.from_file(
            str(tensor_file),
            shared=False,
            size=after_debug.numel(),
            dtype=torch.float32,
        ).reshape(after_debug.shape)
        torch.testing.assert_close(
            dumped[:, : self.actual_bs], after_debug[:, : self.actual_bs]
        )
        self.assertTrue(
            torch.equal(
                dumped[:, self.actual_bs :],
                torch.zeros_like(dumped[:, self.actual_bs :]),
            )
        )

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.assertEqual(generation, metadata["event_generation"])
        self.assertEqual("bad-trace", metadata["event_trace_id"])
        self.assertEqual("test-dual-graph", metadata["event_reason"])
        self.assertEqual(self.graph_bs, metadata["graph_bs"])
        self.assertEqual(self.actual_bs, metadata["current_bs"])
        self.assertEqual([2, 0], metadata["layers"])
        self.assertEqual("bad-trace", metadata["lanes"][0]["trace_id"])
        self.assertEqual(81, metadata["lanes"][0]["sequence_length"])
        self.assertTrue(completion_file.is_file())

        observed = self.runner.retrospective_probe_event()
        self.assertEqual(generation, observed["generation"])
        self.assertEqual(1, observed["ack_rank_mask"] & 1)
        self.assertEqual(0, observed["failure_rank_mask"] & 1)

        inputs.input_ids.add_(1000)
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(probe_buffer.cpu(), after_debug)

        unmatched = self.runner.retrospective_probe_event(
            "unmatched-trace", "test-dual-unmatched", 83
        )
        self.assertTrue(unmatched["published"])
        unmatched_generation = unmatched["generation"]
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        observed = self.runner.retrospective_probe_event()
        self.assertEqual(unmatched_generation, observed["generation"])
        self.assertEqual(0, observed["ack_rank_mask"] & 1)
        self.assertFalse(
            list(output_dir.glob(f"*gen{unmatched_generation}.complete"))
        )


if __name__ == "__main__":
    unittest.main()
