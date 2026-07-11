import json
import os
import tempfile
import unittest

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
        return self.probe.get_capture(graph_bs)

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
        os.environ["RTPLLM_QWEN3_NEXT_GRAPH_PROBE"] = (
            "1" if self.graph_probe_enabled else "0"
        )
        os.environ["RTPLLM_QWEN3_NEXT_GRAPH_PROBE_LAYERS"] = "99,77"

        self.model = _GraphProbeModel()
        self.runner = CudaGraphRunner()
        self.runner.init_decode(
            self.model,
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.tokens_per_block,
            [self.graph_bs],
            self.num_tokens_per_bs,
            True,
            True,
        )
        torch.cuda.synchronize()

    def tearDown(self):
        self.runner = None
        self.temp_dir.cleanup()

    def _build_inputs(self):
        token_rows = self.actual_bs * self.num_tokens_per_bs
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
        inputs.trace_ids = [f"lane-{lane}" for lane in range(self.actual_bs)]

        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.is_target_verify = True
        attention.dtype = get_typemeta(torch.zeros(1, dtype=torch.float32))
        attention.input_lengths = torch.full(
            (self.actual_bs,),
            self.num_tokens_per_bs,
            dtype=torch.int32,
        ).pin_memory()
        attention.prefix_lengths = torch.full(
            (self.actual_bs,),
            self.max_seq_len - self.num_tokens_per_bs,
            dtype=torch.int32,
        ).pin_memory()
        attention.sequence_lengths = torch.empty(
            0, dtype=torch.int32
        ).pin_memory()
        attention.sequence_lengths_plus_1_d = torch.full(
            (self.actual_bs,),
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
            1, self.actual_bs + 1, dtype=torch.int32, device="cuda"
        ).reshape(self.actual_bs, 1)
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
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls + 1, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            records = [json.loads(line) for line in checksum_stream if line.strip()]
        self.assertEqual(4, len(records))
        expected_lanes = [lane for lane in range(self.actual_bs) if lane != 1]
        self.assertEqual(
            [expected_lanes, expected_lanes],
            [[lane["lane"] for lane in record["lanes"]] for record in records[-2:]],
        )

        status_calls = self.model.debug_status_calls
        inputs.attention_inputs.prefix_lengths[: self.actual_bs].fill_(base_prefix + 8)
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(4, sum(1 for line in checksum_stream if line.strip()))

        inputs.attention_inputs.sequence_lengths = (
            inputs.attention_inputs.input_lengths[: self.actual_bs].clone().pin_memory()
        )
        inputs.attention_inputs.sequence_lengths.add_(7)
        status_calls = self.model.debug_status_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls + 1, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(6, sum(1 for line in checksum_stream if line.strip()))

        inputs.attention_inputs.sequence_lengths.add_(1)
        status_calls = self.model.debug_status_calls
        self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(status_calls, self.model.debug_status_calls)
        with open(self.checksum_file, encoding="utf-8") as checksum_stream:
            self.assertEqual(6, sum(1 for line in checksum_stream if line.strip()))


if __name__ == "__main__":
    unittest.main()
