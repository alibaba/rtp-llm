import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.moe_layer import (
    DEFAULT_MOE_CHUNK_TOKENS,
    Dsv4MoeLayer,
    chunked_moe_enabled,
    cp_padded_tokens_per_rank_bound,
    moe_chunk_tokens_from_env,
    resolve_moe_max_tokens_per_rank,
    synchronized_moe_chunk_plan,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _mega_output_capacity,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_ranges_enabled,
)


class _FakeMoe(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.token_chunks = []
        self.input_id_chunks = []
        self.output_buffers = []
        self.ranges_enabled = []

    def forward(self, x, input_ids, observer=None, out=None):
        if x.size(0) > self.capacity:
            raise RuntimeError("chunk overflow")
        self.token_chunks.append(x.size(0))
        self.input_id_chunks.append(None if input_ids is None else input_ids.clone())
        self.ranges_enabled.append(record_function_ranges_enabled())
        result = x * 3
        if out is not None:
            out.copy_(result)
            result = out
            self.output_buffers.append(out)
        if observer is not None:
            observer("final_y", result)
        return result


def _fake_layer(dim=4, capacity=5, is_decode_role=False):
    layer = Dsv4MoeLayer.__new__(Dsv4MoeLayer)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.dim = dim
    layer.max_tokens_per_rank = capacity
    layer._is_decode_role = is_decode_role
    layer._moe = _FakeMoe(capacity)
    layer.strategy_name = "local_loop"
    return layer


class ChunkedMoeTest(unittest.TestCase):
    def test_chunk_helpers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(chunked_moe_enabled())
            self.assertEqual(moe_chunk_tokens_from_env(), DEFAULT_MOE_CHUNK_TOKENS)
        self.assertEqual(cp_padded_tokens_per_rank_bound(200002, 4), 50002)
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "0"}):
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(200002, 200002, 4, 8), 50002
            )

    def test_global_zero_disables_chunking_without_shrinking_capacity(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "0",
                "DSV4_MOE_CHUNK_PREFILL": "1",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
            },
            clear=True,
        ):
            self.assertFalse(chunked_moe_enabled())
            self.assertEqual(moe_chunk_tokens_from_env(), 0)
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(1048576, 65536, 4, 8),
                65536,
            )

    def test_prefill_budget_never_expands_existing_capacity(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MOE_CHUNK_PREFILL": "1", "DSV4_MOE_CHUNK_TOKENS": "65536"},
            clear=True,
        ):
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(1048576, 8192, 4, 8),
                8192,
            )

    def test_prefill_budget_shrinks_to_moe_chunk_capacity(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MOE_CHUNK_PREFILL": "1", "DSV4_MOE_CHUNK_TOKENS": "4096"},
            clear=True,
        ):
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(1048576, 65536, 1, 8),
                4096,
            )

    def test_global_chunk_capacity_takes_priority(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "2048",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
            },
            clear=True,
        ):
            self.assertEqual(moe_chunk_tokens_from_env(), 2048)
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(1048576, 65536, 1, 8),
                2048,
            )

    def test_speculative_decode_budget_accounts_for_generated_tokens(self):
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(
                1048576,
                1048576,
                4,
                1024,
                is_decode_role=True,
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            5120,
        )

    def test_chunking_preserves_tokens_and_input_ids(self):
        layer = _fake_layer(dim=3, capacity=5)
        x = torch.arange(17 * 3, dtype=torch.float32).view(17, 3)
        input_ids = torch.arange(100, 117)
        output = layer(x, input_ids)
        self.assertTrue(torch.equal(output, x * 3))
        self.assertEqual(layer._moe.token_chunks, [5, 5, 5, 2])
        self.assertEqual(
            [chunk.tolist() for chunk in layer._moe.input_id_chunks],
            [
                [100, 101, 102, 103, 104],
                [105, 106, 107, 108, 109],
                [110, 111, 112, 113, 114],
                [115, 116],
            ],
        )
        self.assertEqual(len(layer._moe.output_buffers), 4)
        output_storage = output.untyped_storage().data_ptr()
        for buf in layer._moe.output_buffers:
            self.assertEqual(buf.untyped_storage().data_ptr(), output_storage)

    def test_decode_rejects_oversized_input(self):
        layer = _fake_layer(capacity=4, is_decode_role=True)
        with self.assertRaisesRegex(ValueError, "decode MoE input tokens=5"):
            layer(torch.zeros(5, 4), torch.arange(5))
        self.assertEqual(layer._moe.token_chunks, [])

    def test_input_ids_are_optional(self):
        layer = _fake_layer(dim=3, capacity=5)
        x = torch.arange(7 * 3, dtype=torch.float32).view(7, 3)
        output = layer(x)
        self.assertTrue(torch.equal(output, x * 3))
        self.assertEqual(layer._moe.token_chunks, [5, 2])
        self.assertEqual(layer._moe.input_id_chunks, [None, None])

    def test_cuda_graph_capture_rejects_oversized_input(self):
        layer = _fake_layer(capacity=4)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(
                    ValueError, "CUDA graph capture MoE input tokens=5"
                ):
                    layer(torch.zeros(5, 4), torch.arange(5))
        self.assertEqual(layer._moe.token_chunks, [])

    def test_input_ids_must_match_flattened_tokens(self):
        layer = _fake_layer(dim=2, capacity=4)
        with self.assertRaisesRegex(ValueError, "input_ids has 4 tokens, expected 5"):
            layer(torch.zeros(5, 2), torch.arange(4))

    def test_mega_output_capacity_uses_aligned_buffer_capacity(self):
        buffer = type("Buffer", (), {"num_max_tokens_per_rank": 384})()
        self.assertEqual(_mega_output_capacity(buffer, 17), 384)

    def test_synchronizes_chunk_plan_once_for_the_layer_stack(self):
        first = _fake_layer()
        second = _fake_layer()
        first.strategy_name = "mega_moe_se"
        second.strategy_name = "mega_moe_se"
        layers = [SimpleNamespace(ffn=first), SimpleNamespace(ffn=second)]

        def set_remote_max(token_count, **_):
            token_count.fill_(9)

        with (
            mock.patch(
                "rtp_llm.models_py.modules.dsv4.moe_layer.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.dsv4.moe_layer.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.dsv4.moe_layer.dist.all_reduce",
                side_effect=set_remote_max,
            ) as all_reduce,
        ):
            with synchronized_moe_chunk_plan(layers, 3, torch.device("cpu")):
                self.assertEqual(
                    first._synchronized_chunk_tokens(3, torch.device("cpu")), 9
                )
                self.assertEqual(
                    second._synchronized_chunk_tokens(3, torch.device("cpu")), 9
                )

        all_reduce.assert_called_once()

    def test_dsv4_profiler_switch_disables_generic_moe_ranges(self):
        layer = _fake_layer(capacity=5)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4._profiler._RANGES_ENABLED", False
        ):
            layer(torch.ones(1, 4), torch.zeros(1, dtype=torch.long))

        self.assertEqual(layer._moe.ranges_enabled, [False])

    def test_debug_observer_preserves_tensor_names_and_position_filter(self):
        layer = _fake_layer()
        layer.layer_id = 3
        positions = torch.tensor([4, 9])
        records = []

        def record(level, name, tensor):
            records.append((level, name, tensor.clone()))

        with mock.patch.object(_rt, "should_record_layer", return_value=True):
            with mock.patch.object(_rt, "_DBG_GLOBAL_POS", 9):
                with mock.patch.object(_rt, "record_if_level", side_effect=record):
                    observer = layer._debug_observer(positions)
                    self.assertIsNotNone(observer)
                    for kind in (
                        "input",
                        "topk_weights",
                        "topk_indices",
                        "routed_y",
                        "shared_y",
                        "final_y",
                    ):
                        observer(kind, torch.arange(4).view(2, 2))

        names = [name for _, name, _ in records]
        for suffix in (
            "moe_x_in",
            "moe_topk_weights",
            "moe_topk_indices",
            "moe_routed_y",
            "moe_shared_y",
            "moe_y",
        ):
            self.assertIn(f"L03_{suffix}", names)
            self.assertIn(f"L03_{suffix}_pos9", names)
        position_records = [
            tensor for _, name, tensor in records if name.endswith("pos9")
        ]
        self.assertTrue(all(tensor.shape == (1, 2) for tensor in position_records))


if __name__ == "__main__":
    unittest.main()
