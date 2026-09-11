import itertools
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _mega_output_capacity,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    _FINAL_OUT_CACHE,
    DEFAULT_MOE_CHUNK_TOKENS,
    ChunkedFp8Fp4MoeLayer,
    cp_padded_tokens_per_rank_bound,
    resolve_moe_max_tokens_per_rank,
    synchronized_moe_chunk_plan,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    disable_record_function_ranges,
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
    layer = ChunkedFp8Fp4MoeLayer.__new__(ChunkedFp8Fp4MoeLayer)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.dim = dim
    layer.max_tokens_per_rank = capacity
    layer._is_decode_role = is_decode_role
    layer.chunking_enabled = True
    layer._observer_factory = None
    layer._record_function_scope = nullcontext
    layer._moe = _FakeMoe(capacity)
    layer.strategy_name = "local_loop"
    return layer


class ChunkedMoeTest(unittest.TestCase):
    def test_observer_receives_positions_for_each_chunk(self):
        layer = _fake_layer(dim=2, capacity=3)
        records = []

        def factory(positions):
            def observe(kind, tensor):
                records.append((kind, positions.clone(), tensor.clone()))

            return observe

        layer._observer_factory = factory
        x = torch.arange(14, dtype=torch.float32).view(7, 2)
        layer(x, positions=torch.arange(7))
        self.assertEqual(
            [p.tolist() for _, p, _ in records], [[0, 1, 2], [3, 4, 5], [6]]
        )
        torch.testing.assert_close(torch.cat([t for _, _, t in records]), x * 3)

    def test_caller_controls_profiling_scope(self):
        layer = _fake_layer()
        layer._record_function_scope = disable_record_function_ranges
        layer(torch.ones(7, 4))
        self.assertEqual(layer._moe.ranges_enabled, [False, False])
        self.assertTrue(record_function_ranges_enabled())

    def test_disabled_chunking_skips_stack_and_layer_synchronization(self):
        layer = _fake_layer()
        layer.strategy_name = "mega_moe_se"
        layer.chunking_enabled = False
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.all_reduce") as sync,
        ):
            with synchronized_moe_chunk_plan([layer], 3, torch.device("cpu")):
                layer(torch.ones(3, 4))
        sync.assert_not_called()

    def test_prefill_budget_shrinks_to_moe_chunk_capacity(self):
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(1048576, 65536, 1, 8, chunk_tokens=4096),
            4096,
        )

    def test_cp_prefill_budget_is_partitioned_before_chunking(self):
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(1048576, 8192, 4, 8, chunk_tokens=65536),
            2048,
        )

    def test_chunk_helpers(self):
        self.assertEqual(cp_padded_tokens_per_rank_bound(200002, 4), 50002)
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(
                200002, 200002, 4, 8, chunking_enabled=False
            ),
            50002,
        )
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(200002, 200002, 4, 8),
            DEFAULT_MOE_CHUNK_TOKENS,
        )

    def test_cp_budget_covers_independently_padded_requests(self):
        for cp_size in (2, 3, 4):
            for lengths in itertools.product(range(1, 10), repeat=3):
                padded_local = sum(
                    2 * ((length + 2 * cp_size - 1) // (2 * cp_size))
                    for length in lengths
                )
                capacity = resolve_moe_max_tokens_per_rank(
                    9,
                    sum(lengths),
                    cp_size,
                    8,
                    max_context_batch_size=3,
                    chunking_enabled=False,
                )
                self.assertGreaterEqual(capacity, padded_local)
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(
                4096, 12288, 2, 8, max_context_batch_size=3, chunk_tokens=4096
            ),
            4096,
        )

    def test_nonchunked_forward_reuses_final_output_storage(self):
        _FINAL_OUT_CACHE.clear()
        self.addCleanup(_FINAL_OUT_CACHE.clear)
        layer = _fake_layer(dim=3, capacity=5)
        x = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)

        first = layer(x)
        cache_key = (x.device, layer.dim, x.dtype)
        self.assertEqual(_FINAL_OUT_CACHE[cache_key].size(0), layer.max_tokens_per_rank)
        second = layer(x + 1)

        self.assertTrue(torch.equal(second, (x + 1) * 3))
        self.assertEqual(len(layer._moe.output_buffers), 2)
        self.assertEqual(
            first.untyped_storage().data_ptr(),
            second.untyped_storage().data_ptr(),
        )
        self.assertTrue(
            all(
                buf.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
                for buf in layer._moe.output_buffers
            )
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

    def test_non_speculative_decode_budget_uses_one_token_per_request(self):
        self.assertEqual(
            resolve_moe_max_tokens_per_rank(
                1048576,
                1048576,
                4,
                1024,
                is_decode_role=True,
                is_speculative=False,
                gen_num_per_cycle=4,
            ),
            1024,
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

    def test_pdfusion_decode_forward_skips_world_token_sync(self):
        layer = _fake_layer(capacity=4)
        layer.strategy_name = "mega_moe_se"
        with (
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.all_reduce"
            ) as all_reduce,
        ):
            output = layer(
                torch.ones(3, 4),
                torch.arange(3),
                is_decode_forward=True,
            )

        all_reduce.assert_not_called()
        self.assertEqual(layer._moe.token_chunks, [3])
        self.assertTrue(torch.equal(output, torch.full((3, 4), 3.0)))

    def test_pdfusion_decode_forward_does_not_chunk_oversized_input(self):
        layer = _fake_layer(capacity=4)
        with self.assertRaisesRegex(ValueError, "decode MoE input tokens=5"):
            layer(
                torch.zeros(5, 4),
                torch.arange(5),
                is_decode_forward=True,
            )
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
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.all_reduce",
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

    def test_mixed_non_mega_layer_ignores_synchronized_token_count(self):
        mega = _fake_layer()
        local = _fake_layer()
        mega.strategy_name = "mega_moe_se"
        local.strategy_name = "local_loop"
        layers = [SimpleNamespace(ffn=mega), SimpleNamespace(ffn=local)]

        def set_remote_max(token_count, **_):
            token_count.fill_(9)

        with (
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer.dist.all_reduce",
                side_effect=set_remote_max,
            ),
        ):
            with synchronized_moe_chunk_plan(layers, 3, torch.device("cpu")):
                self.assertEqual(
                    mega._synchronized_chunk_tokens(3, torch.device("cpu")), 9
                )
                self.assertEqual(
                    local._synchronized_chunk_tokens(3, torch.device("cpu")), 3
                )


if __name__ == "__main__":
    unittest.main()
