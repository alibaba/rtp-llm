import bisect
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    concat_and_cache_mla,
    get_typemeta,
    rtp_llm_ops,
)

# Independent expected grids for the C++ capture and replay implementation.
GRID24 = [1, 2, 4, 8, 10, 12, 14, 16, 20, 24]
GRID384 = GRID24 + [
    28,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
]


def padded(rows, grid=GRID24):
    return grid[bisect.bisect_left(grid, rows)]


class Attention:
    def __init__(self, inputs, check_padding=False):
        self.check_padding = check_padding
        self.params = rtp_llm_ops.SparseMlaParams()
        self.params.fill_params(inputs.attention_inputs, 64, False)

    def prepare_cuda_graph(self, inputs):
        lengths = inputs.input_lengths.tolist()
        offsets = inputs.cu_seqlens.tolist()
        assert offsets == [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        assert inputs.cu_seqlens_host.tolist() == offsets
        prefixes = inputs.prefix_lengths.tolist()
        expected_kv = [0] + list(
            torch.tensor([q + k for q, k in zip(lengths, prefixes)]).cumsum(0).tolist()
        )
        # Legacy behavior stays unchanged; adaptive token buckets require
        # consistent zero-length request slots even when token rows are padded.
        if (
            self.check_padding
            or sum(lengths) == self.params.multi_token_decode_total_tokens
        ):
            assert inputs.cu_kv_seqlens.tolist() == expected_kv
        for i, length in enumerate(lengths):
            if length == 0:
                assert inputs.prefix_lengths[i].item() == 0
                assert (
                    inputs.kv_cache_kernel_block_id_device[i].count_nonzero().item()
                    == 0
                )
        self.params.fill_multi_token_decode_cuda_graph_params(
            inputs.input_lengths,
            inputs.prefix_lengths,
            inputs.kv_cache_kernel_block_id_device,
            64,
        )
        if self.check_padding:
            live = sum(lengths)
            assert (self.params.slot_mapping[live:] == -1).all().item()
            assert (self.params.expanded_seq_lens[live:] == 0).all().item()
            assert (self.params.positions_d[live:] == 0).all().item()


class ShapeRecordingModel:
    def __init__(self, check_padding=False):
        self.check_padding = check_padding
        self.captured_rows = []

    def prepare_fmha_impl(self, inputs, *args):
        return Attention(inputs, self.check_padding)

    def forward(self, inputs, attn=None):
        rows = inputs.input_hiddens.shape[0]
        if torch.cuda.is_current_stream_capturing():
            self.captured_rows.append(rows)
        result = PyModelOutputs()
        # Real SparseMLA metadata kernel must retain exact token-to-request,
        # causal position and physical KV write-slot mappings across replays.
        params = attn.params
        bias = params.batch_indice_d + params.positions_d + params.slot_mapping
        result.hidden_states = inputs.input_hiddens + rows + bias[:, None]
        return result


def make_inputs(lengths, value):
    rows = sum(lengths)
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    inputs.input_hiddens = torch.full(
        (rows, 8), value, device="cuda", dtype=torch.float16
    )
    attn = PyAttentionInputs()
    attn.is_prefill = True
    attn.is_target_verify = True
    attn.input_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    attn.prefix_lengths = torch.full_like(attn.input_lengths, 8)
    attn.sequence_lengths = torch.empty(0, device="cuda", dtype=torch.int32)
    attn.cu_seqlens_host = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    )
    attn.cu_seqlens = attn.cu_seqlens_host.cuda()
    attn.cu_kv_seqlens = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            (attn.input_lengths + 8).cumsum(0).int(),
        )
    )
    attn.kv_cache_kernel_block_id_host = torch.ones(
        (len(lengths), 1), dtype=torch.int32
    )
    attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id_host.cuda()
    attn.kv_cache_block_id_host = attn.kv_cache_kernel_block_id_host
    attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
    attn.padding_offset = torch.zeros(rows, device="cuda", dtype=torch.int32)
    attn.total_tokens = rows
    attn.context_total_kv_length = rows + 8 * len(lengths)
    attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))
    inputs.attention_inputs = attn
    return inputs


class AdaptiveGraphTest(unittest.TestCase):
    def test_short_graph_and_ragged_replay(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = ShapeRecordingModel(check_padding=enabled)
                runner = CudaGraphRunner()
                runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, enabled, 4)
                self.assertEqual(
                    len(model.captured_rows), len(GRID24) if enabled else 2
                )
                for value, lengths in enumerate(
                    ([4], [1, 3], [6, 1, 2], [1], [6], [4]), 1
                ):
                    inputs = make_inputs(lengths, value)
                    self.assertTrue(runner.canRun(inputs))
                    result = runner.forward(inputs).hidden_states.clone()
                    batch = 1 if len(lengths) == 1 else 4
                    capacity = padded(sum(lengths)) if enabled else batch * 6
                    expected_bias = torch.tensor(
                        [
                            i + 80 + 2 * j
                            for i, length in enumerate(lengths)
                            for j in range(length)
                        ],
                        device="cuda",
                        dtype=torch.float16,
                    )
                    torch.testing.assert_close(
                        result,
                        inputs.input_hiddens + capacity + expected_bias[:, None],
                        rtol=0,
                        atol=0,
                    )
                del runner

    def test_every_row_and_same_graph_different_request_boundaries(self):
        model = ShapeRecordingModel(check_padding=True)
        runner = CudaGraphRunner()
        # Sparse user BS list must not leave holes in adaptive row coverage.
        runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, True, 4)
        self.assertEqual(sorted(model.captured_rows), GRID24)
        for rows in list(range(1, 25)) + list(range(24, 0, -1)):
            for batch in range((rows + 5) // 6, min(rows, 4) + 1):
                lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                for lengths in (lengths, lengths[::-1]):
                    with self.subTest(rows=rows, lengths=lengths):
                        inputs = make_inputs(lengths, 3)
                        attn = inputs.attention_inputs
                        prefixes = (
                            torch.arange(batch, device="cuda", dtype=torch.int32) + 8
                        )
                        blocks = (
                            torch.arange(batch, device="cuda", dtype=torch.int32) + 1
                        )
                        attn.prefix_lengths = prefixes
                        attn.cu_kv_seqlens = torch.cat(
                            (
                                torch.zeros(1, device="cuda", dtype=torch.int32),
                                (attn.input_lengths + prefixes).cumsum(0).int(),
                            )
                        )
                        attn.kv_cache_kernel_block_id_device = blocks[
                            :, None
                        ].contiguous()
                        attn.kv_cache_kernel_block_id_host = (
                            attn.kv_cache_kernel_block_id_device.cpu()
                        )
                        self.assertTrue(runner.canRun(inputs))
                        result = runner.forward(inputs).hidden_states.clone()
                        expected_bias = torch.tensor(
                            [
                                i + 2 * (8 + i + j) + (i + 1) * 64
                                for i, length in enumerate(lengths)
                                for j in range(length)
                            ],
                            device="cuda",
                            dtype=torch.float16,
                        )
                        expected = (
                            inputs.input_hiddens + padded(rows) + expected_bias[:, None]
                        )
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for lengths in ([1] * 5, [7], []):
            with self.subTest(invalid=lengths), self.assertRaises(Exception):
                runner.canRun(make_inputs(lengths, 0))

    def test_default_dense_rows_single_request(self):
        model = ShapeRecordingModel(check_padding=True)
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [], 3, True, 1)
        self.assertEqual(sorted(model.captured_rows), [1, 2, 3])
        for rows in (3, 1, 2, 1):
            inputs = make_inputs([rows], 2)
            self.assertTrue(runner.canRun(inputs))
            actual = runner.forward(inputs).hidden_states.clone()
            bias = torch.arange(rows, device="cuda", dtype=torch.float16) * 2 + 80
            torch.testing.assert_close(
                actual, inputs.input_hiddens + rows + bias[:, None], rtol=0, atol=0
            )

    @patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_CAPTURE_ROWS": "invalid-ignored"})
    def test_non_adaptive_single_token_decode_keeps_bs_buckets(self):
        class PlainAttention:
            def prepare_cuda_graph(self, inputs):
                pass

        class PlainModel(ShapeRecordingModel):
            def prepare_fmha_impl(self, inputs, *args):
                return PlainAttention()

            def forward(self, inputs, attn=None):
                rows = inputs.input_hiddens.shape[0]
                if torch.cuda.is_current_stream_capturing():
                    self.captured_rows.append(rows)
                result = PyModelOutputs()
                result.hidden_states = inputs.input_hiddens + rows
                return result

        model = PlainModel()
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [1, 4], 1, False, 4)
        self.assertEqual(sorted(model.captured_rows), [1, 4])
        for batch in (4, 2, 1, 3):
            inputs = make_inputs([1] * batch, 2)
            inputs.attention_inputs.is_prefill = False
            inputs.attention_inputs.is_target_verify = False
            inputs.attention_inputs.sequence_lengths = torch.full(
                (batch,), 8, dtype=torch.int32
            ).pin_memory()
            self.assertTrue(runner.canRun(inputs))
            actual = runner.forward(inputs).hidden_states.clone()
            torch.testing.assert_close(
                actual,
                torch.full_like(inputs.input_hiddens, 3 if batch == 1 else 6),
                rtol=0,
                atol=0,
            )

    def test_custom_buckets_capture_replay_and_cost_model_agree(self):
        from rtp_llm.models_py.sampling.adaptive_session import (
            AdaptiveVerificationSession,
        )

        # Non-default, nonuniform buckets; normalization and final closure must
        # match the actual captured graph keys, not a duplicate grid formula.
        grid = [3, 11, 19, 24]
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_CAPTURE_ROWS": "19,3,11,3"}):
            model = ShapeRecordingModel(check_padding=True)
            runner = CudaGraphRunner()
            runner.init_decode(model, 8, 64, 64, 64, [], 6, True, 4)
        self.assertEqual(sorted(model.captured_rows), grid)
        session = AdaptiveVerificationSession(
            4, 5, "cuda", capture_rows=sorted(model.captured_rows)
        )
        for rows in range(1, 25):
            bucket = padded(rows, grid)
            self.assertEqual(session._cost_key(rows), bucket)
            for batch in range((rows + 5) // 6, min(rows, 4) + 1):
                lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                inputs = make_inputs(lengths, 2)
                self.assertTrue(runner.canRun(inputs))
                actual = runner.forward(inputs).hidden_states.clone()
                bias = torch.tensor(
                    [
                        i + 80 + 2 * j
                        for i, length in enumerate(lengths)
                        for j in range(length)
                    ],
                    device="cuda",
                    dtype=torch.float16,
                )
                torch.testing.assert_close(
                    actual,
                    inputs.input_hiddens + bucket + bias[:, None],
                    rtol=0,
                    atol=0,
                )

    def test_manual_dense_and_maximum_only_grids(self):
        from rtp_llm.models_py.sampling.adaptive_session import (
            AdaptiveVerificationSession,
        )

        for grid in (list(range(1, 25)), [24]):
            with self.subTest(grid=grid), patch.dict(
                os.environ, {"RTP_LLM_ADAPTIVE_CAPTURE_ROWS": ",".join(map(str, grid))}
            ):
                model = ShapeRecordingModel(check_padding=True)
                runner = CudaGraphRunner()
                runner.init_decode(model, 8, 64, 64, 64, [], 6, True, 4)
                self.assertEqual(sorted(model.captured_rows), grid)
                session = AdaptiveVerificationSession(4, 5, "cuda", capture_rows=grid)
                for rows in range(24, 0, -1):
                    bucket = padded(rows, grid)
                    self.assertEqual(session._cost_key(rows), bucket)
                    batch = (rows + 5) // 6
                    lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                    inputs = make_inputs(lengths, 2)
                    self.assertTrue(runner.canRun(inputs))
                    actual = runner.forward(inputs).hidden_states.clone()
                    bias = torch.tensor(
                        [
                            i + 80 + 2 * j
                            for i, length in enumerate(lengths)
                            for j in range(length)
                        ],
                        device="cuda",
                        dtype=torch.float16,
                    )
                    torch.testing.assert_close(
                        actual,
                        inputs.input_hiddens + bucket + bias[:, None],
                        rtol=0,
                        atol=0,
                    )

    def test_concurrency_64_sparse_graphs_cover_all_384_live_rows(self):
        model = ShapeRecordingModel(check_padding=True)
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [], 6, True, 64)
        self.assertEqual(sorted(model.captured_rows), GRID384)
        for rows in range(1, 385):
            # Exercise both minimal and maximal request counts at the same key,
            # including row counts across CUDA block/warp boundaries.
            for batch in sorted({(rows + 5) // 6, min(rows, 64)}, reverse=True):
                lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                with self.subTest(rows=rows, batch=batch):
                    inputs = make_inputs(lengths, 2)
                    self.assertTrue(runner.canRun(inputs))
                    actual = runner.forward(inputs).hidden_states.clone()
                    bias = torch.tensor(
                        [
                            i + 80 + 2 * j
                            for i, length in enumerate(lengths)
                            for j in range(length)
                        ],
                        device="cuda",
                        dtype=torch.float16,
                    )
                    torch.testing.assert_close(
                        actual,
                        inputs.input_hiddens + padded(rows, GRID384) + bias[:, None],
                        rtol=0,
                        atol=0,
                    )

    def test_padded_graph_replay_preserves_cache_and_live_outputs(self):
        class CacheModel(ShapeRecordingModel):
            def __init__(self):
                super().__init__(check_padding=True)
                self.cache = torch.full(
                    (8, 64, 576), -7, device="cuda", dtype=torch.bfloat16
                )
                self.scale = torch.ones(1, device="cuda")

            def forward(self, inputs, attn=None):
                rows = inputs.input_hiddens.shape[0]
                values = inputs.input_hiddens[:, :1].to(torch.bfloat16)
                kv = values.expand(rows, 512).contiguous()
                rope = values.expand(rows, 64).contiguous()
                concat_and_cache_mla(
                    kv, rope, self.cache, attn.params.slot_mapping, "auto", self.scale
                )
                return super().forward(inputs, attn)

        model = CacheModel()
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, True, 4)
        # Reuse 8-row graph with full and padded inputs, then switch buckets.
        for lengths in ([2, 2, 2, 2], [3, 2], [6, 1], [5], [1], [6, 6, 5], [2, 2, 1]):
            with self.subTest(lengths=lengths):
                model.cache.fill_(-7)
                inputs = make_inputs(lengths, 3)
                batch = len(lengths)
                blocks = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)[
                    :, None
                ]
                inputs.attention_inputs.kv_cache_kernel_block_id_device = blocks
                inputs.attention_inputs.kv_cache_kernel_block_id_host = blocks.cpu()
                inputs.attention_inputs.kv_cache_block_id_device = blocks
                inputs.attention_inputs.kv_cache_block_id_host = blocks.cpu()
                self.assertTrue(runner.canRun(inputs))
                result = runner.forward(inputs).hidden_states.clone()
                expected_cache = torch.full_like(model.cache, -7)
                for i, length in enumerate(lengths):
                    expected_cache[i + 1, 8 : 8 + length] = 3
                torch.testing.assert_close(model.cache, expected_cache, rtol=0, atol=0)
                bias = torch.tensor(
                    [
                        i + 16 + 2 * j + (i + 1) * 64
                        for i, l in enumerate(lengths)
                        for j in range(l)
                    ],
                    device="cuda",
                )
                torch.testing.assert_close(
                    result,
                    inputs.input_hiddens + padded(sum(lengths)) + bias[:, None],
                    rtol=0,
                    atol=0,
                )

    def test_fused_bf16_fp8_cache_writes_match_unpadded_reference(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla import (
            fused_qk_rope_cat_cache_mla,
        )

        class FusedCacheModel(ShapeRecordingModel):
            def __init__(self, cache_type):
                super().__init__(check_padding=True)
                self.cache_type = cache_type
                fp8 = cache_type == "fp8_ds_mla"
                self.cache = torch.full(
                    (8, 64, 656 if fp8 else 576),
                    17,
                    device="cuda",
                    dtype=torch.uint8 if fp8 else torch.bfloat16,
                )
                self.cos_sin = torch.cat(
                    (torch.ones(64, 32), torch.zeros(64, 32)), dim=1
                ).cuda()
                self.inputs_by_rows = {}

            def write(self, hiddens, slots, positions, cache):
                rows = hiddens.shape[0]
                values = hiddens[:, :1].to(torch.bfloat16)
                q = values[:, :, None].expand(rows, 2, 128).contiguous()
                fused_qk_rope_cat_cache_mla(
                    q,
                    values.expand(rows, 512).contiguous(),
                    values.expand(rows, 64).contiguous(),
                    cache,
                    slots,
                    positions,
                    self.cos_sin,
                    512,
                    64,
                    True,
                    self.cache_type,
                )
                return q

            def forward(self, inputs, attn=None):
                rows = inputs.input_hiddens.shape[0]
                if torch.cuda.is_current_stream_capturing():
                    self.inputs_by_rows[rows] = inputs
                q = self.write(
                    inputs.input_hiddens,
                    attn.params.slot_mapping,
                    attn.params.positions_d,
                    self.cache,
                )
                result = PyModelOutputs()
                result.hidden_states = q[:, 0, -8:].to(torch.float16)
                return result

        for cache_type in ("auto", "fp8_ds_mla"):
            model = FusedCacheModel(cache_type)
            runner = CudaGraphRunner()
            runner.init_decode(model, 8, 64, 64, 64, [], 6, True, 4)
            for lengths in (
                [2, 2, 2, 2],
                [3, 2],
                [6, 1],
                [1, 1, 1],
                [6, 6, 5],
                [1],
                [5],
            ):
                with self.subTest(cache=cache_type, lengths=lengths):
                    rows = sum(lengths)
                    inputs = make_inputs(lengths, 3)
                    batch = len(lengths)
                    blocks = torch.arange(
                        1, batch + 1, device="cuda", dtype=torch.int32
                    )[:, None]
                    for name in ("kv_cache_kernel_block_id", "kv_cache_block_id"):
                        setattr(inputs.attention_inputs, name + "_device", blocks)
                        setattr(inputs.attention_inputs, name + "_host", blocks.cpu())
                    slots = torch.tensor(
                        [
                            (i + 1) * 64 + 8 + j
                            for i, l in enumerate(lengths)
                            for j in range(l)
                        ],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    positions = torch.tensor(
                        [8 + j for l in lengths for j in range(l)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    expected = torch.full_like(model.cache, 17)
                    expected_q = model.write(
                        inputs.input_hiddens, slots, positions, expected
                    )
                    model.cache.fill_(17)
                    # Poison capture input to catch stale padded hidden rows.
                    model.inputs_by_rows[padded(rows)].input_hiddens.fill_(float("nan"))
                    self.assertTrue(runner.canRun(inputs))
                    result = runner.forward(inputs).hidden_states.clone()
                    torch.testing.assert_close(model.cache, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        result, expected_q[:, 0, -8:].to(torch.float16), rtol=0, atol=0
                    )
                    self.assertEqual(result.shape[0], rows)
                    tail = model.inputs_by_rows[padded(rows)].input_hiddens[rows:]
                    self.assertTrue((tail == 0).all().item())


if __name__ == "__main__":
    unittest.main()
