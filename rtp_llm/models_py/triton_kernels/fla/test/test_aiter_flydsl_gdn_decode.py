import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla import aiter_flydsl_decode as adapter
from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_decode import (
    aiter_flydsl_gdn_decode,
    copy_aiter_flydsl_gdn_decode_state_at_block_boundary,
    is_aiter_flydsl_gdn_decode_supported,
    prepare_aiter_flydsl_gdn_decode_state_indices,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


def _make_decode_inputs(
    *,
    batch: int = 2,
    query_length: int = 1,
    key_heads: int = 2,
    value_heads: int = 8,
    dim: int = 128,
    state_dtype: torch.dtype = torch.float32,
):
    # Match the production layout: q/k/v are cross-strided views of one
    # mixed_qkv allocation split along the head dimension.
    mixed_qkv = torch.randn(
        batch,
        query_length,
        key_heads * 2 + value_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = torch.split(mixed_qkv, [key_heads, key_heads, value_heads], dim=2)
    a = torch.randn(
        batch * query_length,
        value_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn_like(a)
    state = torch.randn(
        batch * 3 + 1,
        value_heads,
        dim,
        dim,
        device="cuda",
        dtype=state_dtype,
    )
    return q, k, v, a, b, state


@unittest.skipUnless(torch.cuda.is_available(), "A CUDA/HIP device is required")
class AiterFlydslGdnDecodeCommonTest(unittest.TestCase):
    def test_prepare_decode_indices_honors_noncontiguous_block_map_stride(self):
        padded_block_map = torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        block_map = padded_block_map[:, :3]
        self.assertEqual(block_map.stride(), (5, 1))
        sequence_lengths_plus_1 = torch.tensor(
            [1002, 1025], device="cuda", dtype=torch.int32
        )

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, sequence_lengths_plus_1, 1024
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 4])
        self.assertEqual(write_indices.cpu().tolist(), [1, 5])

    def test_prepare_decode_indices_maps_padding_and_out_of_range_rows_to_dummy(self):
        block_map = torch.tensor(
            [[1, 2], [0, 0], [3, 4]],
            device="cuda",
            dtype=torch.int32,
        )
        sequence_lengths_plus_1 = torch.tensor(
            [1025, 0, 4097], device="cuda", dtype=torch.int32
        )

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, sequence_lengths_plus_1, 1024
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 0, 0])
        self.assertEqual(write_indices.cpu().tolist(), [2, 0, 0])

    def test_prepare_decode_indices_accepts_empty_batch(self):
        block_map = torch.empty((0, 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty((0,), device="cuda", dtype=torch.int32)

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024
        )

        self.assertEqual(read_indices.numel(), 0)
        self.assertEqual(write_indices.numel(), 0)

    def test_prepare_decode_indices_validates_arguments(self):
        valid_map = torch.ones((2, 2), device="cuda", dtype=torch.int32)
        valid_lengths = torch.ones((2,), device="cuda", dtype=torch.int32)
        cases = (
            (
                valid_map.reshape(-1),
                valid_lengths,
                1024,
                "block_map must be 2D",
            ),
            (
                torch.empty((2, 0), device="cuda", dtype=torch.int32),
                valid_lengths,
                1024,
                "at least one block column",
            ),
            (
                valid_map.to(torch.int64),
                valid_lengths,
                1024,
                "block_map must be int32",
            ),
            (
                valid_map,
                valid_lengths.reshape(1, 2),
                1024,
                "sequence_lengths_plus_1 must be 1D",
            ),
            (
                valid_map,
                valid_lengths.to(torch.int64),
                1024,
                "sequence_lengths_plus_1 must be int32",
            ),
            (
                valid_map.cpu(),
                valid_lengths,
                1024,
                "must be on the same device",
            ),
            (
                valid_map,
                valid_lengths[:1],
                1024,
                "count must equal",
            ),
            (
                valid_map,
                valid_lengths,
                0,
                "seq_size_per_block must be positive",
            ),
        )
        for block_map, lengths, block_size, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    prepare_aiter_flydsl_gdn_decode_state_indices(
                        block_map, lengths, block_size
                    )

    def test_shape_gate_accepts_production_stride_and_rejects_invalid_inputs(self):
        valid = _make_decode_inputs()
        q, k, v, a, b, state = valid
        self.assertFalse(q.is_contiguous())
        self.assertFalse(k.is_contiguous())
        self.assertFalse(v.is_contiguous())

        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter, "_get_aiter_flydsl_gdn_decode", return_value=mock.Mock()
            ),
        ):
            self.assertTrue(is_aiter_flydsl_gdn_decode_supported(*valid))

            unaligned_storage = torch.empty(
                state.numel() + 1, device="cuda", dtype=state.dtype
            )
            unaligned_state = unaligned_storage[1:].view_as(state)
            cases = (
                ("q-rank", (q.squeeze(1), k, v, a, b, state)),
                (
                    "multi-token",
                    _make_decode_inputs(query_length=2),
                ),
                (
                    "shape-not-whitelisted",
                    _make_decode_inputs(key_heads=4, value_heads=8),
                ),
                ("q-fp16", (q.to(torch.float16), k, v, a, b, state)),
                ("k-shape", (q, k[:, :, :1], v, a, b, state)),
                ("v-shape", (q, k, v[:, :0], a, b, state)),
                ("state-fp16", (q, k, v, a, b, state.to(torch.float16))),
                ("state-unaligned", (q, k, v, a, b, unaligned_state)),
                (
                    "state-inner-stride",
                    (q, k, v, a, b, state.transpose(-1, -2)),
                ),
                ("a-numel", (q, k, v, a[:-1], b, state)),
            )
            for name, inputs in cases:
                with self.subTest(name=name):
                    self.assertFalse(is_aiter_flydsl_gdn_decode_supported(*inputs))

            with mock.patch.object(
                adapter, "_get_aiter_flydsl_gdn_decode", return_value=None
            ):
                self.assertFalse(is_aiter_flydsl_gdn_decode_supported(*valid))

    def test_copy_state_validates_inner_layout_and_index_shapes(self):
        state = torch.randn(3, 2, 8, 8, device="cuda", dtype=torch.float32)
        indices = torch.tensor([1], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "inner dimensions must be contiguous"):
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state.transpose(-1, -2), indices, indices
            )
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state, indices, torch.tensor([1, 2], device="cuda", dtype=torch.int32)
            )

    def test_decode_entry_validates_auxiliary_arguments(self):
        q, k, v, a, b, state = _make_decode_inputs()
        value_heads = v.shape[2]
        valid_kwargs = {
            "A_log": torch.randn(value_heads, device="cuda", dtype=torch.float32),
            "a": a,
            "dt_bias": torch.randn(value_heads, device="cuda", dtype=torch.bfloat16),
            "q": q,
            "k": k,
            "v": v,
            "b": b,
            "state": state,
            "read_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
            "write_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
        }
        cases = (
            (
                {"A_log": valid_kwargs["A_log"].to(torch.float16)},
                "requires FP32/BF16 A_log",
            ),
            (
                {"A_log": valid_kwargs["A_log"][:1]},
                "one value per value head",
            ),
            (
                {"dt_bias": valid_kwargs["dt_bias"].to(torch.float32)},
                "dt_bias to match q.dtype",
            ),
            (
                {"dt_bias": valid_kwargs["dt_bias"][:1]},
                "one value per value head",
            ),
            (
                {"read_indices": valid_kwargs["read_indices"].to(torch.int64)},
                "batch-sized int32 tensors",
            ),
            ({"scale": 1.0}, "fixed head-dimension scale"),
        )
        with mock.patch.object(
            adapter,
            "is_aiter_flydsl_gdn_decode_supported",
            return_value=True,
        ):
            for changes, message in cases:
                with self.subTest(message=message):
                    kwargs = valid_kwargs | changes
                    with self.assertRaisesRegex(ValueError, message):
                        aiter_flydsl_gdn_decode(**kwargs)

    def test_graph_replay_records_copy_for_a_later_block_boundary(self):
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1002], device="cuda", dtype=torch.int32)
        state = torch.zeros((3, 1, 8, 8), device="cuda", dtype=torch.float32)
        state[1].normal_()
        state[2].fill_(-7)

        # Warm Triton compilation before capture.
        warm_read, warm_write = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024
        )
        copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
            state, warm_read, warm_write
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
                block_map, lengths, 1024
            )
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state, read_indices, write_indices
            )

        # Capture length is not a boundary. Replay the same graph with a
        # boundary length; the device-side indices and copy must update.
        state[2].fill_(-7)
        lengths.fill_(1025)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(state[2], state[1], rtol=0, atol=0)


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(),
    "AITER FlyDSL numerical tests require ROCm",
)
class AiterFlydslGdnDecodeRocmTest(unittest.TestCase):
    def test_required_aiter_symbol_is_available(self):
        adapter._get_aiter_flydsl_gdn_decode.cache_clear()
        self.assertTrue(callable(adapter._get_aiter_flydsl_gdn_decode()))

    def test_mixed_block_boundary_batch_matches_triton_and_preserves_pool(self):
        torch.manual_seed(17)
        batch, dim = 4, 128
        sequence_lengths = [1001, 1024, 1500, 2048]
        for key_heads, value_heads in ((2, 8), (16, 32)):
            for state_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    key_heads=key_heads,
                    value_heads=value_heads,
                    state_dtype=state_dtype,
                ):
                    q, k, v, a, b, _ = _make_decode_inputs(
                        batch=batch,
                        key_heads=key_heads,
                        value_heads=value_heads,
                        dim=dim,
                        state_dtype=state_dtype,
                    )
                    self.assertFalse(q.is_contiguous())
                    self.assertFalse(k.is_contiguous())
                    self.assertFalse(v.is_contiguous())
                    A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
                    dt_bias = torch.randn(
                        value_heads, device="cuda", dtype=torch.bfloat16
                    )
                    block_map = torch.arange(
                        1, batch * 3 + 1, device="cuda", dtype=torch.int32
                    ).reshape(batch, 3)
                    lengths_plus_1 = torch.tensor(
                        [length + 1 for length in sequence_lengths],
                        device="cuda",
                        dtype=torch.int32,
                    )

                    state_elements = value_heads * dim * dim
                    packed_initial = (
                        torch.randn(
                            batch * 3 + 1,
                            state_elements + 12288,
                            device="cuda",
                            dtype=state_dtype,
                        )
                        * 0.01
                    )
                    packed_reference = packed_initial.clone()
                    packed_flydsl = packed_initial.clone()
                    state_reference = packed_reference[:, :state_elements].view(
                        batch * 3 + 1, value_heads, dim, dim
                    )
                    state_flydsl = packed_flydsl[:, :state_elements].view(
                        batch * 3 + 1, value_heads, dim, dim
                    )

                    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
                    output_reference, _ = fused_recurrent_gated_delta_rule(
                        q=q,
                        k=k,
                        v=v,
                        g=g.view(batch, 1, value_heads),
                        beta=beta.view(batch, 1, value_heads),
                        initial_state=state_reference,
                        block_map=block_map,
                        sequence_lengths=lengths_plus_1,
                        seq_size_per_block=1024,
                        use_qk_l2norm_in_kernel=True,
                    )
                    read_indices, write_indices = (
                        prepare_aiter_flydsl_gdn_decode_state_indices(
                            block_map, lengths_plus_1, 1024
                        )
                    )
                    output_flydsl = aiter_flydsl_gdn_decode(
                        A_log=A_log,
                        a=a,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        b=b,
                        state=state_flydsl,
                        read_indices=read_indices,
                        write_indices=write_indices,
                    )
                    torch.cuda.synchronize()

                    torch.testing.assert_close(
                        output_flydsl,
                        output_reference,
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    write_ids = write_indices.to(torch.int64)
                    state_rtol = 5e-3 if state_dtype == torch.float32 else 2e-2
                    state_atol = 5e-4 if state_dtype == torch.float32 else 2e-3
                    torch.testing.assert_close(
                        state_flydsl[write_ids],
                        state_reference[write_ids],
                        rtol=state_rtol,
                        atol=state_atol,
                    )

                    # The state copy/update must not touch the packed conv tail.
                    self.assertTrue(
                        torch.equal(
                            packed_flydsl[:, state_elements:],
                            packed_initial[:, state_elements:],
                        )
                    )
                    # Only write blocks may change. This includes an explicit
                    # exact check that reserved dummy block 0 stays unchanged.
                    untouched = torch.ones(
                        packed_initial.shape[0],
                        device="cuda",
                        dtype=torch.bool,
                    )
                    untouched[write_ids] = False
                    self.assertTrue(
                        torch.equal(
                            packed_flydsl[untouched, :state_elements],
                            packed_initial[untouched, :state_elements],
                        )
                    )

    def test_padding_row_uses_dummy_block_without_affecting_real_request(self):
        torch.manual_seed(23)
        q, k, v, a, b, _ = _make_decode_inputs(batch=2)
        value_heads, dim = v.shape[2], v.shape[3]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1, 2], [0, 0]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1025, 0], device="cuda", dtype=torch.int32)
        state_initial = torch.randn(
            3, value_heads, dim, dim, device="cuda", dtype=torch.float32
        )
        state_reference = state_initial.clone()
        state_flydsl = state_initial.clone()

        g, beta = fused_gdn_gating(A_log, a[:1], b[:1], dt_bias)
        output_reference, _ = fused_recurrent_gated_delta_rule(
            q=q[:1],
            k=k[:1],
            v=v[:1],
            g=g.view(1, 1, value_heads),
            beta=beta.view(1, 1, value_heads),
            initial_state=state_reference,
            block_map=block_map[:1],
            sequence_lengths=lengths[:1],
            seq_size_per_block=1024,
            use_qk_l2norm_in_kernel=True,
        )
        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024
        )
        output_flydsl = aiter_flydsl_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            state=state_flydsl,
            read_indices=read_indices,
            write_indices=write_indices,
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 0])
        self.assertEqual(write_indices.cpu().tolist(), [2, 0])
        torch.testing.assert_close(
            output_flydsl[:1], output_reference, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            state_flydsl[2], state_reference[2], rtol=5e-3, atol=5e-4
        )
        # Padding may update reserved dummy block 0, but never a real block.
        torch.testing.assert_close(state_flydsl[1], state_initial[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
