import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill import try_causal_qsa_prefill


def make_selection(tokens):
    lengths = torch.arange(1, tokens + 1, device="cuda")
    columns = (tokens + 3) // 4
    scores = torch.randn(tokens, columns, device="cuda")
    scores.masked_fill_(
        torch.arange(columns, device="cuda")[None, :] >= lengths[:, None] // 4,
        -torch.inf,
    )
    values, blocks = scores.topk(min(512, columns), dim=-1)
    indices = blocks[:, :, None] * 4 + torch.arange(4, device="cuda")
    indices.masked_fill_(~torch.isfinite(values)[:, :, None], -1)
    selected = torch.full((1, tokens, 2051), -1, device="cuda", dtype=torch.int32)
    selected[0, :, : indices.numel() // tokens] = indices.reshape(tokens, -1)
    tail = torch.arange(3, device="cuda")[None, :]
    selected[0, :, -3:] = torch.where(
        tail < (lengths % 4)[:, None], lengths[:, None] // 4 * 4 + tail, -1
    )
    return selected


def make_inputs(tokens, heads):
    packed = torch.randn(tokens, (heads + 2) * 256, dtype=torch.bfloat16, device="cuda")
    q, k, v = packed.split([heads * 256, 256, 256], dim=-1)
    q = q.reshape(1, tokens, heads, 256).transpose(1, 2)
    k = k.reshape(1, tokens, 1, 256).transpose(1, 2)
    v = v.reshape(1, tokens, 1, 256).transpose(1, 2)
    return packed, q, k, v


def reference(q, k, v, selected):
    tokens = q.shape[2]
    indices = selected[0].long()
    membership = torch.zeros(
        tokens, tokens, dtype=torch.int32, device=q.device
    ).scatter_add_(1, indices.clamp_min(0), (indices >= 0).int())
    scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores += membership.clamp_min(1).double().log()[None, None]
    scores.masked_fill_(membership[None, None] == 0, -torch.inf)
    probability = scores.softmax(-1).nan_to_num(0)
    return (probability @ v.double()).to(q.dtype)


class TmaPrefillTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            self.skipTest("SM120 required")
        torch.manual_seed(471)

    def test_packed_qkv_matches_fp64_with_partial_tiles_holes_and_empty_rows(self):
        for heads, tokens in ((6, 257), (12, 257), (6, 513), (12, 513)):
            with self.subTest(heads=heads, tokens=tokens):
                _, q, k, v = make_inputs(tokens, heads)
                selected = make_selection(tokens)
                selected[:, [0, 13, tokens - 1]] = -1
                selected[:, 16:, :4] = -1
                actual = try_causal_qsa_prefill(q, k, v, selected)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(
                    actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
                )
                self.assertTrue((actual[:, :, [0, 13, tokens - 1]] == 0).all())
                packed_output = actual.transpose(1, 2).reshape(tokens, heads * 256)
                self.assertTrue(packed_output.is_contiguous())
                self.assertEqual(packed_output.data_ptr(), actual.data_ptr())

    def test_bf16_value_range_is_preserved(self):
        _, q, k, v = make_inputs(129, 12)
        selected = make_selection(129)
        unscaled = try_causal_qsa_prefill(q, k, v, selected)
        scale = 2**17
        v.mul_(scale)
        actual = try_causal_qsa_prefill(q, k, v, selected)
        torch.testing.assert_close(actual, unscaled * scale, atol=0, rtol=0)
        # Compare in the original units so the absolute error has the same scale.
        torch.testing.assert_close(
            actual / scale,
            reference(q, k, v, selected) / scale,
            atol=0.0078125,
            rtol=0.008,
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_runtime_uses_packed_views_only_for_indexer_owned_selections(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_gqa_impl

        tokens, heads = 257, 12
        packed, q, k, v = make_inputs(tokens, heads)
        selected = make_selection(tokens)
        configs = SimpleNamespace(
            head_num=heads,
            kv_head_num=1,
            size_per_head=256,
            kernel_tokens_per_block=128,
            indexer_topk=512,
        )
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            input_lengths=torch.tensor([tokens], dtype=torch.int32),
            prefix_lengths=torch.zeros(1, dtype=torch.int32),
            sequence_lengths=torch.zeros(1, dtype=torch.int32),
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            kv_cache_kernel_block_id_device=None,
        )
        writer = SimpleNamespace(
            prepare=lambda _: object(), forward=lambda qkv, *args: qkv
        )
        with patch.object(
            sparse_gqa_impl, "FusedRopeKVCachePrefillOpQKVOut", return_value=writer
        ):
            impl = sparse_gqa_impl.SparseGqaFmhaImpl(configs, inputs)
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            wraps=try_causal_qsa_prefill,
        ) as fast:
            impl.set_trusted_selected_indices(selected[0])
            actual = impl.forward(packed, None)
            fast.assert_called_once()
            for tensor in fast.call_args.args[:3]:
                self.assertEqual(
                    tensor.untyped_storage().data_ptr(),
                    packed.untyped_storage().data_ptr(),
                )
        expected = reference(q, k, v, selected).transpose(1, 2).reshape(tokens, -1)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

        # A different compression ratio can produce the same selection width.
        configs.indexer_topk = 1025
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            side_effect=AssertionError("different block layout reached TMA"),
        ):
            impl.set_trusted_selected_indices(selected[0])
            actual = impl.forward(packed, None)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
        configs.indexer_topk = 512

        # Injected selections may repeat keys and must retain multiplicity.
        selected[..., 4:8] = selected[..., :4]
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            side_effect=AssertionError("untrusted selection reached TMA"),
        ):
            actual = impl.forward(packed, None, selected_indices=selected[0])
        expected = reference(q, k, v, selected).transpose(1, 2).reshape(tokens, -1)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

        # Ragged requests keep their own local key indices in the packed batch.
        inputs.input_lengths = torch.tensor([129, 128], dtype=torch.int32)
        inputs.prefix_lengths = torch.zeros(2, dtype=torch.int32)
        inputs.sequence_lengths = torch.zeros(2, dtype=torch.int32)
        selections = [make_selection(129), make_selection(128)]
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            wraps=try_causal_qsa_prefill,
        ) as fast:
            impl.set_trusted_selected_indices(torch.cat(selections, dim=1)[0])
            actual = impl.forward(packed, None)
            self.assertEqual(fast.call_count, 2)
        expected = torch.cat(
            [
                reference(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], indices)
                .transpose(1, 2)
                .reshape(b - a, -1)
                for (a, b), indices in zip(((0, 129), (129, 257)), selections)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

    def test_graph_replay_refreshes_membership_and_qkv(self):
        packed, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        for _ in range(3):
            try_causal_qsa_prefill(q, k, v, selected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_causal_qsa_prefill(q, k, v, selected)
        selected.fill_(-1)
        graph.replay()
        self.assertTrue((actual == 0).all())
        packed.copy_(torch.randn_like(packed))
        selected.copy_(make_selection(257))
        graph.replay()
        torch.testing.assert_close(
            actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
        )

    def test_unsupported_shapes_and_alignment_return_to_general_kernel(self):
        _, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        self.assertIsNone(try_causal_qsa_prefill(q[:, :3], k, v, selected))
        self.assertIsNone(try_causal_qsa_prefill(q[:, :, :32], k, v, selected))
        self.assertIsNone(try_causal_qsa_prefill(q, k, v, selected[:, :, :-1]))
        self.assertIsNone(try_causal_qsa_prefill(q, k, v, selected.long()))
        misaligned = torch.empty(257, 257, dtype=v.dtype, device=v.device)[:, 1:]
        self.assertIsNone(
            try_causal_qsa_prefill(q, k, misaligned[None, None], selected)
        )

    def test_no_device_scalar_readback(self):
        _, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        try_causal_qsa_prefill(q, k, v, selected)
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("readback")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("readback")
            ),
        ):
            actual = try_causal_qsa_prefill(q, k, v, selected)
        torch.testing.assert_close(
            actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
        )


if __name__ == "__main__":
    unittest.main()
