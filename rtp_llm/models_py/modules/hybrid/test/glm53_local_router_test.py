"""Local GLM routing preserves global token order, top-k, and shared TP sums."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.distributed.sequence_parallel import (
    shard_tokens,
    token_shard_layout,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules import GroupTopK
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router


def route(logits, bias):
    weights = torch.empty((logits.shape[0], 8), device=logits.device)
    ids = torch.empty_like(weights, dtype=torch.int64)
    if logits.shape[0]:
        GroupTopK()(
            topk_weights=weights,
            topk_ids=ids,
            scores=logits,
            correction_bias=bias,
            n_group=1,
            topk_group=1,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
    return weights, ids


class _Routed(nn.Module):
    topk_ids_dtype = torch.int64

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def forward(self, hidden_states, topk_weights, topk_ids, **kwargs):
        return self.callback(hidden_states, topk_weights, topk_ids)


class _Shared(nn.Module):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def forward(self, hidden_states, **kwargs):
        return self.callback(hidden_states, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53LocalRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(530910)
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.old_tf32

    def test_global_chunk_boundaries_topk_and_padding(self):
        weight = torch.randn(4096, 288, device="cuda", dtype=torch.bfloat16).float()
        weight /= 64
        # Include exact and near ties between adjacent expert columns.
        weight[:, 8] = weight[:, 7]
        weight[:, 9] = weight[:, 7]
        weight[0, 9] = torch.nextafter(weight[0, 9], weight.new_tensor(float("inf")))
        gate = Glm53FP32Router(weight)
        bias = torch.zeros(288, device="cuda")
        for tokens in (0, 1, 7, 8, 9, 127, 129, 8191, 8192, 8193, 65535, 65536, 65537):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
                full = gate(x)
                expected_weights, expected_ids = route(full, bias)
                for rank in range(8):
                    layout = token_shard_layout(tokens, 8, rank)
                    actual = gate.forward_shard(shard_tokens(x, layout), layout)
                    valid = layout.local_valid_tokens
                    first = layout.local_start
                    torch.testing.assert_close(
                        actual[:valid], full[first : first + valid], rtol=0, atol=0
                    )
                    weights, ids = route(actual, bias)
                    torch.testing.assert_close(
                        ids[:valid], expected_ids[first : first + valid], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        weights[:valid],
                        expected_weights[first : first + valid],
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(torch.count_nonzero(actual[valid:]).item(), 0)

    def test_aligned_shard_projects_one_eighth_of_rows(self):
        gate = Glm53FP32Router(torch.randn(4096, 288, device="cuda"))
        layout = token_shard_layout(1048576, 8, 3)
        local = torch.randn(
            layout.local_tokens, 4096, device="cuda", dtype=torch.bfloat16
        )
        import torch.nn.functional as functional

        with patch.object(functional, "linear", wraps=functional.linear) as linear:
            gate.forward_shard(local, layout)
        self.assertEqual(linear.call_count, 16)
        self.assertEqual(
            sum(call.args[0].shape[0] for call in linear.call_args_list), 131072
        )
        self.assertTrue(
            all(call.args[0].dtype == torch.float32 for call in linear.call_args_list)
        )

    def test_sp_dispatch_precedes_shared_gather_and_preserves_padding(self):
        for tokens in (1, 7, 8, 9, 17, 65):
            for with_shared in (False, True):
                with self.subTest(tokens=tokens, with_shared=with_shared):
                    self._check_layer(tokens, with_shared, sequence_parallel=True)

    def test_replicated_tp_dispatch_and_shared_allreduce(self):
        for tokens in (1, 9, 65):
            self._check_layer(tokens, True, sequence_parallel=False)

    def test_explicit_fixed_m_configuration_preserves_routing(self):
        self._check_layer(65, True, sequence_parallel=True, gate_chunk_rows=8)

    def test_replicated_shared_uses_local_rows_without_shared_collectives(self):
        for tokens in (1, 7, 8, 9, 65):
            for sp in (False, True):
                self._check_layer(tokens, True, sp, local_shared=True)

    def test_fused_shared_is_added_once_and_padding_stays_zero(self):
        for tokens in (1, 7, 8, 9, 65):
            for sp in (False, True):
                self._check_layer(
                    tokens, True, sp, local_shared=True, fused_shared=True
                )

    def _check_layer(
        self,
        tokens,
        with_shared,
        sequence_parallel,
        gate_chunk_rows=0,
        local_shared=False,
        fused_shared=False,
    ):
        x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
        gate = Glm53FP32Router(torch.randn(64, 288, device="cuda") / 8, chunk_rows=8)
        bias = torch.randn(288, device="cuda") / 100
        ref_logits = (
            fixed_m_linear(gate, x, gate_chunk_rows) if gate_chunk_rows else gate(x)
        )
        ref_w, ref_i = route(ref_logits, bias)

        def expert(h, w, i):
            return h * (((i + 1).float() * w).sum(-1, keepdim=True) / 288).bfloat16()

        expected_routed = expert(x, ref_w, ref_i)
        # Distinct shared TP partitions, accumulated in FP32 for this oracle.
        shared_sum = sum(
            (x.float() * (r + 1) / 128).bfloat16().float() for r in range(8)
        ).bfloat16()
        if local_shared:
            shared_sum = (x.float() / 4).bfloat16()
        for rank in range(8):
            events = []
            layout = token_shard_layout(tokens, 8, rank)
            local = shard_tokens(x, layout)
            if local_shared and sequence_parallel:
                # Input padding is outside the logical model sequence and may
                # contain stale values; output padding must still be zero.
                local[layout.local_valid_tokens :].fill_(3.5)
            layer = GenericMoeLayer.__new__(GenericMoeLayer)
            nn.Module.__init__(layer)
            layer.config = SimpleNamespace(
                has_moe_norm=True,
                moe_n_group=1,
                moe_topk_group=1,
                expert_num=288,
                routed_scaling_factor=2.5,
            )
            layer.gate = gate
            layer.gate_chunk_rows = gate_chunk_rows
            layer.route_local_tokens = True
            layer.shared_expert_local = local_shared
            layer.routed_tp_size = layer.ffn_tp_size = layer.ep_size = 8
            layer.routed_tp_rank = rank
            layer.top_k = 8
            layer.fake_balance_expert = None
            layer.correction_bias = bias
            layer._use_mega_moe_fused_shared = fused_shared
            layer.shared_expert_gate = None
            layer._shared_expert_stream = None

            def dispatch(h, w, i):
                events.append("dispatch")
                torch.testing.assert_close(h, local, rtol=0, atol=0)
                valid, first = layout.local_valid_tokens, layout.local_start
                torch.testing.assert_close(
                    w[:valid], ref_w[first : first + valid], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    i[:valid], ref_i[first : first + valid], rtol=0, atol=0
                )
                self.assertEqual(torch.count_nonzero(w[valid:]).item(), 0)
                self.assertEqual(torch.count_nonzero(i[valid:]).item(), 0)
                output = expert(h, w, i)
                if fused_shared:
                    output = output + (h.float() / 4).bfloat16()
                return output

            def shared(h, **kwargs):
                self.assertFalse(fused_shared, "fused shared must not execute twice")
                events.append("shared")
                torch.testing.assert_close(
                    h, local if local_shared else x, rtol=0, atol=0
                )
                self.assertTrue(kwargs["skip_allreduce"])
                if local_shared:
                    return (h.float() / 4).bfloat16()
                return (h.float() * (rank + 1) / 128).bfloat16()

            def gather(h, logical, group):
                self.assertFalse(local_shared, "local shared must not gather its input")
                events.append("gather")
                self.assertEqual(events[0], "gather" if gate_chunk_rows else "dispatch")
                self.assertEqual(logical, tokens)
                torch.testing.assert_close(h, local, rtol=0, atol=0)
                return x

            layer.fused_moe = _Routed(dispatch)
            layer.shared_expert = _Shared(shared) if with_shared else None
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_gather_trim",
                side_effect=gather,
            ), patch(
                "rtp_llm.models_py.distributed.collective_torch.reduce_scatter_padded",
                side_effect=(
                    AssertionError("local shared must not reduce")
                    if local_shared
                    else None
                ),
                return_value=shard_tokens(shared_sum, layout),
            ), patch(
                "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
                side_effect=(
                    AssertionError("local shared must not reduce")
                    if local_shared
                    else None
                ),
                return_value=shared_sum,
            ), patch(
                "rtp_llm.models_py.model_desc.generic_moe.all_gather",
                side_effect=(
                    AssertionError("SP output must remain local")
                    if local_shared and sequence_parallel
                    else None
                ),
                return_value=(
                    expected_routed + shared_sum if local_shared else expected_routed
                ),
            ):
                actual = layer(
                    local if sequence_parallel else x,
                    sequence_parallel_layout=layout if sequence_parallel else None,
                )
            expected = expected_routed + shared_sum if with_shared else expected_routed
            if sequence_parallel:
                expected = shard_tokens(expected, layout)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(
                events,
                (
                    (
                        ["gather", "dispatch", "shared"]
                        if gate_chunk_rows
                        else ["dispatch", "gather", "shared"]
                    )
                    if with_shared and sequence_parallel and not local_shared
                    else (
                        ["dispatch", "shared"]
                        if with_shared and not fused_shared
                        else ["dispatch"]
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
