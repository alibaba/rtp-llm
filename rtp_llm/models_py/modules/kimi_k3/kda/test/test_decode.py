from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda import decode as kda_decode
from rtp_llm.models_py.modules.kimi_k3.kda import module as kda_module
from rtp_llm.utils.model_weight import W

KimiK3KDADecode = kda_decode.KimiK3KDADecode
_PagedDecodeCache = kda_decode._PagedDecodeCache


class KimiK3KDATargetVerifyTest(TestCase):
    @staticmethod
    def _projection_module(tp_size: int = 8) -> kda_module.KimiK3KDA:
        module = kda_module.KimiK3KDA.__new__(kda_module.KimiK3KDA)
        nn.Module.__init__(module)
        module.attn_tp_size = tp_size
        module.attn_tp_rank = 0
        module.projection_size = 2
        module.eps = 1e-6
        module.weights = {
            W.linear_attn_norm_w: torch.ones(2),
            W.linear_attn_out_w: torch.eye(2),
        }
        return module

    def test_output_projection_always_uses_reduce_scatter(self) -> None:
        module = self._projection_module()
        output = torch.arange(16, dtype=torch.float32).reshape(1, 8, 1, 2)
        output_gate = torch.zeros_like(output)
        expected = torch.arange(2, dtype=torch.float32).reshape(1, 2)

        process_group = object()
        with (
            patch.object(kda_module, "get_process_group", return_value=process_group),
            patch.object(
                kda_module,
                "gemm_reduce_scatter",
                return_value=expected,
            ) as reduce_scatter,
        ):
            projected = module._project_output(
                output,
                output_gate,
            )

        self.assertIs(projected, expected)
        reduce_scatter.assert_called_once()
        projection_input, weight, _ = reduce_scatter.call_args.args
        self.assertEqual(tuple(projection_input.shape), (8, 2))
        self.assertIs(weight, module.weights[W.linear_attn_out_w])
        self.assertIs(reduce_scatter.call_args.args[2], process_group)

    def test_target_verify_dispatches_one_fused_conv_and_recurrence(self) -> None:
        batch = 2
        steps = 2
        projection_size = 2
        page_size = 4
        block_map = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        sequence_lengths = torch.tensor([3, 5], dtype=torch.int32)

        decoder = KimiK3KDADecode(
            weights={},
            cache=None,
            local_heads=1,
            head_dim=projection_size,
            projection_size=projection_size,
            history_size=1,
            gate_lower_bound=-20.0,
            fused_conv=torch.empty(0),
        )

        def rows(values: list[int]) -> torch.Tensor:
            return (
                torch.tensor(values, dtype=torch.float32)
                .repeat_interleave(projection_size)
                .reshape(batch * steps, projection_size)
            )

        q = rows([10, 11, 20, 21])
        k = rows([30, 31, 40, 41])
        v = rows([50, 51, 60, 61])
        gate = rows([70, 71, 80, 81])
        beta = torch.tensor([[90], [91], [100], [101]], dtype=torch.float32)

        block_count = 7
        conv = torch.arange(block_count, dtype=torch.float32).reshape(-1, 1, 1)
        conv = conv.expand(-1, 1, 3 * projection_size).clone()
        ssm = torch.arange(block_count, dtype=torch.float32).reshape(-1, 1, 1, 1)
        ssm = ssm.expand(-1, 1, projection_size, projection_size).clone()
        cache = _PagedDecodeCache(
            ssm=ssm,
            conv=conv,
            block_map=block_map,
            sequence_lengths_plus_one=sequence_lengths,
            page_size=page_size,
        )

        def fake_target_conv(
            q_sequence: torch.Tensor,
            k_sequence: torch.Tensor,
            v_sequence: torch.Tensor,
            _fused_conv: torch.Tensor,
            _conv_cache: torch.Tensor,
            fused_block_map: torch.Tensor,
            fused_lengths: torch.Tensor,
            _page_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.assertEqual(tuple(q_sequence.shape), (batch, steps, projection_size))
            torch.testing.assert_close(fused_block_map, block_map)
            torch.testing.assert_close(fused_lengths, sequence_lengths)
            return q_sequence, k_sequence, v_sequence

        recurrent_calls: list[
            tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]
        ] = []

        def fake_recurrent(
            fused_q: torch.Tensor,
            fused_k: torch.Tensor,
            fused_v: torch.Tensor,
            fused_gate: torch.Tensor,
            fused_beta: torch.Tensor,
            _cu_seqlens: torch.Tensor,
            _ssm_cache: torch.Tensor,
            fused_block_map: torch.Tensor,
            fused_lengths: torch.Tensor,
            _page_size: int,
        ) -> torch.Tensor:
            tensors = (fused_q, fused_k, fused_v, fused_gate, fused_beta)
            recurrent_calls.append(
                (
                    tuple(t.clone() for t in tensors),
                    fused_block_map.clone(),
                    fused_lengths.clone(),
                )
            )
            return fused_q.reshape(1, batch * steps, 1, projection_size)

        with patch.object(
            kda_decode,
            "kimi_kda_short_conv_paged_target_verify",
            side_effect=fake_target_conv,
        ), patch.object(decoder, "_recurrent", side_effect=fake_recurrent):
            output = decoder._target_verify(
                q,
                k,
                v,
                gate,
                beta,
                torch.tensor([0, steps, batch * steps], dtype=torch.int32),
                cache,
            )

        sources = (q, k, v, gate, beta)
        self.assertEqual(len(recurrent_calls), 1)
        tensors, call_map, call_lengths = recurrent_calls[0]
        for actual, source in zip(tensors, sources):
            torch.testing.assert_close(actual, source)
        torch.testing.assert_close(call_map, block_map)
        torch.testing.assert_close(call_lengths, sequence_lengths)

        torch.testing.assert_close(output.reshape(batch * steps, -1), q)


if __name__ == "__main__":
    main()
