from types import SimpleNamespace
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

    def test_decode_sequence_parallel_shards_replicated_projection(self) -> None:
        module = self._projection_module()
        output = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
        output_gate = torch.zeros_like(output)

        with patch.object(
            kda_module,
            "all_reduce",
            side_effect=lambda tensor, *, group: tensor,
        ) as all_reduce:
            projected = module._project_output(
                output,
                output_gate,
                is_target_verify=False,
                sequence_parallel=True,
                hidden_states=SimpleNamespace(is_cuda=True),
                mode="decode",
            )

        self.assertEqual(tuple(projected.shape), (1, 2))
        all_reduce.assert_called_once()

    def test_target_verify_keeps_replicated_projection(self) -> None:
        module = self._projection_module()
        output = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
        output_gate = torch.zeros_like(output)

        with patch.object(
            kda_module,
            "all_reduce",
            side_effect=lambda tensor, *, group: tensor,
        ) as all_reduce:
            projected = module._project_output(
                output,
                output_gate,
                is_target_verify=True,
                sequence_parallel=True,
                hidden_states=SimpleNamespace(is_cuda=True),
                mode="decode",
            )

        self.assertEqual(tuple(projected.shape), (2, 2))
        all_reduce.assert_called_once()

    def test_target_verify_dispatches_shared_replay_with_device_metadata(self) -> None:
        batch = 2
        steps = 2
        projection_size = 2
        page_size = 4
        block_map = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        sequence_lengths = torch.tensor([3, 5], dtype=torch.int32)

        decoder = KimiK3KDADecode(
            weights={
                W.linear_attn_alog: torch.zeros(1),
                W.linear_attn_dt_b_kda: torch.zeros(projection_size),
            },
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

        replay_cache = object()
        replay_inputs = object()
        with patch.object(
            kda_decode,
            "linear_serial_replay",
            return_value=q.reshape(batch * steps, 1, projection_size),
        ) as replay, patch.object(decoder, "_recurrent") as old_recurrent:
            output = decoder._target_verify(
                q,
                k,
                v,
                gate,
                beta,
                torch.tensor([0, steps, batch * steps], dtype=torch.int32),
                cache,
                SimpleNamespace(linear_replay=replay_cache, group_id=2),
                SimpleNamespace(linear_replay=replay_inputs),
            )
        replay.assert_called_once()
        old_recurrent.assert_not_called()
        self.assertIs(replay.call_args.args[-2], replay_cache)
        self.assertIs(replay.call_args.args[-1], replay_inputs)
        self.assertIs(replay.call_args.args[-4], ssm)
        self.assertIs(replay.call_args.args[-3], conv)
        self.assertTrue(replay.call_args.kwargs["vector_gate"])
        self.assertEqual(replay.call_args.kwargs["group_id"], 2)
        torch.testing.assert_close(output.reshape(batch * steps, -1), q)


if __name__ == "__main__":
    main()
