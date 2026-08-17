from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3.kda import decode as kda_decode

KimiK3KDADecode = kda_decode.KimiK3KDADecode
_PagedDecodeCache = kda_decode._PagedDecodeCache


class KimiK3KDATargetVerifyTest(TestCase):
    def test_batch_steps_use_contiguous_inputs_and_distinct_pages(self) -> None:
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
            return torch.tensor(values, dtype=torch.float32).repeat_interleave(
                projection_size
            ).reshape(batch * steps, projection_size)

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
        initial_conv = conv.clone()
        initial_ssm = ssm.clone()
        cache = _PagedDecodeCache(
            ssm=ssm,
            conv=conv,
            block_map=block_map,
            sequence_lengths_plus_one=sequence_lengths,
            page_size=page_size,
        )

        conv_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        recurrent_calls: list[
            tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]
        ] = []

        def fake_short_conv(
            q_step: torch.Tensor,
            k_step: torch.Tensor,
            v_step: torch.Tensor,
            _fused_conv: torch.Tensor,
            _conv_cache: torch.Tensor,
            step_block_map: torch.Tensor,
            step_lengths: torch.Tensor,
            _page_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            conv_calls.append(
                (q_step.clone(), step_block_map.clone(), step_lengths.clone())
            )
            return q_step, k_step, v_step

        def fake_recurrent(
            q_step: torch.Tensor,
            k_step: torch.Tensor,
            v_step: torch.Tensor,
            gate_step: torch.Tensor,
            beta_step: torch.Tensor,
            _cu_seqlens: torch.Tensor,
            _ssm_cache: torch.Tensor,
            step_block_map: torch.Tensor,
            step_lengths: torch.Tensor,
            _page_size: int,
        ) -> torch.Tensor:
            tensors = (q_step, k_step, v_step, gate_step, beta_step)
            recurrent_calls.append(
                (
                    tuple(t.clone() for t in tensors),
                    step_block_map.clone(),
                    step_lengths.clone(),
                )
            )
            self.assertTrue(all(t.is_contiguous() for t in tensors))
            return q_step.reshape(1, batch, 1, projection_size)

        with patch.object(
            kda_decode,
            "kimi_kda_short_conv_paged_decode",
            side_effect=fake_short_conv,
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

        expected_maps = (
            block_map,
            torch.tensor([[2, 2, 3], [4, 6, 6]], dtype=torch.int32),
        )
        sources = (q, k, v, gate, beta)
        for step, indexes in enumerate(([0, 2], [1, 3])):
            expected_lengths = sequence_lengths + step
            torch.testing.assert_close(conv_calls[step][0], q[indexes])
            torch.testing.assert_close(conv_calls[step][1], expected_maps[step])
            torch.testing.assert_close(conv_calls[step][2], expected_lengths)
            tensors, call_map, call_lengths = recurrent_calls[step]
            for actual, source in zip(tensors, sources):
                torch.testing.assert_close(actual, source[indexes])
            torch.testing.assert_close(call_map, expected_maps[step])
            torch.testing.assert_close(call_lengths, expected_lengths)

        torch.testing.assert_close(output.reshape(batch * steps, -1), q)
        torch.testing.assert_close(cache.conv[2], initial_conv[1])
        torch.testing.assert_close(cache.conv[6], initial_conv[5])
        torch.testing.assert_close(cache.ssm[2], initial_ssm[1])
        torch.testing.assert_close(cache.ssm[6], initial_ssm[5])


if __name__ == "__main__":
    main()
