from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda import prefill as kda_prefill


class KimiK3KDAPrefillMetadataTest(TestCase):
    checkpoint_tokens = 1024

    def _prepare(
        self,
        input_lengths: list[int],
        prefix_lengths: list[int],
        **kwargs,
    ) -> kda_prefill.KimiKDAPrefillMetadata:
        cu_seqlens = [0]
        for length in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)
        return kda_prefill.prepare_kimi_kda_prefill_metadata(
            torch.tensor(cu_seqlens, dtype=torch.int32),
            torch.tensor(input_lengths, dtype=torch.int32),
            torch.tensor(prefix_lengths, dtype=torch.int32),
            checkpoint_tokens=self.checkpoint_tokens,
            local_heads=1,
            head_dim=2,
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_checkpoint_boundaries_use_compact_slots(self) -> None:
        expected = {
            self.checkpoint_tokens: (1, [0]),
            self.checkpoint_tokens + 1: (2, [0, 1]),
            2 * self.checkpoint_tokens - 1: (2, [0, 1]),
            2 * self.checkpoint_tokens: (2, [0, 1]),
        }

        for length, (required_slots, store_slots) in expected.items():
            with self.subTest(length=length):
                metadata = self._prepare([length], [0])
                self.assertEqual(metadata.checkpoint_tokens, self.checkpoint_tokens)
                self.assertEqual(metadata.required_slots, required_slots)
                self.assertEqual(
                    metadata.recurrent.store_page_indices.tolist(), store_slots
                )

    def test_aligned_prefix_and_terminal_partial_select_distinct_slots(self) -> None:
        metadata = self._prepare(
            [self.checkpoint_tokens + 1, 1],
            [0, self.checkpoint_tokens],
            materialized_block_maps_host=(
                torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
            ),
        )

        self.assertEqual(metadata.required_slots, 2)
        self.assertEqual(metadata.recurrent.store_sequence_indices.tolist(), [0, 0, 1])
        self.assertEqual(metadata.recurrent.store_page_indices.tolist(), [0, 1, 1])

    def test_mixed_batch_uses_only_materialized_compact_slots(self) -> None:
        metadata = self._prepare(
            [self.checkpoint_tokens + 1, 2 * self.checkpoint_tokens - 1],
            [0, self.checkpoint_tokens],
            materialized_block_maps_host=(
                torch.tensor([[11, 12, 0], [21, 22, 23]], dtype=torch.int32),
                torch.tensor([[31, 32, 0], [41, 42, 0]], dtype=torch.int32),
            ),
        )

        self.assertEqual(metadata.required_slots, 3)
        self.assertEqual(metadata.recurrent.checkpoint_offsets.tolist(), [0, 2, 4])
        self.assertEqual(
            metadata.recurrent.store_checkpoint_indices.tolist(), [0, 1, 2]
        )
        self.assertEqual(metadata.recurrent.store_sequence_indices.tolist(), [0, 0, 1])
        self.assertEqual(metadata.recurrent.store_page_indices.tolist(), [0, 1, 1])
        self.assertEqual(metadata.recurrent.final_checkpoint_indices.tolist(), [1, 3])

    def test_multi_round_mapping_requires_aligned_reusable_prefix(self) -> None:
        first = self._prepare(
            [self.checkpoint_tokens],
            [0],
            active_original_batch_indices=[3],
            continuation_mask=[False],
        )
        second = self._prepare(
            [1],
            [self.checkpoint_tokens],
            active_original_batch_indices=[3],
            continuation_mask=[True],
        )

        self.assertEqual(first.recurrent.store_page_indices.tolist(), [0])
        self.assertEqual(second.recurrent.store_page_indices.tolist(), [1])
        self.assertEqual(second.active_original_batch_indices_host, (3,))
        self.assertEqual(second.continuation_mask_host, (True,))
        with self.assertRaisesRegex(ValueError, "checkpoint-aligned"):
            self._prepare([1], [self.checkpoint_tokens + 1])


class KimiK3KDAGeometryRoutingTest(TestCase):
    checkpoint_tokens = 1024

    def test_prefill_forward_does_not_compare_checkpoint_v_with_physical_b(
        self,
    ) -> None:
        metadata = kda_prefill.prepare_kimi_kda_prefill_metadata(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            checkpoint_tokens=self.checkpoint_tokens,
            local_heads=1,
            head_dim=2,
            device=torch.device("cpu"),
        )
        cache = MagicMock()
        cache.linear_state_block_map_device.return_value = torch.tensor(
            [[1]], dtype=torch.int32
        )
        executor = kda_prefill.KimiK3KDAPrefill.__new__(kda_prefill.KimiK3KDAPrefill)
        nn.Module.__init__(executor)
        executor.cache = cache
        expected = torch.ones((1, 1, 1, 2), dtype=torch.float32)

        with patch.object(
            executor, "_packed_checkpoint_prefill", return_value=expected
        ) as packed:
            actual = executor.forward(
                torch.zeros((1, 6), dtype=torch.float32),
                torch.zeros((1, 2), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                torch.tensor([0, 1], dtype=torch.int32),
                # Deliberately differ from the explicit checkpoint span.
                kv_cache=SimpleNamespace(seq_size_per_block=128),
                attention_inputs=SimpleNamespace(
                    prefix_lengths=torch.tensor([0], dtype=torch.int32),
                    is_target_verify=False,
                ),
                metadata=metadata,
            )

        self.assertIs(actual, expected)
        packed.assert_called_once()

    def test_continuing_rows_use_registry_and_initial_rows_use_cache(self) -> None:
        metadata = kda_prefill.prepare_kimi_kda_prefill_metadata(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor([1024, 0], dtype=torch.int32),
            checkpoint_tokens=self.checkpoint_tokens,
            local_heads=1,
            head_dim=2,
            device=torch.device("cpu"),
            active_original_batch_indices=[2, 0],
            continuation_mask=[True, False],
        )
        executor = kda_prefill.KimiK3KDAPrefill.__new__(kda_prefill.KimiK3KDAPrefill)
        nn.Module.__init__(executor)
        executor.local_heads, executor.head_dim, executor.projection_size = 1, 2, 2
        executor.fused_conv = torch.ones((6, 2))
        executor.cache = MagicMock()
        executor.cache.get_views.return_value = (None, torch.empty((4, 1, 6)))
        physical = torch.full((2, 1, 2, 2), 3.0)
        executor.cache.load_recurrent_state.return_value = physical
        state = kda_prefill.KimiKDACurrentStateRegistry(3).get_or_create(
            0,
            device=torch.device("cpu"),
            conv_dtype=torch.float32,
            history_size=1,
            projection_size=2,
            local_heads=1,
            head_dim=2,
        )
        state.conv.fill_(7)
        state.conv[2].fill_(19)
        state.recurrent.fill_(11)
        state.recurrent[2].fill_(23)
        state.valid_requests.add(2)
        metadata.recurrent_checkpoints.fill_(13)
        metadata.recurrent_checkpoints[0, 1].fill_(29)
        final_conv = torch.full((2, 1, 6), 17.0)
        final_conv[1].fill_(31)
        with patch.object(
            kda_prefill,
            "kimi_kda_short_conv_paged_prefill",
            return_value=(torch.zeros((2, 2)),) * 3 + (final_conv,),
        ) as conv, patch.object(executor, "_cula_checkpoint_prefill") as cula:
            executor._packed_checkpoint_prefill(
                torch.zeros((2, 6)),
                torch.zeros((2, 2)),
                torch.zeros((2, 1)),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                SimpleNamespace(seq_size_per_block=128),
                SimpleNamespace(
                    prefix_lengths=torch.tensor([1024, 0], dtype=torch.int32)
                ),
                torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
                metadata=metadata,
                current_state=state,
            )
        torch.testing.assert_close(
            cula.call_args.args[5],
            torch.stack((torch.full((1, 2, 2), 23.0), physical[1])),
        )
        self.assertEqual(cula.call_args.kwargs["checkpoint_interval"], 1024)
        self.assertEqual(
            executor.cache.load_recurrent_state.call_args.kwargs["checkpoint_tokens"], 1024
        )
        torch.testing.assert_close(
            conv.call_args.kwargs["current_conv_state"],
            torch.stack((torch.full((1, 6), 19.0), torch.full((1, 6), 7.0))),
        )
        self.assertIs(
            conv.call_args.kwargs["continuation_mask"], metadata.continuation_mask
        )
        self.assertTrue(conv.call_args.kwargs["return_final_state"])
        self.assertEqual(conv.call_args.args[6], 1024)
        executor.cache.store_recurrent_checkpoints.assert_called_once()
        torch.testing.assert_close(state.conv[[2, 0]], final_conv)
        torch.testing.assert_close(
            state.recurrent[[2, 0]],
            torch.stack((torch.full((1, 2, 2), 13.0), torch.full((1, 2, 2), 29.0))),
        )
        torch.testing.assert_close(state.recurrent[1], torch.full((1, 2, 2), 11.0))
        torch.testing.assert_close(state.conv[1], torch.full((1, 6), 7.0))
        self.assertEqual(state.valid_requests, {0, 2})


if __name__ == "__main__":
    main()
