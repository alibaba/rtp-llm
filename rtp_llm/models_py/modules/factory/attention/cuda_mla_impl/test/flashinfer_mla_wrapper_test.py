from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferDecodeImpl,
)


class _GraphParams:
    def fill_decode_cuda_graph_params(
        self,
        sequence_lengths_plus_1: torch.Tensor,
        block_table: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        del block_table, seq_size_per_block
        batch_size = sequence_lengths_plus_1.numel()
        self.positions_d = sequence_lengths_plus_1.clamp_min(1) - 1
        self.batch_indice_d = torch.arange(batch_size, dtype=torch.int32)
        self.page_indice_d = torch.arange(batch_size, dtype=torch.int32)
        self.slot_mapping = None


class FlashInferMlaWrapperTest(TestCase):
    def test_graph_group_refresh_uses_current_attention_inputs(self) -> None:
        impl = object.__new__(MlaFlashInferDecodeImpl)
        impl.seq_size_per_block = 4
        impl.fmha_params = _GraphParams()
        impl.fmha_impl = SimpleNamespace(kv_indices_d=torch.empty(2, dtype=torch.int32))
        impl.attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.tensor(
                [[31, 32], [41, 42]], dtype=torch.int32
            )
        )

        current_inputs = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.tensor([2, 5], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[11, 12], [21, 22]], dtype=torch.int32
            ),
        )
        impl.prepare_cuda_graph_group(current_inputs)

        self.assertIs(impl.attn_inputs, current_inputs)
        torch.testing.assert_close(
            impl._device_decode_slot_mapping(),
            torch.tensor([45, 88], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_graph_slot_mapping_uses_reserved_block_for_padding(self) -> None:
        impl = object.__new__(MlaFlashInferDecodeImpl)
        impl.seq_size_per_block = 4
        impl.fmha_params = _GraphParams()
        impl.fmha_impl = SimpleNamespace(kv_indices_d=torch.empty(2, dtype=torch.int32))
        current_inputs = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.tensor([2, 0], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[11, 12], [0, 0]], dtype=torch.int32
            ),
        )

        impl.prepare_cuda_graph_group(current_inputs)

        torch.testing.assert_close(
            impl._device_decode_slot_mapping(),
            torch.tensor([45, 0], dtype=torch.int64),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    main()
