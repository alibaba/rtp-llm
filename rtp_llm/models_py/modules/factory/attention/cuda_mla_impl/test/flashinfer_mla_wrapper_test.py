from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch

from rtp_llm.models_py.model_desc.kimi_k3 import _prepare_mla_fmha_for_group
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferDecodeImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    PagedMlaDecodeImplMixin,
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


class _RecordingImpl:
    def __init__(self) -> None:
        self.calls = []

    def prepare(self, attention_inputs) -> None:
        self.calls.append(("prepare", attention_inputs))

    def prepare_cuda_graph_group(self, attention_inputs) -> None:
        self.calls.append(("graph", attention_inputs))


class _RecordingPagedBackend:
    backend_name = "recording"

    def __init__(self) -> None:
        self.calls = []

    def refresh_cuda_graph_metadata(
        self,
        fmha_params,
        block_table,
        sequence_lengths,
        seq_size_per_block,
    ) -> None:
        self.calls.append(
            (
                fmha_params.slot_mapping,
                fmha_params.positions_d.clone(),
                block_table,
                sequence_lengths,
                seq_size_per_block,
            )
        )


class _PagedImplHarness(PagedMlaDecodeImplMixin):
    _device_decode_slot_mapping = MlaFlashInferDecodeImpl._device_decode_slot_mapping


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

    def test_paged_graph_group_refresh_updates_kv_write_metadata_first(self) -> None:
        impl = _PagedImplHarness()
        impl.seq_size_per_block = 4
        impl.fmha_params = _GraphParams()
        impl.fmha_params.slot_mapping = torch.tensor([999], dtype=torch.int64)
        impl.fmha_impl = _RecordingPagedBackend()
        current_inputs = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.tensor([2, 5], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[11, 12], [21, 22]], dtype=torch.int32
            ),
        )

        impl.prepare_cuda_graph_group(current_inputs)

        self.assertIs(impl.attn_inputs, current_inputs)
        self.assertEqual(len(impl.fmha_impl.calls), 1)
        slot_mapping, positions, block_table, sequence_lengths, page_size = (
            impl.fmha_impl.calls[0]
        )
        self.assertIsNone(slot_mapping)
        torch.testing.assert_close(
            positions, torch.tensor([1, 4], dtype=torch.int32), rtol=0, atol=0
        )
        self.assertIs(block_table, current_inputs.kv_cache_kernel_block_id_device)
        self.assertIs(sequence_lengths, current_inputs.sequence_lengths_plus_1_d)
        self.assertEqual(page_size, 4)
        torch.testing.assert_close(
            impl._device_decode_slot_mapping(),
            torch.tensor([45, 88], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_k3_routes_group_refresh_to_graph_method_during_capture(self) -> None:
        impl = _RecordingImpl()
        attention_inputs = SimpleNamespace(
            sequence_lengths=SimpleNamespace(is_cuda=True)
        )

        with mock.patch(
            "rtp_llm.models_py.model_desc.kimi_k3.torch.cuda.is_current_stream_capturing",
            return_value=True,
        ):
            prepared_group = _prepare_mla_fmha_for_group(
                impl, attention_inputs, selected_group_id=2, prepared_group_id=1
            )

        self.assertEqual(prepared_group, 2)
        self.assertEqual(impl.calls, [("graph", attention_inputs)])

    def test_k3_uses_host_prepare_outside_capture(self) -> None:
        impl = _RecordingImpl()
        attention_inputs = SimpleNamespace(
            sequence_lengths=SimpleNamespace(is_cuda=True)
        )

        with mock.patch(
            "rtp_llm.models_py.model_desc.kimi_k3.torch.cuda.is_current_stream_capturing",
            return_value=False,
        ):
            prepared_group = _prepare_mla_fmha_for_group(
                impl, attention_inputs, selected_group_id=2, prepared_group_id=1
            )

        self.assertEqual(prepared_group, 2)
        self.assertEqual(impl.calls, [("prepare", attention_inputs)])


if __name__ == "__main__":
    main()
