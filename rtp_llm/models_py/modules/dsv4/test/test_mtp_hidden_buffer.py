import types
import unittest

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    MtpHiddenBufferStore,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class MtpHiddenBufferTest(unittest.TestCase):
    def test_accessor_slices_requested_rows(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_hidden_buffer = torch.empty(5, 3, dtype=torch.bfloat16)

        flat = torch.arange(9, dtype=torch.bfloat16).reshape(3, 3)
        V4Transformer._write_mtp_hidden_buffer(v4, flat, is_cuda_graph=False)

        model = types.SimpleNamespace(v4=v4, _is_decode_role=False)
        sliced = DeepSeekV4Model.get_mtp_target_hidden_states(model, 2)
        self.assertTrue(torch.equal(flat[:2], sliced))

    def test_accessor_can_return_last_written_rows(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_hidden_buffer = torch.empty(5, 3, dtype=torch.bfloat16)

        flat = torch.arange(9, dtype=torch.bfloat16).reshape(3, 3)
        V4Transformer._write_mtp_hidden_buffer(v4, flat, is_cuda_graph=False)

        model = types.SimpleNamespace(v4=v4, _is_decode_role=False)
        sliced = DeepSeekV4Model.get_mtp_target_hidden_states(model, -1)
        self.assertTrue(torch.equal(flat, sliced))

    def test_accessor_rejects_requests_beyond_buffer_capacity(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_hidden_buffer = torch.empty(5, 3, dtype=torch.bfloat16)
        model = types.SimpleNamespace(v4=v4, _is_decode_role=False)

        with self.assertRaisesRegex(AssertionError, "requested=6, capacity=5"):
            DeepSeekV4Model.get_mtp_target_hidden_states(model, 6)

    def test_last_hidden_accessor_slices_requested_rows(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_last_hidden_buffer = torch.empty(4, 3, dtype=torch.bfloat16)

        flat = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        V4Transformer._write_mtp_last_hidden_buffer(v4, flat)

        model = types.SimpleNamespace(v4=v4, _is_decode_role=False)
        sliced = DeepSeekV4Model.get_mtp_last_hidden_states(model, 1)
        self.assertTrue(torch.equal(flat[:1], sliced))

    def test_mtp_buffer_allocation_requires_shared_store(self) -> None:
        v4 = types.SimpleNamespace()
        v4.args = types.SimpleNamespace(hc_mult=1, dim=3)

        with self.assertRaisesRegex(AssertionError, "requires a shared store"):
            V4Transformer._allocate_mtp_buffer(
                v4,
                torch.device("cpu"),
                token_capacity=4,
                shared_store=None,
            )

    def test_shared_store_binds_same_tensor_to_multiple_modules(self) -> None:
        store = MtpHiddenBufferStore()
        first = torch.nn.Module()
        second = torch.nn.Module()
        first.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        second.register_buffer("_mtp_hidden_buffer", None, persistent=False)

        store.bind(first, torch.device("cpu"), 5, 3, torch.bfloat16)
        store.bind(second, torch.device("cpu"), 5, 3, torch.bfloat16)
        self.assertEqual(
            first._mtp_hidden_buffer.data_ptr(), second._mtp_hidden_buffer.data_ptr()
        )

    def test_shared_store_rejects_capacity_growth(self) -> None:
        store = MtpHiddenBufferStore()
        module = torch.nn.Module()
        module.register_buffer("_mtp_hidden_buffer", None, persistent=False)

        store.bind(module, torch.device("cpu"), 5, 3, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "cannot grow"):
            store.bind(module, torch.device("cpu"), 7, 3, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
