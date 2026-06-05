import types
import unittest

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    Dsv4MtpHiddenBufferSpec,
    Dsv4SharedRuntimeBufferStore,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class _RuntimeModule(torch.nn.Module):
    """Minimal stand-in for ``V4Transformer`` as a store subscriber.

    The shared store now only owns the cross-model MTP-hidden buffer; the
    per-forward prefill-Q workspace is allocated inside ``forward_layers``
    and is no longer bound here. So a subscriber just needs the real
    ``_bind_runtime_buffers`` (which registers ``_mtp_hidden_buffer``).
    """

    _bind_runtime_buffers = V4Transformer._bind_runtime_buffers


class MtpHiddenBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        Dsv4SharedRuntimeBufferStore._reset_for_test()

    def tearDown(self) -> None:
        Dsv4SharedRuntimeBufferStore._reset_for_test()

    @staticmethod
    def _make_store(
        mtp: bool = False, token_capacity: int = 7
    ) -> Dsv4SharedRuntimeBufferStore:
        if mtp:
            Dsv4SharedRuntimeBufferStore.enable_mtp_hidden()
            mtp_hidden = Dsv4MtpHiddenBufferSpec(
                token_capacity=token_capacity, hc_dim=3
            )
        else:
            mtp_hidden = None
        return Dsv4SharedRuntimeBufferStore.get_or_create(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            mtp_hidden=mtp_hidden,
        )

    def test_shared_store_instance_returns_singleton(self) -> None:
        with self.assertRaisesRegex(AssertionError, "is not bound"):
            Dsv4SharedRuntimeBufferStore.instance()

        first = self._make_store()
        second = Dsv4SharedRuntimeBufferStore.instance()

        self.assertIs(first, second)
        # No MTP requested → no shared storage allocated at all.
        self.assertIsNone(first._mtp_hidden_storage)

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

    def test_shared_store_bind_without_mtp_returns_none(self) -> None:
        store = self._make_store()
        module = _RuntimeModule()

        mtp = store.bind(module)

        self.assertIsNone(mtp)
        self.assertIsNone(module._mtp_hidden_buffer)

    def test_shared_store_allocates_mtp_region_when_enabled_before_first_bind(
        self,
    ) -> None:
        store = self._make_store(mtp=True)
        module = _RuntimeModule()

        mtp = store.bind(module)

        self.assertIsNotNone(mtp)
        self.assertEqual(tuple(mtp.shape), (7, 3))
        self.assertEqual(module._mtp_hidden_buffer.data_ptr(), mtp.data_ptr())

    def test_shared_store_binds_same_buffer_to_multiple_modules(self) -> None:
        store = self._make_store(mtp=True)
        first = _RuntimeModule()
        second = _RuntimeModule()

        mtp1 = store.bind(first)
        mtp2 = store.bind(second)

        self.assertIsNotNone(mtp1)
        self.assertIsNotNone(mtp2)
        self.assertEqual(mtp1.data_ptr(), mtp2.data_ptr())
        self.assertEqual(first._mtp_hidden_buffer.data_ptr(), mtp1.data_ptr())
        self.assertEqual(second._mtp_hidden_buffer.data_ptr(), mtp1.data_ptr())

    def test_shared_store_rejects_capacity_growth(self) -> None:
        self._make_store(mtp=True, token_capacity=7)

        with self.assertRaisesRegex(RuntimeError, "cannot grow MTP hidden capacity"):
            Dsv4SharedRuntimeBufferStore.get_or_create(
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                mtp_hidden=Dsv4MtpHiddenBufferSpec(token_capacity=9, hc_dim=3),
            )

    def test_shared_store_rejects_enabling_mtp_after_allocation(self) -> None:
        self._make_store()

        with self.assertRaisesRegex(RuntimeError, "cannot enable MTP"):
            Dsv4SharedRuntimeBufferStore.enable_mtp_hidden()

    @staticmethod
    def _make_last_hidden_module(cap: int, hc_dim: int) -> torch.nn.Module:
        module = torch.nn.Module()
        module.register_buffer(
            "_mtp_last_hidden_buffer",
            torch.empty(cap, hc_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        module._mtp_last_hidden_valid_tokens = 0
        return module

    def test_last_hidden_write_no_realloc_within_capacity(self) -> None:
        module = self._make_last_hidden_module(cap=4, hc_dim=3)
        original_ptr = module._mtp_last_hidden_buffer.data_ptr()

        flat = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        V4Transformer._write_mtp_last_hidden_buffer(module, flat)

        self.assertEqual(module._mtp_last_hidden_buffer.data_ptr(), original_ptr)
        self.assertEqual(module._mtp_last_hidden_buffer.size(0), 4)
        self.assertTrue(torch.equal(module._mtp_last_hidden_buffer[:2], flat))
        self.assertEqual(module._mtp_last_hidden_valid_tokens, 2)

    def test_last_hidden_write_rejects_overflow(self) -> None:
        module = self._make_last_hidden_module(cap=4, hc_dim=3)
        original_ptr = module._mtp_last_hidden_buffer.data_ptr()

        flat = torch.arange(7 * 3, dtype=torch.bfloat16).reshape(7, 3)
        with self.assertRaisesRegex(AssertionError, "_mtp_last_hidden_buffer overflow"):
            V4Transformer._write_mtp_last_hidden_buffer(module, flat)
        self.assertEqual(module._mtp_last_hidden_buffer.data_ptr(), original_ptr)
        self.assertEqual(module._mtp_last_hidden_valid_tokens, 0)

    def test_last_hidden_buffer_remains_non_persistent_after_write(self) -> None:
        module = self._make_last_hidden_module(cap=2, hc_dim=3)
        self.assertNotIn("_mtp_last_hidden_buffer", module.state_dict())

        flat = torch.arange(2 * 3, dtype=torch.bfloat16).reshape(2, 3)
        V4Transformer._write_mtp_last_hidden_buffer(module, flat)

        self.assertNotIn("_mtp_last_hidden_buffer", module.state_dict())
        self.assertIn("_mtp_last_hidden_buffer", module._buffers)


if __name__ == "__main__":
    unittest.main()
