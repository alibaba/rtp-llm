import gc
import types
import unittest
import weakref
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    Dsv4MtpHiddenBufferSpec,
    Dsv4SharedRuntimeBufferStore,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel


class _RuntimeModule(torch.nn.Module):
    """Minimal stand-in for ``V4Transformer`` as a store subscriber.

    The shared store now only owns the cross-model MTP-hidden buffer; the
    per-forward prefill-Q workspace is allocated inside ``forward_layers``
    and is no longer bound here. So a subscriber just needs the real
    ``_bind_runtime_buffers`` (which registers ``_mtp_hidden_buffer``).
    """

    _bind_runtime_buffers = V4Transformer._bind_runtime_buffers
    _bind_prefill_workspace_dims = V4Transformer._bind_prefill_workspace_dims
    _allocate_mtp_last_hidden_buffer = V4Transformer._allocate_mtp_last_hidden_buffer


class MtpHiddenBufferTest(unittest.TestCase):
    def test_mtp_fusion_restores_tp_embedding_for_normal_and_chunked_paths(self):
        class ContiguousRMSNorm(torch.nn.RMSNorm):
            def forward(self, value):
                # The native fusion norm requires row-contiguous inputs.
                if not value.is_contiguous():
                    raise ValueError("RMSNorm input must be contiguous")
                return super().forward(value)

        dim, hc = 8, 2
        weight = torch.arange(8 * dim, dtype=torch.float32).reshape(8, dim)
        ids = torch.tensor([1, 3, 2, 5, 4])
        positions = torch.tensor([0, 11, 12, 13, 14])
        hidden = torch.arange(5 * hc * dim, dtype=torch.float32).reshape(5, hc, dim)
        for tp_size in (1, 4):
            for chunk_tokens in (2, 100):
                with self.subTest(tp_size=tp_size, chunk_tokens=chunk_tokens):
                    local_dim = dim // tp_size
                    v4 = types.SimpleNamespace(
                        args=types.SimpleNamespace(tp_size=tp_size, dim=dim),
                        embed=torch.nn.Embedding.from_pretrained(weight[:, :local_dim]),
                    )
                    v4._embed = types.MethodType(V4Transformer._embed, v4)
                    model = types.SimpleNamespace(
                        v4=v4,
                        enorm=ContiguousRMSNorm(dim),
                        hnorm=torch.nn.RMSNorm(dim),
                        e_proj=torch.nn.Identity(),
                        h_proj=torch.nn.Identity(),
                        _mtp_fusion_chunk_tokens=lambda: chunk_tokens,
                        _mtp_fusion_chunk_logged=True,
                    )
                    for method in ("_apply_proj", "_build_fused_chunked"):
                        setattr(
                            model,
                            method,
                            types.MethodType(
                                getattr(DeepSeekV4MtpModel, method), model
                            ),
                        )

                    def gather(local, *, tp_size):
                        # Simulate the other column shards of the same checkpoint.
                        full = torch.cat(
                            [local + rank * local_dim for rank in range(tp_size)],
                            dim=-1,
                        )
                        return full.t().contiguous().t()

                    embedded = torch.nn.functional.embedding(ids, weight)
                    embedded[positions == 0] = 0
                    expected = model.hnorm(hidden) + model.enorm(embedded).unsqueeze(1)
                    with patch(
                        "rtp_llm.models_py.modules.dsv4.transformer.tp_gather_hidden",
                        side_effect=gather,
                    ) as collective, patch(
                        "torch.cuda.is_current_stream_capturing", return_value=False
                    ):
                        actual = DeepSeekV4MtpModel._build_fused(
                            model, ids, hidden.clone(), positions
                        )
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(
                        collective.call_count,
                        (3 if chunk_tokens == 2 else 1) if tp_size > 1 else 0,
                    )

    @staticmethod
    def _make_module_model():
        return types.SimpleNamespace(
            module_build_context=object(),
            v4=_RuntimeModule(),
            _shared_runtime_buffers=None,
            _is_speculative=False,
            _prefill_cp_size=1,
            _capture_aux_hidden_layer_ids=(),
            _v4_args=types.SimpleNamespace(dim=4, hc_mult=2),
            _resolve_prefill_q_token_capacity=lambda: 8,
            _resolve_prefill_q_dim=lambda: 16,
            _resolve_mtp_hidden_token_capacity=lambda: 7,
            _resolve_mtp_last_hidden_token_capacity=lambda: 2,
        )

    def test_module_mtp_outputs_are_owned_and_keep_target_handoff_alive(self):
        legacy = self._make_store()
        target, draft = self._make_module_model(), self._make_module_model()
        for model in (target, draft):
            model._is_speculative = True
            model.v4.args = model._v4_args
            model.v4.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
            DeepSeekV4Model._bind_runtime_buffers(model, torch.device("cpu"))
            self.assertEqual(tuple(model.v4._mtp_hidden_buffer.shape), (7, 8))
            self.assertEqual(tuple(model.v4._mtp_last_hidden_buffer.shape), (2, 8))
        target.v4._mtp_hidden_buffer.fill_(3)
        handoff = DeepSeekV4Model.get_mtp_target_hidden_states(target, 4)
        draft.v4._mtp_hidden_buffer.fill_(7)
        self.assertTrue(torch.equal(handoff, torch.full_like(handoff, 3)))
        self.assertNotEqual(handoff.data_ptr(), draft.v4._mtp_hidden_buffer.data_ptr())
        self.assertEqual(legacy._subscribers, [])
        self.assertFalse(Dsv4SharedRuntimeBufferStore.mtp_hidden_requested())

    def test_module_context_owns_store_and_ignores_legacy_mtp_request(self):
        legacy = self._make_store(mtp=True)
        first, second = self._make_module_model(), self._make_module_model()
        for model in (first, second):
            DeepSeekV4Model._bind_runtime_buffers(model, torch.device("cpu"))
            self.assertIsNot(model._shared_runtime_buffers, legacy)
            self.assertIsNone(model.v4._mtp_hidden_buffer)
            self.assertEqual(model.v4._prefill_ws_q_rows, 8)
        self.assertIsNot(first._shared_runtime_buffers, second._shared_runtime_buffers)
        original = first._shared_runtime_buffers
        DeepSeekV4Model._bind_runtime_buffers(first, torch.device("cpu"))
        self.assertIs(first._shared_runtime_buffers, original)
        self.assertEqual(legacy._subscribers, [])

    def test_module_destruction_does_not_leave_weights_in_global_subscribers(self):
        legacy = self._make_store()
        model = self._make_module_model()
        model.v4.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
        DeepSeekV4Model._bind_runtime_buffers(model, torch.device("cpu"))
        module_ref, weight_ref = weakref.ref(model.v4), weakref.ref(model.v4.weight)
        del model
        gc.collect()
        self.assertIsNone(module_ref())
        self.assertIsNone(weight_ref())
        self.assertEqual(legacy._subscribers, [])

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

    def test_target_hidden_buffer_capability_follows_bound_storage(self) -> None:
        v4 = types.SimpleNamespace(_mtp_hidden_buffer=None)
        model = types.SimpleNamespace(v4=v4)

        self.assertFalse(DeepSeekV4Model.has_mtp_hidden_buffer(model))

        v4._mtp_hidden_buffer = torch.empty(1, 3, dtype=torch.bfloat16)
        self.assertTrue(DeepSeekV4Model.has_mtp_hidden_buffer(model))

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
