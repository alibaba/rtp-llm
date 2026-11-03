"""PPU MTP hidden ownership and embedding fusion regressions."""

import gc
import types
import unittest
import weakref
from unittest.mock import patch

import torch

from rtp_llm.platforms.ppu.models.dsv4.impl.model import Dsv4MtpHiddenBufferSpec, Dsv4SharedRuntimeBufferStore
from rtp_llm.platforms.ppu.models.dsv4.runtime_model import PpuDeepSeekV4Model as DeepSeekV4Model
from rtp_llm.platforms.ppu.models.dsv4.mtp_model import PpuDeepSeekV4MtpModel as DeepSeekV4MtpModel
from rtp_llm.platforms.ppu.models.dsv4.transformer import PpuV4Transformer as V4Transformer


class _RuntimeModule(torch.nn.Module):
    _bind_runtime_buffers = V4Transformer._bind_runtime_buffers
    _bind_prefill_workspace_dims = V4Transformer._bind_prefill_workspace_dims
    _allocate_mtp_last_hidden_buffer = V4Transformer._allocate_mtp_last_hidden_buffer


class PpuMtpHiddenTest(unittest.TestCase):
    def test_mtp_constructor_binds_provider_through_platform_parent(self):
        from rtp_llm.platforms.ppu.models.dsv4.impl.model import DeepSeekV4Model as BaseModel
        provider = object()
        captured = []

        def initialize(model, *args, **kwargs):
            captured.append(kwargs["platform_provider"])
            model._v4_args = types.SimpleNamespace(
                n_layers=1, hc_mult=4, compress_ratios=[0],
                fp8_kv_cache=True, max_tokens_per_rank=128,
            )

        with patch.object(BaseModel, "__init__", initialize), patch(
            "rtp_llm.platforms.ppu.models.dsv4.runtime_model.build_provider",
            return_value=provider,
        ) as build:
            model = DeepSeekV4MtpModel(object(), object(), object(), object(), 8)
        build.assert_called_once()
        self.assertEqual(captured, [provider])
        self.assertEqual(model._v4_args.n_hash_layers, 0)

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
                        "rtp_llm.platforms.ppu.models.dsv4.transformer.tp_gather_hidden",
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
            _platform_provider=types.SimpleNamespace(owns_runtime_buffers=True),
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

if __name__ == "__main__":
    unittest.main()
