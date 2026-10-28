"""Gate tests for MRoPE prefill backend selection on sm80 (bf16 KV cache).

On sm80 the PyFlashinfer family is the only surviving prefill candidate for
interleaved-MRoPE models (Qwen3.5 / Kev): every arch-specific prefill backend
is sm90+/Blackwell gated. Before the sm80 gate existed, MRoPE + bf16 KV had
no prefill backend on sm80 at all and the attention factory failed with
"can not find mha type". These tests pin the gate logic only; no flashinfer
kernel is launched.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.ops import KvCacheDataType, RopeStyle


def _mrope_configs(
    fp8_kv_cache_mode=0,
    kv_cache_dtype=KvCacheDataType.BASE,
    need_rope_kv_cache=True,
):
    # Kev/Qwen3.5-4B geometry: head_dim 256, partial rotary factor 0.25
    # (rotary dim 64), interleaved MRoPE sections [11, 11, 10].
    return SimpleNamespace(
        rope_config=SimpleNamespace(
            style=RopeStyle.Mrope,
            mrope_interleaved=True,
            index_factor=3,
            mrope_dim1=11,
            mrope_dim2=11,
            mrope_dim3=10,
            dim=64,
        ),
        fp8_kv_cache_mode=fp8_kv_cache_mode,
        kv_cache_dtype=kv_cache_dtype,
        need_rope_kv_cache=need_rope_kv_cache,
    )


def _mrope_inputs(total_tokens=8, is_cuda_graph=False):
    return SimpleNamespace(
        is_prefill=True,
        is_cuda_graph=is_cuda_graph,
        combo_position_ids=torch.arange(
            3 * total_tokens, dtype=torch.int32, device="cuda"
        ),
        input_lengths=torch.tensor([total_tokens], dtype=torch.int32),
        prefix_lengths=torch.empty(0, dtype=torch.int32),
        is_target_verify=False,
        is_spec_draft_prefill=False,
    )


class PyFlashinferMropeSm80PrefillGateTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_sm80_bf16_mrope_prefill_gate_opens(self):
        configs, inputs = _mrope_configs(), _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: True):
            self.assertTrue(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_non_sm80_bf16_mrope_prefill_gate_stays_closed(self):
        configs, inputs = _mrope_configs(), _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: False):
            self.assertFalse(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_real_arch_gate_matches_is_sm80(self):
        # On any execution pool the unpatched gate must equal the real arch:
        # closed on sm8x/sm90/sm12x (their arch-specific backends win first),
        # open on a real sm80 host.
        configs, inputs = _mrope_configs(), _mrope_inputs()
        self.assertEqual(
            py_flashinfer_mha._supports_prefill_rope(configs, inputs),
            py_flashinfer_mha.is_sm80(),
        )

    def test_non_mrope_unchanged(self):
        configs = SimpleNamespace(rope_config=SimpleNamespace(style=RopeStyle.No))
        inputs = _mrope_inputs()
        self.assertTrue(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_dynamic_fp8_mode2_unchanged(self):
        configs = _mrope_configs(fp8_kv_cache_mode=2)
        inputs = _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: False):
            self.assertTrue(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_mode1_fused_path_unchanged(self):
        configs = _mrope_configs(
            fp8_kv_cache_mode=1, kv_cache_dtype=KvCacheDataType.FP8
        )
        inputs = _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: False):
            self.assertTrue(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_sm80_mode1_still_uses_fused_path(self):
        # The sm80 gate is bf16-KV only; mode 1 keeps its fused rope+FP8
        # KV-write path on every arch.
        configs = _mrope_configs(
            fp8_kv_cache_mode=1, kv_cache_dtype=KvCacheDataType.FP8
        )
        inputs = _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: True):
            self.assertTrue(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_sm80_bf16_rejects_invalid_position_ids(self):
        configs = _mrope_configs()
        inputs = _mrope_inputs(total_tokens=8)
        inputs.combo_position_ids = inputs.combo_position_ids[:-1]
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: True):
            self.assertFalse(
                py_flashinfer_mha._supports_prefill_rope(configs, inputs)
            )

    def test_sm80_bf16_rejects_cuda_graph(self):
        configs = _mrope_configs()
        inputs = _mrope_inputs(is_cuda_graph=True)
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: True):
            self.assertFalse(py_flashinfer_mha._supports_prefill_rope(configs, inputs))

    def test_sm80_bf16_prefill_impls_support(self):
        # End-to-end support() view of the candidates that must survive on
        # sm80: ragged prefill (first prefill, no prefix) and paged prefill
        # (prefix-cache continuation).
        configs, inputs = _mrope_configs(), _mrope_inputs()
        with patch.object(py_flashinfer_mha, "is_sm80", lambda: True):
            self.assertTrue(PyFlashinferPrefillImpl.support(configs, inputs))
            self.assertTrue(PyFlashinferPagedPrefillImpl.support(configs, inputs))


if __name__ == "__main__":
    unittest.main()
