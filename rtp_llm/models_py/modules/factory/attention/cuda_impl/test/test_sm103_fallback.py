import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.ops import fused_rope_kvcache_op


class Sm103FallbackTest(unittest.TestCase):
    def test_kvcache_offset_uses_native_op_by_default(self):
        block_ids = torch.tensor([[1, 3]], dtype=torch.int64)
        native_result = object()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_USE_TORCH_KVCACHE_OFFSET", None)
            with mock.patch.object(
                fused_rope_kvcache_op,
                "_rtp_convert_offset_to_block_array",
                return_value=native_result,
            ) as native_op:
                result = fused_rope_kvcache_op.convert_offset_to_block_array(
                    block_ids
                )

        self.assertIs(result, native_result)
        native_op.assert_called_once_with(block_ids)

    def test_kvcache_offset_uses_torch_fallback_when_enabled(self):
        block_ids = torch.tensor([[1, 3], [4, 5]], dtype=torch.int64)
        expected = torch.tensor(
            [[[[2, 6], [3, 7]]], [[[8, 10], [9, 11]]]], dtype=torch.int32
        )

        with mock.patch.dict(
            os.environ, {"RTP_LLM_USE_TORCH_KVCACHE_OFFSET": "1"}
        ):
            with mock.patch.object(
                fused_rope_kvcache_op, "_rtp_convert_offset_to_block_array"
            ) as native_op:
                result = fused_rope_kvcache_op.convert_offset_to_block_array(
                    block_ids
                )

        torch.testing.assert_close(result, expected)
        native_op.assert_not_called()

    def test_force_python_flashinfer_disables_native_prefill_impls(self):
        native_prefill_impls = {
            "HeadWiseFP8PrefillImpl",
            "HeadWisePrefillImpl",
            "FlashInferTRTLLMSpecDecodeImpl",
            "FlashInferTRTLLMPrefillImpl",
            "TRTMHAImpl",
            "TRTPagedMHAImpl",
            "FlashInferPrefillImpl",
            "CPFlashInferImpl",
        }

        with mock.patch.dict(
            os.environ, {"RTP_LLM_FORCE_PY_FLASHINFER_PREFILL": "true"}
        ):
            for impl_name in native_prefill_impls:
                self.assertTrue(
                    attn_factory._is_fmha_impl_disabled(impl_name, None), impl_name
                )
            self.assertFalse(
                attn_factory._is_fmha_impl_disabled(
                    "PyFlashinferPrefillImpl", None
                )
            )

    def test_cpp_flashinfer_prefill_can_be_disabled_independently(self):
        fmha_config = SimpleNamespace(disable_flash_infer=False)

        with mock.patch.dict(
            os.environ, {"RTP_LLM_DISABLE_CPP_FLASHINFER_PREFILL": "yes"}
        ):
            self.assertTrue(
                attn_factory._is_fmha_impl_disabled(
                    "FlashInferPrefillImpl", fmha_config
                )
            )

    def test_sm100_python_flashinfer_prefill_requires_opt_in(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.empty(0, dtype=torch.int32)
        )

        with mock.patch.object(py_flashinfer_mha, "is_sm_100", return_value=True):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(
                    "RTP_LLM_ENABLE_PY_FLASHINFER_PREFILL_SM100", None
                )
                self.assertFalse(
                    py_flashinfer_mha.PyFlashinferPrefillImpl.support(
                        None, attn_inputs
                    )
                )

            with mock.patch.dict(
                os.environ,
                {"RTP_LLM_ENABLE_PY_FLASHINFER_PREFILL_SM100": "on"},
            ):
                self.assertTrue(
                    py_flashinfer_mha.PyFlashinferPrefillImpl.support(
                        None, attn_inputs
                    )
                )


if __name__ == "__main__":
    unittest.main()
