"""Architecture dispatch truth table for the trtllm-gen / XQA decode gates.

Measured on real hardware (2026-10-14, L20D sm_103, pinned flashinfer
0.6.9+8c4f4dcf + flashinfer_cubin):

- TllmGenFmhaRunner accepts kSM_100/kSM_103 only (fmhaRunner.cuh ctor) and
  the cubin package ships 1130 sm_103a FMHA cubins incl. Bfloat16; both the
  prefill and decode ops pass numerically on sm_103 (trtllm_gen_test 18/18).
- The C++ XQAAttnOp kernels are compiled sm_90a-only (3rdparty/xqa/BUILD)
  and fail with cudaErrorSymbolNotFound on sm_103.
- The flashinfer.xqa JIT module carries Hopper kernels only; on sm_103 it
  builds but fails to dlopen with an undefined launchHopperF8MHAFlashInfer
  symbol.

This test pins the gates to that truth so any future change fails loudly
here instead of silently shifting the dispatch on an untested arch.
"""

import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import trtllm_gen, xqa
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
    _trtllm_gen_cubin_supported,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import (
    XQADecodeImpl,
    XQAImpl,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import PyAttentionInputs


def _mha_configs(dtype=torch.bfloat16):
    cfg = AttentionConfigs()
    cfg.head_num = 32
    cfg.kv_head_num = 8
    cfg.size_per_head = 128
    cfg.dtype = dtype
    cfg.kernel_tokens_per_block = 64
    cfg.use_mla = False
    return cfg


def _prefill_inputs():
    inputs = PyAttentionInputs()
    inputs.is_prefill = True
    inputs.kv_cache_kernel_block_id_device = torch.zeros(
        (1, 1), dtype=torch.int32
    )
    return inputs


def _decode_inputs():
    inputs = PyAttentionInputs()
    inputs.is_prefill = False
    return inputs


class ArchTruthTableTest(unittest.TestCase):
    """Pinned arch -> gate outcomes for the pinned flashinfer stack."""

    def _at_sm(self, sm, fn):
        with mock.patch.object(trtllm_gen, "get_sm", return_value=sm), mock.patch.object(
            trtllm_gen, "is_blackwell", return_value=sm[0] in (10, 12)
        ), mock.patch.object(trtllm_gen, "is_sm12x", return_value=sm[0] == 12), mock.patch.object(
            xqa, "get_sm", return_value=sm
        ), mock.patch.object(
            xqa, "is_sm12x", return_value=sm[0] == 12
        ):
            return fn()

    def test_trtllm_gen_cubin_whitelist(self):
        for sm, supported in [
            ((9, 0), False),  # runner ctor rejects sm_90
            ((10, 0), True),
            ((10, 3), True),  # L20D/B300: cubins shipped, 18/18 measured
            ((12, 0), False),  # no sm_120a cubins in the package
        ]:
            self.assertEqual(
                self._at_sm(sm, _trtllm_gen_cubin_supported),
                supported,
                f"_trtllm_gen_cubin_supported({sm}) should be {supported}",
            )

    def test_trtllm_ops_follow_the_runner_whitelist(self):
        cfg = _mha_configs()
        prefill_op = FlashInferTRTLLMPrefillOp(cfg)
        decode_op = FlashInferTRTLLMDecodeOp(cfg)
        pre = _prefill_inputs()
        dec = _decode_inputs()
        for sm, prefill_ok, decode_ok in [
            ((10, 0), True, True),
            ((10, 3), True, True),
            ((9, 0), False, False),
            ((12, 0), False, False),
        ]:
            def _probe():
                return (
                    prefill_op.support(pre),
                    decode_op.support(dec),
                )

            got_prefill, got_decode = self._at_sm(sm, _probe)
            self.assertEqual(
                (got_prefill, got_decode),
                (prefill_ok, decode_ok),
                f"trtllm-gen ops at {sm} should be {(prefill_ok, decode_ok)}",
            )

    def test_xqa_impls_reject_unsupported_blackwell(self):
        cfg = _mha_configs()
        dec = _decode_inputs()
        # XQAImpl.support delegates to the C++ XQAAttnOp for whitelisted
        # archs; that construction needs a CUDA context, so the whitelist
        # rows only run on a GPU host.
        whitelist_rows = (
            []
            if not torch.cuda.is_available()
            else [((9, 0), True), ((10, 0), True)]
        )
        for sm, xqa_ok in [
            *whitelist_rows,
            ((10, 3), False),  # no sm_103 kernel: measured cudaErrorSymbolNotFound
            ((12, 0), False),  # no sm_120a binding / nb_sub_seq_per_seq rejected
        ]:
            self.assertEqual(
                self._at_sm(sm, lambda: XQAImpl.support(cfg, dec)),
                xqa_ok,
                f"XQAImpl.support at {sm} should be {xqa_ok}",
            )
            self.assertEqual(
                self._at_sm(sm, lambda: XQADecodeImpl.support(cfg, dec)),
                xqa_ok,
                f"XQADecodeImpl.support at {sm} should be {xqa_ok}",
            )


if __name__ == "__main__":
    unittest.main()
