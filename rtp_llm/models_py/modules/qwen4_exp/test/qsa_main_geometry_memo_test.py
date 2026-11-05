"""Per-step main-geometry validation memo (production path) tests.

The layers of one prefill step validate the same main-cache geometry: the
cache, the block table and the two length tensors are the same objects for
every layer of a step. The production path validates that geometry once per
step (module-level memo) instead of re-enqueuing the device verdict per layer,
and the sign/token-count fail-fast checks read the per-step CPU mirrors
instead of the raw ``attn_inputs`` tensors, so a layer of one step never
synchronizes with the device (``RTP_LLM_DEVICE_INPUT`` engines keep those
tensors on the device).
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_gqa_impl
from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_validation import (
    clear_main_geometry_memo,
    main_geometry_validation_stats,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl

_H_Q, _H_KV, _D = 24, 2, 256
_DEV = "cuda"
_PAGE = 4


def _device_work(prof) -> list:
    """Names of device-side kernels and memcpys recorded by the profiler."""
    path = os.path.join(tempfile.mkdtemp(prefix="qsa_geometry_memo_"), "t.json")
    prof.export_chrome_trace(path)
    with open(path) as handle:
        evs = json.load(handle)["traceEvents"]
    return [e["name"] for e in evs if e.get("cat") in ("kernel", "gpu_memcpy")]


class MainGeometryMemoTest(unittest.TestCase):
    class _IdentityRopeWriter:
        def __init__(self, attn_configs):
            pass

        def prepare(self, attn_inputs):
            return object()

        def forward(self, qkv, kv_cache, params):
            return qkv

    def setUp(self):
        clear_main_geometry_memo()
        # Hermetic strict flag: the memo only exists on the production path.
        # An ambient RTP_LLM_QSA_STRICT_VALIDATION=1 (a strict bench export or
        # another module's default) must not flip these assertions; the
        # strict-mode test overrides this to "1" itself.
        strict_off = patch.dict("os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"})
        strict_off.start()
        self.addCleanup(strict_off.stop)

    def _fixture(self, device_sources=False):
        attn_configs = SimpleNamespace(
            head_num=_H_Q,
            kv_head_num=_H_KV,
            size_per_head=_D,
            kernel_tokens_per_block=_PAGE,
        )
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_exact_cuda_graph_batch=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor([4], dtype=torch.int32),
            prefix_lengths=torch.tensor([4], dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[1, 2, 3]], dtype=torch.int32, device=_DEV
            ),
        )
        if device_sources:
            attn_inputs.input_lengths = attn_inputs.input_lengths.to(_DEV)
            attn_inputs.prefix_lengths = attn_inputs.prefix_lengths.to(_DEV)
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            self._IdentityRopeWriter,
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
        # visible KV = prefix 4 + query 4 = 8 tokens = 2 pages of 4.
        required_width = 2 * _H_KV * _PAGE * _D
        runtime = SimpleNamespace(
            main_cache=SimpleNamespace(
                kv_cache_base=torch.zeros(4, required_width, dtype=torch.bfloat16, device=_DEV)
            ),
            main_inputs=attn_inputs,
            validate_before_projection=lambda **kwargs: None,
        )
        hidden = torch.zeros(4, 16, dtype=torch.bfloat16, device=_DEV)
        return impl, runtime, hidden

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_repeated_validation_of_one_step_launches_no_device_work(self):
        impl, runtime, hidden = self._fixture()
        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        # Drain earlier async work: the profiler window must only see this
        # validation's own launches.
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            impl.validate_qsa_before_side_write(runtime, object(), hidden)

        self.assertEqual(
            _device_work(prof),
            [],
            "the layers of one step validate identical geometry and must "
            "not re-enqueue the device verdict",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_memo_counts_and_revalidation_of_changed_geometry(self):
        impl, runtime, hidden = self._fixture()
        before = main_geometry_validation_stats()
        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        after = main_geometry_validation_stats()
        self.assertEqual(after["misses"] - before["misses"], 1)
        self.assertEqual(after["hits"] - before["hits"], 1)

        # A different cache is a new step geometry: it must revalidate and
        # still reject a malformed layout on the host (the check sits behind
        # the memo, so the raise proves the changed geometry revalidated).
        malformed_hnd = torch.zeros(
            4, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV
        ).transpose(1, 2)
        runtime.main_cache = SimpleNamespace(kv_cache_base=malformed_hnd)
        with self.assertRaisesRegex(RuntimeError, "HND inner layout"):
            impl.validate_qsa_before_side_write(runtime, object(), hidden)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_strict_mode_keeps_the_per_call_validation(self):
        impl, runtime, hidden = self._fixture()
        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        before = main_geometry_validation_stats()

        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                impl.validate_qsa_before_side_write(runtime, object(), hidden)

        self.assertTrue(
            _device_work(prof), "strict mode must revalidate every call"
        )
        self.assertEqual(main_geometry_validation_stats()["hits"], before["hits"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_device_source_sign_and_token_checks_stay_on_host_mirrors(self):
        impl, runtime, hidden = self._fixture(device_sources=True)
        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            impl.validate_qsa_before_side_write(runtime, object(), hidden)

        self.assertEqual(
            _device_work(prof),
            [],
            "device-source sign/token checks must read the per-step CPU "
            "mirrors, not the device tensors",
        )


if __name__ == "__main__":
    unittest.main()
