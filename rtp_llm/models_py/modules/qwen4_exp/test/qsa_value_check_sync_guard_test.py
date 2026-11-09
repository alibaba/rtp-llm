"""DtoH-sync guards for the QSA indexer chain's value-dependent checks.

Campaign ledger (H20 profiler evidence): metadata checks (shape/dtype/device/
stride) cost zero device-to-host copies, while ``torch.equal`` / ``bool(any())``
/ ``nonzero`` on device tensors each cost one DtoH memcpy plus one stream
synchronization.  Hot-path entries (per decode layer, per target-verify layer,
per step plan build) must therefore keep their value-dependent checks in a
compliant form:

* host-mirror comparison (pinned host tensor, or the once-per-step mirror
  readback for ``RTP_LLM_DEVICE_INPUT`` engines), or
* ``qsa_assert_valid`` (device comparison plus an enqueued async assert), or
* strict-only legacy host validation (``RTP_LLM_QSA_STRICT_VALIDATION=1``).

Every guard profiles one full hot-path entry in production (non-strict) mode
and asserts zero ``Memcpy DtoH`` and zero ``cudaStreamSynchronize`` events.
Against the pre-cleanup code these guards fail (RED); the compliant forms
make them pass (GREEN).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    clear_step_plan_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_inputs, sparse_gqa_impl
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_base_rope,
    build_qsa_rope,
)
from rtp_llm.ops import RopeStyle

# Production mode: value-dependent device checks take the async/mirror forms.
_PRODUCTION_ENV = {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}


def _profiled(fn):
    """Run ``fn`` under the profiler and return (dtoh, stream_sync) counts."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    events = prof.events()
    dtoh = sum(1 for event in events if "Memcpy DtoH" in event.name)
    sync = sum(1 for event in events if event.name == "cudaStreamSynchronize")
    return dtoh, sync


def _assert_no_dtoh(case, entry, counts):
    dtoh, sync = counts
    case.assertEqual(
        (dtoh, sync),
        (0, 0),
        f"{entry} synchronized with the device in production mode: "
        f"{dtoh} DtoH memcpy, {sync} cudaStreamSynchronize",
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaRopeValueCheckSyncGuardTest(unittest.TestCase):
    """Per-call guards for the indexer RoPE position checks."""

    def setUp(self):
        torch.manual_seed(31)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _base_config(self, index_factor):
        return SimpleNamespace(
            style=RopeStyle.Base,
            index_factor=index_factor,
            dim=64,
            base=10_000,
            scale=1.0,
            indexer_is_neox_style=True,
        )

    def test_build_base_rope_axis_check_has_no_dtoh(self):
        config = self._base_config(3)
        positions = torch.arange(8, device=self.device, dtype=torch.int32)
        transported = positions.unsqueeze(1).expand(-1, 3).reshape(-1)
        kwargs = dict(token_count=8, dtype=torch.bfloat16, device=self.device)
        with patch.dict(os.environ, _PRODUCTION_ENV):
            build_base_rope(transported, config, **kwargs)
            counts = _profiled(
                lambda: build_base_rope(transported, config, **kwargs)
            )
        _assert_no_dtoh(self, "build_base_rope equal-axes check", counts)

    def test_logical_cache_positions_check_has_no_dtoh(self):
        config = self._base_config(1)
        positions = torch.arange(8, device=self.device, dtype=torch.int32)
        logical = torch.arange(8, device=self.device, dtype=torch.int32)
        kwargs = dict(
            token_count=8,
            dtype=torch.bfloat16,
            device=self.device,
            logical_positions=logical,
        )
        with patch.dict(os.environ, _PRODUCTION_ENV):
            build_qsa_rope(positions, config, **kwargs)
            counts = _profiled(lambda: build_qsa_rope(positions, config, **kwargs))
        _assert_no_dtoh(self, "build_qsa_rope logical-positions check", counts)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaDecodeValueCheckSyncGuardTest(unittest.TestCase):
    """Per-layer eager-decode guard for select_decode_tokens (target MRoPE).

    The model calls ``select_decode_tokens`` once per sparse layer per decode
    step (``qwen4_exp.py`` dispatches it inside every sparse attention layer),
    so every value-dependent check on this path runs per layer, not per step.
    """

    PAGE_SIZE = 128
    D = 128

    def setUp(self):
        torch.manual_seed(31)
        clear_step_plan_cache()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.lengths = torch.tensor([7], dtype=torch.int32).pin_memory()
        self.input_lengths = torch.tensor([1], dtype=torch.int32).pin_memory()
        self.prefixes = torch.empty(0, dtype=torch.int32).pin_memory()
        self.positions = torch.tensor([7, 7, 7], dtype=torch.int32, device="cuda")
        self.kv_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
        self.state_table = self.kv_table.clone()
        self.kv = torch.zeros(
            3, (self.PAGE_SIZE // 4) * self.D * 2, dtype=torch.uint8, device="cuda"
        )
        self.state = torch.zeros(3, 8 * self.D, dtype=torch.float32, device="cuda")
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=4,
            compress_ratio=4,
            token_budget=8,
            k_norm_gamma=(torch.randn(self.D, device="cuda") * 0.1).bfloat16(),
            norm_eps=1e-6,
        )
        self.rope = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=64,
            mrope_dim1=12,
            mrope_dim2=10,
            mrope_dim3=10,
            mrope_interleaved=True,
            base=10000,
            scale=1.0,
        )
        self.q = torch.randn(1, 4, self.D, dtype=torch.bfloat16, device="cuda")
        self.raw = torch.randn(1, self.D, dtype=torch.bfloat16, device="cuda")

    def _context(self):
        def inputs(kv_map=None, state_map=None):
            return SimpleNamespace(
                is_prefill=False,
                is_target_verify=False,
                is_cuda_graph=False,
                is_exact_cuda_graph_batch=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=self.input_lengths,
                prefix_lengths=self.prefixes,
                sequence_lengths=self.lengths,
                cu_seqlens_device=None,
                cu_kv_seqlens_device=None,
                combo_position_ids=self.positions,
                kv_cache_kernel_block_id_device=kv_map,
                kv_cache_block_id_device=state_map,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv,
                seq_size_per_block=self.PAGE_SIZE,
            ),
            indexer_kv_inputs=inputs(kv_map=self.kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state,
                seq_size_per_block=self.PAGE_SIZE,
            ),
            indexer_state_inputs=inputs(state_map=self.state_table),
        )

    def test_select_decode_tokens_has_no_dtoh(self):
        context = self._context()

        def call():
            context.select_decode_tokens(
                self.q, self.raw, indexer=self.indexer, rope_config=self.rope
            )
            context.finalize_side_cache()

        with patch.dict(os.environ, _PRODUCTION_ENV):
            call()
            call()
            counts = _profiled(call)
        _assert_no_dtoh(self, "select_decode_tokens (eager decode)", counts)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTargetVerifyValueCheckSyncGuardTest(unittest.TestCase):
    """Per-layer eager target-verify guard for the geometry plan build.

    ``validate_qsa_before_side_write`` rebuilds the target-verify geometry in
    every sparse layer of the step, so the prefix/cu/position checks on this
    path run per layer, not per step.
    """

    D = 128
    RATIO = 4
    QUERY = 4
    PREFIX = 7

    def setUp(self):
        torch.manual_seed(31)
        clear_step_plan_cache()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.kv = torch.zeros(
            8, (64 // self.RATIO) * self.D * 2, dtype=torch.uint8, device="cuda"
        )
        self.state = torch.zeros(8, 8 * self.D, dtype=torch.float32, device="cuda")
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=4,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=torch.ones(self.D, dtype=torch.bfloat16, device="cuda"),
            norm_eps=1e-6,
        )

    def _context(self):
        device = "cuda"
        input_lengths = torch.tensor([self.QUERY], dtype=torch.int32).pin_memory()
        prefix_lengths = torch.tensor([self.PREFIX], dtype=torch.int32).pin_memory()
        sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
        cu_seqlens = torch.tensor(
            [0, self.QUERY], dtype=torch.int32, device=device
        )
        cu_kv_seqlens = torch.tensor(
            [0, self.PREFIX + self.QUERY], dtype=torch.int32, device=device
        )
        positions = torch.arange(
            self.PREFIX, self.PREFIX + self.QUERY, dtype=torch.int32, device=device
        )
        positions = positions.unsqueeze(1).expand(-1, 3).reshape(-1)
        kv_table = torch.tensor([[3, 4]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[6, 7]], dtype=torch.int32, device=device)

        def inputs(kv_map=None, state_map=None):
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=True,
                is_cuda_graph=False,
                is_exact_cuda_graph_batch=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_kv_seqlens,
                combo_position_ids=positions,
                kv_cache_kernel_block_id_device=kv_map,
                kv_cache_block_id_device=state_map,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv,
                seq_size_per_block=64,
            ),
            indexer_kv_inputs=inputs(kv_map=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state,
                seq_size_per_block=8,
            ),
            indexer_state_inputs=inputs(state_map=state_table),
        )

    def test_target_verify_geometry_has_no_dtoh(self):
        context = self._context()
        kwargs = dict(
            indexer=self.indexer,
            token_count=self.QUERY,
            device=self.device,
        )

        def call():
            context._target_verify_geometry(**kwargs)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            call()
            call()
            counts = _profiled(call)
        _assert_no_dtoh(self, "_target_verify_geometry (eager target verify)", counts)

    def test_target_verify_geometry_device_prefixes_amortize_readback(self):
        # Device-input engines (RTP_LLM_DEVICE_INPUT) keep prefix_lengths on
        # the device: the first layer of a step reads the mirror back once and
        # every later layer reuses it without synchronizing.  The three cache
        # regions share one device tensor, matching the engine's per-step
        # inputs invariant.
        context = self._context()
        device_prefixes = context.main_inputs.prefix_lengths.cuda()
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.prefix_lengths = device_prefixes
        kwargs = dict(
            indexer=self.indexer,
            token_count=self.QUERY,
            device=self.device,
        )

        def call():
            context._target_verify_geometry(**kwargs)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            call()
            counts = _profiled(call)
        _assert_no_dtoh(
            self, "_target_verify_geometry (device prefixes, warm mirror)", counts
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaPagedInputsValueCheckSyncGuardTest(unittest.TestCase):
    """Per-layer eager-decode guard for the paged FMHA input planner.

    ``SparseGqaFmhaImpl.forward`` calls ``_prepare_paged_inputs`` in every
    sparse layer of an eager decode step, so its sign/capacity/block-table and
    selection checks run per layer.
    """

    D = 128
    PAGE = 4

    def setUp(self):
        torch.manual_seed(31)

    def _impl(self):
        attn_configs = SimpleNamespace(
            head_num=4,
            kv_head_num=1,
            size_per_head=self.D,
            kernel_tokens_per_block=self.PAGE,
            is_sparse=True,
            use_mla=False,
            use_sparse_gqa_fmha=True,
        )
        sequence_lengths = torch.tensor([4, 6], dtype=torch.int32).pin_memory()
        input_lengths = torch.ones(2, dtype=torch.int32).pin_memory()
        prefix_lengths = torch.zeros(2, dtype=torch.int32).pin_memory()
        block_table = torch.tensor(
            [[1, 2, 0], [3, 4, 0]], dtype=torch.int32, device="cuda"
        )
        attn_inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=False,
            is_exact_cuda_graph_batch=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            sequence_lengths=sequence_lengths,
            kv_cache_kernel_block_id_device=block_table,
        )

        class _IdentityDecodeWriter:
            def __init__(self, configs):
                self.q_width = configs.head_num * configs.size_per_head

            def prepare(self, inputs):
                return object()

            def forward(self, qkv, kv_cache, params):
                return qkv[:, : self.q_width]

        with patch.object(
            sparse_gqa_impl, "FusedRopeKVCacheDecodeOp", _IdentityDecodeWriter
        ):
            return sparse_gqa_impl.SparseGqaFmhaImpl(attn_configs, attn_inputs)

    def test_prepare_paged_inputs_has_no_dtoh(self):
        impl = self._impl()
        q_width = impl.q_width
        kv_width = impl.kv_width
        tokens = 2
        qkv = torch.randn(
            tokens, q_width + 2 * kv_width, dtype=torch.bfloat16, device="cuda"
        )
        cache = SimpleNamespace(
            kv_cache_base=torch.randn(
                5, 2, 1, self.PAGE, self.D, dtype=torch.bfloat16, device="cuda"
            )
        )
        selected = torch.tensor(
            [[0, 2, -1], [1, 3, -1]], dtype=torch.int32, device="cuda"
        )

        def call():
            fmha_inputs._prepare_paged_inputs(impl, qkv, cache, selected)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            call()
            counts = _profiled(call)
        _assert_no_dtoh(self, "_prepare_paged_inputs (eager decode)", counts)


if __name__ == "__main__":
    unittest.main()
