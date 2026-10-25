"""Factory selection and capture/low-latency contracts for Hopper FP8 experts."""

from __future__ import annotations

import dataclasses
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import (
    _MASKED_MAX_N,
    GroupedFP8Strategy,
    _has_grouped_fp8_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
    CudaGroupedFp4Strategy,
    CudaGroupedFp8Strategy,
    CudaLocalLoopStrategy,
    CudaMegaMoeSEStrategy,
    CudaMegaMoeStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)


def _cfg(ep_size: int = 1) -> MoeCfg:
    """A minimal MoeCfg sufficient for ``can_handle`` checks."""
    n_local = 256 // max(ep_size, 1)
    return MoeCfg(
        layer_id=2,
        dim=7168,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=n_local,
        local_expert_start=0,
        local_expert_end=n_local,
        max_tokens_per_rank=8192,
    )


@contextmanager
def _env(**kw):
    """Temporarily set env vars; ``None`` value pops the var."""
    saved = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StrategySelectTest(unittest.TestCase):
    def _config(self, ep=1, storage="fp8", requested="auto", redundant=False):
        return Fp8Fp4MoeRuntimeConfig(
            layer_id=0,
            hidden_size=512,
            moe_inter_dim=256,
            expert_num=16,
            moe_k=4,
            n_shared_experts=1,
            swiglu_limit=7.0,
            ep_size=ep,
            ep_rank=0,
            max_tokens_per_rank=8,
            moe_strategy=requested,
            expert_dtype=storage,
            is_decode_role=True,
            physical_expert_num=32 if redundant else 16,
        )

    def _registry(self):
        registry = StrategyRegistry()
        for cls in (
            CudaMegaMoeSEStrategy,
            CudaMegaMoeStrategy,
            CudaGroupedFp4Strategy,
            CudaLocalLoopStrategy,
            CudaGroupedFp8Strategy,
        ):
            registry.register(cls())
        return registry

    def test_factory_selects_hopper_for_local_and_ep_converted_experts(self):
        for ep in (1, 2, 4, 8):
            for requested in ("auto", "grouped_fp8"):
                with self.subTest(ep=ep, requested=requested), mock.patch.object(
                    GroupedFP8Strategy, "can_handle", return_value=True
                ):
                    self.assertIsInstance(
                        self._registry().get_strategy(
                            self._config(ep, requested=requested)
                        ),
                        CudaGroupedFp8Strategy,
                    )

    def test_hopper_requires_fp8_storage_and_supported_hardware(self):
        for storage, capable in (("fp4", True), ("fp8", False)):
            with self.subTest(storage=storage, capable=capable), mock.patch.object(
                GroupedFP8Strategy, "can_handle", return_value=capable
            ):
                self.assertFalse(
                    CudaGroupedFp8Strategy().can_handle(self._config(storage=storage))
                )

    def test_explicit_fp4_strategy_cannot_consume_converted_experts(self):
        for requested in ("mega_moe", "mega_moe_se", "grouped_fp4", "local_loop"):
            with self.subTest(requested=requested), mock.patch.object(
                GroupedFP8Strategy, "can_handle", return_value=True
            ):
                with self.assertRaises(ValueError):
                    self._registry().get_strategy(
                        self._config(ep=4, requested=requested)
                    )

    def test_redundant_experts_are_rejected(self):
        with mock.patch.object(GroupedFP8Strategy, "can_handle", return_value=True):
            self.assertFalse(
                CudaGroupedFp8Strategy().can_handle(self._config(redundant=True))
            )

    def test_grouped_fp8_kernel_probe(self):
        """env off / no CUDA / kernel absent / non-Hopper each disable it.

        The predicate is ``@functools.cache``d -- it is called once per layer, so
        caching the import and the capability query is deliberate -- which means
        every case here has to clear it first.
        """
        impl = (
            "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper."
            "_m_grouped_fp8_gemm_nt_contiguous_impl"
        )
        cuda = "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8.torch.cuda"

        # The last probe below leaves a mocked True in the cache, and a failing
        # assertion would otherwise skip the trailing clear and leak that answer
        # into every later case in this process.
        self.addCleanup(_has_grouped_fp8_kernel.cache_clear)

        def probe():
            _has_grouped_fp8_kernel.cache_clear()
            return _has_grouped_fp8_kernel()

        with _env(DSV4_USE_GROUPED_FP8="0"):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=False):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=True), mock.patch(
            impl, None
        ):
            self.assertFalse(probe())
        # Gated to Hopper rather than "not SM100" on purpose: SM100 has the FP4
        # kernels, which are faster, so this path would only regress there.
        with mock.patch(f"{cuda}.is_available", return_value=True), mock.patch(
            impl, object()
        ), mock.patch(f"{cuda}.get_device_capability", return_value=(10, 0)):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=True), mock.patch(
            impl, object()
        ), mock.patch(f"{cuda}.get_device_capability", return_value=(9, 0)):
            self.assertTrue(probe())
        _has_grouped_fp8_kernel.cache_clear()

    # --- forced override ---------------------------------------------------


class GroupedFP8CaptureGuardTest(unittest.TestCase):
    """``_assert_one_captured_size``: turn a replay-time NCCL hang into a startup error.

    The hazard is that ``CudaGraphRunner`` selects a graph per rank from that rank's
    own batch size while this strategy's EP collectives sit inside the graph, so two
    ranks on different graphs hang. Neither condition is observable from inside the
    strategy at replay time, but both are observable at capture time: more than one
    captured size, or a captured size the scheduler can exceed.
    """

    CAPTURING = (
        "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8."
        "torch.cuda.is_current_stream_capturing"
    )

    def _strategy(self, ep_size: int, max_tokens_per_rank: int) -> GroupedFP8Strategy:
        # MoeCfg is immutable, so the bound is set by rebuilding it.
        cfg = dataclasses.replace(
            _cfg(ep_size=ep_size),
            max_tokens_per_rank=max_tokens_per_rank,
            is_decode_role=True,
        )
        strat = GroupedFP8Strategy(cfg)
        strat._captured_ns = set()
        return strat

    def test_single_size_at_the_scheduler_bound_is_accepted(self):
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(8)  # same size again is fine

    def test_second_captured_size_raises(self):
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            with self.assertRaises(RuntimeError) as cm:
                strat._assert_one_captured_size(16)
        self.assertIn("more than one batch size", str(cm.exception))
        self.assertIn("concurrency_limit", str(cm.exception))

    def test_captured_size_below_scheduler_bound_raises(self):
        # A rank whose batch exceeds the graph falls back to eager while the others
        # replay -- the same divergence, reached the other way.
        strat = self._strategy(ep_size=4, max_tokens_per_rank=64)
        with mock.patch(self.CAPTURING, return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                strat._assert_one_captured_size(8)
        self.assertIn("may hand it up to", str(cm.exception))

    def test_ep1_and_non_capturing_are_exempt(self):
        # ep_size == 1 has no EP collectives in the graph, and outside capture the
        # runner is not selecting graphs at all.
        strat = self._strategy(ep_size=1, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(999)
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=False):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(999)


class GroupedFP8LowLatencyGateTest(unittest.TestCase):
    """The low-latency buffer must only be built where the role can reach it.

    ``_ll_max_tokens`` rounds the requested tokens-per-rank up to the alignment the
    masked GEMM needs at this ``ep``: the 64-row floor is on ``M_MAX * ep``, not on
    ``M_MAX``, so the unit is ``64 // gcd(64, ep)``.
    """

    def test_ll_max_tokens_alignment_unit_follows_ep(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import (
            _ll_max_tokens,
        )

        with _env(DSV4_MOE_LL_MAX_TOKENS="8"):
            self.assertEqual(_ll_max_tokens(4), 16)  # 64 // gcd(64, 4) = 16
            self.assertEqual(_ll_max_tokens(8), 8)  # 64 // gcd(64, 8) = 8
            self.assertEqual(_ll_max_tokens(1), 64)  # single rank: the full floor
        with _env(DSV4_MOE_LL_MAX_TOKENS="0"):
            self.assertEqual(_ll_max_tokens(4), 16)  # empty/0 still clears the floor

    def test_gate_predicate_separates_the_roles(self):
        """``_ll_gate`` decides whether a 136 MB RDMA buffer is allocated.

        ``resolve_moe_max_tokens_per_rank`` gives a decode role
        ``max_generate_batch_size * tokens_per_batch`` and any other role a >= 4096
        budget, so the predicate is exactly "is this a decode role". Asserted
        through the predicate rather than through arithmetic on ``_ll_max_tokens``:
        an earlier version of this case compared 8 <= 16 and 8192 > 16, which held
        no matter what the gate did.
        """
        with _env(DSV4_MOE_DEEPEP_LL="1", DSV4_MOE_LL_MAX_TOKENS="8"):
            # decode: bound is the concurrency limit, inside the LL capacity
            self.assertTrue(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=4), max_tokens_per_rank=8)
                )
            )
            # prefill: >= 4096 budget, far past it
            self.assertFalse(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=4), max_tokens_per_rank=8192)
                )
            )
            # exactly at the rounded capacity is still reachable
            self.assertTrue(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=4), max_tokens_per_rank=16)
                )
            )
            self.assertFalse(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=4), max_tokens_per_rank=17)
                )
            )
            # single rank: no exchange to accelerate
            self.assertFalse(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=1), max_tokens_per_rank=8)
                )
            )
        # opt-in flag off -> gate closed regardless of the bound
        with _env(DSV4_MOE_DEEPEP_LL="0", DSV4_MOE_LL_MAX_TOKENS="8"):
            self.assertFalse(
                GroupedFP8Strategy._ll_gate(
                    dataclasses.replace(_cfg(ep_size=4), max_tokens_per_rank=8)
                )
            )


class GroupedFP8ShouldMaskTest(unittest.TestCase):
    """``_should_mask`` reads a shape and an env var, never the data.

    Host-only by design -- every rank has to reach the same answer or the group
    deadlocks at the next collective -- so it belongs in the CPU suite. It used to
    be covered from the GPU-only equivalence module, where the capture branch was
    unreachable and the whole case skipped on any CPU lane.
    """

    CAPTURING = (
        "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8."
        "torch.cuda.is_current_stream_capturing"
    )
    WARMUP_FORWARD = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"

    def setUp(self):
        # Three inputs decide this predicate and only one is named per case, so the
        # other two are cleared here. In particular a lane that exports
        # RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD would make every expected-False case
        # return True.
        for name in ("DSV4_MOE_MASKED", self.WARMUP_FORWARD):
            saved = os.environ.get(name)
            os.environ.pop(name, None)
            self.addCleanup(
                lambda n=name, v=saved: (
                    os.environ.pop(n, None)
                    if v is None
                    else os.environ.__setitem__(n, v)
                )
            )

    def test_env_enabled_and_not_capturing(self):
        with _env(DSV4_MOE_MASKED="1"), mock.patch(self.CAPTURING, return_value=False):
            self.assertTrue(GroupedFP8Strategy._should_mask(8))
            self.assertTrue(GroupedFP8Strategy._should_mask(_MASKED_MAX_N))
            # Past the cap and not capturing: the contiguous path is still legal.
            self.assertFalse(GroupedFP8Strategy._should_mask(_MASKED_MAX_N + 1))

    def test_env_disabled_and_not_capturing(self):
        with _env(DSV4_MOE_MASKED="0"), mock.patch(self.CAPTURING, return_value=False):
            self.assertFalse(GroupedFP8Strategy._should_mask(8))

    def test_capture_forces_the_masked_layout(self):
        with _env(DSV4_MOE_MASKED="0"), mock.patch(self.CAPTURING, return_value=True):
            self.assertTrue(GroupedFP8Strategy._should_mask(8))

    def test_pre_capture_warmup_forward_takes_the_masked_layout(self):
        """The forward ``initCapture`` runs before ``captureDecode()``.

        DeepGEMM cannot JIT inside a capture, so the masked kernel has to be
        compiled by that eager forward. Keyed only on
        ``is_current_stream_capturing()`` this held solely when
        ``DSV4_MOE_MASKED=1`` happened to be set, which is a guarantee resting on
        deployment configuration.
        """
        with _env(
            DSV4_MOE_MASKED="0", RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD="1"
        ), mock.patch(self.CAPTURING, return_value=False):
            self.assertTrue(GroupedFP8Strategy._should_mask(8))
            # Fail during eager pre-capture warmup, before capture begins.
            with self.assertRaisesRegex(RuntimeError, "DSV4_MOE_MASKED_MAX_N"):
                GroupedFP8Strategy._should_mask(_MASKED_MAX_N + 1)

    def test_pre_capture_flag_off_leaves_the_contiguous_path(self):
        with _env(
            DSV4_MOE_MASKED="0", RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD="0"
        ), mock.patch(self.CAPTURING, return_value=False):
            self.assertFalse(GroupedFP8Strategy._should_mask(8))

    def test_capture_past_the_cap_raises(self):
        """Capture cannot express the contiguous path, so this must not fall back."""
        with _env(DSV4_MOE_MASKED="1"), mock.patch(self.CAPTURING, return_value=True):
            with self.assertRaisesRegex(RuntimeError, "DSV4_MOE_MASKED_MAX_N"):
                GroupedFP8Strategy._should_mask(_MASKED_MAX_N + 1)

    def test_configured_capacity_accepts_larger_capture_shapes(self):
        with _env(DSV4_MOE_MASKED_MAX_N="2048"), mock.patch(
            self.CAPTURING, return_value=True
        ):
            self.assertTrue(GroupedFP8Strategy._should_mask(2048))
            with self.assertRaisesRegex(RuntimeError, "2049.*2048"):
                GroupedFP8Strategy._should_mask(2049)

    def test_invalid_capacity_is_rejected_without_cuda(self):
        for value in ("0", "-1", "not-an-int"):
            with self.subTest(value=value), _env(
                DSV4_MOE_MASKED_MAX_N=value
            ), mock.patch(self.CAPTURING, return_value=True):
                with self.assertRaisesRegex(ValueError, "DSV4_MOE_MASKED_MAX_N"):
                    GroupedFP8Strategy._should_mask(1)

    def test_graph_capacity_and_role_are_validated_at_initialization(self):
        with _env(DSV4_MOE_DEEPEP_LL="0", DSV4_MOE_MASKED_MAX_N="1024"):
            cfg = dataclasses.replace(
                _cfg(ep_size=4),
                max_tokens_per_rank=512,
                is_decode_role=True,
                enable_cuda_graph=True,
            )
            with self.assertRaisesRegex(ValueError, "DSV4_MOE_MASKED_MAX_N"):
                GroupedFP8Strategy(cfg)
            with _env(DSV4_MOE_MASKED_MAX_N="2048"):
                GroupedFP8Strategy(cfg)
            cfg = dataclasses.replace(cfg, is_decode_role=False)
            with self.assertRaisesRegex(ValueError, "DECODE role"):
                GroupedFP8Strategy(cfg)
            # EP=1 PDFUSION can capture small decode sizes even when its
            # prefill capacity is much larger than the masked limit.
            GroupedFP8Strategy(
                dataclasses.replace(
                    _cfg(ep_size=1),
                    enable_cuda_graph=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
