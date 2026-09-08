"""Benchmark definitions for prefill attention implementations.

To add an implementation benchmark:
1. Import the implementation class and add a PrefillImplBench subclass.
2. Set ``impl`` and override ``should_run_case()`` for planner-only filtering when needed.
3. Implement ``prepare()`` with ``build_instance()`` and return a ``PreparedRun``.
4. Register an instance in ``IMPL_BENCHES``.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from rtp_kernel.fused_rope_kvcache import prefill_fused_rope_kvcache

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferHybridPrefillImpl,
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench.core import (
    BenchCase,
    CaseData,
    PreparedRun,
    ReferencePolicy,
    Tolerance,
    UnsupportedImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    FlashInferTRTLLMFMHAv2PagedPrefillImpl,
    FlashInferTRTLLMFMHAv2PrefillImpl,
)
from rtp_llm.ops import ParallelismConfig


class PrefillImplBench(ABC):
    impl: type

    def should_run_case(self, case: BenchCase) -> bool:
        """
        Reject known mismatches before allocating GPU case data
        Runtime capability depends on built inputs and is checked later by impl.support()
        """
        return case.kv_dtype == "fp8" or case.kv_dtype == case.q_dtype

    def build_instance(
        self, case_data: CaseData, parallelism: ParallelismConfig
    ) -> Any:
        configs = case_data.make_attn_configs()
        try:
            supported = self.impl.support(configs, case_data.attn_inputs)
        except Exception as error:
            raise RuntimeError(f"support() raised: {error}") from error
        if not supported:
            raise UnsupportedImpl("support()==False")
        if not self.impl.support_parallelism_config(parallelism):
            raise UnsupportedImpl("support_parallelism_config()==False")
        return self.impl(configs, case_data.attn_inputs, parallelism)

    @staticmethod
    def tolerance(case: BenchCase) -> Tolerance:
        return (
            Tolerance(0.15, 0.15) if case.kv_dtype == "fp8" else Tolerance(0.015, 0.01)
        )

    @abstractmethod
    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        pass


class PyFlashinferPrefillBench(PrefillImplBench):
    impl = PyFlashinferPrefillImpl

    def should_run_case(self, case: BenchCase) -> bool:
        return case.prefix_len == 0 and super().should_run_case(case)

    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        instance = self.build_instance(case_data, parallelism)
        policy = ReferencePolicy(
            fp8_kv=case_data.case.kv_dtype == "fp8",
            quantize_q=instance.fmha_impl.q_dtype == torch.float8_e4m3fn,
        )
        reference = case_data.reference(policy) if check_correctness else None

        def invoke() -> torch.Tensor:
            query, key, value = instance._split_qkv(case_data.qkv)
            return instance.fmha_impl.forward(query, key, value, None)

        return PreparedRun(
            reference_output=reference,
            tolerance=self.tolerance(case_data.case),
            invoke=invoke,
        )


class PyFlashinferPagedPrefillBench(PrefillImplBench):
    impl = PyFlashinferPagedPrefillImpl

    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        instance = self.build_instance(case_data, parallelism)
        cache = case_data.make_cache(prefix_only=False)
        quantize_q = instance.fmha_impl.q_dtype == torch.float8_e4m3fn
        policy = ReferencePolicy(
            fp8_kv=case_data.case.kv_dtype == "fp8", quantize_q=quantize_q
        )
        reference = case_data.reference(policy) if check_correctness else None

        def invoke() -> torch.Tensor:
            query, _, _ = instance._split_qkv(case_data.qkv)
            return instance.fmha_impl.forward(query, cache)

        return PreparedRun(
            reference_output=reference,
            tolerance=self.tolerance(case_data.case),
            invoke=invoke,
        )


class PyFlashinferHybridPrefillBench(PrefillImplBench):
    impl = PyFlashinferHybridPrefillImpl

    def should_run_case(self, case: BenchCase) -> bool:
        return case.prefix_len > 0 and super().should_run_case(case)

    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        instance = self.build_instance(case_data, parallelism)
        cache = case_data.make_cache(prefix_only=True)
        quantize_q = instance.fmha_impl.q_dtype == torch.float8_e4m3fn
        policy = ReferencePolicy(
            hybrid=True,
            fp8_kv=case_data.case.kv_dtype == "fp8",
            quantize_q=quantize_q,
        )
        reference = case_data.reference(policy) if check_correctness else None

        def invoke() -> torch.Tensor:
            query, key, value = instance._split_qkv(case_data.qkv)
            return instance.fmha_impl.forward(query, key, value, cache)

        return PreparedRun(
            reference_output=reference,
            tolerance=self.tolerance(case_data.case),
            invoke=invoke,
        )


class FlashInferTRTLLMFMHAv2PrefillBench(PrefillImplBench):
    impl = FlashInferTRTLLMFMHAv2PrefillImpl

    def should_run_case(self, case: BenchCase) -> bool:
        return case.prefix_len == 0 and super().should_run_case(case)

    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        instance = self.build_instance(case_data, parallelism)
        policy = ReferencePolicy(
            fp8_kv=case_data.case.kv_dtype == "fp8",
            quantize_q=case_data.case.kv_dtype == "fp8",
        )
        reference = case_data.reference(policy) if check_correctness else None
        return PreparedRun(
            reference_output=reference,
            tolerance=self.tolerance(case_data.case),
            invoke=lambda: instance.fmha_impl.forward(
                case_data.qkv, None, instance.fmha_params
            ),
        )


class FlashInferTRTLLMFMHAv2PagedPrefillBench(PrefillImplBench):
    impl = FlashInferTRTLLMFMHAv2PagedPrefillImpl

    # Isolate attention by omitting RoPE and KV-cache writes from fused QOut.
    @staticmethod
    def _q_out_without_cache_write(
        instance: Any, case_data: CaseData, cache: Any
    ) -> torch.Tensor:
        op = instance.rope_kvcache_impl
        params = instance.rope_params
        config = op.attn_configs
        rope = config.rope_config
        return prefill_fused_rope_kvcache(
            case_data.qkv,
            params.cu_seqlens,
            params.cu_seqlens.size(0) - 1,
            params.max_seq_len,
            config.head_num,
            config.kv_head_num,
            config.size_per_head,
            tokens_per_block=config.kernel_tokens_per_block,
            store_q_no_transpose=True,
            store_q=False,
            store_kv=False,
            store_qkv=False,
            store_qkv_fp8=False,
            store_cache=False,
            use_paged_fmha=params.max_prefix_length > 0,
            kv_cache=cache.kv_cache_base,
            kv_cache_scale=cache.kv_scale_base,
            kv_cache_offset=params.kv_cache_offset,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=None,
            padding_offset=params.padding_offset,
            position_ids=params.position_ids,
            use_logn_attn=config.use_logn_attn,
            rope_style=rope.style,
            rope_dim=rope.dim,
            rope_base=rope.base,
            rope_scale=rope.scale,
            rope_beta_slow=rope.factor1,
            rope_beta_fast=rope.factor2,
            rope_original_max_position_embeddings=rope.max_pos,
            rope_extrapolation_factor=rope.extrapolation_factor,
            rope_mscale=rope.mscale,
            rope_offset=rope.offset,
            rope_index_factor=rope.index_factor,
            rope_mrope_dim1=rope.mrope_dim1,
            rope_mrope_dim2=rope.mrope_dim2,
            rope_mrope_dim3=rope.mrope_dim3,
            prefix_prompt_lengths=params.prefix_lengths,
            max_prefix_length=params.max_prefix_length,
            count_length=params.max_prefix_length > 0,
        )

    def prepare(
        self,
        case_data: CaseData,
        parallelism: ParallelismConfig,
        check_correctness: bool,
    ) -> PreparedRun:
        instance = self.build_instance(case_data, parallelism)
        cache = case_data.make_cache(prefix_only=False)
        policy = ReferencePolicy(
            fp8_kv=case_data.case.kv_dtype == "fp8",
            quantize_q=case_data.case.kv_dtype == "fp8",
        )
        reference = case_data.reference(policy) if check_correctness else None

        def invoke() -> torch.Tensor:
            query = self._q_out_without_cache_write(instance, case_data, cache)
            return instance.fmha_impl.forward(query, cache, instance.fmha_params)

        return PreparedRun(
            reference_output=reference,
            tolerance=self.tolerance(case_data.case),
            invoke=invoke,
        )


IMPL_BENCHES: tuple[PrefillImplBench, ...] = (
    PyFlashinferPrefillBench(),
    PyFlashinferPagedPrefillBench(),
    PyFlashinferHybridPrefillBench(),
    FlashInferTRTLLMFMHAv2PrefillBench(),
    FlashInferTRTLLMFMHAv2PagedPrefillBench(),
)
