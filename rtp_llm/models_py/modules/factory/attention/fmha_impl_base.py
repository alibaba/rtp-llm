from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import KVCache, ParamsBase, PyAttentionInputs


class FMHAImplBase(object):
    fmha_impl: Any
    fmha_params: ParamsBase
    rope_params: Any
    rope_kvcache_impl: Any
    write_cache_store_impl: Any
    attn_inputs: PyAttentionInputs
    support_: bool = False

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        init_params: bool = True,
    ) -> None:
        self.fmha_impl = fmha_impl
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.support_: bool = self.fmha_impl.support(attn_inputs)
        self.fmha_params = None
        self.rope_params = None
        self.write_cache_store_impl = None
        if self.support_ and init_params:
            self.rope_kvcache_impl = rope_kvcache_impl
            self.attn_inputs = attn_inputs
            if self.attn_inputs.is_prefill and self.attn_inputs.cache_store_inputs:
                self.write_cache_store_impl = WriteCacheStoreOp(
                    self.attn_inputs.input_lengths,
                    self.attn_inputs.prefix_lengths,
                    self.attn_inputs.kv_cache_block_id_host,
                    self.attn_inputs.cache_store_inputs,
                )
            self.create_params(attn_inputs)
            if attn_inputs.is_cuda_graph is False:
                self.prepare(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(
                qkv, self.fmha_type(), kv_cache, self.rope_params
            )
        else:
            fmha_input = qkv
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        return res

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.NONE

    def create_params(self, attn_inputs: PyAttentionInputs):
        pass

    def support(self):
        return self.support_

    def support_cuda_graph(self) -> bool:
        return False

    def _update_trt_params(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        new_offset = new_fmha_params.kv_cache_offset
        old_offset = self.fmha_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

    def copy_kv_cache_offset(self, old_offset: torch.Tensor, new_offset: torch.Tensor):
        if new_offset.shape == old_offset.shape:
            old_offset.copy_(new_offset, non_blocking=True)
        else:
            # Build slice indices dynamically
            slice_indices = [
                slice(0, new_offset.size(dim)) for dim in range(new_offset.dim())
            ]
            target_slice = old_offset[tuple(slice_indices)]
            target_slice.copy_(new_offset, non_blocking=True)

    def prepare(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)


class FMHAPrefillImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        super().__init__(fmha_impl, rope_kvcache_impl, attn_inputs)


class FMHADecodeImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        super().__init__(fmha_impl, rope_kvcache_impl, attn_inputs)
