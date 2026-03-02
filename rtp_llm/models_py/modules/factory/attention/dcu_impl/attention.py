import logging
from typing import Any, List, Optional

import torch
from flash_attn import vllm_flash_attn_varlen_func

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention.dcu_impl.rope_kvcache import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
)
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
)


# Pure Python implementation of FMHAParams
class FMHAParams(ParamsBase):
    """Python implementation of FMHAParams for DCU attention operations."""

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        is_prefill: bool = True,
        enable_cuda_graph: bool = False,
        graph_max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph
        self.graph_max_seq_len = graph_max_seq_len

        # Prefill mode
        if is_prefill:
            input_lengths = attn_inputs.input_lengths
            prefix_lengths = (
                attn_inputs.prefix_lengths
                if hasattr(attn_inputs, "prefix_lengths")
                else None
            )

            self.max_seq_len = input_lengths.max().item()
            batch_size = input_lengths.size(0)

            # Create cu_seqlens_q for query (based on input_lengths only)
            self.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=input_lengths.device
            )
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths, 0)

            kv_lengths = torch.zeros_like(input_lengths)
            # Create cu_seqlens_k for key/value (includes prefix_lengths)
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                kv_lengths = input_lengths + prefix_lengths
                self.cu_seqlens_k = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=input_lengths.device
                )
                self.cu_seqlens_k[1:] = torch.cumsum(kv_lengths, 0)
                # Calculate max sequence length including prefix
                max_prefix_length = (
                    prefix_lengths.max().item() if prefix_lengths.numel() > 0 else 0
                )
                self.max_seqlen_k = self.max_seq_len + max_prefix_length
            else:
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.max_seqlen_k = self.max_seq_len

            self.cu_seqlens_q = self.cu_seqlens_q.cuda()
            self.max_seqlen_q = self.max_seq_len
            self.seq_lens = None
            self.seqused_k = input_lengths.cuda()
            self.kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )
            self.prefix_lengths = prefix_lengths
            self.token_q_num = input_lengths.sum().item()
            self.token_kv_num = kv_lengths.sum().item()
        # Decode mode
        else:
            input_lengths = attn_inputs.input_lengths
            batch_size = input_lengths.size(0)
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )

            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
            elif sequence_lengths is not None:
                self.max_seq_len = sequence_lengths.max().item() + 1
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = batch_size
            self.cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=input_lengths.device
            )
            self.cu_seqlens_k = self.cu_seqlens_q

            # Create seq_lens on CUDA
            self.cu_seqlens_q = self.cu_seqlens_q.cuda()
            if sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None
            self.seqused_k = self.seq_lens

    def fillParams(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host=None,
        kv_cache_block_id_device=None,
    ):
        self.sequence_lengths = sequence_lengths
        self.input_lengths = input_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_device is not None:
            self.kv_cache_block_id_device = kv_cache_block_id_device
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
            else:
                self.max_seq_len = self.sequence_lengths.max().item() + 1
            self.max_seqlen_k = self.max_seq_len

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class DcuPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.size_per_head = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.tokens_per_block
        self.total_elems = attn_configs.kv_head_num * attn_configs.size_per_head * attn_configs.tokens_per_block
        self.softmax_scale = 1 / self.size_per_head ** 0.5

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        fmha_params.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_host is not None and kv_cache_block_id_host.numel() > 0:
            fmha_params.kv_cache_block_id_device = kv_cache_block_id_host.cuda()
        else:
            # cuda_graph capture path: legacy kv_cache_block_id_* is empty; fall back to the
            # kernel field (always initialized for cuda graph), promoting 2-D -> 3-D so that
            # the forward path's `[0]` group-index still works.
            fallback = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if fallback is not None and fallback.numel() > 0:
                if fallback.dim() == 2:
                    fallback = fallback.unsqueeze(0)
                fmha_params.kv_cache_block_id_device = fallback
            else:
                fmha_params.kv_cache_block_id_device = None
        #fmha_params.kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        #fmha_params.kv_cache_block_id_device = attn_inputs.kv_cache_block_id_host.cuda()
        return fmha_params

    def reshape_kvcache(self, kv_cache: KVCache):
        block_num = kv_cache.kv_cache_base.shape[1]
        key_cache = kv_cache.kv_cache_base[0].reshape(block_num, self.head_num_kv, self.tokens_per_block, self.size_per_head)
        value_cache = kv_cache.kv_cache_base[1].reshape(block_num, self.head_num_kv, self.size_per_head, self.tokens_per_block)
        return key_cache, value_cache

    def forward(self, qkv, kv_cache: KVCache, fmha_params: FMHAParams):
        q, _, _ = qkv[0],qkv[1],qkv[2]
        q = q.reshape(-1, self.head_num, self.size_per_head)

        key_cache, value_cache = self.reshape_kvcache(kv_cache)
        output = torch.zeros(q.shape, dtype=q.dtype, device=q.device)

        #print(f"PrefillAttnOp: {fmha_params.cu_seqlens_q=}, {fmha_params.max_seqlen_q=}, {fmha_params.seqused_k=}, {fmha_params.max_seqlen_k=}, k: {key_cache[1, 0, :, 0].detach().cpu().tolist()}, v: {value_cache[1, 0, 0, :].detach().cpu().tolist()}, {fmha_params.kv_cache_block_id_device=}")

        res = vllm_flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=fmha_params.cu_seqlens_q,
            max_seqlen_q=fmha_params.max_seqlen_q,
            seqused_k=fmha_params.seqused_k,
            max_seqlen_k=fmha_params.max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=True,
            alibi_slopes=None,
            window_size=(-1, -1),
            block_table=fmha_params.kv_cache_block_id_device[0],
            softcap=0,
            scheduler_metadata=None,
            is_prefix_cache=True,
        )
        token_num = fmha_params.token_q_num
        final_result = res.reshape(token_num, self.head_num * self.size_per_head)
        return final_result


class DcuDecodeAttnOp:
    """DCU decode attention operation."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.size_per_head = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.tokens_per_block
        self.total_elems = attn_configs.kv_head_num * attn_configs.size_per_head * attn_configs.tokens_per_block
        self.softmax_scale = 1 / self.size_per_head ** 0.5
        self.max_seq_len = attn_configs.max_seq_len
        # Updated per-request in prepare(); keep False to avoid stale state before first prepare.
        self.enable_cuda_graph = False
        # Pre-allocated output tensor reused across graph replays to keep captured addresses stable.
        self._graph_output: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        if not self.enable_cuda_graph:
            self._graph_output = None
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
            graph_max_seq_len=self.max_seq_len,
        )
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        fmha_params.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_host is not None and kv_cache_block_id_host.numel() > 0:
            fmha_params.kv_cache_block_id_device = kv_cache_block_id_host.cuda()
        else:
            # During cuda_graph capture/init, the legacy kv_cache_block_id_* fields are not
            # initialized by cuda_graph_runner (only kv_cache_kernel_block_id_* is). Fall back to
            # the kernel field, lifting it to a 3-D [1, batch, blocks] view so that the forward
            # path's `[0]` group-index still works.
            fallback = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if fallback is not None and fallback.numel() > 0:
                if fallback.dim() == 2:
                    fallback = fallback.unsqueeze(0)
                fmha_params.kv_cache_block_id_device = fallback
            else:
                fmha_params.kv_cache_block_id_device = None
        return fmha_params

    def _get_output(self, q: torch.Tensor) -> torch.Tensor:
        """Return a stable pre-allocated output under cuda_graph, or a fresh one otherwise."""
        if self.enable_cuda_graph:
            if self._graph_output is None or self._graph_output.shape != q.shape:
                self._graph_output = torch.empty_like(q)
            return self._graph_output
        return torch.zeros(q.shape, dtype=q.dtype, device=q.device)

    def reshape_kvcache(self, kv_cache: KVCache):
        block_num = kv_cache.kv_cache_base.shape[1]
        key_cache = kv_cache.kv_cache_base[0].reshape(block_num, self.head_num_kv, self.tokens_per_block, self.size_per_head)
        value_cache = kv_cache.kv_cache_base[1].reshape(block_num, self.head_num_kv, self.size_per_head, self.tokens_per_block)
        return key_cache, value_cache

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        q = query.reshape(-1, self.head_num, self.size_per_head)

        key_cache, value_cache = self.reshape_kvcache(kv_cache)
        output = self._get_output(q)

        vllm_flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=fmha_params.cu_seqlens_q,
            max_seqlen_q=fmha_params.max_seqlen_q,
            seqused_k=fmha_params.seqused_k,
            max_seqlen_k=fmha_params.max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=True,
            alibi_slopes=None,
            window_size=(-1, -1),
            block_table=fmha_params.kv_cache_block_id_device[0],
            softcap=0,
            scheduler_metadata=None,
            is_prefix_cache=True,
        )

        return output

class DcuPrefillImpl(FMHAImplBase):
    """DCU prefill attention implementation."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.fmha_impl = DcuPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOp(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        # common.apply_write_cache_store(
        #     self.write_cache_store_impl, self.attn_inputs, kv_cache
        # )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class DcuDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.fmha_impl = DcuDecodeAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Replay path: update fmha_params in-place so graph addresses remain stable.
        # cuda_graph_runner only refreshes kv_cache_kernel_block_id_* during replay (the legacy
        # kv_cache_block_id_* fields are reserved for cache store outside the graph), so we must
        # source the block table from the kernel fields here. Promote 2-D -> 3-D so downstream
        # `[0]` group-index works.
        block_id_host = attn_inputs.kv_cache_kernel_block_id_host
        block_id_device = attn_inputs.kv_cache_kernel_block_id_device
        if block_id_host is not None and block_id_host.dim() == 2:
            block_id_host = block_id_host.unsqueeze(0)
        if block_id_device is not None and block_id_device.dim() == 2:
            block_id_device = block_id_device.unsqueeze(0)
        self.fmha_params.fillParams(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            block_id_host,
            block_id_device,
        )
        # rope's _compute_positions_and_slots also reads kv_cache_block_id_host[0]; expose the
        # promoted kernel host tensor through that legacy field for the duration of replay.
        attn_inputs.kv_cache_block_id_host = block_id_host
        # Update rope positions and slot_mapping in-place for graph replay.
        self.rope_kvcache_impl.update_kv_cache_offset(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
