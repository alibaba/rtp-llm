"""PPU full-prefill graph for one request plus the runner's scratch slot."""

import logging
from typing import Optional

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
    release_py_flashinfer_workspace_buffer,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    PrefillCudaGraphCapability,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops


class PpuFlashinferFullPrefillGraphImpl(FMHAImplBase):
    """Fixed token capacity, with [real tokens, scratch tokens] as two segments.

    CudaGraphRunner waits for the previous forward before prepare_cuda_graph.
    Standalone callers must do the same: both the FlashInfer planner and the
    cache metadata planner reuse pinned host staging memory.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        if not self.support(
            attn_configs, attn_inputs
        ) or not self.support_parallelism_config(parallelism_config):
            raise ValueError(
                "PPU full prefill graph requires BF16, TP1, no prefix "
                "and one request plus scratch"
            )
        self.config = attn_configs
        lengths, table = self._metadata(attn_inputs)
        if lengths[0] != 0:
            raise ValueError(
                "PPU full prefill graph must capture the runner's "
                "[0, capacity] scratch layout"
            )
        self.token_capacity = sum(lengths)
        self.table_shape = tuple(table.shape)
        self.workspace = get_py_flashinfer_workspace_buffer()
        self.qo_indptr = torch.empty(3, dtype=torch.int32, device="cuda")
        self.kv_indptr = torch.empty_like(self.qo_indptr)
        self.indptr_host = torch.empty(3, dtype=torch.int32, pin_memory=True)
        self.input_lengths_host = torch.empty(2, dtype=torch.int32, pin_memory=True)
        self.prefix_lengths_host = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self.empty_sequence_lengths = torch.empty(0, dtype=torch.int32)
        self.output = torch.empty(
            self.token_capacity,
            attn_configs.head_num,
            attn_configs.size_per_head,
            dtype=torch.bfloat16,
            device=self.workspace.device,
        )
        # FA2 graph planning fixes launch scalars using total rows and batch
        # capacity; per-request tiles/lengths are refreshed in its workspace.
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self.qo_indptr,
            kv_indptr_buf=self.kv_indptr,
            backend="fa2",
        )
        self.params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope = MhaRotaryEmbeddingOp(attn_configs)
        self.rope.set_params(self.params)
        self.cache_writer = KVCacheWriteOp(
            attn_configs.kv_head_num,
            attn_configs.size_per_head,
            attn_configs.kernel_tokens_per_block,
        )
        self.cache_writer.set_params(self.params)

        # [0, Tg] capture can need one fewer page than [T, Tg-T] replay.
        # Reserve the worst split before capture, including all token metadata.
        self.input_lengths_host[0] = 1
        self.input_lengths_host[1] = self.token_capacity - 1
        self._fill_cache_params(table, forbid_realloc=False)
        # fill_params enqueues H2D from its own pinned staging buffer. Finish
        # that one-time reserve copy before filling the actual capture layout.
        torch.cuda.current_stream().synchronize()
        self._plan_info = None
        self.prepare_cuda_graph(attn_inputs)
        logging.info(
            "PPU full prefill graph: backend=fa2, real_requests=1, "
            "slots=2, token_capacity=%d",
            self.token_capacity,
        )

    def __del__(self) -> None:
        if hasattr(self, "workspace"):
            release_py_flashinfer_workspace_buffer(self.workspace)

    @classmethod
    def support(cls, config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        # This flag selects the graph-only backend at construction. The C++
        # runner's replay metadata retains its default False: the factory sets
        # the flag on a Python copy, not on the runner-owned inputs.
        return inputs.is_cuda_graph and cls._supports_metadata(config, inputs)

    @staticmethod
    def _supports_metadata(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        lengths = inputs.input_lengths
        prefix = inputs.prefix_lengths
        positions = inputs.combo_position_ids
        return (
            get_device_type() == DeviceType.Ppu
            and inputs.is_prefill
            and not inputs.is_target_verify
            and not config.use_mla
            and not config.is_sparse
            and config.is_causal
            and not config.use_logn_attn
            and config.need_rope_kv_cache
            and config.dtype == torch.bfloat16
            and config.kv_cache_dtype == KvCacheDataType.BASE
            and config.rope_config.style == RopeStyle.Base
            and config.rope_config.dim == config.size_per_head
            and config.q_scaling == 1.0
            and config.softmax_extra_scale == 1.0
            and config.size_per_head == 128
            and config.head_num > 0
            and config.kv_head_num > 0
            and config.head_num % config.kv_head_num == 0
            and config.kernel_tokens_per_block == 64
            and lengths is not None
            and lengths.device.type == "cpu"
            and lengths.dtype == torch.int32
            and lengths.shape == (2,)
            and (prefix is None or (not prefix.is_cuda and not prefix.any().item()))
            and (positions is None or positions.numel() == 0)
            and inputs.cache_store_inputs is None
        )

    @classmethod
    def support_parallelism_config(cls, config: Optional[ParallelismConfig]) -> bool:
        return config is None or (
            config.tp_size == 1
            and config.get_attn_tp_size() == 1
            and not config.prefill_cp_config.is_enabled()
        )

    def support_cuda_graph(self) -> bool:
        return True

    def prefill_cuda_graph_capability(self) -> PrefillCudaGraphCapability:
        return PrefillCudaGraphCapability.FULL_NO_PREFIX

    def _metadata(self, inputs: PyAttentionInputs):
        # The existing graph instance determines execution mode. Revalidate
        # every semantic constraint without requiring the factory's marker.
        if not self._supports_metadata(self.config, inputs):
            raise ValueError(
                "unsupported PPU full prefill graph metadata or configuration"
            )
        lengths = inputs.input_lengths.tolist()
        capacity = sum(lengths)
        if min(lengths) < 0 or capacity <= 0 or capacity > self.config.max_seq_len:
            raise ValueError("PPU full prefill graph has invalid token capacity")
        table = inputs.kv_cache_kernel_block_id
        if (
            table is None
            or table.device.type != "cpu"
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.shape[0] != 2
            or not table.is_contiguous()
            or table.shape[1] * self.config.kernel_tokens_per_block < capacity
        ):
            raise ValueError(
                "PPU full prefill graph requires a fixed CPU int32 page table "
                "for two slots"
            )
        # Host lengths are authoritative, like the full-prefill runner. Check
        # its host cumulative lengths too, without reading device metadata.
        cu = inputs.cu_seqlens
        expected = [0, lengths[0], capacity]
        if (
            cu is None
            or cu.is_cuda
            or cu.dtype != torch.int32
            or cu.tolist() != expected
        ):
            raise ValueError(
                "PPU full prefill graph cu_seqlens disagrees with input_lengths"
            )
        return lengths, table

    def _fill_cache_params(self, table: torch.Tensor, *, forbid_realloc: bool) -> None:
        self.params.fill_params(
            self.prefix_lengths_host,
            self.empty_sequence_lengths,
            self.input_lengths_host,
            table,
            self.config.kernel_tokens_per_block,
            forbid_realloc,
        )

    def prepare_cuda_graph(self, inputs: PyAttentionInputs) -> None:
        lengths, table = self._metadata(inputs)
        if (
            sum(lengths) != self.token_capacity
            or tuple(table.shape) != self.table_shape
        ):
            raise ValueError(
                "PPU full prefill graph cannot change captured token or page-table capacity"
            )
        self.input_lengths_host.copy_(inputs.input_lengths)
        self.indptr_host[0] = 0
        self.indptr_host[1] = lengths[0]
        self.indptr_host[2] = self.token_capacity
        self._fill_cache_params(table, forbid_realloc=True)
        self.wrapper.plan(
            self.indptr_host,
            self.indptr_host,
            self.config.head_num,
            self.config.kv_head_num,
            self.config.size_per_head,
            self.config.size_per_head,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
        )
        # Replanning updates workspace contents, not the launch scalars already
        # captured in the graph. Reject an incompatible dependency plan change.
        plan_info = tuple(self.wrapper._plan_info)
        if self._plan_info is not None and plan_info != self._plan_info:
            raise RuntimeError(
                "PPU full prefill graph FlashInfer plan changed captured launch parameters"
            )
        self._plan_info = plan_info

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if qkv.shape[0] != self.token_capacity or qkv.dtype != torch.bfloat16:
            raise ValueError("PPU full prefill graph requires fixed-capacity BF16 QKV")
        query, key, value = self.rope.forward(qkv)
        self.cache_writer.forward(key, value, kv_cache)
        return self.wrapper.run(query, key, value, out=self.output)
