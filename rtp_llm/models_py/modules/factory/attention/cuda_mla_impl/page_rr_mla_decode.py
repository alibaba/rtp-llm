"""Page-RR Decode attention within the existing RoPE/write/publish pipeline."""

from typing import Dict, List, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather_into
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_collective import (
    merge_page_rr_attention,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaDecodeMetadata,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    _get_tokenspeed_workspace,
    _tokenspeed_workspace_bytes,
    tokenspeed_mla_kernel_supported,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


class PageRRMlaDecodeOp:
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        parallelism: ParallelismConfig,
        weights: List[Dict[str, torch.Tensor]],
        workspace: Optional[torch.Tensor] = None,
    ):
        if attn_configs.is_sparse:
            raise ValueError(
                "Page-RR Decode backend requires dense MLA; "
                "sparse indexer selection is not implemented"
            )
        self.num_heads = attn_configs.head_num
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.qk_rope_head_dim = attn_configs.rope_head_dim
        self.qk_nope_head_dim = attn_configs.nope_head_dim
        self.weights = weights
        self.replicated_heads = parallelism.get_attn_tp_size() == 1
        self.all_heads = self.num_heads * (
            1 if self.replicated_heads else parallelism.tp_size
        )
        self.bmm1_scale = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        ) ** -0.5 * attn_configs.softmax_extra_scale
        projection = next(
            (w[W.mla_kc] for w in weights if W.mla_kc in w and W.mla_vc in w),
            None,
        )
        if projection is None:
            raise ValueError("Page-RR MLA requires absorbed K/V projection weights")
        if (
            attn_configs.kv_cache_dtype != KvCacheDataType.BASE
            or not tokenspeed_mla_kernel_supported(
                self.all_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                attn_configs.kernel_tokens_per_block,
                1,
                projection.dtype,
                projection.device,
            )
        ):
            raise ValueError(
                "Page-RR MLA requires BASE cache and supported TokenSpeed geometry: "
                f"heads={self.all_heads}, latent={self.kv_lora_rank}, "
                f"rope={self.qk_rope_head_dim}, "
                f"page={attn_configs.kernel_tokens_per_block}, dtype={projection.dtype}"
            )
        self.metadata = PageRRMlaDecodeMetadata(
            attn_configs.tokens_per_block,
            attn_configs.kernel_tokens_per_block,
            parallelism.tp_size,
            parallelism.tp_rank,
        )
        # The kernel treats B*Q as independent q1 rows; its split workspace
        # bound therefore does not grow with the speculative query width.
        if workspace is None:
            workspace = _get_tokenspeed_workspace(
                projection.device, self.all_heads, self.kv_lora_rank, 1
            )
        required = _tokenspeed_workspace_bytes(
            projection.device, self.all_heads, self.kv_lora_rank, 1
        )
        if workspace.device != projection.device or workspace.numel() < required:
            raise ValueError("Page-RR MLA model workspace has wrong device or capacity")
        self._workspace = workspace

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: LayerKVCache,
        layer_id: int,
    ) -> torch.Tensor:
        # Construction validates the optional backend and initializes its
        # CuTe bridge before importing the kernel adapter.
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_page_rr import (
            tokenspeed_mla_page_rr_decode,
        )

        batch, queries = self.metadata.local_causal_lens.shape
        tokens = batch * queries
        dim = self.kv_lora_rank + self.qk_rope_head_dim
        # Gather heads directly in projection layout. The resulting transpose
        # is a view accepted by the kernel, not a token-major reorder copy.
        local_query = torch.empty(
            (self.num_heads, tokens, dim), dtype=q_nope.dtype, device=q_nope.device
        )
        torch.bmm(
            q_nope.transpose(0, 1),
            self.weights[layer_id][W.mla_kc],
            out=local_query[..., : self.kv_lora_rank],
        )
        local_query[..., self.kv_lora_rank :].copy_(q_pe.transpose(0, 1))
        query = local_query
        if not self.replicated_heads:
            query = torch.empty(
                (self.all_heads, tokens, dim), dtype=q_nope.dtype, device=q_nope.device
            )
            all_gather_into(local_query, query, Group.TP)
        page_size = self.metadata.kernel_page_size
        partial, lse = tokenspeed_mla_page_rr_decode(
            query.transpose(0, 1).view(batch, queries, self.all_heads, dim),
            kv_cache.kv_cache_base.view(-1, page_size, dim),
            self._workspace,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.metadata.query_block_tables,
            self.metadata.local_causal_lens,
            self.metadata.block_tables.shape[1] * page_size,
            self.bmm1_scale,
        )
        merged = merge_page_rr_attention(
            partial.view(tokens, self.all_heads, self.kv_lora_rank),
            lse.view(tokens, self.all_heads),
            replicated_heads=self.replicated_heads,
        ).to(q_nope.dtype)
        value_weight = self.weights[layer_id][W.mla_vc]
        output = torch.empty(
            (tokens, self.num_heads, value_weight.shape[-1]),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        torch.bmm(merged, value_weight, out=output.transpose(0, 1))
        return output


class PageRRMlaDecodeImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
        cache_group_id: int = 0,
    ):
        self.cache_group_id = cache_group_id
        super().__init__(
            PageRRMlaDecodeOp(
                attn_configs,
                parallelism_config,
                weights,
                getattr(attn_inputs, "cuda_graph_fmha_workspace", None),
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache, attn_configs.rope_config.is_neox_style
            ),
            MlaKVCacheWriteOp(attn_configs.kv_cache_dtype, is_cuda_graph),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
            warmup_flashinfer=False,
        )

    def create_params(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = self.rope_params = self.fmha_impl.metadata
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        self.attn_inputs = attn_inputs
        self.fmha_params.prepare(attn_inputs, forbid_realloc, self.cache_group_id)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs, forbid_realloc=True)

    def cuda_graph_workspace_key(self) -> tuple:
        return (
            "page_rr",
            self.fmha_impl._workspace.device.index,
            self.fmha_impl.all_heads,
            self.fmha_impl.kv_lora_rank,
        )
