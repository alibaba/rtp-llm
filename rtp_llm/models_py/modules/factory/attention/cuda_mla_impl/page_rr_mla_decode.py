"""Decode Page-RR MLA within the existing RoPE/cache-write/projection pipeline."""

import torch

from rtp_llm.ops import (
    DecodeCPMLABackend, DecodeCPMLAFusionMode, DecodeCPMLAA2ABackend, FMHAConfig, KvCacheDataType,
)
from rtp_llm.utils.model_weight import W

from .flashinfer_mla_wrapper import MlaFlashInferImplBase
from .mla_fp8_kernels import quantize_fp8
from .mla_kv_cache_write_op import MlaKVCacheWriteOp
from .page_rr_mla_metadata import PageRRMlaDecodeMetadata
from .rope_emb_new import NewMlaRotaryEmbeddingOp
from .tokenspeed_mla_impl import (
    _get_tokenspeed_workspace,
    _tokenspeed_workspace_bytes,
    tokenspeed_mla_kernel_supported,
)


class PageRRMlaDecodeOp:
    def __init__(
        self,
        attn_configs,
        parallelism,
        weights,
        communicator,
        cache_group_id=0,
        workspace=None,
        q_replicated=False,
        backend=DecodeCPMLABackend.TOKENSPEED,
        fusion_mode=DecodeCPMLAFusionMode.AUTO,
        a2a_backend=DecodeCPMLAA2ABackend.AUTO,
    ):
        if attn_configs.is_sparse:
            raise ValueError("Decode Page-RR requires dense MLA")
        self.num_heads = attn_configs.head_num
        self.all_heads = self.num_heads * parallelism.tp_size
        # The replicated KC tensor is the layout contract for this path:
        # full-head KC means Q-B also produced all heads on this rank.
        self.q_replicated = q_replicated
        self.query_heads = self.all_heads if q_replicated else self.num_heads
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.qk_rope_head_dim = attn_configs.rope_head_dim
        self.qk_nope_head_dim = attn_configs.nope_head_dim
        self.weights = weights
        self.communicator = communicator
        self.fp8_compute = attn_configs.mla_fp8_compute
        self.q_scale = attn_configs.mla_fp8_q_scale
        self.kv_scale = attn_configs.mla_fp8_kv_scale
        self.bmm1_scale = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        ) ** -0.5 * attn_configs.softmax_extra_scale
        if self.fp8_compute:
            self.bmm1_scale *= self.q_scale * self.kv_scale
        projection = next(
            (w[W.mla_kc] for w in weights if W.mla_kc in w and W.mla_vc in w), None
        )
        if projection is None:
            raise ValueError("Decode Page-RR requires absorbed K/V projection weights")
        if projection.shape[0] != self.query_heads:
            raise ValueError(
                "Decode Page-RR KC head layout does not match the configured "
                f"query layout: got {projection.shape[0]}, expected "
                f"{self.query_heads}"
            )
        expected_cache_dtype = (
            KvCacheDataType.FP8 if self.fp8_compute else KvCacheDataType.BASE
        )
        if attn_configs.kv_cache_dtype != expected_cache_dtype:
            raise ValueError(f"Decode Page-RR requires {expected_cache_dtype} cache")
        kernel_dtype = torch.float8_e4m3fn if self.fp8_compute else projection.dtype
        self.fia2a_backend = None
        self.metadata = PageRRMlaDecodeMetadata(
            attn_configs.tokens_per_block,
            attn_configs.kernel_tokens_per_block,
            parallelism.tp_size,
            parallelism.tp_rank,
            cache_group_id,
            expand_query_table=backend == DecodeCPMLABackend.TOKENSPEED,
        )
        if backend == DecodeCPMLABackend.FIA2A:
            from .page_rr_fia2a_backend import PageRRFIA2AMLABackend

            self.fia2a_backend = PageRRFIA2AMLABackend(
                attn_configs, communicator, kernel_dtype, fusion_mode, a2a_backend
            )
        elif backend == DecodeCPMLABackend.TOKENSPEED:
            if not tokenspeed_mla_kernel_supported(
                self.all_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                attn_configs.kernel_tokens_per_block,
                1,
                kernel_dtype,
                projection.device,
            ):
                raise ValueError(
                    "Decode Page-RR requires supported TokenSpeed MLA geometry/dtype"
                )
            # Each B*Q row has its own local causal bound and runs as q1.
            required = _tokenspeed_workspace_bytes(
                projection.device, self.all_heads, self.kv_lora_rank, 1
            )
            if workspace is None:
                workspace = _get_tokenspeed_workspace(
                    projection.device, self.all_heads, self.kv_lora_rank, 1
                )
            if workspace.device != projection.device or workspace.numel() < required:
                raise ValueError("Decode Page-RR workspace has wrong device or capacity")
            self._workspace = workspace
        else:
            raise ValueError("DECODE_CP_MLA_BACKEND must be TOKENSPEED or FIA2A")

    def forward(self, q_nope, q_pe, kv_cache, layer_id):
        batch, queries = self.metadata.local_causal_lens.shape
        tokens = batch * queries
        dim = self.kv_lora_rank + self.qk_rope_head_dim
        expected_heads = self.query_heads
        if q_nope.shape[1] != expected_heads or q_pe.shape[1] != expected_heads:
            raise ValueError(
                "Decode Page-RR query head layout does not match its KC layout: "
                f"q_nope={q_nope.shape[1]}, q_pe={q_pe.shape[1]}, "
                f"expected={expected_heads}"
            )
        local_query = torch.empty(
            (self.query_heads, tokens, dim),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        torch.bmm(
            q_nope.transpose(0, 1),
            self.weights[layer_id][W.mla_kc],
            out=local_query[..., : self.kv_lora_rank],
        )
        local_query[..., self.kv_lora_rank :].copy_(q_pe.transpose(0, 1))
        if self.fp8_compute:
            local_query = quantize_fp8(
                local_query, self.q_scale, name="page_rr_absorbed_q"
            )
        gathered = (
            local_query
            if self.q_replicated
            else self.communicator.query_gather(local_query)
        )
        merged = self._forward_mla(gathered, kv_cache)
        value_weight = self.weights[layer_id][W.mla_vc]
        output = q_nope.new_empty(tokens, self.num_heads, value_weight.shape[-1])
        torch.bmm(merged, value_weight, out=output.transpose(0, 1))
        return output

    def _forward_mla(self, gathered, kv_cache):
        """Absorbed Q through MLA and communication, before V projection."""
        batch, queries = self.metadata.local_causal_lens.shape
        tokens = batch * queries
        dim = self.kv_lora_rank + self.qk_rope_head_dim
        if self.fia2a_backend is not None:
            return self.fia2a_backend.forward(
                gathered,
                kv_cache.kv_cache_base.view(-1, self.metadata.kernel_page_size, dim),
            )
        else:
            # The TS constructor loads its existing CuTe compatibility bridge.
            from .tokenspeed_mla_page_rr import tokenspeed_mla_page_rr_decode

            page_size = self.metadata.kernel_page_size
            partial, lse = tokenspeed_mla_page_rr_decode(
                gathered.transpose(0, 1).view(batch, queries, self.all_heads, dim),
                kv_cache.kv_cache_base.view(-1, page_size, dim),
                self._workspace,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.metadata.query_block_tables,
                self.metadata.local_causal_lens,
                self.metadata.query_block_tables.shape[1] * page_size,
                self.bmm1_scale,
                output_scale=self.kv_scale if self.fp8_compute else 1.0,
                normalize_empty=False,  # combine masks empty rows while packing.
            )
            merged = self.communicator.combine(
                partial.view(tokens, self.all_heads, self.kv_lora_rank),
                lse.view(tokens, self.all_heads),
                self.metadata.local_causal_lens,
            )
            return merged


class PageRRMlaDecodeImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs,
        attn_inputs,
        weights,
        cos_sin_cache,
        fmha_config=None,
        quant_config=None,
        max_seq_len=0,
        is_cuda_graph=False,
        parallelism_config=None,
        *,
        communicator=None,
    ):
        if fmha_config is None:
            fmha_config = FMHAConfig()
        # The target's first layer may be KDA; bind the first *MLA* layer's
        # physical group. Native draft weights/map select their own FULL pool.
        mla_layer = next(
            i for i, w in enumerate(weights) if W.mla_kc in w and W.mla_vc in w
        )
        if communicator is None:
            from .mla_dcp_comm import get_mla_dcp

            projection = weights[mla_layer][W.mla_kc]
            communicator = get_mla_dcp(
                attn_configs,
                projection.device,
                (
                    torch.float8_e4m3fn
                    if attn_configs.mla_fp8_compute
                    else projection.dtype
                ),
            )
        layer_map = attn_inputs.kv_cache_layer_to_group_host
        if layer_map is None or not layer_map.numel():
            # CaptureMemoryHold retains the CPU/pinned map in this field,
            # whereas eager PyWrappedModel supplies the separate host mirror.
            layer_map = attn_inputs.kv_cache_layer_to_group
        if layer_map is not None and layer_map.is_cuda:
            raise ValueError("Decode Page-RR requires a host layer-to-group map")
        group_id = (
            int(layer_map[mla_layer])
            if layer_map is not None and layer_map.numel()
            else 0
        )
        super().__init__(
            PageRRMlaDecodeOp(
                attn_configs,
                parallelism_config,
                weights,
                communicator,
                group_id,
                getattr(attn_inputs, "cuda_graph_fmha_workspace", None),
                q_replicated=bool(parallelism_config.decode_cp_q_replicated),
                backend=fmha_config.decode_cp_mla_backend,
                fusion_mode=fmha_config.decode_cp_mla_fusion_mode,
                a2a_backend=fmha_config.decode_cp_mla_a2a_backend,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache, attn_configs.rope_config.is_neox_style
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
                clear_page_on_boundary=is_cuda_graph,
                fp8_compute=attn_configs.mla_fp8_compute,
                kv_scale=attn_configs.mla_fp8_kv_scale,
            ),
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

    def create_params(self, attn_inputs):
        self.fmha_params = self.rope_params = self.fmha_impl.metadata
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs, forbid_realloc=False):
        self.attn_inputs = attn_inputs
        self.fmha_params.prepare(attn_inputs, forbid_realloc)
        op = self.fmha_impl
        if op.fia2a_backend is not None:
            op.fia2a_backend.prepare(
                self.fmha_params, op.bmm1_scale, op.kv_scale if op.fp8_compute else 1.0,
            )

    def prepare_cuda_graph(self, attn_inputs):
        self.prepare(attn_inputs, forbid_realloc=True)

    def cuda_graph_workspace_key(self):
        if self.fmha_impl.fia2a_backend is not None:
            return None
        return (
            "page_rr",
            self.fmha_impl._workspace.device.index,
            self.fmha_impl.all_heads,
            self.fmha_impl.kv_lora_rank,
        )

    @classmethod
    def support(cls, attn_configs, attn_inputs):
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_mtp_draft_update = bool(getattr(attn_inputs, "is_mtp_draft_update", False))
        return (
            attn_configs.use_mla
            and not attn_configs.is_sparse
            and (not attn_inputs.is_prefill or is_target_verify or is_mtp_draft_update)
        )

    @classmethod
    def support_page_rr_decode(cls) -> bool:
        return True
