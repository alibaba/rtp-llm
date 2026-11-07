"""PPU DSpARK runtime: shared proposal protocol, PPU weights and attention."""

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa import decode_write_swa_fp8
from rtp_llm.models_py.modules.dsv4.ppu.attention.attention_contract import validate_dspark_pool
from rtp_llm.models_py.kernels.ppu.attention.ppu_rmsnorm_rope import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.compute_qkv import decode_compute_qkv
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.ppu.utils import _v4_fp8_linear
from rtp_llm.models_py.model_desc.deepseek_v4_ppu.model import PpuDeepSeekV4Model
from rtp_llm.utils.model_weight import W


class _MainFeatureNorm(torch.nn.Module):
    """Keep the native FP32 reduction and cast after applying the norm weight."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.float(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * value).to(x.dtype)


class PpuDeepSeekV4DSparkModel(DeepSeekV4DSparkModel, PpuDeepSeekV4Model):
    def _load_extra_weights(self, weights):
        gw = weights.global_weights
        self.main_norm = _MainFeatureNorm(gw[W.v4_dspark_main_norm], float(self._v4_args.norm_eps))
        self.main_proj = _v4_fp8_linear(
            gw[W.v4_dspark_main_proj_w], gw[W.v4_dspark_main_proj_s],
            platform_provider=self._platform_provider,
        )

    def _embed_query_tokens(self, query_ids):
        return self.v4._embed(query_ids)

    def _normalize_draft_hidden(self, hidden):
        return self.v4._norm(hidden)

    def _project_committed_kv(self, attn, main_x, context_freqs):
        # Commits must preserve the same BF16/RoPE boundaries as PPU queries.
        return fused_rmsnorm_rope(
            attn._lin(attn.wkv, main_x).contiguous(), attn.kv_norm,
            context_freqs, int(attn.rope_head_dim), eps=float(attn.eps),
        )

    def _forward_dspark_attention(
        self, layer_idx, x, query_positions, prefix_lengths, active_requests,
        block_table, tokens_per_block, graph_metadata,
    ):
        batch_size, gamma, _ = x.shape
        if batch_size == 0:
            return torch.empty_like(x)
        if block_table is None:
            raise RuntimeError("DSpARK attention requires an SWA block table")
        with self._swa_cache_bound(layer_idx, block_table) as (attn, entries, pool):
            qkv = decode_compute_qkv(attn, x, query_positions.reshape(-1))
            req_ids = torch.arange(batch_size, device=x.device).repeat_interleave(gamma)
            slots = self._global_pool_slots(
                block_table, req_ids, query_positions.reshape(-1), entries, tokens_per_block,
            )
            decode_write_swa_fp8(
                kv=qkv.kv, slot_mapping=slots, swa_pool_3d=pool,
                bsz=batch_size, q_len=gamma, head_dim=int(attn.head_dim),
            )
            indices, lengths = self._build_noncausal_indices(
                prefix_lengths, active_requests, block_table, entries, tokens_per_block,
            )
            # Flatten queries as in PPU SGLang: all five rows retain the same
            # noncausal visibility, without requiring native multi-query decode.
            q = qkv.q.reshape(batch_size * gamma, 1, attn.n_heads, attn.head_dim)
            indices = indices.reshape(batch_size * gamma, 1, -1).contiguous()
            lengths = lengths.repeat_interleave(gamma)
            validate_dspark_pool(
                q=q, swa_pool_3d=pool, swa_topk_3d=indices,
                attn_sink=attn.attn_sink, topk_length=lengths,
            )
            sched = get_or_build_sched_meta(
                graph_metadata, platform_provider=self._platform_provider,
                batch_size=batch_size * gamma, q_len=1,
                num_heads=int(attn.n_heads), topk=int(indices.shape[-1]),
                extra_attn_type=None,
            )
            output = attn._get_fp8_decode_op().forward(
                q=q, kv_cache=pool, attn_sink=attn.attn_sink,
                topk_idxs=indices, sched_meta=sched, topk_length=lengths,
            ).reshape(batch_size, gamma, attn.n_heads, attn.head_dim)
            return decode_output_proj(attn, output, qkv.freqs_cis, batch_size, gamma)
