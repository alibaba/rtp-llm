"""V4.1 DSpark proposal and dense TAIL over framework-owned compact SWA.

Proposal KV is temporary and noncausal within the five-row query block. Only
target-feature TAIL commits write persistent draft state. The executor owns
acceptance and the final joint target/draft boundary publication.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.attention import attention_rope
from rtp_llm.models_py.modules.dsv41.block import V41Block
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact, write_compact
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.flashmla import PlanarPages
from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear
from rtp_llm.models_py.modules.dsv41.math import (
    grouped_wo_a,
    hc_pre,
    identity_pre_mix,
    rms_norm,
)
from rtp_llm.models_py.modules.dsv41.moe import V41MoE
from rtp_llm.utils.model_weight import W


class V41DraftAttentionBuffers:
    """One raw-byte proposal pack reused by the three sequential draft stages."""

    def __init__(self, batch_size, device):
        # FlashMLA sparse page size is a multiple of 64. The physical slot
        # multiplier must include padding after history128 + query5.
        self.entries = 192
        stride = ((self.entries * 528 + 511) // 512) * 512
        self.pages = PlanarPages(
            torch.zeros((batch_size + 1, stride), dtype=torch.uint8, device=device),
            CacheRegion.SWA,
            self.entries,
        )
        self.pages.validate(device)
        self.encoded = torch.zeros(
            (batch_size * 5, 528), dtype=torch.uint8, device=device
        )
        self.history_offsets = torch.arange(128, device=device, dtype=torch.int64)
        self.columns = torch.arange(self.entries, device=device, dtype=torch.int64)[
            None, :
        ]
        self.physical = (
            torch.arange(batch_size, device=device, dtype=torch.int64)[:, None] + 1
        ) * self.entries + self.columns
        self.indices = torch.full(
            (batch_size * 5, 1, self.entries), -1, dtype=torch.int32, device=device
        )
        self.lengths = torch.zeros(batch_size * 5, dtype=torch.int32, device=device)


def draft_flashmla_attention(query, query_kv, context, stage, sinks):
    """Use the fixed native reader with a bounded raw-byte pack per request."""
    from flash_mla.flash_mla_interface import FlashMLASchedMeta, flash_mla_with_kvcache

    batch, width = context.batch_size, 5
    if (
        query.shape != (batch * width, 64, 512)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or query_kv.shape != (batch * width, 512)
        or query_kv.dtype != torch.bfloat16
        or query_kv.device != query.device
        or sinks.shape != (64,)
        or sinks.dtype != torch.float32
        or sinks.device != query.device
        or not sinks.is_contiguous()
    ):
        raise ValueError("draft FlashMLA requires BF16 Bx5 queries/KV and FP32 sinks")
    buffers = context.proposal_buffers
    swa = context.swa[stage]
    entries = swa.pages.entries_per_page
    starts = context.starts
    active = context.active
    begin = torch.maximum((starts - 128).clamp_min(0), context.floors[stage])
    lengths = (starts - begin).to(torch.int64)
    page = swa.page_ids.to(torch.int64)
    metadata_ok = (
        (starts > 0)
        & (context.counts > 0)
        & (context.counts <= width)
        & (lengths >= 0)
        & (lengths <= 128)
        & (swa.valid_starts <= begin)
        & (swa.valid_ends == starts)
        & (page > 0)
        & (page < swa.pages.data.shape[0])
    )
    usable = active & metadata_ok
    offsets = buffers.history_offsets
    positions = begin[:, None] + offsets[None, :]
    raw = swa.pages.data[:, : entries * 528].view(-1, entries, 528)
    history = raw[page.clamp(0, raw.shape[0] - 1)[:, None], positions % entries]
    history.masked_fill_(
        ~(usable[:, None] & (offsets[None, :] < lengths[:, None]))[:, :, None], 0
    )
    encoded = encode_compact(
        query_kv.contiguous(),
        CacheRegion.SWA,
        output=buffers.encoded,
        status=context.writer_status[stage],
    )
    combined = torch.cat((history, encoded.output.view(batch, width, 528)), dim=1)

    # Native sparse lengths count a packed prefix. Put the query immediately
    # after actual history, including prefixes shorter than the full window.
    slots, columns = buffers.entries, buffers.columns
    source = torch.where(
        columns < lengths[:, None], columns, 128 + columns - lengths[:, None]
    )
    packed_rows = combined.gather(
        1, source.clamp(0, 132)[:, :, None].expand(-1, -1, 528)
    )
    valid = usable[:, None] & (columns < lengths[:, None] + context.counts[:, None])
    packed_rows.masked_fill_(~valid[:, :, None], 0)
    packed = buffers.pages
    packed.data[1:, : slots * 512].view(batch, slots, 512).copy_(
        packed_rows[:, :, :512]
    )
    packed.data[1:, slots * 512 : slots * 528].view(batch, slots, 16).copy_(
        packed_rows[:, :, 512:]
    )
    indices = torch.where(valid, buffers.physical, -1).to(torch.int32)
    buffers.indices.copy_(indices.repeat_interleave(width, 0).unsqueeze(1))
    buffers.lengths.copy_(
        torch.where(usable, lengths + context.counts, 0)
        .to(torch.int32)
        .repeat_interleave(width)
    )
    output, _ = flash_mla_with_kvcache(
        query.view(batch * width, 1, 64, 512),
        packed.kernel_view(),
        None,
        None,
        512,
        FlashMLASchedMeta(),
        softmax_scale=1.0 / math.sqrt(512),
        causal=False,
        is_fp8_kvcache=True,
        indices=buffers.indices,
        attn_sink=sinks,
        topk_length=buffers.lengths,
    )
    context.reader_status[stage].copy_((active & ~metadata_ok).to(torch.int32))
    valid_queries = usable.repeat_interleave(width) & context.row_active
    return output.view(batch * width, 64, 512).masked_fill(
        ~valid_queries[:, None, None], 0
    )


class V41DraftAttention(nn.Module):
    def __init__(self, stage, weights):
        super().__init__()
        if stage not in (0, 1, 2):
            raise ValueError("draft attention must belong to one of three stages")
        self.stage = stage
        for name in ("wq_a", "wq_b", "wkv", "wo_b"):
            setattr(
                self,
                name,
                V41Block32Linear(
                    weights[f"attn.{name}.weight"], weights[f"attn.{name}.scale"]
                ),
            )
        for name, key in (
            ("q_norm", "q_norm.weight"),
            ("kv_norm", "kv_norm.weight"),
            ("sinks", "attn_sink"),
        ):
            self.register_buffer(name, weights["attn." + key])
        self.register_buffer("wo_a", weights["attn.wo_a.weight"].view(8, 1024, 4096))

    @torch.inference_mode()
    def forward(self, hidden, context):
        qr = rms_norm(self.wq_a(hidden), self.q_norm)
        query = attention_rope(
            self.wq_b(qr).view(-1, 64, 512), context.positions, global_branch=False
        )
        kv = attention_rope(
            rms_norm(self.wkv(hidden), self.kv_norm),
            context.positions,
            global_branch=False,
        )
        output = draft_flashmla_attention(
            query.contiguous(), kv, context, self.stage, self.sinks
        )
        output = attention_rope(
            output, context.positions, global_branch=False, inverse=True
        )
        return self.wo_b(
            grouped_wo_a(output.reshape(-1, 8, 4096), self.wo_a).flatten(1)
        )


class V41DraftModel(nn.Module):
    def __init__(self, config, weights, *, ep_size, ep_rank, max_tokens_per_rank):
        super().__init__()
        self.register_buffer("embedding", weights.global_weights[W.embedding])
        self.register_buffer("norm", weights.global_weights[W.final_ln_gamma])
        self.committer = V41PrefillDraftCommit.from_model_weights(config, weights)
        blocks = []
        block_names = (
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        )
        for stage, installed in enumerate(weights.weights):
            local = {
                name.removeprefix("v41."): value for name, value in installed.items()
            }
            blocks.append(
                V41Block(
                    V41DraftAttention(stage, local),
                    V41MoE.from_weights(
                        config,
                        stage,
                        local,
                        ep_size=ep_size,
                        ep_rank=ep_rank,
                        max_tokens_per_rank=max_tokens_per_rank,
                        draft=True,
                    ),
                    {name: local[name] for name in block_names},
                )
            )
        if len(blocks) != 3:
            raise ValueError("V4.1 DSpark must bind all three draft stages")
        self.blocks = nn.ModuleList(blocks)

    @torch.inference_mode()
    def propose(self, context):
        context.begin_forward()
        ids = context.token_ids.masked_fill(~context.row_active, 0)
        hidden = F.embedding(ids, self.embedding).unsqueeze(1).repeat(1, 4, 1)
        hidden.masked_fill_(~context.row_active[:, None, None], 0)
        pre_mix = identity_pre_mix(hidden)
        text_rows = torch.zeros_like(context.row_active)
        for block in self.blocks:
            hidden, pre_mix = block(hidden, pre_mix, context, text_rows)
            hidden.masked_fill_(~context.row_active[:, None, None], 0)
        return rms_norm(hc_pre(hidden, pre_mix), self.norm).masked_fill(
            ~context.row_active[:, None], 0
        )

    @torch.inference_mode()
    def commit(self, context):
        context.begin_forward()
        aux = context.aux.masked_fill(~context.row_active[:, None], 0)
        main = rms_norm(self.committer.main_projection(aux), self.committer.main_norm)
        for stage in range(3):
            swa = context.swa[stage]
            values = rms_norm(
                self.committer.stage_projections[stage](main),
                getattr(self.committer, f"stage_norm_{stage}"),
            )
            values = attention_rope(values, context.positions, global_branch=False)
            slots = (
                swa.page_ids.to(torch.int64)[:, None] * swa.pages.entries_per_page
                + context.positions.view(context.batch_size, 6)
                % swa.pages.entries_per_page
            )
            slots = torch.where(
                context.row_active.view(context.batch_size, 6), slots, -1
            )
            write_compact(
                values.contiguous(),
                swa.pages,
                slots.flatten(),
                status=context.writer_status[stage],
            )
            end = context.starts + context.counts
            swa.valid_starts.copy_(
                torch.where(
                    context.active,
                    torch.maximum(
                        swa.valid_starts,
                        (end - swa.pages.entries_per_page).clamp_min(0),
                    ),
                    swa.valid_starts,
                )
            )
            swa.valid_ends.copy_(torch.where(context.active, end, swa.valid_ends))
        return torch.zeros(
            (context.batch_size * 6, 5120), dtype=torch.bfloat16, device=context.device
        )
