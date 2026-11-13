"""PPU projection/codec seams for the inherited DSpark layer and commit loops."""

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu._quant import dequantize_swa_kv_rows
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.grouped_projection import project_wo_a
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _rope_tail
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm
from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import (
    dequantize_draft_swa,
    try_prepare_draft_swa,
    try_rotate_q_and_pack_swa,
    try_write_rotary_swa,
    unique_commit_swa_slots,
)
from rtp_llm.utils.model_weight import W


class SharedPPUCommitAttention(nn.Module):
    def __init__(self, views, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.wkv = views.linear(layer_id, W.v4_attn_wkv_w)
        self.kv_norm = PPUSharedRMSNorm(views.tensor(layer_id, W.v4_attn_kv_norm), views.geometry.norm_eps)
        self._cp_ctx = None
        self.runtime = None

    def set_cp_ctx(self, context):
        self._cp_ctx = context


class SharedPPUDSparkBlock(SharedPPUBlock):
    def __init__(self, views, layer_id, ops, *, enpm_group=None):
        if views.commit_only:
            nn.Module.__init__(self)
            self.layer_id = layer_id
            self.attn = SharedPPUCommitAttention(views, layer_id)
            self.engram = None
        else:
            super().__init__(views, layer_id, ops, enpm_group=enpm_group)


def _project_commit_kv(attn, main_x, quantized_input=None):
    raw = (
        attn.wkv(main_x) if quantized_input is None
        else attn.wkv.forward_quantized(*quantized_input)
    )
    return attn.kv_norm(raw)


def project_commit(attn, main_x, positions, freqs, *, quantized_input=None):
    return _rope_tail(
        _project_commit_kv(attn, main_x, quantized_input),
        freqs[positions.long().clamp_min(0)],
    )


def _swa_geometry(cache, layer_id):
    domain = cache._table_tag(layer_id, "swa_kv")
    return domain, cache.entries[domain], cache.spans[domain], cache.pool(layer_id, "swa_kv").shape[0]


def graph_commit_layer(
    model, state, layer_idx, main_x, requests, positions, ends,
    *, quantized_input=None,
):
    """The public commit loop calls this with the actual feature-row map."""
    runtime = model.shared_runtime
    cache = state.cache
    g = runtime.geometry
    private = getattr(cache, "is_private_draft_commit", False)
    # The public loop passes the same feature-row map to every draft layer.
    # This state belongs to one capture and is rebuilt on every invocation.
    if not hasattr(state, "shared"):
        state.shared = {}
    shared = state.shared
    if "commit_positions" not in shared:
        req = requests.long().clamp_min(0)
        end = ends[req]
        live = (
            (requests >= 0) & state.live[req] & (positions >= 0)
            & (positions >= end - g.window) & (positions < end)
            & (positions < cache.max_seq_len)
        )
        pos = positions.clamp(0, cache.max_seq_len - 1).long()
        raw = torch.where(live, pos, -1)
        shared["commit_positions"] = (req, pos, raw)
        if private:
            cache.bind_commit(req, live, state.live)
            shared["commit_private_rows"] = live.sum()
    req, pos, raw = shared["commit_positions"]
    attn = model.v4.layers[layer_idx].attn
    values = _project_commit_kv(attn, main_x, quantized_input)
    table = runtime.freqs_for(layer_idx)
    rope_key = ("commit_rope", id(table))
    if rope_key not in shared:
        shared[rope_key] = table[pos]
    freqs = shared[rope_key]
    if private:
        if not try_write_rotary_swa(values, freqs, cache, layer_idx, raw, journal=True):
            packed = model._shared_ops.encode_kv(_rope_tail(values, freqs), pool_tag="swa_kv")
            cache.write_rows(layer_idx, "swa_kv", raw, packed)
        state.writes[layer_idx].copy_(shared["commit_private_rows"])
        return
    geometry = _swa_geometry(cache, layer_idx)
    slot_key = ("commit_swa", geometry)
    if slot_key not in shared:
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import position_slots

        domain, entries, span, blocks = geometry
        slots = position_slots(
            raw, cache.tables[domain][req], span=span, entries=entries,
            ratio=1, full=False, pool_blocks=blocks,
        )
        shared[slot_key] = (unique_commit_swa_slots(slots), (slots >= 0).sum())
    slots, writes = shared[slot_key]
    if not try_write_rotary_swa(values, freqs, cache, layer_idx, slots):
        packed = model._shared_ops.encode_kv(_rope_tail(values, freqs), pool_tag="swa_kv")
        cache.write_rows(layer_idx, "swa_kv", slots, packed)
    state.writes[layer_idx].copy_(writes)


def _project_query_output(
    attn, g, output, freqs, shape, *, valid_mask=None, proposal_batched_wo_a=False,
):
    if valid_mask is not None:
        from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import _rotate_graph_value

        output = _rotate_graph_value(
            output, freqs, inplace=True, inverse=True, valid_mask=valid_mask,
            preserve_signed_zero=True,
        )
    else:
        output = _rope_tail(output, freqs, inverse=True)
    output = output.reshape(-1, g.o_groups, g.n_heads * g.head_dim // g.o_groups)
    def bf16_projection(output, weight):
        if valid_mask is not None and proposal_batched_wo_a:
            from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import _try_batched_wo_a

            projected = _try_batched_wo_a(output, weight, proposal_q3=True)
            if projected is not None:
                return projected
        if valid_mask is not None and weight.is_contiguous() and not (
            torch.is_grad_enabled() and (output.requires_grad or weight.requires_grad)
        ):
            projected = output.new_empty((output.shape[0], g.o_groups, g.o_lora_rank))
            for i, group_weight in enumerate(weight.view(g.o_groups, g.o_lora_rank, -1)):
                torch.mm(output[:, i], group_weight.T, out=projected[:, i])
        else:
            projected = torch.stack([
                F.linear(output[:, i].contiguous(), weight[i * g.o_lora_rank:(i + 1) * g.o_lora_rank].contiguous())
                for i in range(g.o_groups)
            ], 1)
        return projected

    projected = project_wo_a(output, attn.wo_a, bf16_projection=bf16_projection)
    return attn.wo_b(projected.flatten(1)).reshape(shape)


def forward_graph_query(model, layer_id, x, positions, prefixes, active):
    """Read committed bytes plus an ephemeral noncausal block, without writes."""
    from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import _compact_workspace_indices

    runtime = model.shared_runtime
    state, g = runtime.current, runtime.geometry
    attn = model.v4.layers[layer_id].attn
    batch, width = x.shape[:2]
    cache = state.cache
    flat = x.reshape(-1, g.hidden_size)
    table = state.freqs[layer_id]
    key = ("draft_rope", id(table))
    if key not in state.shared:
        state.shared[key] = table[positions.reshape(-1).clamp(0, cache.max_seq_len - 1)]
    freqs = state.shared[key]
    _, q, fresh, _ = attn.project_query_kv(flat, reuse_ptpc=True, fuse_qr_norm=True)
    q = q.reshape(-1, g.n_heads, g.head_dim)
    packed = try_rotate_q_and_pack_swa(q, fresh, freqs)
    if packed is None:
        q = _rope_tail(q, freqs)
        fresh = model._shared_ops.encode_kv(_rope_tail(fresh, freqs), pool_tag="swa_kv")
    else:
        q, fresh = packed

    key = ("draft_swa", _swa_geometry(cache, layer_id), g.window, width)
    if key not in state.shared:
        state.shared[key] = try_prepare_draft_swa(
            cache, layer_id, prefixes, active, window=g.window, width=width,
        )
    metadata = state.shared[key]
    if metadata is not None:
        slots, indices, lengths = metadata
        kv = dequantize_draft_swa(cache, layer_id, slots, fresh, width=width)
    else:
        context = prefixes.long()[:, None] - g.window + torch.arange(g.window, device=x.device)
        slots = cache.slots(layer_id, "swa_kv", torch.where(active[:, None], context, -1))
        committed = cache.read_rows(layer_id, "swa_kv", slots)
        bank_width = g.window + width
        bank = torch.cat((committed, fresh.reshape(batch, width, -1)), dim=1)
        kv = dequantize_swa_kv_rows(bank.flatten(0, 1)).reshape(-1, 1, g.head_dim)
        live = torch.cat((slots >= 0, active[:, None].expand(batch, width)), dim=1)
        ordinals = torch.arange(batch * bank_width, device=x.device, dtype=torch.int32).reshape(batch, bank_width)
        indices = torch.where(live, ordinals, -1)
        # Empty owners use their own dummy row to keep sparse MLA well-defined.
        indices[:, 0] = torch.where(active, indices[:, 0], ordinals[:, 0])
        indices = F.pad(indices, (0, (-bank_width) % 128), value=-1)
        indices, lengths = _compact_workspace_indices(indices[:, None])
        indices = indices.repeat_interleave(width, dim=0)
        lengths = lengths.repeat_interleave(width)
    output, _, _ = flash_mla_sparse_fwd_ppu(
        q.contiguous(), kv, indices, g.sm_scale(), attn_sink=attn.attn_sink, topk_length=lengths,
    )
    return _project_query_output(
        attn, g, output, freqs, x.shape, valid_mask=state.metadata.valid_mask,
        proposal_batched_wo_a=getattr(state, "proposal_batched_wo_a", False),
    )


def unsampled_anchor_metadata(prefixes, active, *, width, window, virtual_span):
    """PPU virtual workspace slots for gamma+1 noncausal query rows.

    The V4.1 public builder uses gamma for both query and proposal width.
    Its signature has no separate query-width argument. The existing PPU
    workspace needs this layout adapter when the sampler skips anchor logits;
    query construction, layer loop and sampling still remain public.
    """
    committed = prefixes.long().clamp(max=window)
    columns = torch.arange((window + width + 127) // 128 * 128, device=prefixes.device)
    positions = prefixes.long()[:, None] - committed[:, None] + columns[None, :]
    sizes = (committed + width).int()
    valid = active.bool()[:, None] & (columns[None, :] < sizes[:, None])
    bases = (torch.arange(prefixes.numel(), device=prefixes.device) + 1) * virtual_span
    slots = torch.where(valid, bases[:, None] + positions, -1).int()
    return slots.repeat_interleave(width, dim=0), torch.where(active.bool(), sizes, 0).int()


def forward_query(model, layer_id, x, positions, prefixes, active, indices):
    """All query KV precede attention; every query sees context+whole block.

    Proposal rows live only in call-local workspaces. They cannot overwrite
    the committed target-feature ring used by a later proposal or PD handoff.
    Public _proposal_metadata supplies the exact noncausal visibility indices.
    """
    runtime, g = model.shared_runtime, model.shared_weight_views.geometry
    state = runtime.current
    attn = model.v4.layers[layer_id].attn
    batch, width = x.shape[:2]
    if state.idle:
        return torch.zeros_like(x)
    flat = x.reshape(-1, g.hidden_size)
    freqs = runtime.freqs_for(layer_id)[positions.reshape(-1)]
    _, q, fresh, _ = attn.project_query_kv(flat, reuse_ptpc=True, fuse_qr_norm=True)
    q = _rope_tail(q.reshape(-1, g.n_heads, g.head_dim), freqs)
    fresh = _rope_tail(fresh, freqs)
    cache = state.cache
    packed = model._shared_ops.encode_kv(fresh, pool_tag="swa_kv")
    local_query = torch.arange(batch, device=x.device).repeat_interleave(width)
    cache.stage_query(layer_id, positions.reshape(-1), local_query, packed, prefixes.tolist())
    global_indices, lengths = indices
    slots = global_indices.reshape(batch * width, -1).long()
    workspace = []
    mapped = torch.full_like(slots, -1, dtype=torch.int32)
    offset = 0
    for rid in range(batch):
        if not bool(active[rid]):
            continue
        prefix = int(prefixes[rid])
        domain = torch.arange(max(0, prefix - g.window), prefix + width, device=x.device)
        packed_rows = cache.read_swa(layer_id, rid, domain)
        decoded = x.new_empty((domain.numel(), g.head_dim))
        model._shared_ops.decode_kv(packed_rows, pool_tag="swa_kv", descriptor=cache.descriptors(layer_id)["swa_kv"], out=decoded)
        # Public indices address a call-local virtual workspace. Native ring
        # addresses cannot distinguish window128+query3 generations when a
        # ring has exactly128 entries. Keep every context/query row distinct.
        physical = (rid + 1) * g.max_seq_len + domain
        sub = slots[rid * width:(rid + 1) * width]
        # Match slots against this request's latest ring generations. K140
        # bounds context; proposal scratch is addressed by absolute position.
        matches = sub[..., None] == physical[None, None, :]
        if bool(((sub >= 0) & ~matches.any(-1)).any()):
            raise RuntimeError("DSpark noncausal metadata refers to an unavailable native slot")
        local = matches.long().argmax(-1) + offset
        mapped[rid * width:(rid + 1) * width] = torch.where(sub >= 0, local, -1).int()
        workspace.append(decoded)
        offset += domain.numel()
    if not workspace:
        cache.release_swa(layer_id)
        return torch.zeros_like(x)
    kv = torch.cat(workspace).reshape(-1, 1, g.head_dim)
    output, _, _ = flash_mla_sparse_fwd_ppu(
        q.contiguous(), kv, mapped[:, None].contiguous(), g.sm_scale(),
        attn_sink=attn.attn_sink, topk_length=lengths.repeat_interleave(width).contiguous(),
    )
    cache.release_swa(layer_id)
    return _project_query_output(attn, g, output, freqs, x.shape)
