"""PPU projection/codec seams for the inherited DSpark layer and commit loops."""

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu._quant import dequantize_swa_kv_rows
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _rope_tail
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm
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


def project_commit(attn, main_x, positions, freqs):
    return _rope_tail(attn.kv_norm(attn.wkv(main_x)), freqs[positions.long().clamp_min(0)])


def graph_commit_layer(model, state, layer_idx, main_x, requests, positions, ends):
    """The public commit loop calls this with the actual feature-row map."""
    runtime = model.shared_runtime
    cache = state.cache
    g = runtime.geometry
    req = requests.long().clamp_min(0)
    end = ends[req]
    live = (
        (requests >= 0) & state.live[req] & (positions >= 0)
        & (positions >= end - g.window) & (positions < end)
        & (positions < cache.max_seq_len)
    )
    pos = positions.clamp(0, cache.max_seq_len - 1)
    attn = model.v4.layers[layer_idx].attn
    values = project_commit(attn, main_x, pos, runtime.freqs_for(layer_idx))
    packed = model._shared_ops.encode_kv(values, pool_tag="swa_kv")
    from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import position_slots

    tag = cache._table_tag(layer_idx, "swa_kv")
    slots = position_slots(
        torch.where(live, pos, -1), cache.tables[tag][req],
        span=cache.spans[tag], entries=cache.entries[tag], ratio=1, full=False,
        pool_blocks=cache.pool(layer_idx, "swa_kv").shape[0],
    )
    cache.write_rows(layer_idx, "swa_kv", slots, packed)
    state.writes[layer_idx].copy_((slots >= 0).sum())


def _project_query_output(attn, g, output, freqs, shape):
    output = _rope_tail(output, freqs, inverse=True).reshape(
        -1, g.o_groups, g.n_heads * g.head_dim // g.o_groups
    )
    projected = torch.stack([
        F.linear(output[:, i].contiguous(), attn.wo_a.weight[i * g.o_lora_rank:(i + 1) * g.o_lora_rank].contiguous())
        for i in range(g.o_groups)
    ], 1)
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
    freqs = state.freqs[layer_id][positions.reshape(-1).clamp(0, cache.max_seq_len - 1)]
    q = _rope_tail(attn.wq_b(attn.query_latent(flat)).reshape(-1, g.n_heads, g.head_dim), freqs)
    fresh = _rope_tail(attn.kv_norm(attn.wkv(flat)), freqs)
    fresh = model._shared_ops.encode_kv(fresh, pool_tag="swa_kv").reshape(batch, width, -1)

    context = prefixes.long()[:, None] - g.window + torch.arange(g.window, device=x.device)
    slots = cache.slots(layer_id, "swa_kv", torch.where(active[:, None], context, -1))
    committed = cache.read_rows(layer_id, "swa_kv", slots)
    bank_width = g.window + width
    bank = torch.cat((committed, fresh), dim=1)
    kv = dequantize_swa_kv_rows(bank.flatten(0, 1)).reshape(-1, 1, g.head_dim)
    live = torch.cat((slots >= 0, active[:, None].expand(batch, width)), dim=1)
    ordinals = torch.arange(batch * bank_width, device=x.device, dtype=torch.int32).reshape(batch, bank_width)
    indices = torch.where(live, ordinals, -1)
    # Empty owners use their own dummy row to keep sparse MLA well-defined.
    indices[:, 0] = torch.where(active, indices[:, 0], ordinals[:, 0])
    # The PPU sparse MLA contract uses a 128-aligned candidate width.
    indices = F.pad(indices, (0, (-bank_width) % 128), value=-1)
    indices, lengths = _compact_workspace_indices(indices[:, None])
    output, _, _ = flash_mla_sparse_fwd_ppu(
        q.contiguous(), kv, indices.repeat_interleave(width, dim=0), g.sm_scale(),
        attn_sink=attn.attn_sink, topk_length=lengths.repeat_interleave(width),
    )
    output = torch.where(state.metadata.valid_mask[:, None, None], output, 0)
    return _project_query_output(attn, g, output, freqs, x.shape)


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
    q = _rope_tail(attn.wq_b(attn.query_latent(flat)).reshape(-1, g.n_heads, g.head_dim), freqs)
    fresh = _rope_tail(attn.kv_norm(attn.wkv(flat)), freqs)
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
