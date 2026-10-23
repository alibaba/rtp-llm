"""Full-head GLM53 KDA on the model's existing zigzag token shard.

Projection weights are BF16, convolution weights remain FP32, and published
states retain the native TP head partition. No per-layer token transpose.
"""
import os
from functools import lru_cache

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
from rtp_llm.models_py.distributed.glm53_collective_gemm import canonical_bf16_weight
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.fla.block import load_initial_state_from_block_map
from rtp_llm.models_py.triton_kernels.kimi_kda.checkpoint import (
    prepare_kda_recurrent_checkpoint_metadata,
    store_kda_recurrent_checkpoints,
)

from .kcp_bm_pipeline import scope
from .kcp_recurrence import checkpoint_prefill, device_plan
from .kcp_checkpoint import exchange_conv_states, publish_conv_states
from .kcp_conv_halo import halo_plan, exchange_halo


def gather(x, group):
    x = x.contiguous()
    result = x.new_empty((dist.get_world_size(group), *x.shape))
    dist.all_gather_into_tensor(result.flatten(0, 1), x, group=group)
    return result


def gather_cat(x, dim, group):
    return torch.cat(tuple(gather(x, group).unbind(0)), dim=dim).contiguous()


@triton.jit
def _initial_conv(X, HALO, INITIAL, KIND, INDEX, OUT,
                  WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    kind = tl.load(KIND + row)
    index = tl.load(INDEX + row).to(tl.int64)
    value = tl.load(X + index * WIDTH + c, (c < WIDTH) & (kind == 0), 0)
    value += tl.load(HALO + index * WIDTH + c, (c < WIDTH) & (kind == 1), 0)
    value += tl.load(INITIAL + index * WIDTH + c, (c < WIDTH) & (kind == 2), 0)
    tl.store(OUT + row * WIDTH + c, value, c < WIDTH)


@lru_cache(maxsize=128)
def conv_plan(cu_values, world, span, rank, device_index, stream, history):
    plan, ids, cpu_cu, cu, *_ = device_plan(
        cu_values, world, span, rank, device_index, stream
    )
    kinds, indices, tail_indices, halo_rows = halo_plan(cu_values, world, span, rank, history)
    device = torch.device('cuda', device_index)
    n = len(ids)
    # Synthetic per-segment cache: a read-only initial page and a separate
    # final page. Reuse the GLM FP32-weight convolution kernel unchanged.
    block_map = torch.stack((torch.arange(n, device=device),
                             torch.arange(n, 2 * n, device=device)), dim=1).int()
    prefix = torch.full((n,), span, dtype=torch.int32, device=device)
    tails = torch.tensor(tail_indices, dtype=torch.int64, device=device)
    return (torch.tensor(kinds, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int64, device=device),
            prepare_causal_conv1d_metadata(cu, device), block_map, prefix, tails, halo_rows)


class Glm53ZigzagKDA:
    def __init__(self, module):
        self.module = module
        self.group = get_process_group(Group.TP)
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        p = module.prefill_kda
        if p.gate_lower_bound != -5.0 or p.head_k_dim != 128:
            raise ValueError('GLM53 zigzag KCP requires D128 and lower_bound=-5')
        if p.conv_weights.dtype != torch.float32:
            raise ValueError('GLM53 convolution weights must retain FP32 storage')
        parts = [canonical_bf16_weight(x) for x in module.input_projections()]
        if any(x is None for x in parts):
            raise ValueError('GLM53 KCP requires native unquantized BF16 KDA weights')
        local_width = p.local_num_v_heads * p.head_v_dim
        self.local_width = local_width
        self.width = local_width * self.world
        self.heads = p.local_num_v_heads * self.world
        self.dim = p.head_v_dim
        self.history = p.linear_conv_kernel_dim - 1
        qkv = gather(parts[0], self.group).view(self.world, parts[0].shape[0], 3, local_width)
        qkv = qkv.permute(1, 2, 0, 3).contiguous().flatten(1)
        beta = gather_cat(parts[1], 1, self.group)
        self.input_widths = (3 * self.width, self.heads, parts[2].shape[1], parts[3].shape[1])
        self.input_weight = torch.cat((qkv, beta, parts[2], parts[3]), dim=1).contiguous()
        self.forget_weight = gather_cat(canonical_bf16_weight(module.f_b_proj), 1, self.group)
        self.gate_weight = gather_cat(canonical_bf16_weight(module.g_b_proj), 1, self.group)
        self.output_weight = gather_cat(canonical_bf16_weight(module.out_proj), 0, self.group)
        conv = gather(p.conv_weights, self.group)
        self.conv_weight = conv.view(self.world, 3, local_width, -1).transpose(0, 1).flatten(0, 2).contiguous()
        self.a_log = gather_cat(p.alog.float().flatten(), 0, self.group)
        self.dt_bias = gather_cat(p.dt_bias.float().flatten(), 0, self.group)

    @torch.inference_mode()
    def __call__(self, x, inputs, kv_cache, meta):
        m, p, group = self.module, self.module.prefill_kda, self.group
        if kv_cache is None:
            raise ValueError('Full-model GLM53 KCP requires the native paged cache')
        span = kv_cache.seq_size_per_block
        cu_cpu, lengths_cpu, prefixes_cpu = p._host_lengths(inputs)
        if any(int(v) % span for v in prefixes_cpu.tolist()):
            raise ValueError('GLM53 KCP prefix must align with the native linear page')
        plan, ids, cpu_cu, cu, _, _, _, checkpoint_index = device_plan(
            tuple(cu_cpu.tolist()), self.world, span, self.rank, x.device.index,
            int(torch.cuda.current_stream(x.device).cuda_stream))
        if x.shape[0] != plan['local_tokens']:
            raise ValueError('KDA input does not match the shared MLA zigzag layout')
        with scope('glm53_kcp/projection'):
            projected = x @ self.input_weight
            mixed, beta, forget_low, gate_low = torch.split(projected, self.input_widths, dim=1)
            mixed = mixed.contiguous()
            raw_gate, output_gate = forget_low @ self.forget_weight, gate_low @ self.gate_weight
        cache_tensor = kv_cache.kv_cache_base.reshape(kv_cache.kv_cache_base.shape[0], -1)
        conv_cache, ssm_cache = p._get_conv_states(cache_tensor), p._get_ssm_states(cache_tensor)
        block_map = inputs.kv_cache_kernel_block_id_device
        zero_initial = not any(prefixes_cpu.tolist())
        initial = x.new_zeros((len(lengths_cpu), p.local_num_v_heads, self.dim, self.dim), dtype=torch.float32)
        if not zero_initial:
            load_initial_state_from_block_map(inputs.prefix_lengths, block_map, ssm_cache, initial, span)
            rows = torch.arange(len(lengths_cpu), device=x.device)
            blocks = block_map[rows, ((inputs.prefix_lengths.long() - 1) // span).clamp_min(0)].long()
            local_conv = torch.where((inputs.prefix_lengths > 0)[:, None, None],
                                     conv_cache.index_select(0, blocks.clamp_min(0)), 0)
            full_conv = gather(local_conv, group).view(self.world, len(lengths_cpu), self.history, 3, self.local_width)
            full_conv = full_conv.permute(1, 2, 3, 0, 4).contiguous().flatten(2)
        else:
            full_conv = mixed.new_zeros((len(lengths_cpu), self.history, 3 * self.width))
        kind, index, conv_meta, dummy_map, dummy_prefix, tail_index, halo_rows = conv_plan(
            tuple(cu_cpu.tolist()), self.world, span, self.rank, x.device.index,
            int(torch.cuda.current_stream(x.device).cuda_stream), self.history)
        with scope('glm53_kcp/conv_halo'):
            if os.environ.get('GLM53_KCP_DIRECTED_HALO', '0') == '1':
                from .kcp_conv_halo_directed import directed_plan, exchange_halo_directed
                directed = directed_plan(tuple(cu_cpu.tolist()), self.world, span,
                    self.rank, self.history, x.device.index,
                    int(torch.cuda.current_stream(x.device).cuda_stream))
                tails, index = exchange_halo_directed(mixed, directed, group)
            else:
                tails = exchange_halo(mixed, tail_index, halo_rows, group)
            synthetic_cache = mixed.new_empty((2 * len(ids), self.history, 3 * self.width))
            _initial_conv[(len(ids) * self.history, triton.cdiv(3 * self.width, 256))](
                mixed, tails.flatten(0, 1), full_conv, kind, index, synthetic_cache,
                WIDTH=3 * self.width, BLOCK=256)
        with scope('glm53_kcp/conv_fp32_weight'):
            q, k, v = causal_conv1d_fn(
                mixed.T, self.conv_weight, bias=None,
                conv_states=synthetic_cache.transpose(1, 2),
                query_start_loc=cu, block_map=dummy_map, seq_size_per_block=span,
                prefix_lengths=dummy_prefix, metadata=conv_meta, output_groups=3).unbind(0)
        if meta.kda_checkpoint_meta is None or meta.kda_checkpoint_page_size != span:
            meta.kda_checkpoint_meta = prepare_kda_recurrent_checkpoint_metadata(lengths_cpu, prefixes_cpu, span, x.device)
            checkpoint_shape = (1, meta.kda_checkpoint_meta.total_checkpoints,
                                p.local_num_v_heads, self.dim, self.dim)
            if os.environ.get('GLM53_KCP_PEER_CHECKPOINT', '0') == '1':
                from .kcp_peer_checkpoint import output_buffer
                meta.kda_checkpoint_states = output_buffer(checkpoint_shape, group, x.device)[0]
            else:
                meta.kda_checkpoint_states = initial.new_empty(checkpoint_shape)
            meta.kda_checkpoint_page_size = span
        shape = (1, x.shape[0], self.heads, self.dim)
        publish_masks = None
        if os.environ.get('GLM53_KCP_FILTER_CHECKPOINT', '0') == '1':
            if os.environ.get('GLM53_KCP_PEER_CHECKPOINT', '0') != '1':
                raise ValueError('Filtered native publication requires peer checkpoint transport')
            from .kcp_masked_checkpoint import publication_masks
            with scope('kcp/native_publication_mask'):
                publish_masks = publication_masks(meta.kda_checkpoint_meta,
                    block_map, ssm_cache.shape[0], group)
        output = checkpoint_prefill(
            q.reshape(shape), k.reshape(shape), v.reshape(shape), raw_gate.reshape(shape),
            beta.reshape(1, x.shape[0], self.heads).contiguous(), initial,
            cu_seqlens_cpu=cu_cpu, checkpoint_interval=span, checkpoint_states=meta.kda_checkpoint_states,
            a_log=self.a_log, dt_bias=self.dt_bias, lower_bound=p.gate_lower_bound,
            scale=self.dim ** -0.5, group=group, input_is_cp=True, zero_initial_state=zero_initial,
            checkpoint_publish_masks=publish_masks)
        with scope('glm53_kcp/publish_conv_cache'):
            incoming = exchange_conv_states(synthetic_cache[len(ids):], plan['padded_segments'],
                                            self.local_width, group)
            cm = meta.kda_checkpoint_meta
            publish_conv_states(incoming, checkpoint_index, cm, block_map, conv_cache)
            store_kda_recurrent_checkpoints(meta.kda_checkpoint_states, cm, block_map, ssm_cache)
        kv_cache.cache_store_segment_sizes = list(p.cache_store_segment_sizes)
        from rtp_llm.ops import compute_ops

        compute_ops.write_cache_store(
            inputs.input_lengths, inputs.prefix_lengths, inputs.kv_cache_block_id_host,
            inputs.cache_store_inputs, kv_cache)
        # Preserve native cache-store publication before the output projection.
        with scope('glm53_kcp/output_projection'):
            # KDA's gated RMSNorm belongs to the attention module; the model
            # final norm is applied only after all decoder layers.
            normed = m.norm(output.reshape(-1, self.dim), output_gate.reshape(-1, self.dim))
            return normed.reshape(x.shape[0], self.width) @ self.output_weight
