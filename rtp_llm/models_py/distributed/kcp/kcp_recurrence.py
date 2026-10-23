"""KDA zigzag recurrence with paged checkpoints and direct CP inputs.

The direct path receives full heads on a local token shard and skips Q/K/V/g/beta
layout exchange. TP-format checkpoint publication and continuation remain intact.
"""
import torch
import torch.distributed as dist
import os
from pathlib import Path
from functools import lru_cache

from .kcp_plan import make_plan, make_checkpoint_plan
from .kcp_bm_pipeline import scope, summary_allgather, scratch_buffer
from .kcp_checkpoint import return_checkpoints


@lru_cache(maxsize=128)
def device_plan(cu_values, world, span, rank, device_index, stream_pointer):
    from fla.ops.utils.index import prepare_chunk_indices

    plan = make_plan(cu_values, world, span)
    ids = plan["by_rank"][rank]
    rows = [plan["segments"][i] for i in ids]
    tokens = plan["local_tokens"]
    device = torch.device("cuda", device_index)
    cpu_cu = torch.tensor([plan['local_offsets'][i] for i in ids] + [tokens], dtype=torch.int32)
    cu = cpu_cu.to(device)
    return (plan, ids, cpu_cu, cu,
            prepare_chunk_indices(cu, 64, cu_seqlens_cpu=cpu_cu),
            torch.tensor(plan["flattened_indices"], device=device),
            torch.tensor([plan["initial_indices"][i] for i in ids], device=device),
            torch.tensor(plan["checkpoint_indices"], device=device))


def gather_heads(state, group):
    world = dist.get_world_size(group)
    outgoing = state.unsqueeze(0).contiguous()
    incoming = state.new_empty((world, *state.shape))
    dist.all_gather_into_tensor(incoming, outgoing, group=group)
    return incoming.transpose(0, 1).contiguous().flatten(1, 2)


@lru_cache(maxsize=128)
def checkpoint_device_plan(cu_values, world, page, rank, device, stream, summary_interval):
    metadata = make_checkpoint_plan(cu_values, world, page, summary_interval)
    base = device_plan(cu_values, world, metadata['summary_interval'], rank, device, stream)
    target = torch.device('cuda', device)
    offsets = metadata['checkpoint_offsets'][rank]
    return (base, offsets[-1], metadata['padded_checkpoints'],
            torch.tensor(offsets, dtype=torch.int32, device=target),
            torch.tensor(metadata['checkpoint_indices'], dtype=torch.int64, device=target))


@torch.inference_mode()
def checkpoint_prefill(q, k, v, raw_gate, raw_beta, initial_state, *,
                       cu_seqlens_cpu, checkpoint_interval, checkpoint_states,
                       a_log, dt_bias, lower_bound, scale, group=None,
                       bm_groups=1, overlap=False, strided=False, cache_merge_metadata=False,
                       input_is_cp=False, zero_initial_state=False, summary_interval=32768,
                       checkpoint_publish_masks=None):
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from cula.kda.chunk_intra import chunk_kda_fwd_intra
    from cula.ops.kda.beta_sigmoid import beta_sigmoid
    from .kcp_merge import launch_merge_local
    from cula.ops.kda.sm100.delta_h import chunk_gated_delta_rule_fwd_h
    from cula.ops.kda.sm100.fwd_o import chunk_gla_fwd_o

    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("KCP requires BF16 and head_dim=128")
    if a_log.numel() != q.shape[2] or (dt_bias is not None and dt_bias.numel() != q.shape[2] * q.shape[3]):
        raise ValueError('KCP gate parameters require heads A_log and heads*head_dim dt_bias elements')
    device = q.device
    base, checkpoint_count, padded_checkpoints, checkpoint_offsets, checkpoint_index = checkpoint_device_plan(
        tuple(cu_seqlens_cpu.tolist()), world, checkpoint_interval, rank, device.index,
        int(torch.cuda.current_stream(device).cuda_stream), summary_interval)
    plan, ids, cpu_cu, cu, chunk_indices, gather_index, initial_index, _ = base
    if not input_is_cp:
        raise ValueError('Unified zigzag recurrence requires directly projected local CP rows')
    with scope("kcp/load_initial_head_state"):
        full_initial = (initial_state.new_zeros(
            initial_state.shape[0], initial_state.shape[1] * world, *initial_state.shape[2:]
        ) if zero_initial_state else gather_heads(initial_state, group))
    _, tokens, heads, dim = q.shape
    with scope("kcp/intra"):
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = beta_sigmoid(raw_beta.contiguous())
        gate = kda_gate_chunk_cumsum(g=raw_gate, A_log=a_log, dt_bias=dt_bias,
                                    scale=RCP_LN2, chunk_size=64, cu_seqlens=cu,
                                    chunk_indices=chunk_indices, lower_bound=lower_bound)
        w, u, _, kg, aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=gate, beta=beta, scale=scale, cu_seqlens=cu,
            chunk_size=64, chunk_indices=chunk_indices, safe_gate=True,
            disable_recompute=False, use_tf32_inverse=True, unified_gref=False)
    capture = os.environ.get('RTP_KCP_CAPTURE_DIR')
    if capture:
        path = Path(capture, f'intra_rank{rank}.pt')
        if not path.exists():
            torch.save({'kg':kg.cpu(),'w':w.cpu(),'u':u.cpu(),'gate':gate.cpu(),
                        'q':q.cpu(),'aqk':aqk.cpu(),'v':v.cpu(),
                        'cu':cpu_cu,'plan':plan,'ids':ids,'initial':full_initial.cpu(),
                        'rank':rank},path)
    with scope("kcp/prefix"):
        if plan["non_first"]:
            prefix_backend = os.environ.get('GLM53_KCP_HEAD_PREFIX', '0')
            if prefix_backend == 'auto':
                # CP8 measurements favor head all-to-all for batches >=3,
                # direct peer access for B2, and retain B1's original path.
                requests = full_initial.shape[0]
                prefix_backend = '0'
                if world == 8:
                    prefix_backend = ('1' if requests >= 3 else
                                      'remote' if requests == 2 else '0')
            if prefix_backend in ('1', 'remote'):
                if prefix_backend == 'remote':
                    from .kcp_remote_prefix import summary_remote_prefix as prefix_fn
                else:
                    from .kcp_head_prefix import summary_head_prefix as prefix_fn
                if bm_groups != 1 or overlap:
                    raise ValueError('Head prefix uses one chronological summary exchange')
                local_initial = prefix_fn(kg, w, u, gate, cu, full_initial,
                    gather_index, plan, group=group, local_segments=len(ids))
            else:
                gathered = summary_allgather(kg, w, u, gate, cu, group=group, groups=bm_groups,
                    overlap=overlap, last_rank_skips=False, pad_sequences_to=plan["padded_segments"],
                    strided=strided)
                local_initial = launch_merge_local(gathered.flatten(0, 1), gather_index,
                                      plan["merge_starts"], plan['local_merge_counts'][rank],
                                      plan["merge_seq_ids"], full_initial, rank,
                                      plan['padded_segments'], len(ids), residual=True)
        else:
            local_initial = full_initial.index_select(0, initial_index).contiguous()
    with scope("kcp/recurrence_output"):
        local_checkpoints = scratch_buffer(group, (1, checkpoint_count, heads, dim, dim),
                                           torch.float32, device, 'checkpoint_local')
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=gate, initial_state=local_initial, output_final_state=False,
            cu_seqlens=cu, chunk_indices=chunk_indices, cu_seqlens_cpu=cpu_cu,
            use_intracard_cp=False, checkpoint_states=local_checkpoints,
            checkpoint_interval=checkpoint_interval, checkpoint_offsets=checkpoint_offsets)
        output = torch.zeros_like(v)
        chunk_gla_fwd_o(q=q, v=v_new, g=gate, A=aqk, h=h, o=output, scale=scale,
                       cu_seqlens=cu, chunk_size=64, chunk_indices=chunk_indices, is_varlen=True)
    with scope("kcp/checkpoint_return"):
        if checkpoint_publish_masks is None:
            return_checkpoints(local_checkpoints, checkpoint_index, checkpoint_states,
                               padded_checkpoints, group)
        else:
            # Native cache publication supplies its valid destination pages.
            # All local dense checkpoints are still computed above; the generic
            # dense-output API remains the default when no mask is supplied.
            from .kcp_masked_checkpoint import return_checkpoints as masked_return
            masked_return(local_checkpoints, checkpoint_index, checkpoint_states,
                          padded_checkpoints, group, checkpoint_publish_masks)
    return output
