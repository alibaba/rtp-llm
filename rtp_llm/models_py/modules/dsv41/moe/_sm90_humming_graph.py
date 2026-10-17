"""Bounded, fixed-shape routing for SM90 native W4A8 decode graphs.

The normal DeepEP protocol and its BF16 combine boundary are unchanged.
Only intranode dispatch is supported. Every rank must use the same configured
input-row bound. Routing *values* are rebuilt on device on every invocation;
neither a captured route nor a received-token count is cached on the host.
"""

import torch


def static_group_routes(indices, local_start, local_end):
    """Return all route slots, legal fixed-capacity offsets and validity.

    Valid slots preserve the eager stable expert/slot ordering. Invalid slots
    sort after every valid expert and become zero dummy rows of the last
    expert. Thus offsets[-1] is the physical row capacity, not a GPU scalar
    used as a host shape. Repeated offsets represent empty experts.
    """
    if indices.ndim != 2 or local_end <= local_start:
        raise ValueError("Expected [rows, topk] indices and nonempty expert range")
    experts = local_end - local_start
    if indices.is_cuda and indices.numel() <= 256 and experts <= 256:
        from ._sm90_route_kernels import static_route_metadata

        return static_route_metadata(indices, local_start, local_end)[:3]
    flat = indices.reshape(-1)
    valid = (flat >= local_start) & (flat < local_end)
    expert = flat - local_start
    keys = torch.where(valid, expert, experts)
    slots = torch.argsort(keys, stable=True)
    expert_ids = torch.arange(experts, device=indices.device, dtype=flat.dtype)
    counts = ((expert[:, None] == expert_ids[None, :]) & valid[:, None]).sum(
        dim=0, dtype=torch.int32
    )
    # Invalid rows are physical padding of the final expert, not valid routes.
    counts[-1] += (~valid).sum(dtype=torch.int32)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    return slots, offsets, valid


def forward_local_static(
    local, x, weights, indices, local_start, local_end, *, output_dtype=torch.float32
):
    """Native A8/K32 experts with no data-dependent shapes or host sync."""
    from ._sm90_route_kernels import (
        prepare_static_routes,
        reduce_routes,
        reduce_static_routes,
        routed_swiglu,
    )

    if not local.native_a8 or any(
        linear.promotion_group_size != 32 for linear in (local.w13, local.w2)
    ):
        raise RuntimeError("SM90 graph routing requires native W4A8/K32")
    if indices.shape != weights.shape or indices.shape[0] != x.shape[0]:
        raise ValueError("Route tensors must match input rows")
    topk = indices.shape[1]
    if not 1 <= topk <= 8:
        raise ValueError("SM90 graph routing supports topk 1..8")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("SM90 graph output must preserve FP32/BF16 combine boundary")
    if x.shape[0] == 0:
        return torch.zeros(x.shape, device=x.device, dtype=output_dtype)
    fused = indices.numel() <= 256 and 0 < local_end - local_start <= 256
    if fused:
        slots, offsets, valid, inverse, grouped_x, clean_weights = (
            prepare_static_routes(
                x,
                weights,
                indices,
                local_start,
                local_end,
            )
        )
    else:
        slots, offsets, valid = static_group_routes(indices, local_start, local_end)
        grouped_valid = valid.index_select(0, slots)
        # DeepEP does not promise initialized recv_x/weights for padded rows.
        # Mask by selection, never by multiplication (NaN * 0 is still NaN).
        grouped_x = torch.where(
            grouped_valid[:, None], x.index_select(0, slots // topk), 0
        ).contiguous()
        clean_weights = torch.where(valid.reshape_as(weights), weights, 0).contiguous()
    gate_up = local.w13(grouped_x, offsets)
    hidden = routed_swiglu(
        gate_up,
        clean_weights,
        slots,
        clamp_limit=local.cfg.swiglu_limit,
        output_dtype=x.dtype,
    )
    down = local.w2(hidden, offsets)
    if fused:
        return reduce_static_routes(
            down,
            inverse,
            valid,
            x.shape[0],
            topk,
            output_dtype=output_dtype,
        )
    # Keep dummy slots exactly zero even if a backend ever leaves them unwritten.
    down = torch.where(grouped_valid[:, None], down, 0).contiguous()
    return reduce_routes(down, slots, x.shape[0], topk).to(output_dtype)


def forward_normal_static(
    strategy,
    x,
    weights,
    indices,
    max_rows,
    *,
    local_forward=None,
):
    """Original normal dispatch/combine, with a static intranode recv bound."""
    from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPMode, DeepEPWrapper

    wrapper = DeepEPWrapper._instance
    if wrapper is None or wrapper.mode != DeepEPMode.NORMAL:
        raise RuntimeError("SM90 graph MoE requires initialized DeepEP normal mode")
    if wrapper.config.use_deepep_internode:
        raise RuntimeError("SM90 graph MoE currently supports intranode DeepEP only")
    if max_rows <= 0 or x.shape[0] > max_rows:
        raise ValueError("Input exceeds configured SM90 graph row bound")
    cfg = strategy.cfg
    if wrapper.config.ep_size != cfg.ep_size:
        raise RuntimeError("DeepEP and model EP sizes differ")
    buf = wrapper.buffer
    if (
        indices.shape[-1] == 6
        and indices.dtype == torch.int64
        and weights.dtype == torch.float32
    ):
        from ._sm90_dispatch_pad import pad_deepep_routes

        indices_p, weights_p = pad_deepep_routes(indices, weights, pad_to=8)
    else:
        indices_p, weights_p = strategy._dispatch._pad_topk_for_deepep(indices, weights)
    per_rank, per_rdma, per_expert, in_rank, _ = buf.get_dispatch_layout(
        indices_p, cfg.n_routed_experts
    )
    recv_x, recv_idx, recv_weights, _, handle, _ = buf.dispatch(
        x,
        None,
        per_rank,
        per_rdma,
        in_rank,
        per_expert,
        indices_p,
        weights_p,
        expert_alignment=1,
        num_worst_tokens=max_rows * cfg.ep_size,
    )
    # recv_idx is local, including -1 for nonlocal and capacity padding.
    compute_local = forward_local_static if local_forward is None else local_forward
    local_output = compute_local(
        strategy._local,
        recv_x.contiguous(),
        recv_weights.contiguous(),
        recv_idx,
        0,
        cfg.local_expert_end - cfg.local_expert_start,
        output_dtype=x.dtype,
    )
    # The local backend applies route weights exactly once. Normal combine
    # remains unweighted and receives the existing BF16 protocol boundary.
    combined, _, _ = buf.combine(local_output, handle)
    return combined.float()


def forward_elastic_static(strategy, x, weights, indices, *, local_forward):
    """DeepEP V2 decode with GPU receive counts and fixed Graph capacity."""
    from ._sm90_dispatch_pad import pad_deepep_routes

    cfg = strategy.cfg
    indices_p, weights_p = pad_deepep_routes(indices, weights, pad_to=8)
    recv_x, recv_idx, recv_weights, handle, _ = strategy._comm_buffer.dispatch(
        x,
        topk_idx=indices_p,
        topk_weights=weights_p,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=strategy._decode_graph_max_rows,
        expert_alignment=1,
        num_sms=strategy._v2_num_sms,
        do_cpu_sync=False,
        do_expand=False,
    )
    # The FI prepare kernel masks unused receive capacity before reading it.
    # Keep counts on device so every Graph replay can use different routes.
    local_output = local_forward(
        strategy._local,
        recv_x,
        recv_weights,
        recv_idx,
        0,
        cfg.n_local_experts,
        output_dtype=x.dtype,
        recv_count=handle.psum_num_recv_tokens_per_scaleup_rank[-1],
    )
    # Local FI already applied the route weights exactly once.
    combined, _, _ = strategy._comm_buffer.combine(local_output, handle)
    return combined.float()
