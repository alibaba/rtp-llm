"""Fixed-capacity DeepEP V2 dispatch/combine for SM90 decode graphs.

Local expert compute is supplied by the MoE strategy. Receive counts stay on
device and routing is recomputed on every replay.
"""


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
