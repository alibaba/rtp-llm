"""Shared capability checks for the fused grouped top-k router."""

GROUP_TOPK_WARP_SIZE = 32


def group_topk_supported(
    *,
    num_experts: int,
    n_group: int,
    topk_group: int,
    top_k: int,
    renormalize: bool,
) -> bool:
    """Return whether the CUDA GroupTopK kernel can represent this routing.

    The kernel scores each group with its largest two expert scores. A
    single-expert group leaves the second value at ``-inf`` and can silently
    trigger the kernel's uniform-routing fallback. Group and expert selection
    also share one 32-lane warp. ``GROUP_TOPK_WARP_SIZE`` mirrors WARP_SIZE and
    the WarpSelect capability in ``no_aux_tc_kernels.cu``.
    """

    return (
        num_experts > 0
        and n_group > 0
        and num_experts % n_group == 0
        and num_experts // n_group >= 2
        and 0 < topk_group <= n_group
        and 0 < top_k <= topk_group * (num_experts // n_group)
        and n_group <= GROUP_TOPK_WARP_SIZE
        and top_k <= GROUP_TOPK_WARP_SIZE
        and not (top_k == 1 and renormalize)
    )


__all__ = ["GROUP_TOPK_WARP_SIZE", "group_topk_supported"]
