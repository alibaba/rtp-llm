"""PPU loader preparation; imported only after implementation binding."""


def split_routed_output(t, tp, tp_rank, *, moe_pure_tp_mode=False, **kwargs):
    if moe_pure_tp_mode:
        if tp <= 0 or not 0 <= tp_rank < tp or t.shape[1] % tp:
            raise ValueError("PPU routed intermediate dimension must divide TP")
        return t.narrow(1, tp_rank * (t.shape[1] // tp), t.shape[1] // tp)
    return t


def split_shared_weight(t, tp, tp_rank, *, projection, scale=False, **kwargs):
    """Prepare the TP owner's tensor at both initial load and weight update."""
    import torch

    if tp != 4 or not 0 <= tp_rank < tp:
        raise ValueError("PPU shared expert requires TP4 and rank in [0, 4)")
    dtype = torch.float8_e8m0fnu if scale else torch.float8_e4m3fn
    block = 1 if scale else 128
    if t.ndim != 2 or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(f"Shared expert requires a contiguous 2D {dtype} tensor")
    if min(t.shape) <= 0:
        raise ValueError("Shared expert dimensions must be positive")
    if projection == "w13":
        if t.shape[0] % (2 * block * tp) or t.shape[1] % block:
            raise ValueError("Shared w13 must preserve block-128 TP scale geometry")
        half = t.shape[0] // 2
        width = half // tp
        start = tp_rank * width
        result = torch.cat(
            (
                t[start : start + width].view(torch.uint8),
                t[half + start : half + start + width].view(torch.uint8),
            ),
            dim=0,
        ).view(t.dtype)
    elif projection == "w2":
        if t.shape[0] % block or t.shape[1] % (block * tp):
            raise ValueError("Shared w2 must preserve block-128 TP scale geometry")
        width = t.shape[1] // tp
        result = t[:, tp_rank * width : (tp_rank + 1) * width].contiguous()
    else:
        raise ValueError(f"Unknown shared projection: {projection}")
    # Own the actual DeepGEMM scale representation. A second converted copy in
    # the linear would become stale after ModelWeights.update_layer_weight.
    return result.to(torch.float32) if scale else result


def routed_tp_preparation():
    from rtp_llm.model_loader.weight_preparation import WeightPreparation
    from rtp_llm.utils import model_weight as mw

    out = (
        mw.W.v4_routed_w1_w,
        mw.W.v4_routed_w1_s,
        mw.W.v4_routed_w3_w,
        mw.W.v4_routed_w3_s,
    )
    down = (mw.W.v4_routed_w2_w, mw.W.v4_routed_w2_s)
    return WeightPreparation(
        split_strategies=tuple((name, split_routed_output) for name in out)
        + tuple((name, mw.sp_moe_neg1) for name in down),
        preshard_layouts=tuple(
            ((name, mw.stack_, 1), (0, (0,), False, split_routed_output))
            for name in out
        )
        + tuple(
            ((name, mw.stack_, 1), (1, (0,), False, mw.sp_moe_neg1)) for name in down
        ),
    )


def tp_moe_shared_fp32_preparation():
    from functools import partial
    from rtp_llm.model_loader.weight_preparation import WeightPreparation
    from rtp_llm.utils.model_weight import W

    routed = routed_tp_preparation()
    return WeightPreparation(
        split_strategies=routed.split_strategies
        + tuple(
            (name, partial(split_shared_weight, projection=projection, scale=scale))
            for name, projection, scale in (
                (W.v4_shared_w13_w, "w13", False),
                (W.v4_shared_w13_s, "w13", True),
                (W.v4_shared_w2_w, "w2", False),
                (W.v4_shared_w2_s, "w2", True),
            )
        ),
        preshard_layouts=routed.preshard_layouts,
    )
