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

    if tp <= 0 or not 0 <= tp_rank < tp:
        raise ValueError("PPU shared expert requires a positive TP size and valid rank")
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


def tp_split_strategies():
    from functools import partial
    from rtp_llm.utils import model_weight as mw

    out = (
        mw.W.v4_routed_w1_w,
        mw.W.v4_routed_w1_s,
        mw.W.v4_routed_w3_w,
        mw.W.v4_routed_w3_s,
    )
    down = (mw.W.v4_routed_w2_w, mw.W.v4_routed_w2_s)
    return {
        **dict.fromkeys(out, split_routed_output),
        **dict.fromkeys(down, mw.sp_moe_neg1),
        **{
            name: partial(split_shared_weight, projection=projection, scale=scale)
            for name, projection, scale in (
                (mw.W.v4_shared_w13_w, "w13", False),
                (mw.W.v4_shared_w13_s, "w13", True),
                (mw.W.v4_shared_w2_w, "w2", False),
                (mw.W.v4_shared_w2_s, "w2", True),
            )
        },
    }


def configure_weight_splits(info):
    """Specialize only this PPU model's descriptors, leaving global tables intact."""
    from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
    from rtp_llm.platforms.ppu.models.dsv4.routed_weight import PpuRoutedWeight

    strategies = tp_split_strategies()

    def configure(weight):
        if weight.name in strategies:
            if type(weight) is MoeAtomicWeight:
                return PpuRoutedWeight(**MoeAtomicWeight.extract_params(PpuRoutedWeight, weight, None))
            table = getattr(weight, "gpt_style_tp_strategy", None)
            if table is not None:
                weight.gpt_style_tp_strategy = {**table, weight.name: strategies[weight.name]}
        children = getattr(weight, "sub_weights", {})
        for name, child in children.items():
            children[name] = configure(child)
        return weight

    for layer in info.layer_weights:
        layer[:] = [configure(weight) for weight in layer]
    info.weights[:] = [configure(weight) for weight in info.weights]
