"""PPU loader preparation; imported only after implementation binding."""


def split_routed_output(t, tp, tp_rank, *, moe_pure_tp_mode=False, **kwargs):
    if moe_pure_tp_mode:
        if tp <= 0 or not 0 <= tp_rank < tp or t.shape[1] % tp:
            raise ValueError("PPU routed intermediate dimension must divide TP")
        return t.narrow(1, tp_rank * (t.shape[1] // tp), t.shape[1] // tp)
    return t


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
