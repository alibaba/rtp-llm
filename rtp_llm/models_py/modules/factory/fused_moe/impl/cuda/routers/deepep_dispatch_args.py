def get_low_latency_dispatch_quant_args(
    *,
    use_fp8: bool,
    is_block_quantized: bool,
    is_per_act_token: bool,
    use_e8m0: bool,
) -> dict[str, bool]:
    if use_fp8 and is_block_quantized and use_e8m0:
        return {"round_scale": True, "use_ue8m0": True}
    if use_fp8 and is_per_act_token:
        return {"pertoken_quant": True}
    return {}
