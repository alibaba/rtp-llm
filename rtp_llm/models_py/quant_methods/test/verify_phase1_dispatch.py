"""Phase 1 派发骨架静态自检（不加载模型、不需要 ckpt）。

验证点:
1. 双注册表填充正确（fp8 系在 Linear 注册表、none 在 Linear 注册表）。
2. Linear 层派发 → 拿到对应 Linear 方法（行为同改动前）。
3. prefix 命中 ignore → 拿到 UnquantizedLinearMethod（新能力)。
4. source_config 的 ignore 字段能被自动提取（走法1 富配置打通）。

运行（容器内，bazel 环境）:
    python -m rtp_llm.models_py.quant_methods.test.verify_phase1_dispatch
或在 server_test 的 runfiles python 下直接跑本文件。
"""


def main() -> None:
    # 触发注册
    import rtp_llm.models_py.quant_methods.fp8  # noqa: F401
    import rtp_llm.models_py.quant_methods.unquantized  # noqa: F401
    from rtp_llm.models_py.quant_methods.base import (
        _LINEAR_METHOD_REGISTRY,
        _MOE_METHOD_REGISTRY,
        QuantizationConfig,
    )
    from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod

    ok = True

    # 1. 注册表
    for k in ("none", "", "fp8", "fp8_online", "fp8_per_channel", "fp8_block"):
        present = k in _LINEAR_METHOD_REGISTRY
        print(f"[reg] linear[{k!r}] present = {present}")
        ok &= present
    print(
        f"[reg] moe registry keys = {list(_MOE_METHOD_REGISTRY.keys())} "
        f"(Phase 1 预期为空，Phase 2 再填)"
    )

    # 2. Linear 派发(用一个非 MoE 的假层即可，get_quant_method 只按类型分流)
    class _FakeLinear:  # 非 BaseMoEExperts → 走 Linear 分支
        pass

    fake = _FakeLinear()

    qc_none = QuantizationConfig(quant_type="none")
    m = qc_none.get_quant_method(fake, "model.layers.0.mlp.gate_proj")
    print(f"[dispatch] none -> {type(m).__name__}")
    ok &= isinstance(m, UnquantizedLinearMethod)

    qc_fp8 = QuantizationConfig(quant_type="fp8")
    m = qc_fp8.get_quant_method(fake, "model.layers.0.mlp.gate_proj")
    print(f"[dispatch] fp8  -> {type(m).__name__}")
    ok &= type(m).__name__ == "Fp8LinearMethod"

    # 3. ignore 命中 → 未量化(即使 quant_type=fp8)
    qc_ig = QuantizationConfig(quant_type="fp8", ignored_layers=["lm_head"])
    m = qc_ig.get_quant_method(fake, "model.lm_head")
    print(f"[ignore] fp8 + ignore lm_head -> {type(m).__name__}")
    ok &= isinstance(m, UnquantizedLinearMethod)
    # 非 ignore 模块仍走 fp8
    m = qc_ig.get_quant_method(fake, "model.layers.0.mlp.gate_proj")
    ok &= type(m).__name__ == "Fp8LinearMethod"

    # 4. source_config 自动提取 ignore（走法1）
    class _FakeSrc:
        ignore_patterns = ["re:.*lm_head.*", "vision"]

    qc_src = QuantizationConfig(quant_type="fp8", source_config=_FakeSrc())
    print(f"[source_config] extracted ignored_layers = {qc_src.ignored_layers}")
    ok &= qc_src.ignored_layers == ["re:.*lm_head.*", "vision"]
    ok &= qc_src.is_layer_ignored("model.visionxxx") is True

    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
