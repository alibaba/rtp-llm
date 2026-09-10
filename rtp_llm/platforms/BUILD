# Descriptors stay lightweight; concrete PPU adapters and JIT sources ship
# only in the PPU distribution. Tests may request ppu_sources explicitly.
_ENTRYPOINTS = ["__init__.py", "ppu/__init__.py", "ppu/models/__init__.py", "ppu/models/dsv4/__init__.py", "ppu/models/dsv4/register.py", "ppu/models/dsv4/manifest.py"]

filegroup(
    name = "ppu_sources",
    srcs = glob(["ppu/**/*.py", "ppu/**/*.cuh", "ppu/**/*.h", "ppu/**/*.json", "ppu/**/*.patch", "ppu/**/NOTICE*"], exclude = ["**/test/**", "**/tests/**"] + _ENTRYPOINTS),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "platform_sources",
    srcs = _ENTRYPOINTS + select({
        "@//:using_ppu": [":ppu_sources"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_test(
    name = "bootstrap_test",
    srcs = ["ppu/test/test_platform_bootstrap.py"],
    main = "ppu/test/test_platform_bootstrap.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models"],
)

py_test(
    name = "fp8_contract_test",
    srcs = ["ppu/test/test_fp8_contracts.py"],
    main = "ppu/test/test_fp8_contracts.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
)

py_test(
    name = "mxfp4_masked_test",
    srcs = ["ppu/test/test_mxfp4_masked.py"],
    main = "ppu/test/test_mxfp4_masked.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "decode_hc_test",
    srcs = ["ppu/test/test_decode_hc.py"],
    main = "ppu/test/test_decode_hc.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mxfp4_deepep_test",
    srcs = ["ppu/test/test_mxfp4_deepep.py"],
    main = "ppu/test/test_mxfp4_deepep.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "decode_attention_test",
    srcs = ["ppu/test/test_decode_attention.py"],
    main = "ppu/test/test_decode_attention.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "decode_shared_expert_test",
    srcs = ["ppu/test/test_decode_shared_expert.py"],
    main = "ppu/test/test_decode_shared_expert.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "test_ppu_grouped_fp4_strategy_source",
    srcs = ["ppu/models/dsv4/test/test_ppu_grouped_fp4_strategy_source.py"],
    main = "ppu/models/dsv4/test/test_ppu_grouped_fp4_strategy_source.py",
    data = [":platform_sources", ":ppu_sources", "//rtp_llm/models_py:dsv4_moe_registry_source"],
    tags = ["source-only"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "decode_state_slots_test",
    srcs = ["ppu/test/test_decode_state_slots.py"],
    main = "ppu/test/test_decode_state_slots.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_metadata_graph_test",
    srcs = ["ppu/test/test_decode_metadata_graph.py"],
    main = "ppu/test/test_decode_metadata_graph.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_qkv_test",
    srcs = ["ppu/test/test_decode_qkv.py"],
    main = "ppu/test/test_decode_qkv.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_hc_norm_test",
    srcs = ["ppu/test/test_decode_hc_norm.py"],
    main = "ppu/test/test_decode_hc_norm.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_fp8_quant_test",
    srcs = ["ppu/test/test_decode_fp8_quant.py"],
    main = "ppu/test/test_decode_fp8_quant.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_indexer_overlap_test",
    srcs = ["ppu/test/test_decode_indexer_overlap.py"],
    main = "ppu/test/test_decode_indexer_overlap.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_moe_hint_test",
    srcs = ["ppu/test/test_decode_moe_hint.py"],
    main = "ppu/test/test_decode_moe_hint.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_moe_output_test",
    srcs = ["ppu/test/test_decode_moe_output.py"],
    main = "ppu/test/test_decode_moe_output.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_shared_schedule_test",
    srcs = ["ppu/test/test_decode_shared_schedule.py"],
    main = "ppu/test/test_decode_shared_schedule.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_tp_shared_weights_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_tp_shared_weights.py"],
    main = "ppu/models/dsv4/test/test_ppu_tp_shared_weights.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "topk_padding_test",
    srcs = ["ppu/test/test_topk_padding.py"],
    main = "ppu/test/test_topk_padding.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_hc_head_options_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_hc_head_options.py"],
    main = "ppu/models/dsv4/test/test_ppu_hc_head_options.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "silu_mul_mxfp4_test",
    srcs = ["ppu/test/test_silu_mul_mxfp4.py"],
    main = "ppu/test/test_silu_mul_mxfp4.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# Explicit PPU regression entry; checkpoint/multirank tests keep their existing
# manual targets. This suite uses generated inputs and one M890P.
test_suite(
    name = "ppu_decode_regression_tests",
    tests = [
        ":decode_hc_test",
        ":decode_state_slots_test",
        ":decode_metadata_graph_test",
        ":decode_qkv_test",
        ":decode_hc_norm_test",
        ":decode_fp8_quant_test",
        ":decode_indexer_overlap_test",
        ":decode_moe_hint_test",
        ":decode_moe_output_test",
        ":decode_shared_schedule_test",
        ":ppu_tp_shared_weights_test",
        ":topk_padding_test",
        ":ppu_hc_head_options_test",
        ":silu_mul_mxfp4_test",
        "//rtp_llm/models_py/modules/dsv4/test:test_tp_norm",
        "//rtp_llm/models_py/modules/dsv4/test:test_tp_prefill_workspace_dims",
        "//rtp_llm/models_py/modules/dsv4/test:test_chunk_options",
    ],
)
