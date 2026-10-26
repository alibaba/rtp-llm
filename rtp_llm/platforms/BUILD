load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_py_test")

cc_library(
    name = "mtp_policy",
    srcs = ["MtpPolicy.cc"],
    hdrs = ["MtpPolicy.h", "ppu/MtpPolicy.h"],
    copts = copts(),
    deps = ["//rtp_llm/cpp/cache"] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_test_wrapper(
    name = "mtp_policy_test",
    srcs = ["ppu/test/MtpPolicyTest.cc"],
    copts = copts(),
    deps = [
        ":mtp_policy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

# Imported model contracts require the same CUDA13 closure as the original tests.
[dsv4_py_test(
    name = name,
    src = "ppu/test/test_" + name + ".py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    timeout = "moderate",
) for name in [
    "mtp_weights", "mtp_hidden", "compressor_norm_dispatch", "mtp_swa_ring_page",
    "chunked_moe", "deepep_graph_dispatch", "ppu_deep_gemm_layout_compat",
    "strategy_select", "tp_norm", "v4_router_contract", "v4_tp_reduce_contract",
]]

# PPU operators and JIT sources are packaged only for the PPU build.
[py_test(
    name = name,
    srcs = ["ppu/test/test_" + name + ".py"],
    main = "ppu/test/test_" + name + ".py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
) for name in [
    "isolated_metadata", "rich_contract", "chunk_options", "platform_provider",
    "prefill_communication", "tp_prefill_workspace_dims",
]]

_ENTRYPOINTS = ["__init__.py", "model_dispatch.py", "ppu/__init__.py"]

[py_test(
    name = name + "_test",
    srcs = ["ppu/test/test_" + name + ".py"],
    main = "ppu/test/test_" + name + ".py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
) for name in ["mega_moe_contract", "mtp_input", "decode_metadata_graph", "decode_state_slots"]]

py_test(
    name = "decode_fusions_test",
    srcs = ["ppu/test/test_decode_fusions.py"],
    main = "ppu/test/test_decode_fusions.py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "platform_boundary_test",
    srcs = ["ppu/test/test_platform_boundary.py"],
    main = "ppu/test/test_platform_boundary.py",
    data = ["model_dispatch.py", "ppu/models/dsv4/resources.py"],
)

py_test(
    name = "ep_chunk_rounds_test",
    srcs = ["ppu/test/test_ep_chunk_rounds.py"],
    main = "ppu/test/test_ep_chunk_rounds.py",
    data = ["ppu/models/dsv4/ppu_ep_moe.py"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "indexer_cache_test",
    srcs = ["ppu/test/test_indexer_cache.py"],
    main = "ppu/test/test_indexer_cache.py",
    data = ["ppu/models/dsv4/cache.py"],
)

py_test(
    name = "dsv4_config_test",
    srcs = ["ppu/test/test_dsv4_config.py"],
    main = "ppu/test/test_dsv4_config.py",
    data = ["ppu/models/dsv4/manifest.py"],
)

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
    name = "fp8_contract_test",
    srcs = ["ppu/test/test_fp8_contracts.py"],
    main = "ppu/test/test_fp8_contracts.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
)

py_test(
    name = "fused_permute_test",
    srcs = ["ppu/test/test_fused_permute.py"],
    main = "ppu/test/test_fused_permute.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
    name = "flash_mla_runtime_test",
    srcs = ["ppu/test/test_flash_mla_runtime.py"],
    main = "ppu/test/test_flash_mla_runtime.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm/models_py:flash-mla", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "fp4_indexer_request_slices_test",
    srcs = ["ppu/test/test_fp4_indexer_request_slices.py"],
    main = "ppu/test/test_fp4_indexer_request_slices.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10_CU13"},
)

test_suite(
    name = "ppu_source_regression_tests",
    tests = [
        ":fp4_indexer_request_slices_test",
    ],
)

py_test(
    name = "fp4_indexer_chunk_state_test",
    srcs = ["ppu/test/test_fp4_indexer_chunk_state.py"],
    main = "ppu/test/test_fp4_indexer_chunk_state.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "bf16_radix_topk_test",
    srcs = ["ppu/test/test_bf16_radix_topk.py"],
    main = "ppu/test/test_bf16_radix_topk.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["gpu", "manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mtp_queries_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_mtp_queries.py"],
    main = "ppu/models/dsv4/test/test_ppu_mtp_queries.py",
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
