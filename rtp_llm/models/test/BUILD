py_test(
    name = "dsv41_kv_cache_test",
    srcs = ["dsv41_kv_cache_test.py"],
    data = ["//rtp_llm/config/test:testdata/deepseek_v41_flash.json"],
    deps = ["//rtp_llm:sdk"],
)

py_test(
    name = "dsv41_cache_production_test",
    srcs = ["dsv41_cache_production_test.py"],
    data = [
        "//rtp_llm/config/test:testdata/deepseek_v41_flash.json",
        "//rtp_llm/cpp/cache/test:cache_config_creator_py_test",
    ],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "dsv41_cache_codec_test",
    srcs = ["dsv41_cache_codec_test.py"],
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_delayed_hc_test",
    srcs = ["dsv41_delayed_hc_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_delayed_hc_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_engram_hash_test",
    srcs = ["dsv41_engram_hash_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_engram_hash_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources", "//rtp_llm/config/test:testdata/deepseek_v41_flash.json"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch", "//rtp_llm:triton"],
    env = {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
)

py_test(
    name = "dsv41_engram_host_test",
    srcs = ["dsv41_engram_host_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_engram_host_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch", "//rtp_llm:triton"],
    env = {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
)

py_test(
    name = "dsv41_engram_inject_test",
    srcs = ["dsv41_engram_inject_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_engram_inject_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch", "//rtp_llm:triton"],
    env = {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
)

py_test(
    name = "dsv41_mxfp8_test",
    srcs = ["dsv41_mxfp8_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_mxfp8_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch", "//rtp_llm:triton"],
    env = {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
)

py_test(
    name = "dsv41_engram_test",
    srcs = [
        "dsv41_engram_test.py",
        "dsv41_engram_host_test.py",
        "dsv41_engram_inject_test.py",
        "dsv41_mxfp8_test.py",
        "dsv41_cache_codec_test.py",
    ],
    main = "dsv41_engram_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_engram_weights_test",
    srcs = [
        "dsv41_engram_weights_test.py",
        "dsv41_engram_test.py",
        "dsv41_engram_host_test.py",
        "dsv41_engram_inject_test.py",
        "dsv41_mxfp8_test.py",
        "dsv41_cache_codec_test.py",
    ],
    main = "dsv41_engram_weights_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_engram_stage_test",
    srcs = [
        "dsv41_engram_stage_test.py",
        "dsv41_engram_weights_test.py",
        "dsv41_engram_hash_test.py",
        "dsv41_engram_test.py",
        "dsv41_engram_host_test.py",
        "dsv41_engram_inject_test.py",
        "dsv41_mxfp8_test.py",
        "dsv41_cache_codec_test.py",
    ],
    main = "dsv41_engram_stage_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_dispatch_test",
    srcs = ["dsv41_dispatch_test.py", "dsv41_expert_test.py", "dsv41_mxfp8_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_dispatch_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_moe_test",
    srcs = ["dsv41_moe_test.py", "dsv41_dispatch_test.py", "dsv41_routed_test.py", "dsv41_expert_test.py", "dsv41_mxfp8_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_moe_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_routed_test",
    srcs = ["dsv41_routed_test.py", "dsv41_expert_test.py", "dsv41_mxfp8_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_routed_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_expert_test",
    srcs = ["dsv41_expert_test.py", "dsv41_mxfp8_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_expert_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_pp_test",
    srcs = ["dsv41_pp_test.py", "dsv41_cache_transfer_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_pp_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_cache_transfer_test",
    srcs = ["dsv41_cache_transfer_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_cache_transfer_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "dsv41_cache_triton_test",
    srcs = ["dsv41_cache_triton_test.py", "dsv41_cache_codec_test.py"],
    main = "dsv41_cache_triton_test.py",
    data = ["//rtp_llm/models_py:dsv41_cache_codec_sources"],
    deps = ["//rtp_llm:sdk", "//rtp_llm:torch", "//rtp_llm:triton"],
    env = {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
)

py_test(
    name = "dsv4_kv_cache_test",
    srcs = ["dsv4_kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "H20"},
)

# Production PD transfer-geometry regression: drives the real
# descs builder + real C++ SpecBuilder via the test-only binding and asserts
# the expected wire geometry per tag/kernel, padding visibility, state/SWA
# kernel independence, layer coverage, and invalid-shape rejection. CPU-only.
py_test(
    name = "pd_transfer_geometry_production_test",
    srcs = ["pd_transfer_geometry_production_test.py"],
    data = [
        "//rtp_llm/cpp/cache/test:cache_config_creator_py_test",
    ],
    deps = ["//rtp_llm:testlib"],
)
