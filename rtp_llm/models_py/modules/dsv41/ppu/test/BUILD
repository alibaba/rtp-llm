package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_prefill_dp8",
    srcs = ["test_prefill_dp8.py", "test_native_cache.py"],
    main = "test_prefill_dp8.py",
    deps = ["//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "0"},
    tags = ["cpu"],
)

py_test(
    name = "test_moe_megamoe",
    srcs = ["test_moe_megamoe.py"],
    deps = ["//rtp_llm/models_py:v41_ppu_moe"],
    env = {"GPU_COUNT": "0"},
    tags = ["cpu"],
)

py_test(
    name = "test_moe_w4a8",
    srcs = ["test_moe_w4a8.py"],
    deps = ["//rtp_llm/models_py:v41_ppu_moe"],
    env = {
        "GPU_COUNT": "0",
        "RTP_V41_TEST_W4A8_DEVICE": "0",
    },
    tags = ["cpu"],
)

sh_test(
    name = "test_moe_w4a8_device",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["moe-device"],
    env = {
        "GPU_COUNT": "1",
        "RTP_V41_TEST_W4A8_DEVICE": "1",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
