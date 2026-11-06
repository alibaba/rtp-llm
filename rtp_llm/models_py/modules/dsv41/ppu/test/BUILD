package(default_visibility = ["//visibility:public"])

sh_test(
    name = "mhc_mega_integration_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["mhc-integration"],
    env = {"GPU_COUNT": "1", "RTP_V41_TEST_MHC_DEVICE": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "dspark_contract_test",
    srcs = ["run_dspark_contracts.sh"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    timeout = "long",
)

sh_test(
    name = "dspark_native_contract_role_test",
    srcs = ["run_dspark_contracts.sh"],
    args = ["native"],
    env = {
        "GPU_COUNT": "8",
        "RTP_V41_TEST_DSPARK_SUITE": "contracts",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "dspark_native_role_test",
    srcs = ["run_dspark_contracts.sh"],
    args = ["native"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "ced_contract_test",
    srcs = ["run_ced_contracts.sh"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    timeout = "long",
)

sh_test(
    name = "ced_native_role_test",
    srcs = ["run_ced_contracts.sh"],
    args = ["native"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

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

sh_test(
    name = "s5_kernel_surface_probe_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["s5-kernel-surface"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "s5_mhc_mega_probe_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["s5-mhc-mega"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
