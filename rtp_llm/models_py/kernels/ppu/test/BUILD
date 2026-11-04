load("@arch_config//:arch_select.bzl", "requirement")

requirement(["deep_gemm", "numpy", "torch"])

py_test(
    name = "test_deepgemm_warmup",
    srcs = ["test_deepgemm_warmup.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
    ] + [":torch"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_deepgemm_wrapper",
    srcs = ["test_deepgemm_wrapper.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:internal_kernels"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_deepgemm_masked_graph",
    srcs = ["test_deepgemm_masked_graph.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "PPU-ZW810E",
        "gpu_count": "1",
    },
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        ":deep_gemm",
        ":numpy",
        ":torch",
    ],
    visibility = ["//visibility:public"],
)
