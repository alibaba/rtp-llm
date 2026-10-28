load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch"])

py_test(
    name = "test_int8_quant",
    srcs = ["test_int8_quant.py"],
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
        "//rtp_llm/models_py:ppu_modules",
    ] + [":torch"],
    visibility = ["//visibility:public"],
)
