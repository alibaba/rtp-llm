load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
])

py_test(
    name = "test_fa3_decode_padding",
    srcs = ["test_fa3_decode_padding.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py:ppu_modules",
    ] + [":torch"],
)

py_test(
    name = "test_beam_search_cascade",
    srcs = ["test_beam_search_cascade.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_full_prefill_graph",
    srcs = ["test_full_prefill_graph.py"],
    deps = [
        "//rtp_llm/models_py:ppu_entrypoint",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
