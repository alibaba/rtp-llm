load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

cuda13_sm100_py_test(
    name = "legacy_fp8_indexer_score_test",
    srcs = ["legacy_fp8_indexer_score_test.py"],
    deps = ["//rtp_llm:testlib"],
)

cuda13_sm100_py_test(
    name = "legacy_fp8_indexer_score_perf",
    srcs = ["legacy_fp8_indexer_score_test.py"],
    args = ["--bench"],
    main = "legacy_fp8_indexer_score_test.py",
    tags = [
        "manual",
        "perf",
    ],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "linear_replay_test",
    srcs = ["linear_replay_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)
