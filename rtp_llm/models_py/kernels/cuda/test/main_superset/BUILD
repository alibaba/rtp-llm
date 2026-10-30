load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

package(default_visibility = ["//visibility:public"])

_DEPS = [
    "//rtp_llm:torch",
    "@pip_dsv4_test_pytest//:pkg",
]

cuda13_sm100_py_test(
    name = "deepgemm_main_superset_test",
    srcs = ["deepgemm_main_superset_test.py"],
    deps = _DEPS + ["//rtp_llm:deep_gemm"],
    timeout = "long",
)

cuda13_sm100_py_test(
    name = "flashmla_main_superset_test",
    srcs = ["flashmla_main_superset_test.py"],
    deps = _DEPS + ["//rtp_llm:flash-mla"] + select({
        "@//:using_cuda13_arm": ["//rtp_llm:flash-mla-legacy"],
        "//conditions:default": [],
    }),
    timeout = "long",
)

[
    py_test(
        name = "deepgemm_main_superset_" + suffix + "_test",
        srcs = ["deepgemm_main_superset_test.py"],
        main = "deepgemm_main_superset_test.py",
        deps = _DEPS + ["//rtp_llm:deep_gemm"],
        env = {"GPU_COUNT": "1"},
        tags = [gpu],
        exec_properties = {"gpu": gpu, "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        timeout = "long",
    )
    for suffix, gpu in [
        ("sm90", "H20_CU13"),
        ("sm120", "RTX_5000_PRO_CU13"),
    ]
]

test_suite(
    name = "main_superset_cuda13_x86",
    tests = [
        ":deepgemm_main_superset_test_x86",
        ":flashmla_main_superset_test_x86",
    ],
)
