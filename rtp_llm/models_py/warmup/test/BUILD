load("//bazel:arch_select.bzl", "requirement")
deep_gemm = ["deep_gemm"]
requirement(deep_gemm)
py_test(
    name = "deepgemm_warmup_test",
    srcs = ["deepgemm_warmup_test.py"],
    deps = [
        "//rtp_llm/models_py/warmup:warmup",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12": deep_gemm,
        "//conditions:default": [],
    }),
    data = [
        "//:rtp_compute_ops",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)
