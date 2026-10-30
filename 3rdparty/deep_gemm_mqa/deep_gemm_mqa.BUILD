load("@rules_python//python:defs.bzl", "py_library")

# Keep the old group-128 MQA native/JIT resources in a separate import namespace.
# No GEMM, RS, MoE or legacy Python fallback is exposed by this dependency.
py_library(
    name = "mqa",
    srcs = ["deep_gemm_mqa/__init__.py"],
    data = glob([
        "deep_gemm_mqa/*.so",
        "deep_gemm_mqa/include/**",
        "deep_gemm_mqa/csrc/**",
        "deep_gemm_mqa/LICENSE",
    ]),
    imports = ["."],
    visibility = ["//visibility:public"],
)
