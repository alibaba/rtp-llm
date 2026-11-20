load(":host_binding.bzl", "host_binding")

package(default_visibility = ["//rtp_llm:__subpackages__"])

_CUDA13 = select({
    "@//:using_cuda13_x86": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

filegroup(
    name = "native_sources",
    srcs = glob(["native/**/*"], exclude = ["native/bind.cpp", "native/exports.map"]),
)

host_binding(
    name = "host",
    out = "rtp_dg120_grouped.so",
    source = "native/bind.cpp",
    exports = "native/exports.map",
    native_sources = ":native_sources",
    compiler = "//bazel:dsv4_host_compiler.sh",
    compiler_identity = "@dsv4_host_tools//:compiler.identity",
    cuda_headers = "@local_config_cuda//cuda:cuda-include",
    cudart = "@local_config_cuda//cuda:cudart",
    python_headers = "@local_config_python//:python_include",
    torch = "@pip_gpu_cuda13_torch_torch//:pkg",
    deep_gemm = "@pip_gpu_cuda13_torch_deep_gemm//:pkg",
    exec_compatible_with = ["@platforms//os:linux", "@platforms//cpu:x86_64"],
    target_compatible_with = _CUDA13,
)

py_library(
    name = "grouped_decode",
    srcs = glob(["*.py"]),
    data = [":native_sources", ":host"],
    deps = [
        "//rtp_llm:deep_gemm",
        "//rtp_llm:jit_cache_env",
        "//rtp_llm:torch",
    ],
    target_compatible_with = _CUDA13,
)
