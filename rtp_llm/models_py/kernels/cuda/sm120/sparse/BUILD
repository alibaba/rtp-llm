package(default_visibility = ["//rtp_llm:__subpackages__"])

filegroup(
    name = "native_sources",
    srcs = glob(["csrc/**", "include/**"]) + ["LICENSE", "SOURCE_PROVENANCE.json"],
)

py_library(
    name = "sparse",
    srcs = glob(["**/*.py"]),
    data = [":native_sources"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:jit_cache_env",
        "//rtp_llm:torch",
    ],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
