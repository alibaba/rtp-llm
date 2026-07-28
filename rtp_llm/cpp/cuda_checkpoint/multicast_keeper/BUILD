package(default_visibility = ["//visibility:public"])

cc_library(
    name = "keeper_protocol",
    hdrs = ["keeper_protocol.h"],
)

cc_binary(
    name = "keeper_lite_holder",
    srcs = ["keeper_lite_holder.c"],
    deps = [":keeper_protocol"],
)

cc_binary(
    name = "keeper_lite_creator",
    srcs = ["keeper_lite_creator.cc"],
    copts = ["-std=c++17"],
    deps = [
        ":keeper_protocol",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "mc_shim_unified.so",
    srcs = ["mc_shim_unified.c"],
    copts = ["-fPIC"],
    linkopts = [
        "-ldl",
        "-pthread",
    ],
    linkshared = 1,
    deps = [":keeper_protocol"],
)

# Production runtime payload. The full CUDA wheel consumes this filegroup and
# preserves these package-relative paths; the frontend-only wheel does not.
filegroup(
    name = "runtime_files",
    srcs = [
        ":keeper_lite_creator",
        ":keeper_lite_holder",
        ":mc_shim_unified.so",
    ],
)

exports_files([
    "README.md",
    "UPSTREAM.md",
])
