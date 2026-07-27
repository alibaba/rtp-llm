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

sh_binary(
    name = "multicast_keeper",
    srcs = ["multicast_keeper.sh"],
    data = [
        ":keeper_lite_creator",
        ":keeper_lite_holder",
        ":mc_shim_unified.so",
    ],
)

cc_binary(
    name = "libcuda.so.1",
    srcs = ["tests/fake_cuda_driver.c"],
    linkopts = ["-Wl,-soname,libcuda.so.1"],
    linkshared = 1,
    testonly = 1,
)

py_test(
    name = "multicast_keeper_test",
    srcs = ["tests/multicast_keeper_test.py"],
    data = [
        ":keeper_lite_creator",
        ":keeper_lite_holder",
        ":libcuda.so.1",
        ":mc_shim_unified.so",
        ":multicast_keeper",
    ],
    python_version = "PY3",
    timeout = "short",
)

py_binary(
    name = "checkpoint_driver_action",
    srcs = ["tests/checkpoint_driver_action.py"],
    deps = ["//rtp_llm:utils"],
)

sh_test(
    name = "multicast_keeper_gpu_test",
    srcs = ["tests/run_multicast_checkpoint_gpu_test.sh"],
    args = [
        "$(location :keeper_lite_holder)",
        "$(location :keeper_lite_creator)",
        "$(location :mc_shim_unified.so)",
        "$(location tests/multicast_checkpoint_worker.py)",
        "$(location :checkpoint_driver_action)",
    ],
    data = [
        ":keeper_lite_creator",
        ":keeper_lite_holder",
        ":mc_shim_unified.so",
        ":checkpoint_driver_action",
        "tests/multicast_checkpoint_worker.py",
    ],
    tags = [
        "gpu",
        "manual",
    ],
    timeout = "long",
)

sh_binary(
    name = "multinode_multicast_keeper_test",
    srcs = ["tests/run_multinode_multicast_keeper_test.sh"],
    args = [
        "$(location :multicast_keeper)",
        "$(location :keeper_lite_holder)",
        "$(location :keeper_lite_creator)",
        "$(location :mc_shim_unified.so)",
        "$(location tests/multinode_multicast_worker.py)",
    ],
    data = [
        ":keeper_lite_creator",
        ":keeper_lite_holder",
        ":mc_shim_unified.so",
        ":multicast_keeper",
        "tests/multinode_multicast_worker.py",
    ],
)

exports_files([
    "README.md",
    "UPSTREAM.md",
])
