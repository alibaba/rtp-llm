load("@//:def.bzl", "copts", "rocm_copts")
load("@//bazel:arch_select.bzl", "torch_deps")

genrule(
    name = "cpp_libraries",
    srcs = glob(include = ["**/*"], exclude = ["**/*_hip.*"]),
    outs = [
        "libdeep_ep_rocm.so",
        "csrc/config_hip.hpp",
        "csrc/deep_ep_hip.hpp",
        "csrc/event_hip.hpp",
        "csrc/kernels/exception_hip.cuh",
        "csrc/kernels/configs_hip.cuh",
        "csrc/kernels/api_hip.cuh",
        "csrc/kernels/launch_hip.cuh"],
    cmd = """
    if test -f "external/deep_ep_rocm/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so"; then
        cp external/deep_ep_rocm/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so $(location libdeep_ep_rocm.so)
        cp external/deep_ep_rocm/csrc/config_hip.hpp $(location csrc/config_hip.hpp)
        cp external/deep_ep_rocm/csrc/deep_ep_hip.hpp $(location csrc/deep_ep_hip.hpp)
        cp external/deep_ep_rocm/csrc/event_hip.hpp $(location csrc/event_hip.hpp)
        cp external/deep_ep_rocm/csrc/kernels/exception_hip.cuh $(location csrc/kernels/exception_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/configs_hip.cuh $(location csrc/kernels/configs_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/api_hip.cuh $(location csrc/kernels/api_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/launch_hip.cuh $(location csrc/kernels/launch_hip.cuh)
    else
        cd external/deep_ep_rocm
        LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib/llvm/lib/:$$LD_LIBRARY_PATH LIBRARY_PATH=/usr/local/lib/ AITER_MOE=1 ROCM_HOME=/opt/rocm/ OMPI_DIR=/lib ROCSHMEM_DIR=/opt/nvshmem \\
        /opt/conda310/bin/python3 setup.py --variance rocm build develop --force-nvshmem-api
        cd ../..
        cp external/deep_ep_rocm/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so $(location libdeep_ep_rocm.so)
        cp external/deep_ep_rocm/csrc/config_hip.hpp $(location csrc/config_hip.hpp)
        cp external/deep_ep_rocm/csrc/deep_ep_hip.hpp $(location csrc/deep_ep_hip.hpp)
        cp external/deep_ep_rocm/csrc/event_hip.hpp $(location csrc/event_hip.hpp)
        cp external/deep_ep_rocm/csrc/kernels/exception_hip.cuh $(location csrc/kernels/exception_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/configs_hip.cuh $(location csrc/kernels/configs_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/api_hip.cuh $(location csrc/kernels/api_hip.cuh)
        cp external/deep_ep_rocm/csrc/kernels/launch_hip.cuh $(location csrc/kernels/launch_hip.cuh)
    fi
    """,
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "deep_ep",
    srcs = ["libdeep_ep_rocm.so"],
    hdrs = [
        ":csrc/config_hip.hpp", 
        ":csrc/deep_ep_hip.hpp", 
        ":csrc/event_hip.hpp", 
        ":csrc/kernels/exception_hip.cuh",
        ":csrc/kernels/configs_hip.cuh",
        ":csrc/kernels/api_hip.cuh",
        ":csrc/kernels/launch_hip.cuh"],
    deps = [":cpp_libraries"] + torch_deps(),
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)
