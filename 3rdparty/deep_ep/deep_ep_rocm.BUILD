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
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so $(location libdeep_ep_rocm.so)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/config_hip.hpp $(location csrc/config_hip.hpp)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/deep_ep_hip.hpp $(location csrc/deep_ep_hip.hpp)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/event_hip.hpp $(location csrc/event_hip.hpp)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/kernels/api_hip.cuh $(location csrc/kernels/api_hip.cuh)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/kernels/configs_hip.cuh $(location csrc/kernels/configs_hip.cuh)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/kernels/exception_hip.cuh $(location csrc/kernels/exception_hip.cuh)
        cp -f /opt/conda310/lib/python3.10/site-packages/deep_ep/include/kernels/launch_hip.cuh $(location csrc/kernels/launch_hip.cuh)
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
