load("@arch_config//:arch_select.bzl", "requirement")

package(default_visibility = ["//rtp_llm:__subpackages__"])

requirement(["grpcio", "protobuf"])

# Shared coordination and contracts do not depend on frontend or GPU adapters.
py_library(
    name = "core",
    srcs = [
        "__init__.py",
        "lease.py",
        "shutdown.py",
        "status.py",
        "timing.py",
        "validation.py",
    ],
)

py_library(
    name = "control",
    srcs = [
        "controller.py",
        "quiesce.py",
        "rpc.py",
    ],
    deps = [
        ":core",
        ":grpcio",
        ":protobuf",
        "//rtp_llm/aios/kmonitor:kmonitor_py",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        "//rtp_llm/metrics",
    ],
)

py_library(
    name = "resources",
    srcs = [
        "resources/__init__.py",
        "resources/gpu_mem_probe.py",
        "resources/nccl_memory.py",
        "resources/nccl_memory_backend.py",
        "resources/nccl_memory_policy.py",
        "resources/sleep_gpu_reclaim.py",
    ],
    deps = [
        ":core",
        "//rtp_llm:torch",
    ],
)
