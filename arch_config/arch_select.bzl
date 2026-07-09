# to wrapper target relate with different system config
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")  # TODO(pip_unify): Remove after internal overlay migration
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")

def copy_all_so():
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:rtp_compute_ops")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_cuda12_arm": [requirement_cuda12_arm(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name)],
                # Default falls through to cuda12_9 (the canonical x86 GPU build).
                "//conditions:default": [requirement_gpu_cuda12_9(name)],
            }),
            visibility = ["//visibility:public"],
        )

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def transfer_rdma_deps():
    native.alias(
        name = "transfer_rdma_impl",
        actual = "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:no_rdma_impl",
    )

def transfer_backend_deps():
    native.alias(
        name = "transfer_backend_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_base_impl",
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "@rtp_llm//rtp_llm/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )

def subscribe_deps():
    native.alias(
        name = "subscribe_deps",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl"
    )

def whl_deps():
    return select({
        # Kept in sync with deps/requirements_rocm.txt and requirements_lock_rocm.txt:
        # amdsmi is served from the rtp-opensource bucket (not sinian-metrics), and
        # fast-safetensors/fastsafetensors/torch/torchvision/triton are pinned so the
        # ROCm wheel's install_requires matches the CI/lockfile dependency set.
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/amd-smi/amd_smi.tar",
            "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.17.dev79%2Bg2570b35f9.d20260623-cp310-cp310-linux_x86_64.whl",
            "triton@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl",
            "triton-kernels@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl",
            "fast-safetensors==0.7.3+torch2.1.2.rocm",
            "fastsafetensors==0.1.19rc5+ali",
            "torch==2.9.1+git7e1940d",
            "torchvision==0.24.0+rocm7.2.0.gitb919bd0c",
        ],
        # aarch64 (arm) is intentionally on torch 2.9.0 while x86 stays on 2.8.0.
        # Keep each in lock-step with its lockfile: 2.9.0 <-> requirements_lock_cuda12_arm.txt,
        # 2.8.0 <-> requirements_lock_torch_gpu_cuda12_9.txt. Bump both together only
        # after regenerating the corresponding lockfile.
        "@rtp_llm//:using_cuda12_arm": ["torch==2.9.0+cu129"],
        # Default covers x86 cuda12_9 only.
        "//conditions:default": ["torch==2.8.0+cu129"],
    })

def platform_deps():
    return select({
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0"],
        "//conditions:default": ["decord==0.6.0"],
    })

def torch_deps():
    deps = select({
        "@rtp_llm//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_arm": [
            "@torch_2.9_py310_cuda_arm//:torch_api",
            "@torch_2.9_py310_cuda_arm//:torch",
            "@torch_2.9_py310_cuda_arm//:torch_libs",
        ],
        "//conditions:default": [
            "@torch_2.8_py310_cuda//:torch_api",
            "@torch_2.8_py310_cuda//:torch",
            "@torch_2.8_py310_cuda//:torch_libs",
        ],
    })
    return deps

def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = "@flashinfer_cpp//:flashinfer"
    )

def flashmla_deps():
    native.alias(
        name = "flashmla",
        actual = "@flashmla//:flashmla"
    )

def deep_ep_py_deps():
    native.alias(
        name = "deep_ep_py",
        actual = "@rtp_llm//rtp_llm:empty_target",
    )

def cuda_register():
    native.alias(
        name = "cuda_register",
        actual = select({
            "//conditions:default": "@rtp_llm//rtp_llm/models_py/bindings/cuda/ops:gpu_register",
        }),
        visibility = ["//visibility:public"],
    )

def triton_deps(names):
    return select({
        "//conditions:default": [],
    })

def internal_deps():
    return []

def jit_deps():
    return []

def select_py_bindings():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:cuda_bindings_register"
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings/rocm:rocm_bindings_register"
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
        ],
    })

def no_block_copy_link_deps():
    """Deps for the cc_library that defines execNoBlockCopy / warmupNoBlockCopy (per device)."""
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:no_block_copy",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })
