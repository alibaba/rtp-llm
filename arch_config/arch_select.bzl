# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_gpu_cuda13="requirement")
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
                "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12(name)],
                "@rtp_llm//:using_cuda13_x86": [requirement_gpu_cuda13(name)],
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name)],
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
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
        "@rtp_llm//:using_cuda13_x86": [
            "torch@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "torchvision@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "deep_gemm@http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/deep_gemm/cuda13_b200/4af4ac732eae77acb57ab3ac59e3ceb796b797b5/deep_gemm-2.5.0%2Blocal-cp310-cp310-linux_x86_64.whl",
            "flash-mla@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/flash_mla-1.0.0%2B9241ae3-cp310-cp310-linux_x86_64.whl",
            "rtp-kernel@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/rtp_kernel-0.1.0%2Bcu13.4a1a7e3-cp310-cp310-linux_x86_64.whl",
            "fast-safetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0507/fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_x86_64.whl",
            "fastsafetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0502/fastsafetensors-0.1.20%2Bali-cp310-cp310-linux_x86_64.whl",
            "tilelang==0.1.9",
        ],
        "@rtp_llm//:using_cuda12": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar",
            "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.17.dev79%2Bg2570b35f9.d20260623-cp310-cp310-linux_x86_64.whl",
            "triton@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl",
            "triton-kernels@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl",
        ],
        "//conditions:default": ["torch==2.1.2"],
    })

def platform_deps():
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0", "av==16.1.0"],
        "//conditions:default": ["decord==0.6.0", "av==16.1.0"],
    })

def torch_deps():
    deps = select({
        "@rtp_llm//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@rtp_llm//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api",
            "@torch_2.3_py310_cpu_aarch64//:torch",
            "@torch_2.3_py310_cpu_aarch64//:torch_libs",
        ],
        "@rtp_llm//:cuda_pre_12_9": [
            "@torch_2.6_py310_cuda//:torch_api",
            "@torch_2.6_py310_cuda//:torch",
            "@torch_2.6_py310_cuda//:torch_libs",
        ],
        "@rtp_llm//:using_cuda13_x86": [
            "@torch_2.11_py310_cuda//:torch_api",
            "@torch_2.11_py310_cuda//:torch",
            "@torch_2.11_py310_cuda//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_9_x86": [
            "@torch_2.8_py310_cuda//:torch_api",
            "@torch_2.8_py310_cuda//:torch",
            "@torch_2.8_py310_cuda//:torch_libs",
        ],
        "//conditions:default": [
            "@torch_2.1_py310_cpu//:torch_api",
            "@torch_2.1_py310_cpu//:torch",
            "@torch_2.1_py310_cpu//:torch_libs",
        ]
    })
    return deps

def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = select({
            "@rtp_llm//:using_cuda13_x86": "@flashinfer_cpp_cu13//:flashinfer",
            "//conditions:default": "@flashinfer_cpp//:flashinfer",
        })
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
