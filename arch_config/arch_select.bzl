# Wrapper targets for different system configs.
# Python deps are managed by pip/pyproject.toml; Bazel only keeps C++ platform selections here.

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def rdma_transport_deps():
    # Open-source builds expose the same factory API but have no RDMA provider.
    native.alias(
        name = "rdma_transport_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/rdma_transport:rdma_transport_no_impl",
        visibility = ["//visibility:public"],
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
            # Keep the ROCm AITER/FlyDSL pins synchronized with
            # deps/requirements{,_lock}_rocm.txt. AMD SMI itself is supplied
            # by the ROCm runtime; its legacy bare .tar URL is not a valid
            # Python-native wheel dependency.
            "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.21.dev80%2Bg987203ba5.d20260825-cp310-cp310-linux_x86_64.whl",
            "flydsl==0.3.1",
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
def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = select({
            "@rtp_llm//:using_cuda13_x86": "@flashinfer_cpp_cu13//:flashinfer",
            "//conditions:default": "@flashinfer_cpp//:flashinfer",
        })
    )

def cuda_register():
    native.alias(
        name = "cuda_register",
        actual = select({
            "//conditions:default": "@rtp_llm//rtp_llm/models_py/bindings/cuda/ops:gpu_register",
        }),
        visibility = ["//visibility:public"],
    )

def triton_deps(names = []):
    return select({
        "//conditions:default": [],
    })

def internal_deps():
    return []

def telemetry_test_deps():
    # The tracing SDK is optional at runtime. SDK-specific test methods skip
    # explicitly when it is unavailable, while Trace-off tests remain independent
    # of interpreter-wide packages. The lock carrying the SDK supplies it through
    # the architecture-specific dependency selector.
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

def cuda13_test_exec_properties(gpu_count = 1):
    """GPU test pools; internal builds override CUDA13 pool names."""
    return select({
        "@rtp_llm//:using_cuda13_arm": {"gpu": "SM100_ARM", "gpu_count": str(gpu_count)},
        "@rtp_llm//:using_cuda13_x86": {"gpu": "SM100_X86", "gpu_count": str(gpu_count)},
        "@rtp_llm//:using_cuda12_arm": {"gpu": "SM100_ARM", "gpu_count": str(gpu_count)},
        "//conditions:default": {"gpu": "A10", "gpu_count": str(gpu_count)},
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

def torch_deps():
    """Torch cc deps; same as //bazel:defs.bzl."""
    return [
        "@torch//:torch_api",
        "@torch//:torch",
        "@torch//:torch_libs",
    ]

# ---------------------------------------------------------------------------
# Compatibility shim for legacy BUILD files that still call requirement().
# Python packages are now managed by pip/pyproject.toml; this function creates
# empty placeholder targets so Bazel package loading does not break while
# callers are migrated.
# ---------------------------------------------------------------------------

def requirement(packages):
    """Create empty py_library aliases for each pip package name.

    Callers reference them as ':<package>' in deps.  The actual packages are
    installed via pip; these targets only exist to satisfy Bazel's loading
    phase.
    """
    for pkg in packages:
        safe_name = pkg.replace("-", "_").replace(".", "_")
        native.py_library(
            name = safe_name,
            srcs = [],
            visibility = ["//visibility:public"],
        )
