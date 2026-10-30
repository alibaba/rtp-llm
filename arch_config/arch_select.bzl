load("@pip_arm_torch//:requirements.bzl", requirement_arm = "requirement")

# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm = "requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9 = "requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12 = "requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_gpu_cuda13 = "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")
load("@pip_ppu_torch//:requirements.bzl", requirement_ppu = "requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")

def copy_all_so():
    copy_so("@rtp_llm//:mm_rdma_exporter")
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:th_grammar_tokenizer_info")
    copy_so("@rtp_llm//:rtp_compute_ops")

_CUDA13_X86_DEFERRED = ["flash-attn-3"]
_CUDA13_ARM_DEFERRED = ["flash_attn", "flash-attn-3"]

# xgrammar's wheel metadata pulls apache-tvm-ffi (and triton on x86), which only
# the cuda12_9/cuda13 locks carry; dash_sc imports it optionally and degrades
# gracefully, so the other platforms resolve it to nothing.
_DSV4_PLATFORM_ONLY = ["xgrammar"]

# Names with no PPU wheel in the vendor lock (CUDA/ROCm-only deps, or no
# cp312 wheel). requirement() resolves them to an empty library on PPU.
_PPU_LOCK_SKIPPED = [
    "aiter",
    "amdsmi",
    "decord",
    "fast-safetensors",
    "fastsafetensors",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "nvidia-cutlass-dsl",
    "pyrsmi",
    "rtp-kernel",
    "tilelang",
    "triton-kernels",
    "xfastertransformer_devel",
    "xfastertransformer_devel_icx",
    "xgrammar",
]

def requirement(names):
    for name in names:
        if name == "flash-mla-legacy":
            # Compatibility label: current CUDA13 FlashMLA includes legacy ops.
            native.py_library(
                name = name,
                deps = select({
                    "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm("flash-mla")],
                    "//conditions:default": [],
                }),
                visibility = ["//visibility:public"],
            )
            continue
        if name == "tokenspeed-mla":
            native.py_library(
                name = name,
                deps = select({
                    "@rtp_llm//:using_cuda13_x86": [requirement_gpu_cuda13(name)],
                    "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm(name)],
                    "//conditions:default": [],
                }),
                visibility = ["//visibility:public"],
            )
            continue
        cuda13_x86_deps = [] if name in _CUDA13_X86_DEFERRED else [requirement_gpu_cuda13(name)]
        cuda13_arm_deps = [] if name in _CUDA13_ARM_DEFERRED else [requirement_cuda13_arm(name)]
        ppu_deps = [] if name in _PPU_LOCK_SKIPPED else [requirement_ppu(name)]
        if name in _DSV4_PLATFORM_ONLY:
            native.py_library(
                name = name,
                deps = select({
                    "@rtp_llm//:using_cuda13_x86": cuda13_x86_deps,
                    "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                    "@rtp_llm//:using_cuda13_arm": cuda13_arm_deps,
                    "//conditions:default": [],
                }),
                visibility = ["//visibility:public"],
            )
            continue
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12(name)],
                "@rtp_llm//:using_cuda13_x86": cuda13_x86_deps,
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_cuda13_arm": cuda13_arm_deps,
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm("amd-aiter" if name == "aiter" else name)],
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl",
    )

def rdma_transport_deps():
    # Open-source builds expose the same factory API but have no RDMA provider.
    native.alias(
        name = "rdma_transport_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/rdma_transport:rdma_transport_no_impl",
        visibility = ["//visibility:public"],
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "@rtp_llm//rtp_llm/cpp/embedding_engine:embedding_engine_arpc_server_impl",
    )

def subscribe_deps():
    native.alias(
        name = "subscribe_deps",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl",
    )

def whl_deps():
    return select({
        "@rtp_llm//:using_cuda13_x86": [
            "torch@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "torchvision@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            # DeepGEMM 98a672e for CUDA 13.2.
            "deep_gemm@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.2+98a672e.cu132-cp310-cp310-linux_x86_64.whl",
            "flash-mla@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/flash_mla/flash_mla-1.0.0%2B3ca6856.cu132-cp310-cp310-linux_x86_64.whl#sha256=53dc99022a69667ad20059809db1e91b7b5f8546f95d4d613cea173332deaf58",
            "deep-ep@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep-ep/deep_ep-2.1.0%2Ba56d615-cp310-cp310-linux_x86_64.whl",
            "fast-hadamard-transform@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/fast-hadamard-transform/fast_hadamard_transform-1.1.0%2Be7706fa.cu132.torch2.11.cxx11abitrue-cp310-cp310-linux_x86_64.whl",
            "flash_attn@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/flash-attn/flash_attn-2.8.3.post1%2Bcu13torch2.11cxx11abitrue.r1-cp310-cp310-linux_x86_64.whl",
            "flashinfer-python@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/flashinfer-python/flashinfer_python-0.6.9+8c4f4dcf-py3-none-any.whl",
            "flashinfer-cubin@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/flashinfer-cubin/flashinfer_cubin-0.6.9+8c4f4dcf-py3-none-any.whl",
            "rtp-kernel@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/rtp-kernel/rtp_kernel-0.1.0%2B94fd20eb.cu132-cp310-cp310-linux_x86_64.whl",
            "fast-safetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0507/fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_x86_64.whl",
            "fastsafetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0502/fastsafetensors-0.1.20%2Bali-cp310-cp310-linux_x86_64.whl",
            "tilelang==0.1.15",
            "apache-tvm-ffi==0.1.12",
            "tokenspeed-mla@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/tokenspeed-mla/tokenspeed_mla-0.2.11+8b513a59.cu132-py3-none-any.whl#sha256=d66de94e5072c4e509d9b83b115616ddbc91bc1f50a6fb438deb5dfeecf089c3",
            "nvidia-cutlass-dsl==4.5.3",
        ],
        "@rtp_llm//:using_cuda13_arm": [
            "torch@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/arm_pkg/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "torchvision@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/arm_pkg/torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "deep_gemm@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.2+98a672e.cu132-cp310-cp310-linux_aarch64.whl",
            "flash-mla@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/flash_mla/flash_mla-1.0.0%2B3ca6856.cu132-cp310-cp310-linux_aarch64.whl#sha256=d960b9b2ae558f7c088af277657dc8814a8802dee1df5b1eca24573490a2f94d",
            "deep-ep@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep-ep/deep_ep-2.1.0%2Ba56d615-2-cp310-cp310-linux_aarch64.whl",
            "flashinfer-python@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/flashinfer-python/flashinfer_python-0.6.9+8c4f4dcf-py3-none-any.whl",
            "flashinfer-cubin@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/flashinfer-cubin/flashinfer_cubin-0.6.9+8c4f4dcf-py3-none-any.whl",
            "rtp-kernel@http://artlab.alibaba-inc.com/1/pypi/rtp_llm/rtp-kernel/rtp_kernel-0.1.0%2B94fd20eb.cu132-cp310-cp310-linux_aarch64.whl",
            "fast-safetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0513/arm_pkg/fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_aarch64.whl",
            "fastsafetensors@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0513/arm_pkg/fastsafetensors-0.1.20%2Bali-cp310-cp310-linux_aarch64.whl",
            "tilelang@https://search-ad.oss-cn-hangzhou-zmf-internal.aliyuncs.com/rtp-llm-deps/dsv41/20261009/tilelang-0.1.15%2Bcuda.gita35f8ddf-cp39-abi3-linux_aarch64.whl",
            "apache-tvm-ffi==0.1.12",
            "tokenspeed-mla@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/tokenspeed-mla/tokenspeed_mla-0.2.11+8b513a59.cu132-py3-none-any.whl#sha256=d66de94e5072c4e509d9b83b115616ddbc91bc1f50a6fb438deb5dfeecf089c3",
            "nvidia-cutlass-dsl==4.5.3",
        ],
        "@rtp_llm//:using_cuda12": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar",
            # Keep the ROCm AITER/FlyDSL pins synchronized with
            # deps/requirements{,_lock}_rocm.txt.
            "amd-aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/amd_aiter-0.1.23.dev134%2Bg8449f416b-cp310-cp310-linux_x86_64.whl",
            "flydsl==0.3.2",
            "triton@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.8.0%2Bamd.rocm7.2.0.git111ff227-cp310-cp310-linux_x86_64.whl",
            "triton-kernels@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.git111ff227-py3-none-any.whl",
        ],
        "//conditions:default": ["torch==2.1.2"],
    })

def ml_pin_reqs():
    """Wheel pins that diverge between the PPU vendor lock and the CUDA locks.

    The PPU lock (deps/requirements_lock_torch_ppu.txt) pins the transformers
    4.57.1 family and carries no xgrammar (its wheel chain is CUDA-only; the
    PPU bazel deps exclude it the same way). Keep wheel metadata aligned with
    the lock of the environment the wheel is installed into.
    """
    return select({
        "@rtp_llm//:using_ppu": [
            "transformers==4.57.1",
            "sentence-transformers==2.7.0",
            "pyarrow==17.0.0",
        ],
        "//conditions:default": [
            "transformers==5.2.0",
            "sentence-transformers==5.2.0",
            "pyarrow==14.0.0",
            "xgrammar==0.2.8",
        ],
    })

def platform_deps():
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_cuda13_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2", "decord==0.6.0", "av==16.1.0"],
        # decord has no cp312 wheel; PPU video input is unsupported.
        "@rtp_llm//:using_ppu": ["av==16.1.0"],
        "//conditions:default": ["decord==0.6.0", "av==16.1.0"],
    })

def cuda13_x86_test_gpu():
    """Pool label for CUDA 13 x86 test execution.

    Returned through @arch_config so builds can resolve their own pool
    label; the default value names the SM103 machine class only.
    """
    return "SM103_TEST"

def torch_deps():
    deps = select({
        "@rtp_llm//:using_ppu": ["@torch_2.11_py312_ppu//:torch_api", "@torch_2.11_py312_ppu//:torch", "@torch_2.11_py312_ppu//:torch_libs"],
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
        "@rtp_llm//:using_cuda13_arm": [
            "@torch_2.11_py310_cuda-aarch64//:torch_api",
            "@torch_2.11_py310_cuda-aarch64//:torch",
            "@torch_2.11_py310_cuda-aarch64//:torch_libs",
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
        ],
    })
    return deps

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
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:cuda_bindings_register",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings/rocm:rocm_bindings_register",
        ],
        "@rtp_llm//:using_ppu": [
            "@rtp_llm//rtp_llm/models_py/bindings/ppu:ppu_bindings_register",
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
        "@rtp_llm//:using_ppu": [
            "@rtp_llm//rtp_llm/models_py/bindings/ppu:no_block_copy",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })

def transfer_backend_deps():
    native.alias(
        name = "transfer_backend_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_base_impl",
    )
