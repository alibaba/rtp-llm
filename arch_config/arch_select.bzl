load("@pip_arm_torch//:requirements.bzl", requirement_arm = "requirement")

# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu = "requirement")
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm = "requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9 = "requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12 = "requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_gpu_cuda13 = "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")
load("@pip_ppu_torch//:requirements.bzl", requirement_ppu = "requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")

# Endpoint and mirror seams, shared by open- and closed-source builds:
#   * every custom wheel below is served from the public rtp-opensource
#     bucket (rtp_llm/<arch>/); official torch wheels resolve from the
#     mirrors.aliyun.com pytorch mirror. Both are shared by open and closed
#     builds alike.
load("@internal_impls//:impls.bzl", "EMBEDDING_ARPC_IMPL", "SUBSCRIBE_IMPL")

# RDMA arch-selects live in the main repo's bazel package (see
# bazel/rdma_arch_select.bzl for why) and are re-exported here for the
# canonical @arch_config//:arch_select.bzl consumers. Both trees use this file
# verbatim; the internal overlay no longer shadows arch_config.
load("@rtp_llm//bazel:rdma_arch_select.bzl", _rdma_cache_store_deps = "cache_store_deps", _rdma_rdma_transport_deps = "rdma_transport_deps", _rdma_transfer_backend_deps = "transfer_backend_deps")

def cache_store_deps():
    _rdma_cache_store_deps()

def rdma_transport_deps():
    _rdma_rdma_transport_deps()

def transfer_backend_deps():
    _rdma_transfer_backend_deps()

def copy_all_so():
    copy_so("@rtp_llm//:mm_rdma_exporter")
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:th_grammar_tokenizer_info")
    copy_so("@rtp_llm//:rtp_compute_ops")

_CUDA13_X86_DEFERRED = ["flash-attn-3", "flashinfer-jit-cache"]
_CUDA13_ARM_DEFERRED = ["flash_attn", "flash-attn-3", "flashinfer-jit-cache"]

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
            native.py_library(
                name = name,
                deps = select({
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
                "@rtp_llm//:using_cuda12_arm": [requirement_cuda12_arm(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm("amd-aiter" if name == "aiter" else name)],
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "@rtp_llm//:using_ppu": ppu_deps,
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = EMBEDDING_ARPC_IMPL,
    )

def subscribe_deps():
    native.alias(
        name = "subscribe_deps",
        actual = select({
            "@rtp_llm//:cuda_pre_12_9": SUBSCRIBE_IMPL,
            "@rtp_llm//:using_cu12_9_or_13_x86": SUBSCRIBE_IMPL,
            "@rtp_llm//:using_rocm": SUBSCRIBE_IMPL,
            "//conditions:default": "@rtp_llm//rtp_llm/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl",
        }),
    )

def whl_deps():
    return select({
        "@rtp_llm//:using_cuda13_x86": [
            # torch 2.11+cu130 mirrored on OSS (artlab pytorch mirror has no cu130 yet,
            # source upstream is download.pytorch.org/whl/cu130).
            "torch==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            # GLM5 kernels shared by the CUDA13 inference paths.
            "deep-gemm@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.0+6d6aaad.cu132-cp310-cp310-linux_x86_64.whl",
            "flash-mla@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/flash_mla/flash_mla-1.0.0+49e2000.cu132-cp310-cp310-linux_x86_64.whl",
            "deep-ep==2.1.0+a56d615",
            "fast-hadamard-transform@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/whls/fast_hadamard/fast_hadamard_transform-1.1.0-cp310-cp310-linux_x86_64.whl",
            "flash_attn==2.8.3.post1+cu13torch2.11cxx11abitrue.r1",
            "flashinfer-python@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp-kernel/rtp_kernel_260608/flashinfer_python-0.6.12-py3-none-any.whl",
            "flashinfer-cubin@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp-kernel/rtp_kernel_260608/flashinfer_cubin-0.6.12-py3-none-any.whl",
            # Unified rtp_kernel from cuda_extension feat/port-nvfp4-xqa@94fd20eb;
            # NVFP4 mm only (no XQA wrappers).
            "rtp-kernel==0.1.0+94fd20eb.cu132",
            # fastsafetensors / fast_safetensors rebuilt against torch 2.11+cu130
            # Project name `fast-safetensors`; import name
            # `fast_safetensors`.  Required by --load_method fastsafetensors smokes.
            "fast-safetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fast-safetensors/fast_safetensors-0.7.4.dev0%2Btorch2.11.0.cu130.aone67401633-cp310-cp310-linux_x86_64.whl",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_x86_64.whl",
            "tilelang==0.1.15",
            "apache-tvm-ffi==0.1.12",
            # FlashAttention 3 remains deferred on CUDA13.
            "nvidia-cutlass-dsl==4.5.0",
            "nvidia-cutlass-dsl-libs-base==4.5.0",
        ],
        "@rtp_llm//:using_cuda13_arm": [
            "torch==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            # DeepGEMM opt_glm5 + SM12x, CI-built against torch 2.11+cu130.
            "deep_gemm==2.8.1+057ca59",
            "flash-mla==1.0.0+2e5429f",
            "flash-mla-legacy==1.0.0+92fd68b.rtp1",
            "deep-ep==2.1.0+a56d615",
            "flashinfer-python==0.6.9",
            "flashinfer-cubin==0.6.11",
            "rtp-kernel==0.1.0+94fd20eb.cu132",
            "fast-safetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fast-safetensors/fast_safetensors-0.7.4.dev0%2Btorch2.11.0.cu130.aone67401633-cp310-cp310-linux_aarch64.whl",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_aarch64.whl",
            "tilelang==0.1.15+cuda.gita35f8ddf",
            "apache-tvm-ffi==0.1.12",
            # FlashAttention wheels remain deferred on CUDA13 ARM.
            "fast-hadamard-transform@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/fast_hadamard_transform-1.0.4.post1-cp310-cp310-linux_aarch64.whl",
            "nvidia-cutlass-dsl==4.5.0",
            "nvidia-cutlass-dsl-libs-base==4.5.0",
        ],
        "@rtp_llm//:using_cuda12_9_x86": [
            # torch 2.8.0+cu129: the upstream wheel (http.bzl torch_2.8_py310_cuda
            # fetches the same file); the artlab pytorch mirror copy is the
            # closed-source equivalent.
            "torch==2.8.0+cu129",
            "torchvision==0.23.0+cu129",
            "flash_attn==2.8.3",
            "flash-attn-3==3.0.0b1",
            "fast-safetensors@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/fast_safetensors-0.7.4.dev0%2Btorch2.8.0.cu129.aone67401633-cp310-cp310-linux_x86_64.whl",
            # deep-ep follows the cuda12_9 lock pin (the internal d2fa38c build,
            # re-hosted on rtp-opensource rtp_llm/cu129/).
            "deep-ep==1.2.1.10.1+d2fa38c.cu129",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_x86_64.whl",
            "flashinfer-python==0.6.9+e87d610f",
            "flashinfer-cubin==0.6.9+e87d610f",
            "cuda-tile==1.4.0",
            "deep_gemm==2.1.1+ee8d9c7.cu129",
            "flash-mla==1.0.0+fcaae18.cu129",
            "fast-hadamard-transform==1.0.4.post1",
            "nvidia-cutlass-dsl==4.4.2",
            "nvidia-cutlass-dsl-libs-base==4.4.2",
            "rtp-kernel@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/rtp-kernel/rtp_kernel-0.1.0%2Bbdafb874.cu129-cp310-cp310-linux_x86_64.whl",
        ],
        "@rtp_llm//:cuda_pre_12_9": [
            # torch 2.6.0+cu126: the aliyun pytorch mirror wheel (same file http.bzl
            # torch_2.6_py310_cuda fetches).
            "torch==2.6.0+cu126",
            "torchvision==0.21.0+cu126",
            "flash_attn==2.7.4.post1",
            "fast-safetensors==0.7.3+torch2.1.2.cu121",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_x86_64.whl",
            "deep-ep==1.2.1+29d31c0",
            "deep_gemm==2.2.0+9b680f4",
            "flashinfer-python==0.6.0",
            "flashinfer-cubin==0.6.0",
            "apache-tvm-ffi==0.1.8.post2",
        ],
        "@rtp_llm//:using_cuda12_arm": [
            # Legacy ARM cu12 lane (cuda13_arm is the supported one).  torch and
            # torchvision resolve from upstream; the remaining kernels come from
            # the rtp-maga/rtp-opensource mirrors.
            "torch==2.9.0+cu129",
            "torchvision==0.24.0",
            "flash_attn==2.8.3",
            "flash-attn-3==3.0.0b1",
            "fast-safetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fast-safetensors/fast_safetensors-0.7.4.dev0%2Btorch2.9.0.cu129.aone67354779-cp310-cp310-linux_aarch64.whl",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_aarch64.whl",
            "flashinfer-python==0.6.9+e87d610f",
            "flashinfer-cubin==0.6.9+e87d610f",
            "deep_gemm==2.1.1+ee8d9c7.cu129",
            # No OSS mirror for the PAI aarch64 build; this pin mirrors the
            # cuda12_arm lock, which also points at the artlab path.
            "deep-ep==1.2.1.11+unknown.pai",
            "rtp-kernel@https://artlab.alibaba-inc.com/1/pypi/rtp_llm/rtp-kernel/rtp_kernel-0.1.0%2Bbdafb874.cu129-cp310-cp310-linux_aarch64.whl",
            "flash-mla==1.0.0+fcaae18.cu129",
            "fast-hadamard-transform==1.0.4.post1",
        ],
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/rocm/amd_smi.tar",
            # Keep the ROCm AITER/FlyDSL pins synchronized with
            # deps/requirements{,_lock}_rocm.txt.
            "amd-aiter==0.1.23.dev134+g8449f416b",
            "flydsl==0.3.2",
            "triton==3.8.0+amd.rocm7.2.0.git111ff227",
            "triton-kernels==1.0.0+amd.rocm7.2.0.git111ff227",
            "fast-safetensors==0.7.3+torch2.1.2.rocm",
            "fastsafetensors@https://artlab.alibaba-inc.com/1/pypi/simple/fastsafetensors/fastsafetensors-0.3.4.dev20260908%2Bali.fuseshm.gfb2c381b.aone70048596-cp310-cp310-linux_x86_64.whl",
        ],
        "@rtp_llm//:using_ppu": [
            "torch==2.11.0+cu130alios7u2oe",
            "torchvision==0.26.0+cu130torch2.11.0alios7u2ce",
            "triton==3.7.0+cu130alios7u2oe",
            "flash_attn==2.8.2+cu130torch2.11.0alios7u2oe",
            "flash-attn-3==2.8.2+cu130torch2.11.0alios7u2oe",
            "flashinfer-python==0.6.8.post1+cu130torch2.11.0alios7u2oe",
            "deep-ep==2.2.1+cu130torch2.11.0alios7u2oe",
            "deep-gemm==1.0.0+cu130torch2.11.0alios7u2oe",
            "flash-mla==2.0.0+cu130torch2.11.0alios7u2oe",
            "tilelang==0.1.13",
            "apache-tvm-ffi==0.1.12",
            "torch-c-dlpack-ext==0.1.5",
        ],  # PPU SDK 2.2.0 / cu130 / alios7u2 / cp312 wheels
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
            "xgrammar==0.2.5",
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

def internal_deps():
    return select({
        "@rtp_llm//:cuda_pre_12_9": ["@rtp_llm//rtp_llm:fast-safetensors", "@rtp_llm//rtp_llm:fastsafetensors"],
        "@rtp_llm//:using_cuda12_9_x86": ["@rtp_llm//rtp_llm:fast-safetensors", "@rtp_llm//rtp_llm:fastsafetensors"],
        "@rtp_llm//:using_cuda13_x86": ["@rtp_llm//rtp_llm:fast-safetensors", "@rtp_llm//rtp_llm:fastsafetensors"],
        "@rtp_llm//:using_cuda12_arm": ["@rtp_llm//rtp_llm:fast-safetensors", "@rtp_llm//rtp_llm:fastsafetensors"],
        "@rtp_llm//:using_cuda13_arm": ["@rtp_llm//rtp_llm:fast-safetensors", "@rtp_llm//rtp_llm:fastsafetensors"],
        "@rtp_llm//:using_rocm": [],
        "@rtp_llm//:using_ppu": ["@rtp_llm//rtp_llm:flashinfer-python", "@rtp_llm//rtp_llm:flash-attn-3", "@rtp_llm//rtp_llm:flash-mla"],  # PPU SDK 2.2.0 wheels
        "//conditions:default": [],
    })

def telemetry_test_deps():
    return select({
        "@rtp_llm//:using_cuda12_9_x86": [
            requirement_gpu_cuda12_9("opentelemetry-sdk"),
            requirement_gpu_cuda12_9("opentelemetry-exporter-otlp-proto-http"),
        ],
        "@rtp_llm//:using_cuda13_x86": [
            requirement_gpu_cuda13("opentelemetry-sdk"),
            requirement_gpu_cuda13("opentelemetry-exporter-otlp-proto-http"),
        ],
        "//conditions:default": [],
    })

def cuda13_x86_test_gpu():
    """Pool label for CUDA 13 x86 test execution (SM103 machine class)."""
    return "SM103_CU13"

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
        "@rtp_llm//:using_cuda13_arm": [
            "@torch_2.11_py310_cuda-aarch64//:torch_api",
            "@torch_2.11_py310_cuda-aarch64//:torch",
            "@torch_2.11_py310_cuda-aarch64//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_9_x86": [
            "@torch_2.8_py310_cuda//:torch_api",
            "@torch_2.8_py310_cuda//:torch",
            "@torch_2.8_py310_cuda//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_arm": [
            "@torch_2.9_py310_cuda-aarch64//:torch_api",
            "@torch_2.9_py310_cuda-aarch64//:torch",
            "@torch_2.9_py310_cuda-aarch64//:torch_libs",
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
        "@rtp_llm//:using_ppu": [requirement_ppu(name) for name in names],
        "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name) for name in names],
        "//conditions:default": [],
    })

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
