# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_gpu_cuda13="requirement")
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm="requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")
load("@rtp_llm//bazel:cuda13_packages.bzl", "CUDA13_UNAVAILABLE_REQUIREMENTS")

_PLATFORM_CONFIG_ERROR = (
    "No supported RTP-LLM platform configuration matched. Use an explicit " +
    "--config (cpu, arm, rocm, cuda12_2, cuda12_6, cuda12_9, " +
    "cuda12_9_arm, cuda13, or cuda13_arm); bare --define is unsupported."
)

# Packages available only on CUDA GPU platforms; CPU, ARM CPU, and ROCm
# branches intentionally use empty deps. CUDA 13 wheel availability belongs in
# bazel/cuda13_packages.bzl, not in this platform classification.
_GPU_ONLY_REQUIREMENTS = [
    "apache-tvm-ffi",
    "deep-ep",
    "deep-gemm",
    "fast-hadamard-transform",
    "flash-attn-3",
    "flash-mla",
    "flash-attn",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "rtp-kernel",
    "tilelang",
]

_ROCM_ONLY_REQUIREMENTS = [
    "aiter",
    "amdsmi",
    "pyrsmi",
    "triton-kernels",
]

_ARM_UNAVAILABLE_REQUIREMENTS = [
    "av",
    "decord",
    "xfastertransformer-devel",
    "xfastertransformer-devel-icx",
]

_CUDA12_ARM_UNAVAILABLE_REQUIREMENTS = [
    "apache-tvm-ffi",
    "deep-ep",
    "deep-gemm",
    "fast-safetensors",
    "fastsafetensors",
    "flash-attn",
    "flash-attn-3",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "nvidia-cutlass-dsl",
    "rtp-kernel",
    "tilelang",
]

def _normalize_requirement_name(name):
    """Normalize a package name following PEP 503 comparison rules."""
    normalized = name.lower().replace("_", "-").replace(".", "-")
    return "-".join([part for part in normalized.split("-") if part])

def copy_all_so():
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:rtp_compute_ops")

def requirement(names):
    for name in names:
        normalized_name = _normalize_requirement_name(name)
        is_rocm_only = normalized_name in _ROCM_ONLY_REQUIREMENTS
        is_arm_unavailable = is_rocm_only or normalized_name in _ARM_UNAVAILABLE_REQUIREMENTS
        cuda12_arm_deps = (
            []
            if is_arm_unavailable or normalized_name in _CUDA12_ARM_UNAVAILABLE_REQUIREMENTS
            else [requirement_cuda12_arm(name)]
        )
        cuda13_x86_deps = (
            []
            if is_rocm_only or normalized_name in CUDA13_UNAVAILABLE_REQUIREMENTS
            else [requirement_gpu_cuda13(name)]
        )
        cuda13_arm_deps = (
            []
            if is_arm_unavailable or normalized_name in CUDA13_UNAVAILABLE_REQUIREMENTS
            else [requirement_cuda13_arm(name)]
        )
        # cuda12 x86 (cuda_pre_12_9) and cuda12_9 x86 have no per-platform
        # "unavailable" table on purpose: they are the baseline GPU platforms
        # where every _GPU_ONLY_REQUIREMENTS wheel is qualified and present in
        # the lock. Per-platform unavailability is expressed only for the
        # constrained newer platforms (cuda12_arm via
        # _CUDA12_ARM_UNAVAILABLE_REQUIREMENTS, cuda13 via
        # CUDA13_UNAVAILABLE_REQUIREMENTS). Consequently a package genuinely
        # missing from a cuda12 x86 lock fails loudly at analysis ("no such
        # target"), NOT as a silent runtime ImportError; only rocm-only packages
        # intentionally degrade to empty deps here, as on every non-rocm branch.
        # If a wheel ever becomes unavailable specifically on cuda12 x86, add a
        # table here mirroring the ones above rather than relying on that gap.
        cuda12_x86_deps = [] if is_rocm_only else [requirement_gpu_cuda12(name)]
        cuda12_9_x86_deps = [] if is_rocm_only else [requirement_gpu_cuda12_9(name)]
        generic_deps = (
            []
            if normalized_name in _GPU_ONLY_REQUIREMENTS or is_rocm_only
            else [requirement_cpu(name)]
        )
        arm_deps = (
            []
            if normalized_name in _GPU_ONLY_REQUIREMENTS or is_arm_unavailable
            else [requirement_arm(name)]
        )
        rocm_deps = [] if normalized_name in _GPU_ONLY_REQUIREMENTS else [requirement_gpu_rocm(name)]
        # Intentionally no default: platform-dependent targets must use a
        # supported --config chain instead of silently importing CPU wheels.
        # PPU is internal-only; .internal_bazelrc replaces this arch_config.
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:cuda_pre_12_9": cuda12_x86_deps,
                "@rtp_llm//:using_cuda13_x86": cuda13_x86_deps,
                "@rtp_llm//:using_cuda12_9_x86": cuda12_9_x86_deps,
                "@rtp_llm//:using_cuda12_arm": cuda12_arm_deps,
                "@rtp_llm//:using_cuda13_arm": cuda13_arm_deps,
                "@rtp_llm//:using_rocm": rocm_deps,
                "@rtp_llm//:using_arm": arm_deps,
                "@rtp_llm//:using_cpu": generic_deps,
            }, no_match_error = _PLATFORM_CONFIG_ERROR),
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
    # Intentionally no default; a missing platform config must fail analysis.
    # This list intentionally describes dependencies embedded in the produced
    # wheel metadata; it is not a replacement for the complete pip resolver
    # inputs or their hashed locks. Keep overlapping CUDA 13 URLs synchronized
    # with requirements_torch_gpu_cuda13.txt, requirements_cuda13_arm.txt, and
    # their locks until a generated manifest can serve both representations.
    # bazel/check_cuda13_wheel_consistency.py enforces this for every package
    # that appears in both this table and the source requirements.
    #
    # nvidia-cutlass-dsl / apache-tvm-ffi / triton are deliberately NOT listed
    # here: they are pip resolver pins, not ABI-bound wheels shipped in the
    # wheel metadata. Their installed version is constrained by the source
    # requirements pin + hashed lock, and asserted at runtime by
    # //rtp_llm/models_py/standalone/test:cuda13_dependency_import_test against
    # CUDA13_EXPECTED_DEPENDENCY_VERSIONS in bazel/cuda13_packages.bzl.
    return select({
        "@rtp_llm//:using_cuda13_x86": [
            "torch@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "torchvision@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "deep_gemm@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/deep-gemm/deep_gemm-2.5.0%2B8a4dfba-cp310-cp310-linux_x86_64.whl",
            "deep_ep@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/deep-ep/deep_ep-1.2.1.12%2B37fda1c.base-cp310-cp310-linux_x86_64.whl",
            "flash-mla@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/flash-mla/flash_mla-1.0.0%2B9241ae3-cp310-cp310-linux_x86_64.whl",
            "rtp-kernel@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/rtp-kernel/rtp_kernel-0.1.0%2Bcu13.4a1a7e3-cp310-cp310-linux_x86_64.whl",
            "fast-safetensors@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/fast-safetensors/fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_x86_64.whl",
            "fastsafetensors@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/x86_64/fastsafetensors/fastsafetensors-0.1.20%2Bali-cp310-cp310-linux_x86_64.whl",
            "flashinfer-python==0.6.9",
            "tilelang==0.1.9",
        ],
        "@rtp_llm//:using_cuda13_arm": [
            "torch@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/arm_pkg/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "torchvision@https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/arm_pkg/torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "deep_gemm@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/deep-gemm/deep_gemm-2.5.0%2B8a4dfba-cp310-cp310-linux_aarch64.whl",
            "deep_ep@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/deep-ep/deep_ep-1.2.1.12%2B37fda1c.base-cp310-cp310-linux_aarch64.whl",
            "flash-mla@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/flash-mla/flash_mla-1.0.0%2B92fd68b-cp310-cp310-linux_aarch64.whl",
            "rtp-kernel@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/rtp-kernel/rtp_kernel-0.1.0-49c379b-cp310-cp310-linux_aarch64.whl",
            "fast-safetensors@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/fast-safetensors/fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_aarch64.whl",
            "fastsafetensors@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/fastsafetensors/fastsafetensors-0.1.20%2Bali-cp310-cp310-linux_aarch64.whl",
            "flashinfer-python==0.6.9",
            "tilelang@https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/cuda13/wheels/aarch64/tilelang/tilelang-0.1.9%2Bcuda.git441c3b06-cp38-abi3-linux_aarch64.whl",
        ],
        "@rtp_llm//:cuda_pre_12_9": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_cuda12_9_x86": ["torch==2.8.0+cu129"],
        "@rtp_llm//:using_cuda12_arm": [
            "torch@https://download.pytorch.org/whl/cu129/torch-2.9.0%2Bcu129-cp310-cp310-manylinux_2_28_aarch64.whl",
            "torchvision@https://download.pytorch.org/whl/cu128/torchvision-0.24.0-cp310-cp310-manylinux_2_28_aarch64.whl",
        ],
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar",
            "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.17.dev79%2Bg2570b35f9.d20260623-cp310-cp310-linux_x86_64.whl",
            "triton@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl",
            "triton-kernels@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl",
        ],
        "@rtp_llm//:using_arm": ["torch==2.1.2"],
        "@rtp_llm//:using_cpu": ["torch==2.1.2"],
    }, no_match_error = _PLATFORM_CONFIG_ERROR)

def platform_deps():
    # The default is intentional here: these optional media packages are used
    # by x86 CPU/CUDA builds and do not select an ABI-sensitive core runtime.
    # ARM and ROCm remain explicit because their package sets differ.
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda13_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0", "av==16.1.0"],
        "//conditions:default": ["decord==0.6.0", "av==16.1.0"],
    })

def torch_deps():
    # Intentionally no default; a missing platform config must fail analysis.
    # PPU is internal-only; .internal_bazelrc replaces this arch_config.
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
        "@rtp_llm//:using_cuda13_arm": [
            "@torch_2.11_py310_cuda_aarch64//:torch_api",
            "@torch_2.11_py310_cuda_aarch64//:torch",
            "@torch_2.11_py310_cuda_aarch64//:torch_libs",
        ],
        "@rtp_llm//:using_cuda12_arm": [
            "@torch_2.9_py310_cuda_aarch64//:torch_api",
            "@torch_2.9_py310_cuda_aarch64//:torch",
            "@torch_2.9_py310_cuda_aarch64//:torch_libs",
        ],
        "@rtp_llm//:using_cpu": [
            "@torch_2.1_py310_cpu//:torch_api",
            "@torch_2.1_py310_cpu//:torch",
            "@torch_2.1_py310_cpu//:torch_libs",
        ],
    }, no_match_error = _PLATFORM_CONFIG_ERROR)
    return deps

def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = select({
            "@rtp_llm//:using_cuda13_arm": "@flashinfer_cpp_cu13//:flashinfer",
            "@rtp_llm//:using_cuda13_x86": "@flashinfer_cpp_cu13//:flashinfer",
            "//conditions:default": "@flashinfer_cpp//:flashinfer",
        })
    )

def deep_ep_py_deps():
    native.alias(
        name = "deep_ep_py",
        actual = select({
            "@rtp_llm//:using_cuda13_x86": requirement_gpu_cuda13("deep-ep"),
            "@rtp_llm//:using_cuda13_arm": requirement_cuda13_arm("deep-ep"),
            "//conditions:default": "@rtp_llm//rtp_llm:empty_target",
        }),
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
