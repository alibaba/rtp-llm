# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@xpu_pip_gate//:requirements.bzl", requirement_xpu="requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")

# Packages not available in XPU pip environment (CUDA/ROCm-only).
# Names are PEP 503 normalized (lowercase, hyphens). See _normalize_pkg().
_XPU_EXCLUDED_PACKAGES = [
    "pynvml", "cpm-kernels", "xfastertransformer-devel",
    "xfastertransformer-devel-icx", "decord", "av", "onnx", "bitsandbytes",
    "pyrsmi", "amdsmi", "fast-safetensors", "fastsafetensors", "blobfile", "pyopenssl",
    "pyarrow", "pyodps", "matplotlib",
    # CUDA/ROCm GPU kernel packages — not available for XPU
    "apache-tvm-ffi", "flashinfer-python", "flashinfer-cubin",
    "nvidia-cutlass-dsl", "flashinfer-jit-cache",
    "fast-hadamard-transform", "flash-mla", "tilelang",
    "deep-gemm", "deep-ep", "rtp-kernel",
    "flash-attn", "flash-attn-3", "aiter",
    # ROCm-only triton kernels wheel; XPU ships only triton-xpu (see remap).
    "triton-kernels",
]

# Packages with different names in the XPU pip environment.
_XPU_PACKAGE_REMAP = {
    "triton": "triton-xpu",
}

# Version-pinned packages whose pins in _whl_reqs_static (CUDA/ROCm) differ
# from the versions in requirements_lock_xpu.txt.  The XPU wheel must
# advertise the locked versions so `pip install` does not conflict.
_XPU_VERSION_OVERRIDES = {
    "fastapi": "fastapi==0.138.2",
    "grpcio": "grpcio==1.78.0",
    "grpcio-tools": "grpcio-tools==1.78.0",
    "protobuf": "protobuf==6.33.6",
    "sentencepiece": "sentencepiece==0.2.1",
    "setuptools": "setuptools==81.0.0",
    "tiktoken": "tiktoken==0.13.0",
    "timm": "timm==1.0.27",
    "uvicorn": "uvicorn==0.49.0",
}

def copy_all_so():
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:rtp_compute_ops")

def _normalize_pkg(name):
    """Normalize package name per PEP 503 (lowercase, replace [-_.] with -)."""
    return name.lower().replace("_", "-").replace(".", "-")

def requirement(names):
    for name in names:
        normalized = _normalize_pkg(name)
        if normalized in _XPU_EXCLUDED_PACKAGES:
            xpu_dep = []
        elif normalized in _XPU_PACKAGE_REMAP:
            xpu_dep = [requirement_xpu(_XPU_PACKAGE_REMAP[normalized])]
        else:
            xpu_dep = [requirement_xpu(name)]
        native.py_library(
            name = name,
            deps = select({
                "@rtp_llm//:cuda_pre_12_9": [requirement_gpu_cuda12(name)],
                "@rtp_llm//:using_cuda12_9_x86": [requirement_gpu_cuda12_9(name)],
                "@rtp_llm//:using_rocm": [requirement_gpu_rocm(name)],
                "@rtp_llm//:using_xpu": xpu_dep,
                "@rtp_llm//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def _strip_req_version(req):
    """Extract the bare package name from a PEP 508 requirement string."""
    name = req
    for sep in ["==", ">=", "<=", "~=", "!=", ">", "<", "@", "[", " ", ";"]:
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip()

def filter_xpu_whl_reqs(reqs):
    """Return a select() that strips CUDA/ROCm-only packages from wheel
    metadata when building for XPU and applies the XPU package remap, so the
    XPU wheel does not declare dependencies unavailable for that platform."""
    xpu_reqs = []
    for req in reqs:
        normalized = _normalize_pkg(_strip_req_version(req))
        if normalized in _XPU_EXCLUDED_PACKAGES:
            continue
        elif normalized in _XPU_VERSION_OVERRIDES:
            xpu_reqs.append(_XPU_VERSION_OVERRIDES[normalized])
        elif normalized in _XPU_PACKAGE_REMAP:
            xpu_reqs.append(_XPU_PACKAGE_REMAP[normalized])
        else:
            xpu_reqs.append(req)
    return select({
        "@rtp_llm//:using_xpu": xpu_reqs,
        "//conditions:default": reqs,
    })

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
        "@rtp_llm//:using_cuda12": ["torch==2.6.0+cu126"],
        "@rtp_llm//:using_rocm": [
            "pyrsmi==0.2.0",
            "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar",
            "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.17.dev79%2Bg2570b35f9.d20260623-cp310-cp310-linux_x86_64.whl",
            "triton@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl",
            "triton-kernels@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl",
        ],
        "@rtp_llm//:using_xpu": ["torch==2.12.0+xpu", "torchvision==0.27.0+xpu"],
        "//conditions:default": ["torch==2.1.2"],
    })

def platform_deps():
    return select({
        "@rtp_llm//:using_arm": [],
        "@rtp_llm//:using_cuda12_arm": [],
        "@rtp_llm//:using_rocm": ["pyyaml==6.0.2","decord==0.6.0", "av==16.1.0"],
        "@rtp_llm//:using_xpu": [],
        "//conditions:default": ["decord==0.6.0", "av==16.1.0"],
    })

def torch_deps():
    deps = select({
        "@rtp_llm//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@rtp_llm//:using_xpu": [
            "@torch_xpu//:torch_api",
            "@torch_xpu//:torch",
            "@torch_xpu//:torch_libs",
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
            "@rtp_llm//:using_xpu": "@rtp_llm//rtp_llm:empty_target",
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
        "@rtp_llm//:using_xpu": [
            "@rtp_llm//rtp_llm/models_py/bindings/xpu:xpu_bindings_register",
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
        "@rtp_llm//:using_xpu": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })
