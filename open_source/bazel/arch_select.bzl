# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("//bazel:defs.bzl", "copy_so", "copy_so_inst")
load("//rtp_llm/cpp/cuda/deep_gemm:template.bzl", "dpsk_gemm_so_num", "qwen_gemm_so_num")

def copy_all_so():
    copy_so("//:th_transformer")
    copy_so("//:th_transformer_config")
    copy_so("//:rtp_compute_ops")
    copy_so("//rtp_llm/cpp/kernels/decoder_masked_multihead_attention:mmha1")
    copy_so("//rtp_llm/cpp/kernels/decoder_masked_multihead_attention:mmha2")
    copy_so("//rtp_llm/cpp/kernels/decoder_masked_multihead_attention:dmmha")
    copy_so("//rtp_llm/cpp/cuda:fa")
    copy_so("//rtp_llm/cpp/cuda/cutlass:fpA_intB")
    copy_so("//rtp_llm/cpp/cuda/cutlass:moe")
    copy_so("//rtp_llm/cpp/cuda/cutlass:moe_sm90")
    copy_so("//rtp_llm/cpp/cuda/cutlass:int8_gemm")
    copy_so("@flashinfer//:flashinfer_single_prefill")
    copy_so("@flashinfer//:flashinfer_single_decode")
    copy_so("@flashinfer//:flashinfer_batch_paged_prefill")
    copy_so("@flashinfer//:flashinfer_batch_paged_decode")
    copy_so("@flashinfer//:flashinfer_batch_ragged_prefill")
    # num of so
    copy_so_inst("//rtp_llm/cpp/cuda/deep_gemm:deepgemm_dpsk", dpsk_gemm_so_num)
    copy_so_inst("//rtp_llm/cpp/cuda/deep_gemm:deepgemm_qwen", qwen_gemm_so_num)
    copy_so("@flashinfer//:flashinfer_sm90")
    copy_so("@deep_ep//:deep_ep_cu")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "@//:using_cuda12": [requirement_gpu_cuda12(name)],
                "@//:using_rocm": [requirement_gpu_rocm(name)],
                "@//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "//rtp_llm/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )

def subscribe_deps():
    native.alias(
        name = "subscribe_deps",
        actual = "//rtp_llm/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl"
    )

def whl_deps():
    return select({
        "@//:using_cuda12": ["torch==2.6.0+cu126"],
        "@//:using_rocm": ["pyrsmi", "amdsmi@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis%2FAMD%2Famd_smi%2Fali%2Famd_smi.tar", "aiter@https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/RTP/aiter-0.1.0%2Bgit.94934e7d.date.202508252339-py3-none-any.whl"],
        "//conditions:default": ["torch==2.1.2"],
    })

def platform_deps():
    return select({
        "@//:using_arm": [],
        "@//:using_rocm": ["pyyaml","decord==0.6.0"],
        "//conditions:default": ["decord==0.6.0"],
    })

def torch_deps():
    deps = select({
        "@//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api",
            "@torch_2.3_py310_cpu_aarch64//:torch",
            "@torch_2.3_py310_cpu_aarch64//:torch_libs",
        ],
        "@//:using_cuda": [
            "@torch_2.6_py310_cuda//:torch_api",
            "@torch_2.6_py310_cuda//:torch",
            "@torch_2.6_py310_cuda//:torch_libs",
        ],
        "//conditions:default": [
            "@torch_2.1_py310_cpu//:torch_api",
            "@torch_2.1_py310_cpu//:torch",
            "@torch_2.1_py310_cpu//:torch_libs",
        ]
    })
    return deps

def fa_deps():
    native.alias(
        name = "fa",
        actual = "@flash_attention//:fa"
    )

    native.alias(
        name = "fa_hdrs",
        actual = "@flash_attention//:fa_hdrs",
    )

def flashinfer_deps():
    native.alias(
        name = "flashinfer",
        actual = "@flashinfer//:flashinfer"
    )

def flashmla_deps():
    native.alias(
        name = "flashmla",
        actual = "@flashmla//:flashmla"
    )

def deep_ep_deps():
    native.alias(
        name = "deep_ep",
        actual = "@deep_ep//:deep_ep"
    )

def deep_ep_py_deps():
    native.alias(
        name = "deep_ep_py",
        actual = "//rtp_llm:empty_target",
    )

def deep_gemm_deps():
    native.alias(
        name = "deep_gemm",
        actual = "@deep_gemm_ext//:deep_gemm",
        visibility = ["//visibility:public"],
    )

def kernel_so_deps():
    return select({
        "@//:using_cuda": [":libmmha1_so", ":libmmha2_so", ":libdmmha_so", ":libfa_so", ":libfpA_intB_so", ":libint8_gemm_so", ":libmoe_so", ":libmoe_sm90_so", ":libflashinfer_single_prefill_so", ":libflashinfer_single_decode_so", ":libflashinfer_batch_paged_prefill_so", ":libflashinfer_batch_paged_decode_so", ":libflashinfer_batch_ragged_prefill_so", ":libflashinfer_sm90_so", ":libdeepgemm_dpsk_inst_so", ":libdeepgemm_qwen_inst_so"],
        "@//:using_rocm": [":libmmha1_so", ":libmmha2_so", ":libdmmha_so"],
        "//conditions:default":[],
    })

def arpc_deps():
    native.cc_library(
        name = ""
    )

def trt_plugins():
    native.alias(
        name = "trt_plugins",
        actual = "//rtp_llm/cpp/cuda/nv_trt_plugins:nv_trt_plugins",
    )

def cuda_register():
    native.alias(
        name = "cuda_register",
        actual = select({
            "//conditions:default": "//rtp_llm/cpp/devices/cuda_impl:gpu_register",
        })
    )

def triton_deps(names):
    return select({
        "//conditions:default": [],
    })

def internal_deps():
    return []

def jit_deps():
    return select({
        "//:using_cuda": ["//rtp_llm/cpp/cuda/deep_gemm:jit_includes"],
        "//conditions:default": [],
    })

def select_py_bindings():
    return select({
        "//:using_cuda12": [
            "//rtp_llm/models_py/bindings/cuda:cuda_bindings_register"
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm:rocm_bindings_register"
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
            "//rtp_llm/models_py/bindings:dummy_register",
        ],
        "//conditions:default": [
            "//rtp_llm/cpp/devices/cpu_impl:cpu_impl",
            "//rtp_llm/models_py/bindings:dummy_register",
        ],
    })
