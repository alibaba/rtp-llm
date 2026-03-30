# Open source arch_select - wrapper targets for different system configs
# Python deps are managed by pip/pyproject.toml, not Bazel.
# torch C++ deps come from torch_local_repository (@torch).
load("//bazel:defs.bzl", "copy_so")

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
    copy_so("@flashinfer_cpp//:flashinfer_single_prefill")
    copy_so("@flashinfer_cpp//:flashinfer_single_decode")
    copy_so("@flashinfer_cpp//:flashinfer_batch_paged_prefill")
    copy_so("@flashinfer_cpp//:flashinfer_batch_paged_decode")
    copy_so("@flashinfer_cpp//:flashinfer_batch_ragged_prefill")
    copy_so("@flashinfer_cpp//:flashinfer_single_prefill_256")
    copy_so("@flashinfer_cpp//:flashinfer_single_decode_256")
    copy_so("@flashinfer_cpp//:flashinfer_batch_paged_prefill_256")
    copy_so("@flashinfer_cpp//:flashinfer_batch_paged_decode_256")
    copy_so("@flashinfer_cpp//:flashinfer_batch_ragged_prefill_256")
    copy_so("@flashinfer_cpp//:flashinfer_sm90")
    copy_so("@deep_ep//:deep_ep_cu")

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def transfer_rdma_deps():
    native.alias(
        name = "transfer_rdma_impl",
        actual = "//rtp_llm/cpp/cache/connector/p2p/transfer:no_rdma_impl",
    )

def transfer_backend_deps():
    native.alias(
        name = "transfer_backend_arch_select_impl",
        actual = "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_base_impl",
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

def torch_deps():
    return [
        "@torch//:torch_api",
        "@torch//:torch",
        "@torch//:torch_libs",
    ]

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
        actual = "@flashinfer_cpp//:flashinfer"
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
    pass

def kernel_so_deps():
    return select({
        "@//:using_cuda": [":libmmha1_so", ":libmmha2_so", ":libdmmha_so", ":libfa_so", ":libfpA_intB_so", ":libint8_gemm_so", ":libmoe_so", ":libmoe_sm90_so", ":libflashinfer_single_prefill_so", ":libflashinfer_single_decode_so", ":libflashinfer_batch_paged_prefill_so", ":libflashinfer_batch_paged_decode_so", ":libflashinfer_batch_ragged_prefill_so", ":libflashinfer_sm90_so", ":libflashinfer_single_prefill_256_so", ":libflashinfer_single_decode_256_so", ":libflashinfer_batch_paged_prefill_256_so", ":libflashinfer_batch_paged_decode_256_so", ":libflashinfer_batch_ragged_prefill_256_so"],
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
            "//conditions:default": "//rtp_llm/cpp/cuda/ops:gpu_register",
        }),
        visibility = ["//visibility:public"],
    )

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
        "//conditions:default": [
            "//rtp_llm/models_py/bindings:dummy_register",
        ],
    })
