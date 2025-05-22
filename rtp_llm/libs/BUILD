load("//bazel:defs.bzl", "copy_so", "copy_so_inst")
load("//bazel:arch_select.bzl", "kernel_so_deps")
load("//rtp_llm/cpp/deep_gemm:template.bzl", "dpsk_gemm_so_num", "qwen_gemm_so_num")
package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_so("//:th_transformer")
copy_so("//rtp_llm/cpp/kernels:mmha1")
copy_so("//rtp_llm/cpp/kernels:mmha2")
copy_so("//rtp_llm/cpp/kernels:dmmha")
copy_so("//rtp_llm/cpp/cuda:fa")
copy_so("//rtp_llm/cpp/cutlass:fpA_intB")
copy_so("//rtp_llm/cpp/cutlass:moe")
copy_so("//rtp_llm/cpp/cutlass:moe_sm90")
copy_so("//rtp_llm/cpp/cutlass:int8_gemm")
copy_so("@flashinfer//:flashinfer_single_prefill")
copy_so("@flashinfer//:flashinfer_single_decode")
copy_so("@flashinfer//:flashinfer_batch_paged_prefill")
copy_so("@flashinfer//:flashinfer_batch_paged_decode")
copy_so("@flashinfer//:flashinfer_batch_ragged_prefill")
# num of so
copy_so_inst("//rtp_llm/cpp/deep_gemm:deepgemm_dpsk", dpsk_gemm_so_num)
copy_so_inst("//rtp_llm/cpp/deep_gemm:deepgemm_qwen", qwen_gemm_so_num)
copy_so("@flashinfer//:flashinfer_sm90")
copy_so("@acext//:acext_1")
copy_so("@acext//:acext_2")
copy_so("@acext//:acext_3")
copy_so("@deep_ep//:deep_ep_cu")

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:cpp_libraries"],
    outs = [
        "libasm_mla_decode_fwd_torch.so",
        "libmodule_gemm_a8w8_blockscale.so",
        "libmodule_quant.so"
    ],
    cmd = """
        cp ./bazel-out/k8-opt/bin/external/aiter/csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so $(location libasm_mla_decode_fwd_torch.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_gemm_a8w8_blockscale.so $(location libmodule_gemm_a8w8_blockscale.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_quant.so $(location libmodule_quant.so);
    """,
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
