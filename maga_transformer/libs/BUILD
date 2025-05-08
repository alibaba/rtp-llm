load("//bazel:defs.bzl", "copy_so")
load("//bazel:arch_select.bzl", "kernel_so_deps")
package(default_visibility = ["//maga_transformer:__subpackages__"])

copy_so("//:th_transformer")
copy_so("//maga_transformer/cpp/kernels:mmha1")
copy_so("//maga_transformer/cpp/kernels:mmha2")
copy_so("//maga_transformer/cpp/kernels:dmmha")
copy_so("//maga_transformer/cpp/cuda:fa")
copy_so("//maga_transformer/cpp/cutlass:fpA_intB")
copy_so("//maga_transformer/cpp/cutlass:moe")
copy_so("//maga_transformer/cpp/cutlass:moe_sm90")
copy_so("//maga_transformer/cpp/cutlass:int8_gemm")
copy_so("@flashinfer//:flashinfer_single_prefill")
copy_so("@flashinfer//:flashinfer_single_decode")
copy_so("@flashinfer//:flashinfer_batch_paged_prefill")
copy_so("@flashinfer//:flashinfer_batch_paged_decode")
copy_so("@flashinfer//:flashinfer_batch_ragged_prefill")
copy_so("//maga_transformer/cpp/deep_gemm:deepgemm_dpsk_inst")
copy_so("//maga_transformer/cpp/deep_gemm:deepgemm_qwen_inst")
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
