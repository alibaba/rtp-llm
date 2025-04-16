load("//bazel:defs.bzl", "copy_so")
load("//bazel:arch_select.bzl", "kernel_so_deps")
package(default_visibility = ["//maga_transformer:__subpackages__"])

copy_so("//:th_transformer")
copy_so("//src/fastertransformer/kernels:mmha1")
copy_so("//src/fastertransformer/kernels:mmha2")
copy_so("//src/fastertransformer/kernels:dmmha")
copy_so("//src/fastertransformer/cuda:fa")
copy_so("//src/fastertransformer/cutlass:fpA_intB")
copy_so("//src/fastertransformer/cutlass:moe")
copy_so("//src/fastertransformer/cutlass:moe_sm90")
copy_so("//src/fastertransformer/cutlass:int8_gemm")
copy_so("@flashinfer//:flashinfer_single_prefill")
copy_so("@flashinfer//:flashinfer_single_decode")
copy_so("@flashinfer//:flashinfer_batch_paged_prefill")
copy_so("@flashinfer//:flashinfer_batch_paged_decode")
copy_so("@flashinfer//:flashinfer_batch_ragged_prefill")
copy_so("@flashinfer//:flashinfer_sm90")
copy_so("@acext//:acext_1")
copy_so("@acext//:acext_2")
copy_so("@acext//:acext_3")
copy_so("@deep_ep//:deep_ep_cu")

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
