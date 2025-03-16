load("//bazel:defs.bzl", "copy_so")
load("//bazel:arch_select.bzl", "kernel_so_deps", "libacext_so")
package(default_visibility = ["//maga_transformer:__subpackages__"])

libacext_so()

copy_so("//:th_transformer")
copy_so("//src/fastertransformer/kernels:mmha1")
copy_so("//src/fastertransformer/kernels:mmha2")
copy_so("//src/fastertransformer/kernels:dmmha")
copy_so("//src/fastertransformer/cuda:fa")
copy_so("//src/fastertransformer/cutlass:fpA_intB")
copy_so("//src/fastertransformer/cutlass:moe")
copy_so("//src/fastertransformer/cutlass:int8_gemm")
copy_so("@flashinfer//:flashinfer_single")
copy_so("@flashinfer//:flashinfer_batch_paged_prefill")
copy_so("@flashinfer//:flashinfer_batch_paged_decode")
copy_so("@flashinfer//:flashinfer_batch_ragged_prefill")
copy_so("@flashinfer//:flashinfer_sm90")

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
