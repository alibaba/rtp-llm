load("//bazel:defs.bzl", "copy_so")
load("//bazel:arch_select.bzl", "th_transformer_so", "kernel_so_deps", "libacext_so")
package(default_visibility = ["//maga_transformer:__subpackages__"])

th_transformer_so()
libacext_so()

copy_so(":th_transformer");
copy_so("//src/fastertransformer/kernels:mmha1")
copy_so("//src/fastertransformer/kernels:mmha2")
copy_so("//src/fastertransformer/kernels:dmmha")
copy_so("//src/fastertransformer/cuda:fa")
copy_so("//src/fastertransformer/cutlass:fpA_intB")
copy_so("//src/fastertransformer/cutlass:moe")
copy_so("//src/fastertransformer/cutlass:int8_gemm")
copy_so("@flashinfer//:flashinfer_0")
copy_so("@flashinfer//:flashinfer_1")
copy_so("@flashinfer//:flashinfer_2")

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
