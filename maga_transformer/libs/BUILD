load("//bazel:defs.bzl", "copy_target_to")
load("//bazel:arch_select.bzl", "th_transformer_so", "kernel_so_deps")
package(default_visibility = ["//maga_transformer:__subpackages__"])

th_transformer_so()

genrule(
    name = "libth_transformer_so",
    srcs = [":th_transformer_so"],
    outs = [
        "libth_transformer.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "libmmha_so",
    srcs = ["//src/fastertransformer/kernels:mmha"],
    outs = [
        "libmmha.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "libfa_so",
    srcs = ["//src/fastertransformer/cuda:fa"],
    outs = [
        "libfa.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "libfpA_intB_so",
    srcs = ["//src/fastertransformer/cutlass:fpA_intB"],
    outs = [
        "libfpA_intB.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "libmoe_so",
    srcs = ["//src/fastertransformer/cutlass:moe"],
    outs = [
        "libmoe.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "libint8_gemm_so",
    srcs = ["//src/fastertransformer/cutlass:int8_gemm"],
    outs = [
        "libint8_gemm.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
