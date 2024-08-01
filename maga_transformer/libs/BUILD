load("//bazel:defs.bzl", "copy_target_to")
load("//bazel:arch_select.bzl", "th_transformer_so")
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

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + select({
        "//:using_cuda": [":libmmha_so"],
        "//:using_rocm": [":libmmha_so"],
        "//conditions:default":[],
    })
)
