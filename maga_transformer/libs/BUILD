load("//bazel:defs.bzl", "copy_target_to")
load("//bazel:arch_select.bzl", "th_transformer_so")
package(default_visibility = ["//maga_transformer:__subpackages__"])

th_transformer_so()

filegroup(
    name = "libs",
    srcs = [],
    data = ["libth_transformer_so"]
)