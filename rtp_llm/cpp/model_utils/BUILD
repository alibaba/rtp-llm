load("//:def.bzl", "copts")
load("@//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "model_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = copts(),
)
