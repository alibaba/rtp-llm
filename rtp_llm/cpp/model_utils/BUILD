load("//:def.bzl", "copts")

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
        "//rtp_llm/cpp/utils:utils",
    ],
    copts = copts(),
)
