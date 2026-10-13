package(default_visibility = ["//:__subpackages__"])
load("@arch_config//:arch_select.bzl", "requirement")

requirement(["requests"])

py_library(
    name = "tipc",
    data = [
        "csrc/common.h",
        "csrc/export.cc",
        "csrc/ipc.cc",
        "csrc/ipc.h",
    ],
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
        ":requests",
    ],
)
