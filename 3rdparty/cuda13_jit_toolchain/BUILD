exports_files(["repositories.bzl"])

py_library(
    name = "runtime_setup",
    srcs = ["//rtp_llm:utils/jit_toolchain.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "toolchain_test",
    srcs = ["toolchain_test.py"],
    deps = [":runtime_setup", "@cuda13_jit_gcc//:runtime"],
    tags = ["manual"],
    timeout = "short",
)
