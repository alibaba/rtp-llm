package(default_visibility = ["//visibility:public"])

# Vendored dlpack header (single file). xgrammar's matcher.h includes
# <dlpack/dlpack.h>, but xgrammar's upstream pulls dlpack as a git
# submodule from github.com/dmlc/dlpack which is unreachable from the
# internal build environment. We vendor the v0.8 header verbatim instead.
cc_library(
    name = "dlpack_headers",
    hdrs = ["include/dlpack/dlpack.h"],
    includes = ["include"],
)
