# Build overlay for the xgrammar source pinned in repositories.bzl.
# picojson is in-tree; dlpack is provided by @rtp_llm//3rdparty/dlpack.

# Public xgrammar/*.h headers; dlpack flows transitively via xgrammar/matcher.h.
cc_library(
    name = "xgrammar_headers",
    hdrs = glob([
        "include/xgrammar/*.h",
    ]),
    includes = [
        "include",
    ],
    deps = ["@rtp_llm//3rdparty/dlpack:dlpack_headers"],
    visibility = ["//visibility:public"],
)

# Internal cpp/ headers used by xgrammar source files; private — callers use xgrammar/*.h only.
cc_library(
    name = "xgrammar_internal_headers",
    hdrs = glob([
        "cpp/*.h",
        "cpp/support/*.h",
        "3rdparty/picojson/picojson.h",
    ]),
    includes = [
        "cpp",
        "3rdparty/picojson",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "xgrammar",
    # cpp/testing.cc must stay in: it defines xgrammar::PrintTokenByIds, used by
    # matcher/compiled_grammar operator<<.
    srcs = glob([
        "cpp/*.cc",
        "cpp/support/*.cc",
    ]),
    defines = [
        "XGRAMMAR_ENABLE_CPPTRACE=0",
    ],
    copts = [
        "-std=c++17",
        "-fexceptions",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
    ],
    deps = [
        ":xgrammar_headers",
        ":xgrammar_internal_headers",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
