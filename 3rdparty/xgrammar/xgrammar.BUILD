# xgrammar built from source (mlc-ai/xgrammar @ v0.1.32).

cc_library(
    name = "xgrammar_headers",
    hdrs = glob([
        "include/xgrammar/*.h",
        "3rdparty/dlpack/include/dlpack/*.h",
    ]),
    includes = [
        "include",
        "3rdparty/dlpack/include",
    ],
    visibility = ["//visibility:public"],
)

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
    srcs = glob(
        [
            "cpp/*.cc",
            "cpp/support/*.cc",
        ],
        exclude = [
            "cpp/nanobind/**",
        ],
    ),
    defines = [
        "XGRAMMAR_ENABLE_CPPTRACE=0",
    ],
    copts = [
        "-std=c++17",
        "-fexceptions",
        "-Wno-sign-compare",
        "-Wno-unused-variable",
    ],
    deps = [
        ":xgrammar_headers",
        ":xgrammar_internal_headers",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
