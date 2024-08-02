load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
)

cc_library(
    name = "ck_headers",
    hdrs = glob([
        "inlcude/**/*.h",
        "include/**/*.inc",
        "include/**/*.hpp",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":ck_library_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_library_headers",
    srcs = glob(["library/src/utility/**/*.cpp"]),
    hdrs = glob([
        "library/inlcude/**/*.h",
        "library/include/**/*.inc",
        "library/include/**/*.hpp",
    ]),
    strip_include_prefix = "library/include",
    copts = rocm_default_copts(),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "ck_fmha_example_headers",
    hdrs = glob([
        "example/ck_tile/01_fmha/*.hpp",
    ]),
    deps = [
        ":ck_headers",
    ],
    strip_include_prefix = "example/ck_tile/01_fmha",
    visibility = ["//visibility:public"],
)

exports_files(["example/ck_tile/01_fmha/generate.py"])
