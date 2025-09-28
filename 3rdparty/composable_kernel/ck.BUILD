load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
)

cc_library(
    name = "ck_headers",
    deps = [
        ":ck_library_headers",
        ":ck_headers_real",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_headers_real",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.inc",
        "include/**/*.hpp",
    ]) + [":config_h"],
    copts = rocm_default_copts(),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "ck_library_headers",
    srcs = glob(["library/src/utility/**/*.cpp"]),
    hdrs = glob([
        "library/include/**/*.h",
        "library/include/**/*.inc",
        "library/include/**/*.hpp",
    ]),
    strip_include_prefix = "library/include",
    copts = rocm_default_copts(),
    deps = [
        ":ck_headers_real",
    ],
)

cc_library(
    name = "ck_fmha_example_headers",
    hdrs = glob([
        "example/ck_tile/01_fmha/*.hpp",
    ]),
    copts = rocm_default_copts(),
    deps = [
        ":ck_headers",
    ],
    strip_include_prefix = "example/ck_tile/01_fmha",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_layernorm2d_example_headers",
    hdrs = glob([
        "example/ck_tile/02_layernorm2d/*.hpp",
    ]),
    deps = [
        ":ck_headers",
    ],
    strip_include_prefix = "example/ck_tile/02_layernorm2d",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_rmsnorm2d_example_headers",
    hdrs = glob([
        "example/ck_tile/10_rmsnorm2d/*.hpp",
    ]),
    deps = [
        ":ck_headers",
    ],
    strip_include_prefix = "example/ck_tile/10_rmsnorm2d",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_fused_moe_example_headers",
    srcs = glob([
        "example/ck_tile/15_fused_moe/instances/*.cpp",
    ]),
    hdrs = glob([
        "example/ck_tile/15_fused_moe/**/*.hpp",
    ]),
    deps = [
        ":ck_headers",
    ],
    copts = rocm_default_copts(),
    strip_include_prefix = "example/ck_tile/15_fused_moe",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ck_fmha_rmsnorm2d_libraries",
    srcs = glob(["**/*.so"]),
    visibility = ["//visibility:public"],
    tags = ["rocm", "local"],
)

genrule(
    name = "config_h",
    srcs = [
        "include/ck/config.h.in",
    ],
    outs = [
        "include/ck/config.h",
    ],
    cmd = """
       awk '{gsub(/^#cmakedefine DTYPES \"@DTYPES@\"/, "/* #undef DTYPES*/");
             gsub(/^#cmakedefine CK_ENABLE_ALL_DTYPES @CK_ENABLE_ALL_DTYPES@/, "#define CK_ENABLE_ALL_DTYPES ON");
             gsub(/^#cmakedefine CK_ENABLE_INT8 @CK_ENABLE_INT8@/, "/* #undef CK_ENABLE_INT8*/");
             gsub(/^#cmakedefine CK_ENABLE_FP8 @CK_ENABLE_FP8@/, "/* #undef CK_ENABLE_FP8*/");
             gsub(/^#cmakedefine CK_ENABLE_BF8 @CK_ENABLE_BF8@/, "/* #undef CK_ENABLE_BF8*/");
             gsub(/^#cmakedefine CK_ENABLE_FP16 @CK_ENABLE_FP16@/, "/* #undef CK_ENABLE_FP16*/");
             gsub(/^#cmakedefine CK_ENABLE_BF16 @CK_ENABLE_BF16@/, "/* #undef CK_ENABLE_BF16*/");
             gsub(/^#cmakedefine CK_ENABLE_FP32 @CK_ENABLE_FP32@/, "/* #undef CK_ENABLE_FP32*/");
             gsub(/^#cmakedefine CK_ENABLE_FP64 @CK_ENABLE_FP64@/, "/* #undef CK_ENABLE_FP64*/");
             gsub(/^#cmakedefine CK_ENABLE_DL_KERNELS @CK_ENABLE_DL_KERNELS@/, "/* #undef CK_ENABLE_DL_KERNELS*/");
             gsub(/^#cmakedefine CK_ENABLE_INSTANCES_ONLY @CK_ENABLE_INSTANCES_ONLY@/, "/* #undef CK_ENABLE_INSTANCES_ONLY*/");
             gsub(/^#cmakedefine CK_USE_XDL @CK_USE_XDL@/, "#define CK_USE_XDL ON");
             gsub(/^#cmakedefine CK_USE_WMMA @CK_USE_WMMA@/, "/* #undef CK_USE_WMMA*/");
             gsub(/^#cmakedefine/, "//cmakedefine");print;}' $(<) > $(@)
    """,
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

exports_files(["example/ck_tile/01_fmha/generate.py"])
exports_files(["example/ck_tile/02_layernorm2d/generate.py"])
