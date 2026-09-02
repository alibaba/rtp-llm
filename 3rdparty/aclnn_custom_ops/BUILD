package(default_visibility = ["//visibility:public"])

# SoC version for the AscendC custom-op build; override with
# --define aclnn_soc_version=<soc> when targeting other chips (e.g. ascend910b).
ACLNN_SOC_VERSION = "ascend950"

genrule(
    name = "build_aclnn_ops",
    srcs = ["@aclnn_custom_ops_src//:all_srcs"],
    outs = [
        "lib/libcust_opapi.so",
        "lib/libcust_opsproto_rt2.0.so",
        "lib/libcust_opmaster_rt2.0.so",
        "opp/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    ],
    cmd = "bash $(location @aclnn_custom_ops_src//:build_for_bazel.sh) " +
          "$(RULEDIR) 'apply_top_k_top_p_custom' " + ACLNN_SOC_VERSION,
    tools = ["@aclnn_custom_ops_src//:build_for_bazel.sh"],
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["local"],
)

# Runtime payload only (prebuilt .so/OPP from the genrule above); no CcInfo,
# consumers get the artifacts through data/runfiles.
cc_library(
    name = "aclnn_custom_ops",
    data = [
        ":build_aclnn_ops",
        "lib/libcust_opapi.so",
        "lib/libcust_opsproto_rt2.0.so",
        "lib/libcust_opmaster_rt2.0.so",
        "opp/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    ],
    deps = [
        "@local_config_ascend//ascend:ascend_headers",
    ],
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
