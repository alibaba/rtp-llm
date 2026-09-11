package(default_visibility = ["//visibility:public"])

load(":aclnn_build.bzl", "aclnn_custom_ops_build")

# SoC version for the AscendC custom-op build.
# Only support ascend950.
# Other SoC versions are not supported.
ACLNN_SOC_VERSION = "ascend950"

aclnn_custom_ops_build(
    name = "build_aclnn_ops",
    srcs = ["@aclnn_custom_ops_src//:all_srcs"],
    script = "@aclnn_custom_ops_src//:build_for_bazel.sh",
    op_names = "apply_top_k_top_p_custom",
    soc_version = ACLNN_SOC_VERSION,
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["local", "no-remote-cache"],
)

# Runtime payload only (prebuilt .so/OPP from the rule above); no CcInfo,
# consumers get the artifacts through data/runfiles.
cc_library(
    name = "aclnn_custom_ops",
    data = [
        ":build_aclnn_ops",
    ],
    deps = [
        "@local_config_ascend//ascend:ascend_headers",
    ],
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
