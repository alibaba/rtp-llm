package(default_visibility = ["//visibility:public"])

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
          "$(RULEDIR) 'apply_top_k_top_p_custom' ascend950",
    tools = ["@aclnn_custom_ops_src//:build_for_bazel.sh"],
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["local"],
)

cc_library(
    name = "aclnn_custom_ops",
    data = [
        "lib/libcust_opapi.so",
        "lib/libcust_opsproto_rt2.0.so",
        "lib/libcust_opmaster_rt2.0.so",
        "opp/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    ],
    deps = [
        ":build_aclnn_ops",
        "@local_config_ascend//ascend:ascend_headers",
    ],
)
