load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "npu_bridge",
    srcs = [
        "NPUBridge.cpp",
        "NPUStorageImpl.cpp",
    ],
    hdrs = [
        "NPUBridge.h",
        "NPUStorageImpl.h",
        "aclnn_apply_top_k_top_p_custom.h",
        "ascend_apply_top_k_top_p_custom.h",
        "op_api_common.h",
    ],
    deps = torch_deps() + select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
        ],
        "//conditions:default": [],
    }),
    # 仅在 Ascend 平台编译：op_api_common.h 依赖 torch_npu 头文件，
    # 非 Ascend 平台下 torch_deps() 不包含 torch_npu，会导致头文件校验失败。
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)
