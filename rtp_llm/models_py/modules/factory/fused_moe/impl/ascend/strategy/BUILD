py_library(
    name = "ascend_strategies",
    srcs = glob(["*.py"]),
    deps = [
        "//rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors:ascend_executors",
        "//rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/routers:ascend_routers",
    ],
    visibility = ["//visibility:public"],
)
