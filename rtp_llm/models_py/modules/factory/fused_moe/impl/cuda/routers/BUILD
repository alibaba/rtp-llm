py_library(
    name = "deepep_dispatch_args",
    srcs = ["deepep_dispatch_args.py"],
    visibility = [
        "//rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/routers/test:__pkg__",
    ],
)

py_library(
    name = "routers",
    srcs = glob(
        ["*.py"],
        exclude = ["deepep_dispatch_args.py"],
    ),
    deps = [":deepep_dispatch_args"],
    visibility = ["//rtp_llm/models_py:__pkg__"],
)
