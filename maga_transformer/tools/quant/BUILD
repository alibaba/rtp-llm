package(default_visibility = ["//maga_transformer:__subpackages__", "//internal_source:__subpackages__"])

py_library(
    name = "quant",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//maga_transformer:utils",
        "//maga_transformer:config",
        "//maga_transformer:structure",
        "//maga_transformer:models",
        "//maga_transformer:openai_api",
        "//maga_transformer/server:server",
        "//maga_transformer/tools:model_assistant_api"
    ]
)

