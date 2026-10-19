package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "prepare_kev_checkpoint",
    srcs = ["prepare_kev_checkpoint.py"],
    deps = [
        ":convert_kev_checkpoint",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "prepare_kev_checkpoint_test",
    srcs = ["prepare_kev_checkpoint_test.py"],
    deps = [":prepare_kev_checkpoint", "//rtp_llm:torch"],
)

py_library(
    name = "convert_kev_checkpoint",
    srcs = ["convert_kev_checkpoint.py"],
)

# Standalone CPU mocks: no accelerator, RTP ops, or model downloads.
py_test(
    name = "convert_kev_checkpoint_test",
    srcs = ["convert_kev_checkpoint_test.py"],
    deps = [":convert_kev_checkpoint"],
)

py_library(
    name = "model_assistant_api",
    srcs = glob([
        "api/*.py",
    ]),
)

py_library(
    name = "model_assistant",
    srcs = [
           "model_assistant_server.py",
           ":model_assistant_api"
    ]
)

