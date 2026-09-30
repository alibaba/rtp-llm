package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "checkpoint_inventory_lib",
    srcs = ["checkpoint_inventory.py"],
    deps = ["//rtp_llm:config", "//rtp_llm:safetensors_header"],
)

py_binary(
    name = "checkpoint_inventory",
    srcs = ["checkpoint_inventory.py"],
    deps = ["//rtp_llm:config", "//rtp_llm:safetensors_header"],
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
