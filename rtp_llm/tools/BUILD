package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "dsv41_trace",
    srcs = ["dsv41_trace.py"],
    deps = ["//rtp_llm:torch"],
)

py_library(
    name = "dsv41_trace_compare",
    srcs = ["dsv41_trace_compare.py"],
    deps = ["//rtp_llm:torch"],
)

py_library(
    name = "dsv41_reference_probe",
    srcs = ["dsv41_reference_probe.py"],
    deps = [":dsv41_trace", "//rtp_llm:safetensors_header", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_attention_replay",
    testonly = True,
    srcs = ["dsv41_attention_replay.py", "dsv41_attention_history_replay.py", "dsv41_native_history_replay.py"],
    deps = [
        ":dsv41_reference_probe",
        ":dsv41_trace_compare",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
)

py_library(
    name = "dsv41_mhc_boundary_replay",
    testonly = True,
    srcs = ["dsv41_mhc_boundary_replay.py"],
    deps = [":dsv41_attention_replay", "//rtp_llm:torch"],
)

py_library(
    name = "checkpoint_inventory_lib",
    srcs = ["checkpoint_inventory.py"],
    deps = ["//rtp_llm:config", "//rtp_llm:safetensors_header"],
)

py_library(
    name = "dsv41_sparse_arithmetic_replay",
    testonly = True,
    srcs = ["dsv41_sparse_arithmetic_replay.py"],
    deps = [":dsv41_attention_replay", "//rtp_llm:torch", "//rtp_llm:triton"],
)

py_library(
    name = "dsv41_sparse_owner_replay",
    testonly = True,
    srcs = ["dsv41_sparse_owner_replay.py"],
    deps = [":dsv41_attention_replay", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_ffn_boundary_replay",
    testonly = True,
    srcs = ["dsv41_ffn_boundary_replay.py"],
    deps = [":dsv41_attention_replay", "//rtp_llm:torch"],
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
