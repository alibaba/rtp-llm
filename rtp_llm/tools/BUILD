package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "dsv41_pp_smoke",
    testonly = True,
    srcs = ["dsv41_pp_smoke.py"],
    deps = [":dsv41_phase_smoke"],
)

py_library(
    name = "dsv41_phase_smoke",
    testonly = True,
    srcs = ["dsv41_phase_smoke.py"],
)

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
    name = "dsv41_engram_boundary_replay",
    testonly = True,
    srcs = ["dsv41_engram_boundary_replay.py"],
    deps = [":dsv41_attention_replay", ":dsv41_mhc_boundary_replay", "//rtp_llm:torch", "//rtp_llm:transformers"],
)

py_library(
    name = "dsv41_query_boundary_replay",
    testonly = True,
    srcs = ["dsv41_query_boundary_replay.py"],
    deps = [":dsv41_attention_replay", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_query_owner_replay",
    testonly = True,
    srcs = ["dsv41_query_owner_replay.py"],
    deps = [":dsv41_query_boundary_replay", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_output_projection_replay",
    testonly = True,
    srcs = ["dsv41_output_projection_replay.py"],
    deps = [":dsv41_query_owner_replay", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_output_owner_replay",
    testonly = True,
    srcs = ["dsv41_output_owner_replay.py"],
    deps = [":dsv41_output_projection_replay", "//rtp_llm:torch"],
)

py_library(
    name = "dsv41_sparse_precision_replay",
    testonly = True,
    srcs = ["dsv41_sparse_precision_replay.py"],
    deps = [":dsv41_output_owner_replay", "//rtp_llm:torch", "//rtp_llm:triton"],
)

py_library(
    name = "dsv41_sparse_order_replay",
    testonly = True,
    srcs = ["dsv41_sparse_order_replay.py"],
    deps = [":dsv41_sparse_precision_replay", "//rtp_llm:torch", "//rtp_llm:triton"],
)

py_library(
    name = "dsv41_sparse_prefill_owner_replay",
    testonly = True,
    srcs = ["dsv41_sparse_prefill_owner_replay.py"],
    deps = [
        ":dsv41_output_owner_replay",
        ":dsv41_sparse_order_replay",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_sparse_prefill_reference_testlib",
        "//rtp_llm:torch",
    ],
)

py_library(
    name = "dsv41_deep_attention_history_replay",
    testonly = True,
    srcs = ["dsv41_deep_attention_history_replay.py", "dsv41_deep_native_history_replay.py"],
    deps = [
        ":dsv41_attention_replay",
        ":dsv41_sparse_prefill_owner_replay",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_deep_history_testlib",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_decode_tilelang_reference_testlib",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_mhc_collapse_reference_testlib",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
)

py_library(
    name = "dsv41_block_norm_replay",
    testonly = True,
    srcs = ["dsv41_block_norm_replay.py"],
    deps = [
        ":dsv41_corrected_history_replay",
        ":dsv41_mhc_collapse_replay",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_block_norm_reference_testlib",
        "//rtp_llm:torch",
    ],
)

py_library(
    name = "dsv41_kv_boundary_replay",
    testonly = True,
    srcs = ["dsv41_kv_boundary_replay.py"],
    deps = [
        ":dsv41_block_norm_replay",
        ":dsv41_query_boundary_replay",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
)

py_library(
    name = "dsv41_corrected_history_replay",
    testonly = True,
    srcs = ["dsv41_corrected_history_replay.py"],
    deps = [
        ":dsv41_deep_attention_history_replay",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
)

py_library(
    name = "dsv41_mhc_collapse_replay",
    testonly = True,
    srcs = ["dsv41_mhc_collapse_replay.py"],
    deps = [
        ":dsv41_deep_attention_history_replay",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_mhc_collapse_reference_testlib",
        "//rtp_llm:torch",
    ],
)

py_library(
    name = "dsv41_deep_decode_sparse_replay",
    testonly = True,
    srcs = ["dsv41_deep_decode_sparse_replay.py"],
    deps = [
        ":dsv41_mhc_collapse_replay",
        ":dsv41_sparse_precision_replay",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
)

py_library(
    name = "dsv41_deep_decode_tile_replay",
    testonly = True,
    srcs = ["dsv41_deep_decode_tile_replay.py"],
    deps = [
        ":dsv41_deep_decode_sparse_replay",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
)

py_library(
    name = "dsv41_decode_tilelang_replay",
    testonly = True,
    srcs = ["dsv41_decode_tilelang_replay.py"],
    deps = [
        ":dsv41_deep_decode_sparse_replay",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:v41_decode_tilelang_reference_testlib",
        "//rtp_llm:torch",
    ],
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
