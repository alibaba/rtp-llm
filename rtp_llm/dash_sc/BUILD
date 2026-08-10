# BUILD for rtp_llm/dash_sc: Python frontend gRPC (same pattern as model_rpc).
load("//bazel:arch_select.bzl", "requirement")

requirement([
    "grpcio",
    "jsonschema",
    "xgrammar",
])

# Scripts and data to be packaged with the wheel (dash_sc gRPC client).
filegroup(
    name = "grpc_client_scripts",
    srcs = [
        "grpc_client_run.sh",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "xgrammar_core_dump_control",
    srcs = ["inference/core_dump_control.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "dash_sc_grpc_server",
    srcs = [
        "__init__.py",
        "access_log.py",
        "access_record.py",
        "app.py",
        "client.py",
        "codec.py",
        "grpc_metrics.py",
        "json_format_e2e_smoke.py",
        "inference/__init__.py",
        "inference/__main__.py",
        "inference/grammar_validator.py",
        "inference/servicer.py",
        "proxy/__init__.py",
        "proxy/__main__.py",
        "proxy/service_route.py",
        "proxy/service_route_config.py",
        "proxy/servicer.py",
        "repetition_monitor.py",
        "server.py",
        "status.py",
        "structural_tag.py",
        "think.py",
    ],
    data = [":grpc_client_scripts"],
    visibility = ["//visibility:public"],
    deps = [
        ":xgrammar_core_dump_control",
        "//rtp_llm/server:server",
        ":grpcio",
        ":jsonschema",
        ":xgrammar",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm:config",
        "//rtp_llm/dash_sc/proto:predict_v2_py_proto",
        "//rtp_llm:frontend",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm:orjson",
        "//rtp_llm:sentencepiece",
        "//rtp_llm:tiktoken",
        "//rtp_llm:utils",
    ],
)

py_binary(
    name = "json_format_e2e_smoke",
    srcs = ["json_format_e2e_smoke.py"],
    main = "json_format_e2e_smoke.py",
    visibility = ["//visibility:public"],
)

py_test(
    name = "codec_test",
    srcs = ["test/codec_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "grammar_validator_test",
    srcs = ["test/inference/grammar_validator_test.py"],
    data = ["//:th_transformer_config"],
    main = "test/inference/grammar_validator_test.py",
    deps = [":dash_sc_grpc_server"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "core_dump_control_test",
    srcs = ["test/inference/core_dump_control_test.py"],
    main = "test/inference/core_dump_control_test.py",
    deps = [":xgrammar_core_dump_control"],
)

py_test(
    name = "grammar_validator_core_dump_test",
    srcs = ["test/inference/grammar_validator_core_dump_test.py"],
    data = ["//:th_transformer_config"],
    main = "test/inference/grammar_validator_core_dump_test.py",
    tags = [
        "local",
        "manual",
        "no-remote",
    ],
    deps = [
        ":dash_sc_grpc_server",
        ":xgrammar",
        ":xgrammar_core_dump_control",
    ],
)

py_test(
    name = "think_test",
    srcs = ["test/think_test.py"],
    main = "test/think_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "inference_servicer_test",
    srcs = ["test/inference/servicer_test.py"],
    main = "test/inference/servicer_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "mrcr_smoke_test",
    srcs = ["test/inference/mrcr_smoke_test.py"],
    data = ["test/data/mrcr_deepseek_v4_think_leak_cases.json"],
    main = "test/inference/mrcr_smoke_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "proxy_servicer_test",
    srcs = ["test/proxy/servicer_test.py"],
    main = "test/proxy/servicer_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "access_log_test",
    srcs = ["test/access_log_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "repetition_monitor_test",
    srcs = ["test/repetition_monitor_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "app_test",
    srcs = ["test/app_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_binary(
    name = "grammar_validation_smoke",
    srcs = ["grammar_validation_smoke.py"],
    main = "grammar_validation_smoke.py",
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:testlib"],
)

py_library(
    name = "grammar_validation_smoke_lib",
    srcs = ["grammar_validation_smoke.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dash_sc_grpc_server",
        "//rtp_llm:testlib",
    ],
)
