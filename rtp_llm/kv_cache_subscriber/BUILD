load("//bazel:arch_select.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

requirement([
    "grpcio",
])

py_library(
    name = "kv_cache_subscriber",
    srcs = glob(
        ["*.py"],
        exclude = [
            "__main__.py",
            "test_*.py",
        ],
    ),
    deps = [
        ":grpcio",
        "//rtp_llm:aiohttp",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
)

py_binary(
    name = "subscriber",
    srcs = ["__main__.py"],
    main = "__main__.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "models_test",
    srcs = ["test_models.py"],
    main = "test_models.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "service_test",
    srcs = [
        "test_service.py",
        "test_utils.py",
    ],
    main = "test_service.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "source_test",
    srcs = ["test_source.py"],
    main = "test_source.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "reporter_test",
    srcs = [
        "test_reporter.py",
        "test_utils.py",
    ],
    main = "test_reporter.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "config_test",
    srcs = [
        "test_config.py",
        "test_utils.py",
    ],
    main = "test_config.py",
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "manager_test",
    srcs = ["test_manager.py"],
    main = "test_manager.py",
    deps = [":kv_cache_subscriber"],
)

test_suite(
    name = "tests",
    tests = [
        ":config_test",
        ":manager_test",
        ":models_test",
        ":reporter_test",
        ":service_test",
        ":source_test",
    ],
)
