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
    deps = [":kv_cache_subscriber"],
)

py_test(
    name = "service_test",
    srcs = ["test_service.py"],
    deps = [":kv_cache_subscriber"],
)
