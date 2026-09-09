package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_baseline_modules",
    srcs = ["test_baseline_modules.py"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
)

py_test(
    name = "test_lifecycle",
    srcs = ["test_lifecycle.py", "test_factory.py"],
    deps = ["//rtp_llm/models_py:pluggable"],
)

py_test(
    name = "test_factory",
    srcs = ["test_factory.py"],
    deps = ["//rtp_llm/models_py:pluggable"],
)

py_test(
    name = "test_control",
    srcs = ["test_control.py"],
    deps = ["//rtp_llm/models_py:pluggable", "//rtp_llm:torch"],
)

py_test(
    name = "test_platform",
    srcs = ["test_platform.py"],
    deps = ["//rtp_llm/models_py:pluggable"],
)

py_test(
    name = "test_dsv4_plan",
    srcs = ["test_dsv4_plan.py"],
    deps = ["//rtp_llm/models_py:pluggable"],
)

py_test(
    name = "test_dsv4_resources",
    srcs = ["test_dsv4_resources.py"],
    deps = ["//rtp_llm/models_py:pluggable", "//rtp_llm:models", "//rtp_llm:torch"],
)
