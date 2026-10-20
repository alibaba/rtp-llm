load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS")

# Exercise the smoke comparers without starting a model server or using a GPU.
py_test(
    name = "normal_comparer_test",
    srcs = ["normal_comparer_test.py"],
    data = ["//rtp_llm/test/smoke:smoke_framework_srcs"],
    imports = ["../.."],
    legacy_create_init = 0,
    deps = SMOKE_FRAMEWORK_DEPS,
)

py_test(
    name = "openai_comparer_test",
    srcs = ["openai_comparer_test.py"],
    data = ["//rtp_llm/test/smoke:smoke_framework_srcs"],
    imports = ["../.."],
    legacy_create_init = 0,
    deps = SMOKE_FRAMEWORK_DEPS,
)
