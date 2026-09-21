load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS")

py_test(
    name = "entry_env_test",
    srcs = ["entry_env_test.py"],
    data = ["//rtp_llm/test/smoke:smoke_framework_srcs"],
    imports = ["../.."],
    legacy_create_init = 0,
    deps = SMOKE_FRAMEWORK_DEPS,
)
