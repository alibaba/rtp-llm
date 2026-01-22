load("//bazel:arch_select.bzl", get_triton_deps = "triton_deps", "internal_deps")

py_test(
    name = "schedule_meta_test",
    srcs = ["schedule_meta_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
    ],
    tags = ["manual"],
)
