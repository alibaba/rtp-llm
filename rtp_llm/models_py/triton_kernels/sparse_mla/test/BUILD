load("//bazel:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy"])

py_test(
    name = "merge_states_kv_cascade_test",
    srcs = ["merge_states_kv_cascade_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
        "//rtp_llm/test/utils:test_util",
    ] + [":triton", ":torch", ":numpy"],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)
