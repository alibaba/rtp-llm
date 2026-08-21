load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_decode_ktp_layout",
    srcs = ["test_decode_ktp_layout.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "test_decode_ktp_cache_model",
    srcs = ["test_decode_ktp_cache_model.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "test_mla_cache_tp",
    srcs = ["test_mla_cache_tp.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "test_pd_mla_shard_fanin",
    srcs = ["test_pd_mla_shard_fanin.py"],
    deps = ["//rtp_llm:testlib"],
)
