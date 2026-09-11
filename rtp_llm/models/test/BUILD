_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "dsv4_kv_cache_test",
    srcs = ["dsv4_kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "deepseek_v4_dspark_commit_only_test",
    srcs = ["deepseek_v4_dspark_commit_only_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "moe_config_propagation_test",
    srcs = ["moe_config_propagation_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "moe_cuda13_config_propagation_test",
    srcs = ["moe_cuda13_config_propagation_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)
