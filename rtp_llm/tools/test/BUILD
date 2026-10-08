py_test(
    name = "dsv41_reference_probe_test",
    srcs = ["dsv41_reference_probe_test.py"],
    deps = ["//rtp_llm/tools:dsv41_reference_probe"],
    target_compatible_with = select({
        "@//:using_cuda13_sm120": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "dsv41_trace_compare_test",
    srcs = ["dsv41_trace_compare_test.py"],
    deps = ["//rtp_llm/tools:dsv41_trace_compare"],
)

py_test(
    name = "checkpoint_inventory_test",
    srcs = ["checkpoint_inventory_test.py"],
    data = ["//rtp_llm/config/test:testdata/deepseek_v41_flash.json"],
    deps = ["//rtp_llm/tools:checkpoint_inventory_lib"],
)
