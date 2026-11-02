load("//rtp_llm/models_py/modules/qwen4_exp/test:defs.bzl", "qwen4_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "fixtures",
    srcs = ["fixtures.py"] + ["ple_page_reference.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
    ],
)

py_library(
    name = "qsa_fixtures",
    srcs = ["qsa_fixtures.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
    ],
)

qwen4_test(
    name = "mtp_test",
    src = "mtp_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "ple_cache_test",
    src = "ple_cache_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "ple_graph_test",
    src = "ple_graph_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "ple_host_test",
    src = "ple_host_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "ple_prefetch_test",
    src = "ple_prefetch_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "qsa_cache_test",
    src = "qsa_cache_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "qsa_graph_test",
    src = "qsa_graph_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "qsa_mtp_test",
    src = "qsa_mtp_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "qsa_wiring_test",
    src = "qsa_wiring_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

qwen4_test(
    name = "wiring_test",
    src = "wiring_test.py",
    gpu = "L20_CU13",
    deps = [
        ":fixtures",
        ":qsa_fixtures",
    ],
)

test_suite(
    name = "qwen4_runtime",
    tests = [
        ":mtp_test",
        ":ple_cache_test",
        ":ple_graph_test",
        ":ple_host_test",
        ":ple_prefetch_test",
        ":qsa_cache_test",
        ":qsa_graph_test",
        ":qsa_mtp_test",
        ":qsa_wiring_test",
        ":wiring_test",
    ],
)
