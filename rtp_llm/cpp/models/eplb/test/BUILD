load("//:def.bzl", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "th_eplb_stage_test_lib",
    srcs = ["eplb_stage_test.cc"],
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:rank_layout",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    copts = copts() + ["-fno-access-control"],
    alwayslink = True,
)

cc_binary(
    name = "th_eplb_stage_test",
    deps = [":th_eplb_stage_test_lib"],
    linkshared = 1,
)

py_test(
    name = "eplb_stage_test",
    srcs = ["eplb_stage_test.py"],
    data = [":th_eplb_stage_test"],
    deps = ["//rtp_llm:testlib"],
    timeout = "long",
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "4"},
    tags = ["H20_CU13"],
    env = {"GPU_COUNT": "4"},
)

cc_library(
    name = "th_eplb_py_wrapper_test_lib",
    srcs = [
        "eplb_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/pybind:th_compute_lib",
        "//rtp_llm/cpp/pybind:th_utils",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_eplb_py_wrapper_test",
    deps = [":th_eplb_py_wrapper_test_lib"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "eplb_py_wrapper_test",
    srcs = [
        "eplb_py_wrapper_test.py",
    ],
    data = [
        ":th_eplb_py_wrapper_test",
    ],
    deps = [
        "//rtp_llm:torch",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
