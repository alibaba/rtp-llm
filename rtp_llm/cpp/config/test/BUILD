load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

# Same cuda13 handling as model_rpc/test and multimodal_processor/test: without it the module
# fails to load with "libtorch_nvshmem.so: cannot open shared object file" before any case runs.
cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

# extractVitConfig() maps a Python object, so its cases need a live interpreter. Same wiring as
# rtp_llm/cpp/api_server/test: the gtest cases are linked into a pybind module that a py_test
# imports and runs.
py_test(
    name = "config_unittest",  # needs a py file of the same name
    python_version = "PY3",
    srcs_version = "PY3",
    srcs = [
        "config_unittest.py",
    ],
    deps = [
        ":config_unittest_pylib",
        "//rtp_llm:config_ops",
    ],
    data = [
        # rtp_llm/ops/__init__.py walks runfiles for libth_transformer_config.so at import time and
        # throws if it's missing, so the py_test has to ship the .so alongside the pybind module.
        "//:th_transformer_config",
        "//:th_transformer",
        "//:rtp_compute_ops",
    ],
    imports = [
        ".",
    ],
    # The pybind module links torch, which needs libcuda.so.1 just to import -- so this has
    # to land on a GPU node even though the assertions need no GPU compute.
    exec_properties = {'gpu':'A10'},
)

py_library(
    name = "config_unittest_pylib",
    srcs = [],
    deps = [],
    data = [
        ":config_unittest_lib.so",
    ],
)

cc_binary(
    name = "config_unittest_lib.so",
    srcs = [],
    deps = [
        ":config_unittest_lib",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    data = ["//:rtp_compute_ops"],
    linkshared = 1,
)

cc_library(
    name = "config_unittest_lib",
    srcs = [
        "TestMain.cc",
        "VitConfigExtractTest.cc",
    ],
    deps = [
        "//rtp_llm/cpp/config:vit_config_extract",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    copts = test_copts,
    alwayslink = 1,
)
