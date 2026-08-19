load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

cc_test(
    name = "collective_backend_test",
    srcs = ["CollectiveBackendTest.cc"],
    deps = [
        "//rtp_llm/cpp/comm:collective_backend_impl",
        "//rtp_llm/cpp/comm:collective_backend_pybind_hdrs",
        "//rtp_llm/cpp/runtime:cuda_runtime_impl",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    copts = copts(),
    linkstatic = True,
)
