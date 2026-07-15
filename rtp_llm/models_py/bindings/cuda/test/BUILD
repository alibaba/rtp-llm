load("//:def.bzl", "any_cuda_copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

any_cuda_deps = select({
    "@//:using_cuda": [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "vocab_prune_mapping_test",
    srcs = ["VocabPruneMappingTest.cc"],
    copts = any_cuda_copts(),
    deps = any_cuda_deps + [
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {"gpu": "H20"},
)
