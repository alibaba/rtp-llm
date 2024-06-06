load("//:def.bzl", "copts", "torch_deps")


test_linkopts = [
    "-lpython3.10",
    "-ltorch",
#     # "-ltorch_cpu",
#     # "-ltorch_python",
]


test_deps = [
    "@com_google_googletest//:gtest",
] + torch_deps() 


cc_test(
    name = "qbuffer_test",
    srcs = [
        "QBufferTest.cc",
    ],
    copts = copts(),
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/core:buffer",
        "//src/fastertransformer/core:types",
        "//src/fastertransformer/core:buffer_torch",
        
    ],
    visibility = ["//visibility:public"],
)