load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_binary(
    name = "engram_native_dev_492ecf45.so",
    srcs = ["engram_native.cc"],
    copts = copts() + [
        "-DTORCH_EXTENSION_NAME=engram_native_dev_492ecf45",
        "-fvisibility=hidden",
    ],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
    ],
    linkshared = True,
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_library(
    name = "native_runtime",
    data = [":engram_native_dev_492ecf45.so"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_embedding_contract",
    size = "small",
    srcs = ["test_embedding_contract.py"],
    main = "test_embedding_contract.py",
    deps = ["//rtp_llm/models_py:engram_model_cpu"],
    env = {"GPU_COUNT": "0"},
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
