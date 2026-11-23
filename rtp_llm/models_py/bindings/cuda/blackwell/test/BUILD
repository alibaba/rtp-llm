load("@rtp_blackwell_config//:pins.bzl", "CUDA_EXTENSION")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "wheel_assets_test",
    srcs = ["wheel_assets_test.py"],
    args = ["$(rootpath //rtp_llm:rtp_llm)"],
    data = ["//rtp_llm"],
    tags = ["manual"],
    target_compatible_with = select({
        "//:using_glm53_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) if CUDA_EXTENSION != None else ["@platforms//:incompatible"],
)
