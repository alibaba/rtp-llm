load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("@//3rdparty/tensorrt_llm:template.bzl", "header", "source")

py_library(
    name = "setup_py",
    srcs = [
        "cpp/kernels/fmha_v2/setup.py",
    ],
    deps = [],
)

genrule(
    name = "generate_cu",
    tools = [":setup_py"],
    cmd = "loc=$(locations @tensorrt_llm//:setup_py); loc=$${loc%/*};cd $$loc && /opt/conda310/bin/python -m setup && cd -; rm -rf $(RULEDIR)/generated && mv $$loc/generated $(RULEDIR)",
    outs = header + source,
    tags=["local"],
)

cc_library(
    name = "fmha_v2_lib_header",
    hdrs = glob([
        "cpp/kernels/fmha_v2/src/**/*.h",
    ]),
    copts = cuda_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",        
    ],
    strip_include_prefix = "cpp/kernels/fmha_v2/src",
)

cc_library(
    name = "fmha_v2_lib",
    hdrs = header,
    srcs = source,
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":fmha_v2_lib_header",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
    strip_include_prefix = "generated",
)
