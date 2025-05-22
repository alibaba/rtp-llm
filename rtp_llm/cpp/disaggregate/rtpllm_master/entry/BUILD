load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "master_init_parameter",
    srcs = ["MasterInitParameter.cpp"],
    hdrs = ["MasterInitParameter.h"],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "random_generator",
    hdrs = ["RandomRequestIdGenerator.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "http_server",
    srcs = ["MasterHttpServer.cpp"],
    hdrs = ["MasterHttpServer.h", "Response.h"],
    deps = [
        ":master_init_parameter",
        ":random_generator",
        "//rtp_llm/cpp/api_server/common:common_service",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/tokenize:tokenize_service"
    ],
)

cc_library(
    name = "entry",
    srcs = ["RtpLLMMasterEntry.cpp", "Init.cpp"],
    hdrs = ["RtpLLMMasterEntry.h"],
    deps = [
        ":master_init_parameter",
        ":http_server",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "rtpllm_master",
    deps = [
        ":entry",
    ],
    linkshared = 1,    
    visibility = ["//visibility:public"],
)