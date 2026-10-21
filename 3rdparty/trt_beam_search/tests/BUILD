load("//:def.bzl", "copts", "any_cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "test_kernels",
    testonly = True,
    srcs = ["test_kernels.cu"],
    hdrs = ["test_api.h", "frozen/addCumLogProbs.cuh", "frozen/beamSearchKernelsTemplate.h",
            "frozen/CudaBeamSearchOp.cc"],
    deps = [
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
        "//3rdparty/trt_beam_search:trt_beam_search_header",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
    ],
    copts = any_cuda_copts() + ["-I3rdparty/trt_beam_search"],
)

cc_binary(
    name = "beam_search_test_ops.so",
    testonly = True,
    srcs = ["test_ops.cc", "test_sample.cc"],
    deps = [
        ":test_kernels",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:common_defines",
        "//rtp_llm/models_py/bindings/core:type_convert",
        "//rtp_llm/cpp/utils:debug_utils",
    ] + torch_deps(),
    copts = copts(),
    linkshared = True,
)

py_library(
    name = "sampling_test_support",
    testonly = True,
    srcs = ["sampling_test_support.py"],
    data = [":beam_search_test_ops.so", "frozen/manifest.json", "frozen/beamSearchKernelsTemplate.h",
            "frozen/addCumLogProbs.cuh", "frozen/CudaBeamSearchOp.cc"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "beam_search_parallel_test",
    srcs = ["beam_search_parallel_test.py"],
    deps = [":sampling_test_support"],
    tags = ["RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO"},
)
