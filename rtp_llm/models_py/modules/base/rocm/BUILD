py_library(
    name = "rocm_base",
    srcs = glob(
        ["*.py"],
        exclude = [
            "trt_allreduce.py",
            "vllm_custom_allreduce.py",
        ],
    ),
    deps = [
        "//rtp_llm:torch",
    ],
    data = [
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "trt_allreduce",
    srcs = ["trt_allreduce.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    data = [
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "vllm_custom_allreduce",
    srcs = ["vllm_custom_allreduce.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    data = [
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
