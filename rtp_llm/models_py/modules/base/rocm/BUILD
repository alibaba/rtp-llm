py_library(
    name = "rocm_base",
    srcs = glob(
        ["*.py"],
        exclude = ["trt_allreduce.py"],
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
