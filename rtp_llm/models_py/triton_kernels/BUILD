py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"],
    visibility = ["//visibility:public"]
)

py_library(
    name = "causal_conv1d",
    srcs = glob(["causal_conv1d/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "moe",
    srcs = glob(["moe/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "sparse_mla",
    srcs = glob(["sparse_mla/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "sparse_msa",
    srcs = glob([
        "sparse_msa/*.py",
        "sparse_msa/common/*.py",
        "sparse_msa/prefill/*.py",
        "sparse_msa/decode/*.py",
    ]),
    deps = select({
        "@//:using_cuda": ["//rtp_llm:flashinfer-python"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"]
)

py_library(
    name = "triton_kernels",
    # Keep root-level kernels in the wheel.  Subdirectories have their own
    # libraries above, but this package previously had no srcs at all, so a
    # newly added root module could pass source-tree tests and still be absent
    # from the installed package.
    srcs = glob(["*.py"]),
    deps = [
        ":fla",
        ":causal_conv1d",
        ":moe",
        ":common",
        ":sparse_mla",
        ":sparse_msa",
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"]
)
