load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc"
    ],
    hdrs = [
        "allocator.h"
    ],
    copts = copts(),
    deps = [
        "//src/fastertransformer/utils:logger"
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_cuda",
    srcs = [
        "allocator_cuda.cc"
    ],
    hdrs = [
        "allocator_cuda.h"
    ],
    copts = cuda_copts(),
    deps = [
        ":allocator",
        "//src/fastertransformer/cuda:cuda_utils",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_torch",
    srcs = [
        "allocator_torch.cc"
    ],
    hdrs = [
        "allocator_torch.h"
    ],
    copts = copts(),
    deps = torch_deps() + [
        ":allocator_cuda",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_hdr",
    hdrs = [
        "Tensor.h",
    ],
    deps = [
        ":allocator",
    ],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "Tensor",
    srcs = glob([
        "Tensor.cc",
    ]),
    deps = [
        ":tensor_hdr",
        "//src/fastertransformer/cuda:cuda_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
