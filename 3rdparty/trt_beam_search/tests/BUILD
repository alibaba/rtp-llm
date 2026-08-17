load("//:def.bzl", "any_cuda_copts", "cc_test_wrapper")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "topk_bench",
    srcs = ["topk_bench.cu"],
    copts = any_cuda_copts(),
    deps = [
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ] + select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
)

cc_test = cc_test_wrapper

# Runs the bench's host-reference semantic checks (--check mode) as a proper test
# target so GPU CI exercises the topk path partition (WarpSort gate / radix) and
# the sorted/unsorted output contracts.
cc_test(
    name = "topk_semantic_test",
    srcs = ["topk_bench.cu"],
    args = ["--check"],
    copts = any_cuda_copts(),
    deps = [
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ] + select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
)
