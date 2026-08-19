load("//:def.bzl", "any_cuda_copts", "cc_test_wrapper")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "topk_bench",
    srcs = ["topk_bench.cu"],
    copts = any_cuda_copts(),
    tags = ["manual"],
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
# target so GPU CI exercises the topk path partition (WarpSort gate / radix), the
# sorted/unsorted output contracts, and multi-vs-one-block bitwise consistency.
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

# Focused regression for the V2 id-mapping math (addCumLogProbs strides, gatherId
# divisors): runs the stage pipeline on a small deterministic input and compares
# the final token ids / beam lineage against a host reference element-wise.
cc_test(
    name = "beam_mapping_test",
    srcs = ["beam_mapping_test.cu"],
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
