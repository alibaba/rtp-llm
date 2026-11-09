load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "any_cuda_copts", "cc_test_wrapper", "copts")

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
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda",
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
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    exec_properties = select({
        "@//:using_cuda": {"gpu": "L20_CU13"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
    # A select needs its own CUDA branch: with only the rocm branch present, CUDA
    # falls through to "//conditions:default": {} -- empty exec_properties, which is
    # not a GPU request at all. Both tests then landed on a CPU-only worker and died
    # with "libcuda.so.1: cannot open shared object file" (exit 127) in run 69714248.
    tags = ["L20_CU13"],
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
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    exec_properties = select({
        "@//:using_cuda": {"gpu": "L20_CU13"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
    # A select needs its own CUDA branch: with only the rocm branch present, CUDA
    # falls through to "//conditions:default": {} -- empty exec_properties, which is
    # not a GPU request at all. Both tests then landed on a CPU-only worker and died
    # with "libcuda.so.1: cannot open shared object file" (exit 127) in run 69714248.
    tags = ["L20_CU13"],
)

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
    imports = ["."],
    data = [":beam_search_test_ops.so", "frozen/manifest.json", "frozen/beamSearchKernelsTemplate.h",
            "frozen/addCumLogProbs.cuh", "frozen/CudaBeamSearchOp.cc"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "beam_search_parallel_test",
    srcs = ["beam_search_parallel_test.py"],
    deps = [":sampling_test_support"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

# The frozen sampler always queries and executes auto routing. Exercise the
# candidate's forced paths in separate processes because the switch is cached.
[
    py_test(
        name = "beam_search_parallel_force_%d_test" % path,
        srcs = ["beam_search_parallel_test.py"],
        main = "beam_search_parallel_test.py",
        deps = [":sampling_test_support"],
        env = {"RTP_LLM_BEAM_TOPK_FORCE_PATH": str(path)},
        tags = ["RTX_5000_PRO_CU13"],
        exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
    )
    for path in [1, 2]
]

py_test(
    name = "beam_search_ppu_test",
    srcs = ["beam_search_parallel_test.py"],
    main = "beam_search_parallel_test.py",
    args = [
        "BeamSearchParallelTest.test_large_variable_beams_masked_full_vs_compact_math_reference",
        "BeamSearchParallelTest.test_unmasked_full_vocab_changes_cross_parent_ranking",
    ],
    deps = [":sampling_test_support"],
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)
