load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + rocm_copts()

test_linkopts = [
    "-L/opt/rocm/lib",
    "-lhipsolver",
    "-lhiprtc",
    "-lrccl",
    "-lhipblas",
    "-lMIOpen",
    "-lrocblas",
    "-lhipblaslt",
    "-lrocm-core",
    "-lhipfft",
    "-lroctracer64",
    "-lhiprand",
    "-lrocm-dbgapi",
    "-lamd_comgr",
    "-lamdocl64",
    "-lamdhip64",
    "-lrocrand",
    "-lrocm_smi64",
    "-lroctx64",
    "-lhsa-runtime64",
    "-lhipsparse",
    "-lhiprtc-builtins",
    "-lrocsolver",
    "-L/opt/amdgpu/lib64",
    "-ldrm_amdgpu",
    "-ldrm"
]

test_deps = [
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:rocblas",
    "//rtp_llm/cpp/cuda:nccl_util",
    "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/base_tests:base_tests",
] + torch_deps()

cc_test(
    name = "rocm_memory_test",
    srcs = [
        "RocmAllocatorTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rocm_basic_test",
    srcs = [],
    env = device_test_envs(),
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/devices/base_tests:basic_test_cases"
    ],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rocm_ops_test",
    srcs = [
        "RocmOpsTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "gemm_op_test",
    srcs = [
        "ops/ROCmGemmOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rocm_act_op_test",
    srcs = [
        "ops/RocmActOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rocm_ffn_op_test",
    srcs = [
        "ops/RocmFFnOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps + select({
        "@//:using_aiter_src": [
            "@aiter_src//:module_aiter_enum",
            "@aiter_src//:module_moe_asm",
            "@aiter_src//:module_moe_ck2stages",
            "@aiter_src//:module_quant",
            "@aiter_src//:module_moe_sorting",
            "@aiter_src//:module_gemm_a8w8",
        ],
        "//conditions:default": [
            "@aiter//:module_aiter_enum",
            "@aiter//:module_moe_asm",
            "@aiter//:module_moe_ck2stages",
            "@aiter//:module_quant",
            "@aiter//:module_moe_sorting",
            "@aiter//:module_gemm_a8w8",
        ],
    }),
    tags = ["rocm"],
)

cc_test(
    name="rocm_attention_op_test",
    srcs=[
        "ops/ROCmAttentionOpTest.cc",
    ],
    data=[],
    env = device_test_envs() | { "AITER_ASM_DIR": "./external/aiter/aiter_meta/hsa/gfx942/"},
    copts=test_copts + copts(),
    linkopts=test_linkopts,
    deps=test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "embedding_lookup_test",
    srcs = [
        "ops/EmbeddingLookupTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rocm_softmax_op_test",
    srcs = [
        "ops/RocmSoftmaxOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)


cc_test(
    name="sampler_test",
    srcs=[
        "ROCmSamplerTest.cc",
    ],
    data=[],
    env=device_test_envs(),
    copts=test_copts + copts(),
    linkopts=test_linkopts,
    deps=test_deps,
    tags=["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name="layernorm_test",
    srcs=[
        "ops/LayernormTest.cc",
    ],
    data=[],
    env=device_test_envs(),
    copts=test_copts + copts(),
    linkopts=test_linkopts,
    deps=test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_test(
    name = "rccl_test",
    srcs = [
        "RcclTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

cc_binary(
    name = "custom_ar_test",
    srcs = [
        "ops/CustomAllReduceTest.cc",
    ],
    deps = test_deps + [
        "//rtp_llm/cpp/devices/torch_impl:torch_reference_impl",
        "//rtp_llm/cpp/models:models",
    ],
    visibility = ["//visibility:public"],
    linkstatic = False,
    copts = test_copts,
)

cc_test(
    name = "rocm_custom_all_reduce_test",
    srcs = [
        "ops/ROCmCustomAllReduceTest.cc"
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        "//rtp_llm/cpp/devices/rocm_impl/test:custom_ar_test"
    ],
    tags = ["custom_ar_ut", "rocm"],
)
