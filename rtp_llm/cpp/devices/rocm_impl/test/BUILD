load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + rocm_copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
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
    "-lhsa-amd-aqlprofile64",
    "-lroctx64",
    "-lrocprofiler64v2",
    "-lrocprofiler-sdk",
    "-lhsa-runtime64",
    "-lrocprofiler-register",
    "-lrocprofiler64",
    "-lrocprofiler-sdk-roctx",
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
    name = "rocm_basic_test",
    srcs = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
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

# cc_test(
#     name = "rocm_ffn_op_test",
#     srcs = [
#         "ops/RocmFFnOpTest.cc",
#     ],
#     data = [],
#     env = device_test_envs(),
#     copts = test_copts + copts(),
#     linkopts = test_linkopts,
#     deps = test_deps,
#     tags = ["rocm"],
#     exec_properties = {'gpu':'MI308X'},
# )

cc_test(
    name="rocm_attention_op_test",
    srcs=[
        "ops/ROCmAttentionOpTest.cc",
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
