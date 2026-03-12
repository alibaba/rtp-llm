load("//:def.bzl", "cc_test_wrapper", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

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
    name = "rocm_mask_logits_op_test",
    srcs = [
        "ROCmMaskLogitsOpTest.cc",
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
    name="rocm_speculative_sampling",
    srcs=[
        "ops/SpeculativeSampling.cc",
    ],
    data=[],
    env = device_test_envs(),
    copts=test_copts + copts(),
    linkopts=test_linkopts,
    deps=test_deps + [
        "//rtp_llm/cpp/rocm:speculative_sampling"
    ],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)
