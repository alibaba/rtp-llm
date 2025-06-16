load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("@//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "deep_ep_hdrs",
    hdrs = glob([
        "csrc/*.hpp",
        "csrc/kernels/*.cuh",
    ]),
    includes = ["csrc"],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

deps = torch_deps() + [
    "@local_config_cuda//cuda:cuda_headers",
    "@nvshmem//:nvshmem_device",
    ":deep_ep_hdrs",
]
this_copts = cuda_default_copts_without_arch() + [
    '-nvcc_options=relocatable-device-code=true',
    '-nvcc_options=ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage',
    '--cuda-include-ptx=sm_90', '--cuda-gpu-arch=sm_90',
]

cc_library(
    name = "runtime_cu",
    srcs = ["csrc/kernels/runtime.cu"],
    deps = deps,
    copts = this_copts,
)

cc_library(
    name = "internode_cu",
    srcs = ["csrc/kernels/internode.cu"],
    deps = deps,
    copts = this_copts,
)

cc_library(
    name = "internode_ll_cu",
    srcs = ["csrc/kernels/internode_ll.cu"],
    deps = deps,
    copts = this_copts,
)

cc_library(
    name = "intranode_cu",
    srcs = ["csrc/kernels/intranode.cu"],
    deps = deps,
    copts = this_copts,
)

genrule(
    name = "deep_ep_cu",
    srcs = [
        ":runtime_cu",
        ":intranode_cu",
        ":internode_cu",
        ":internode_ll_cu",
    ],
    outs = ["libdeep_ep_cu.so"],
    cmd = """
    read -r o1 _ <<< "$(locations runtime_cu)"
    read -r o2 _ <<< "$(locations intranode_cu)"
    read -r o3 _ <<< "$(locations internode_cu)"
    read -r o4 _ <<< "$(locations internode_ll_cu)"
    /usr/local/cuda/bin/nvcc --compiler-options "-fPIC" -gencode=arch=compute_90,code=sm_90 --device-link $$o1 $$o2 $$o3 $$o4 -L/usr/lib64 -lnvshmem -o tmp.o && /usr/local/cuda/bin/nvcc -shared -l:libnvshmem.a $$o1 $$o2 $$o3 $$o4 tmp.o -o $(OUTS)
    """,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_ep",
    srcs = [
        'csrc/deep_ep.cpp',
        ":deep_ep_cu",
    ],
    hdrs = [
        "@//3rdparty/deep_ep:deep_ep_api.h",
    ],
    strip_include_prefix = "3rdparty/deep_ep",
    deps = [
        ":deep_ep_hdrs",
    ],
    implementation_deps = [
        "@nvshmem//:nvshmem_host",
    ],
    copts = copts() + [
        '-Wno-reorder',
        '-Wno-unused-variable',
    ],
    linkopts = [
        "-l:nvshmem_bootstrap_uid.so",
    ],
    visibility = ["//visibility:public"],
)
