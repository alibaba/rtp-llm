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
    "@nvshmem//:nvshmem_host",
    ":deep_ep_hdrs",
]
this_copts = cuda_default_copts_without_arch() + [
    '-nvcc_options=relocatable-device-code=true',
    '-nvcc_options=ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage',
    '--cuda-include-ptx=sm_90', '--cuda-gpu-arch=sm_90',
]

# NVSHMEM 3.4.5 places device-side state (nvshmemi_device_state_d) in
# libnvshmem_device.a and strips it from the shared libnvshmem_host.so's
# symbol table with hidden visibility.  Without explicitly linking the static
# device archive here, ld can't resolve the reference emitted by cudaGetSymbolAddress
# in csrc/kernels/runtime.cu.
this_linkopts = [
    "-L/usr/lib64",
    "-l:libnvshmem_device.a",
    "-l:libnvshmem.a",
]

cc_library(
    name = "runtime_cu",
    srcs = ["csrc/kernels/runtime.cu"],
    deps = deps,
    copts = this_copts,
    linkopts = this_linkopts,
)

cc_library(
    name = "internode_cu",
    srcs = ["csrc/kernels/internode.cu"],
    deps = deps,
    copts = this_copts,
    linkopts = this_linkopts,
)

cc_library(
    name = "internode_ll_cu",
    srcs = ["csrc/kernels/internode_ll.cu"],
    deps = deps,
    copts = this_copts,
    linkopts = this_linkopts,
)

cc_library(
    name = "intranode_cu",
    srcs = ["csrc/kernels/intranode.cu"],
    deps = deps,
    copts = this_copts,
    linkopts = this_linkopts,
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
    # NVSHMEM 3.4.5 ships the device-side state (nvshmemi_device_state_d) and
    # the nvshmemi_transfer_* templates as weak symbols inside
    # /usr/lib64/libnvshmem_device.a.  Pass the archive directly to both the
    # --device-link step (nvlink) and the final shared-library link (host ld)
    # so both layers resolve.  Extract the archive objects first so nvlink
    # considers every TU even if it appears unused at that point.
    mkdir -p nvshmem_device_objs && (cd nvshmem_device_objs && ar x /usr/lib64/libnvshmem_device.a)
    /usr/local/cuda/bin/nvcc --compiler-options "-fPIC" -gencode=arch=compute_90,code=sm_90 --device-link $$o1 $$o2 $$o3 $$o4 nvshmem_device_objs/*.o -L/usr/lib64 -l:libnvshmem.a -o tmp.o && /usr/local/cuda/bin/nvcc -shared $$o1 $$o2 $$o3 $$o4 tmp.o nvshmem_device_objs/*.o -L/usr/lib64 -l:libnvshmem.a -o $(OUTS)
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
