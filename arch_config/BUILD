# Package-root BUILD of @arch_config (open-source stub).
# Holds the absence stubs referenced by the select branches in arch_select.bzl: a profile
# without public supply points at a stub here, so hitting that profile fails explicitly at
# [analysis time] instead of falling back to another profile's supply.
# Other configs do not reference these stubs → not analyzed → no fail triggered
# (select only analyzes the chosen branch).
load("@rtp_llm//deps:absent.bzl", "absent_dep")

absent_dep(
    name = "torch_cuda12_9_arm_absent",
    dep_name = "torch",
    profile = "cuda12_9_arm",
    reason = "cuda12_9_arm has no public C++ torch view (provided by the internal overlay); to use it on the open-source side, add a cc_view in deps.json",
    visibility = ["//visibility:public"],
)

# cuda13 (x86 + arm) has no public pip supply at all: torch 2.11+cu130 and every wheel built
# against it are published only to the internal index, so this tree declares no cuda13 hub or
# lock. requirement() routes both cuda13 configs here so a cuda13 build fails explicitly
# instead of silently taking another profile's wheels (cpu for x86, cuda12_9_arm for arm).
absent_dep(
    name = "python_cuda13_absent",
    dep_name = "python deps",
    profile = "cuda13",
    reason = "cuda13 has no public pip supply (torch 2.11+cu130 and the wheels built against it are only published to the internal index); the internal overlay supplies cuda13",
    visibility = ["//visibility:public"],
)

absent_dep(
    name = "python_cuda13_arm_absent",
    dep_name = "python deps",
    profile = "cuda13_arm",
    reason = "cuda13_arm has no public pip supply (aarch64 torch 2.11+cu130 and the wheels built against it are only published to the internal index); the internal overlay supplies cuda13_arm",
    visibility = ["//visibility:public"],
)

# The C++ torch view for cuda13 (torch 2.11+cu130) has no public cc_view either: it is derived
# from the same internal-only wheel. torch_deps() points the using_cuda13_x86 branch here
# rather than at the cuda12 view, which would link a different ABI behind the same flag.
absent_dep(
    name = "torch_cuda13_absent",
    dep_name = "torch",
    profile = "cuda13",
    reason = "cuda13 has no public C++ torch view (torch 2.11+cu130 is only published to the internal index; the internal overlay provides it); to use it on the open-source side, add a cc_view in deps.json",
    visibility = ["//visibility:public"],
)

absent_dep(
    name = "torch_cuda13_arm_absent",
    dep_name = "torch",
    profile = "cuda13_arm",
    reason = "cuda13_arm has no public C++ torch view (aarch64 torch 2.11+cu130 is only published to the internal index; the internal overlay provides it); to use it on the open-source side, add a cc_view in deps.json",
    visibility = ["//visibility:public"],
)

# The CUDA-13 C++ flashinfer view is a separate source build (its own BUILD file + cuda13
# patches) that this tree does not declare; flashinfer_deps() points the using_cuda13_x86
# branch here instead of at the CUDA-12 build, so the mismatch cannot pass silently.
absent_dep(
    name = "flashinfer_cuda13_absent",
    dep_name = "flashinfer",
    profile = "cuda13",
    reason = "cuda13 has no public C++ flashinfer view (the internal overlay provides it); to use it on the open-source side, declare the cuda13 flashinfer archive in deps/extensions/git_deps.bzl",
    visibility = ["//visibility:public"],
)

absent_dep(
    name = "flashinfer_cuda13_arm_absent",
    dep_name = "flashinfer",
    profile = "cuda13_arm",
    reason = "cuda13_arm has no public C++ flashinfer view (the internal overlay provides it); to use it on the open-source side, declare the cuda13 ARM flashinfer archive in deps/extensions/git_deps.bzl",
    visibility = ["//visibility:public"],
)

# The public package keeps an analysis-time failure for the private CUDA13 nvshmem target; the
# internal arch_config overlay replaces it with a real cc_import from the selected torch wheel.
absent_dep(
    name = "torch_nvshmem_cuda13",
    dep_name = "torch nvshmem library (libtorch_nvshmem.so)",
    profile = "cuda13",
    reason = "it is shipped inside the cuda13 torch wheel, which is only published to the internal index (the internal overlay provides it); to use it on the open-source side, declare a cuda13 pip supply in deps.json",
    visibility = ["//visibility:public"],
)

# The cuda13 CUTLASS view names are consumed by 3rdparty/flashmla/flashmla.BUILD's
# select branches; the internal arch_config maps the same names to the real repos.
absent_dep(
    name = "cutlass_cuda13",
    dep_name = "cutlass (CUDA 13 build)",
    profile = "cuda13",
    reason = "the cuda13 CUTLASS C++ view (v3.8, gates cuTensorMapEncode on __CUDACC_VER_MAJOR__>12) is only provided by the internal overlay; to use it on the open-source side, declare it in deps/extensions/git_deps.bzl",
    visibility = ["//visibility:public"],
)

absent_dep(
    name = "cutlass_cuda13_utils",
    dep_name = "cutlass utils (CUDA 13 build)",
    profile = "cuda13",
    reason = "the cuda13 CUTLASS C++ view is only provided by the internal overlay; to use it on the open-source side, declare it in deps/extensions/git_deps.bzl",
    visibility = ["//visibility:public"],
)
