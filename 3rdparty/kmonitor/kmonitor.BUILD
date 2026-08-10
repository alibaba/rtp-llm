load("@bazel_skylib//lib:selects.bzl", "selects")

# Havenask consumes this BUILD as an external-repository override and can only
# mirror the positive ARM CUDA defines. The mirrored names are intentionally
# limited to using_cuda12_arm, using_cuda13_arm, and using_cuda_arm; their
# predicates must not be mechanically compared with root BUILD because root
# specialization also owns explicit-false values. Any new ARM CUDA variant
# must update the root BUILD/.bazelrc contract and this positive-only mirror.

config_setting(
    name='hack_get_set_env',
    define_values={'hack_get_set_env': 'true'},
    visibility=['//visibility:public']
)

config_setting(
    name = "using_cuda12_arm",
    values = {"define": "using_cuda12_arm=true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda13_arm",
    values = {"define": "using_cuda13_arm=true"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "using_cuda_arm",
    match_any = [
        ":using_cuda12_arm",
        ":using_cuda13_arm",
    ],
    visibility = ["//visibility:public"],
)
