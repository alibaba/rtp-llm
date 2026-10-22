exports_files(["nvcomp.BUILD"])

# Materialize the linked library as a top-level output for Native packaging.
# The wheel-backed cc_import source otherwise remains in the external repo.
genrule(
    name = "nvcomp_runtime",
    srcs = select({
        "//:using_cuda13_arm": ["@nvcomp_cuda13_arm//:shared_library"],
        "//:using_cuda13_x86": ["@nvcomp_cuda13_x86//:shared_library"],
        "//conditions:default": [],
    }),
    outs = ["libnvcomp.so.5"],
    cmd = "cp \"$(SRCS)\" \"$@\"",
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)
