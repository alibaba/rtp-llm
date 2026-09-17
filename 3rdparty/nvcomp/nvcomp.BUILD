package(default_visibility = ["//visibility:public"])

# The Python wheel omits two generated C API headers. The version header is not
# used by this binding; the export annotation does not change the C ABI.
genrule(
    name = "generated_headers",
    outs = ["compat/nvcomp/version.h", "compat/nvcomp_export.h"],
    cmd = "printf '#pragma once\n' > $(location compat/nvcomp/version.h); " +
          "printf '#pragma once\n#define NVCOMP_EXPORT __attribute__((visibility(\"default\")))\n' > $(location compat/nvcomp_export.h)",
)

cc_import(
    name = "nvcomp_shared",
    shared_library = "nvidia/nvcomp/libnvcomp.so.5",
)

cc_library(
    name = "crc32",
    hdrs = [
        "nvidia/nvcomp/include/nvcomp.h",
        "nvidia/nvcomp/include/nvcomp/crc32.h",
        "nvidia/nvcomp/include/nvcomp/shared_types.h",
        ":generated_headers",
    ],
    includes = ["nvidia/nvcomp/include", "compat"],
    deps = [":nvcomp_shared", "@local_config_cuda//cuda:cuda_headers"],
)

filegroup(name = "shared_library", srcs = ["nvidia/nvcomp/libnvcomp.so.5"])
