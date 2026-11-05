cc_library(
    name = "elfutils-libelf_headers",
    hdrs = ["gelf.h", "libelf.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "elfutils-libelf",
    shared_library = "libelf.so",
    deps = [":elfutils-libelf_headers"],
    visibility = ["//visibility:public"],
)
