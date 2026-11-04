cc_library(
    name = "lz4",
    linkopts = ["-llz4"],
    visibility = ["//visibility:public"],
)


# hdrs = [ "include/" + hdr for hdr in hdrs ]
# outs = [static_lib]

# rpms = ["@lz4//file:file", "@lz4-devel//file:file", "@lz4-static//file:file"]
# include_path = "usr/include"
# lib_path = "usr/lib64"

# bash_cmd = "for e in $(SRCS); do rpm2cpio $$e | cpio -idm; done"
# bash_cmd += "&& cp -L " + lib_path + "/*.a" + " $(@D)/"

# genrule(
#     name = "lz4_static" + "_files_genrule",
#     srcs = rpms,
#     outs = outs,
#     cmd = bash_cmd,
#     visibility = ["//visibility:public"],
# )

# filegroup(
#     name = "lz4_static_files",
#     srcs = [static_lib],
#     visibility = ["//visibility:public"],
# )

# cc_import(
#     name = "lz_static_import",
#     static_library = static_lib,
#     alwayslink=True
# )

# cc_library(
#     name = "lz4_static",
#     hdrs = ["lz4_hdrs_fg"],
#     deps = ["lz_static_import"],
#     visibility = ["//visibility:public"],
#     strip_include_prefix = "include",
#     include_prefix = None,
#     linkstatic=True
# )

