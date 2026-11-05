cc_library(
    name = "tbb_headers",
    hdrs = glob(["tbb/**/*.h", "tbb/**/*"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "tbb",
    shared_library = "libtbb.so",
    deps = [":tbb_headers"],
    visibility = ["//visibility:public"],
)
