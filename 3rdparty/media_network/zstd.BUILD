cc_library(
    name = "zstddec",
    srcs = glob([
        "lib/common/*.c",
        "lib/common/*.h",
        "lib/decompress/*.S",
        "lib/decompress/*.c",
        "lib/decompress/*.h",
    ]),
    hdrs = ["lib/zstd.h", "lib/zstd_errors.h"],
    strip_include_prefix = "lib",
    visibility = ["//visibility:public"],
)
