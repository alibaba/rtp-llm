package(default_visibility = ["//visibility:public"])

cc_library(
    name = "zstd",
    srcs = glob([
        "lib/*.c",
        "lib/common/*.c",
        "lib/compress/*.c",
        "lib/decompress/*.c",
        "lib/decompress/*.S",
        "lib/decompress/*.s",
        "lib/dict/*.c",
    ]),
    hdrs = glob(["lib/*.h", "lib/common/*.h", "lib/compress/*.h", "lib/decompress/*.h", "lib/dict/*.h"]),
    includes = ["lib"],
    copts = ["-fPIC"],
)
