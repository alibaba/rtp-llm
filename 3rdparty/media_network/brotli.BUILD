licenses(["notice"])  # MIT

cc_library(
    name = "brotlidec",
    srcs = glob([
        "c/common/*.c",
        "c/common/*.h",
        "c/dec/*.c",
        "c/dec/*.h",
    ]),
    hdrs = glob(["c/include/brotli/*.h"]),
    strip_include_prefix = "c/include",
    visibility = ["//visibility:public"],
)
