load("//:def.bzl", "copts")


cc_library(
    name = "hie_bfloat16",
    hdrs = [
        "hie_bfloat16.hpp",
    ],
    srcs = [
        "bfloat16_impl.hpp",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "hie_bfloat16_cmath",
    hdrs = [
        "hie_bfloat16_cmath.hpp",
    ],
    srcs = [
        "bfloat16_cmath_impl.hpp",
    ],
    deps = [
        ":hie_bfloat16",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)