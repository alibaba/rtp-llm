cc_library(
    name = "curl",
    srcs = glob(["lib/**/*.c", "lib/**/*.h"]),
    hdrs = glob(["include/curl/*.h"]),
    copts = [
        "-D_GNU_SOURCE",
        "-DBUILDING_LIBCURL",
        "-DCURL_MAX_WRITE_SIZE=65536",
        "-DHAVE_CONFIG_H",
        "-Iexternal/media_curl/lib",
    ],
    includes = ["include"],
    deps = [
        "@media_boringssl//:ssl",
        "@media_cares//:ares",
        "@rtp_llm//3rdparty/media_network:curl_config",
        "@zlib_archive//:zlib",
    ],
    visibility = ["//visibility:public"],
)
