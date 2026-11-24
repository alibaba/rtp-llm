cc_library(
    name = "ares",
    srcs = glob(["src/lib/**/*.c", "src/lib/**/*.h"]),
    hdrs = glob(["include/*.h"]),
    copts = ["-D_GNU_SOURCE", "-DHAVE_CONFIG_H"],
    includes = ["include", "src/lib", "src/lib/include"],
    deps = ["@rtp_llm//3rdparty/media_network:cares_config"],
    visibility = ["//visibility:public"],
)
