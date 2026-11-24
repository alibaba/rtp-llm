exports_files(glob(["*.BUILD", "*.h"]))

cc_library(
    name = "cares_config",
    hdrs = ["ares_build.h", "ares_config.h"],
    includes = ["."],
    visibility = ["@media_cares//:__pkg__"],
)

cc_library(
    name = "curl_config",
    hdrs = ["curl_config.h"],
    includes = ["."],
    visibility = ["@media_curl//:__pkg__"],
)
