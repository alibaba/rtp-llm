cc_library(
    name = "nacos_include_header",
    hdrs = glob([
        "include/*.h",
        "include/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "nacos_config_header",
    hdrs = glob([
        "src/config/*.h",
    ]),
    strip_include_prefix = "src/config",
)

cc_library(
    name = "nacos_crypto_inner",
    srcs = [
        "src/crypto/hmac_sha1/hmac/hmac_sha1.cpp",
        "src/crypto/hmac_sha1/sha/sha1.cpp",
        "src/crypto/md5/md5.cpp",
    ],
    hdrs = [
        "src/crypto/base64/base64.h",
        "src/crypto/hmac_sha1/hmac/hmac.h",
        "src/crypto/hmac_sha1/sha/sha.h",
        "src/crypto/md5/md5.h"
    ],
    deps = [
        ":nacos_include_header",
    ],
)

cc_library(
    name = "nacos_crypto_header",
    hdrs = glob([
        "src/crypto/*.h",
    ]),
    strip_include_prefix = "src/crypto",
)

cc_library(
    name = "nacos_debug_header",
    hdrs = glob([
        "src/debug/*.h",
    ]),
    strip_include_prefix = "src/debug",
)

cc_library(
    name = "nacos_factory_header",
    hdrs = glob([
        "src/factory/*.h",
    ]),
    strip_include_prefix = "src/factory",
)

cc_library(
    name = "nacos_http_delegate_header",
    hdrs = glob([
        "src/http/delegate/*.h",
    ]),
    strip_include_prefix = "src/http/delegate",
)

cc_library(
    name = "nacos_http_header",
    hdrs = glob([
        "src/http/*.h",
    ]),
    strip_include_prefix = "src/http",
)

cc_library(
    name = "nacos_init_header",
    hdrs = glob([
        "src/init/*.h",
    ]),
    strip_include_prefix = "src/init",
)

cc_library(
    name = "nacos_json_rapidjson",
    hdrs = glob([
        "src/json/rapidjson/*.h",
        "src/json/rapidjson/error/*.h",
        "src/json/rapidjson/internal/*.h",
        "src/json/rapidjson/misinttypes/*.h",
    ]),
)

cc_library(
    name = "nacos_json_header",
    hdrs = glob([
        "src/json/*.h",
    ]),
    strip_include_prefix = "src/json",
)

cc_library(
    name = "nacos_listen_header",
    hdrs = glob([
        "src/listen/*.h",
    ]),
    strip_include_prefix = "src/listen",
)

cc_library(
    name = "nacos_log_header",
    hdrs = glob([
        "src/log/*.h",
    ]),
    strip_include_prefix = "src/log",
)

cc_library(
    name = "nacos_naming_beat_header",
    hdrs = glob([
        "src/naming/beat/*.h",
    ]),
    strip_include_prefix = "src/naming/beat",
)

cc_library(
    name = "nacos_naming_cache_header",
    hdrs = glob([
        "src/naming/cache/*.h",
    ]),
    strip_include_prefix = "src/naming/cache",
)

cc_library(
    name = "nacos_naming_subscribe_header",
    hdrs = glob([
        "src/naming/subscribe/*.h",
    ]),
    strip_include_prefix = "src/naming/subscribe",
)

cc_library(
    name = "nacos_naming_header",
    hdrs = glob([
        "src/naming/*.h",
    ]),
    strip_include_prefix = "src/naming",
)

cc_library(
    name = "nacos_security_header",
    hdrs = glob([
        "src/security/*.h",
    ]),
    strip_include_prefix = "src/security",
)

cc_library(
    name = "nacos_server_header",
    hdrs = glob([
        "src/server/*.h",
    ]),
    strip_include_prefix = "src/server",
)

cc_library(
    name = "nacos_thread_header",
    hdrs = glob([
        "src/thread/*.h",
    ]),
    strip_include_prefix = "src/thread",
)

cc_library(
    name = "nacos_utils_header",
    hdrs = glob([
        "src/utils/*.h",
    ]),
    strip_include_prefix = "src/utils",
)

cc_library(
    name = "nacos_sdk",
    srcs = glob([
        "src/*.cpp",
        "src/config/*.cpp",
        "src/constant/*.cpp",
        "src/crypto/*.cpp",
        "src/factory/*.cpp",
        "src/http/*.cpp",
        "src/http/delegate/*.cpp",
        "src/init/*.cpp",
        "src/json/*.cpp",
        "src/listen/*.cpp",
        "src/log/*.cpp",
        "src/naming/*.cpp",
        "src/naming/**/*.cpp",
        "src/security/*.cpp",
        "src/server/*.cpp",
        "src/thread/*.cpp",
        "src/utils/*.cpp",
    ]),
    deps = [
        "@curl//:curl",
        ":nacos_include_header",
        ":nacos_config_header",
        ":nacos_crypto_inner",
        "nacos_crypto_header",
        ":nacos_debug_header",
        ":nacos_factory_header",
        ":nacos_http_header",
        ":nacos_http_delegate_header",
        ":nacos_init_header",
        ":nacos_json_rapidjson",
        ":nacos_json_header",
        ":nacos_listen_header",
        ":nacos_log_header",
        ":nacos_naming_beat_header",
        ":nacos_naming_cache_header",
        ":nacos_naming_subscribe_header",
        ":nacos_naming_header",
        ":nacos_security_header",
        ":nacos_server_header",
        ":nacos_thread_header",
        ":nacos_utils_header",
    ],
    copts = [
        "-Wno-unused-result",
        "-Wno-uninitialized",
    ],
    visibility = ["//visibility:public"],
)
