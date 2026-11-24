load(":gen/sources.bzl", "bcm_internal_headers", "bcm_sources", "bcm_sources_asm", "crypto_headers", "crypto_internal_headers", "crypto_sources", "crypto_sources_asm", "ssl_headers", "ssl_internal_headers", "ssl_sources")

cc_library(
    name = "ssl",
    srcs = bcm_sources + crypto_sources + ssl_sources + [
        src
        for src in bcm_sources_asm + crypto_sources_asm
        if src.endswith(".S")
    ] + bcm_internal_headers + crypto_internal_headers + ssl_internal_headers,
    hdrs = crypto_headers + ssl_headers,
    copts = ["-DBORINGSSL_IMPLEMENTATION"],
    includes = ["include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
