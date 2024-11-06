load("//bazel:arch_select.bzl", "subscribe_deps")
subscribe_deps()

cc_binary(
    name = "local_subscribe_service_main",
    srcs = [
        "LocalSubscribeServiceMain.cpp",
    ],
    deps = [
        ":subscribe_deps",
    ],
    linkstatic = True,
)