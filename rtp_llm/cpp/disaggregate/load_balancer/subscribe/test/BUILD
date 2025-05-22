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

cc_binary(
    name = "nacos_subscribe_service_main",
    srcs = [
        "NacosSubscribeServiceMain.cpp",
    ],
    deps = [
        ":subscribe_deps",
        "@nacos_sdk_cpp//:nacos_sdk",
    ],
    linkstatic = True,
)