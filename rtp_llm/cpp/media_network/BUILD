load("@arch_config//:arch_select.bzl", "torch_deps")

load(":binding_headers.bzl", "binding_headers")

package(default_visibility = ["//rtp_llm:__subpackages__"])

binding_headers(
    name = "pybind_headers",
    deps = torch_deps(),
)

cc_binary(
    name = "_media_network.so",
    srcs = ["MediaNetwork.cc"],
    copts = ["-std=c++17"],
    linkshared = True,
    deps = [
        ":pybind_headers",
        "@curl//:curl",
        "@cares//:ares",
        "@local_config_python//:python_headers",
    ],
)

py_test(
    name = "media_network_test",
    srcs = ["media_network_test.py"],
    data = [":_media_network.so"],
)
