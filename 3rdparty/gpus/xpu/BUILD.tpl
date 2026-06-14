
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_xpu",
    values = {
        "define": "using_xpu=true",
    },
)

cc_library(
    name = "xpu_headers",
    hdrs = glob(["include/**"]),
    includes = [
        ".",
        "include",
        "include/sycl",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sycl_runtime",
    srcs = %{sycl_runtime_srcs},
    data = %{sycl_runtime_srcs},
    includes = [
        ".",
        "include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ze_loader",
    srcs = %{ze_loader_srcs},
    data = %{ze_loader_srcs},
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

# Meta-target: depend on this to get the full XPU runtime.
cc_library(
    name = "xpu",
    visibility = ["//visibility:public"],
    deps = [
        ":xpu_headers",
        ":sycl_runtime",
        ":ze_loader",
    ],
)

%{copy_rules}
