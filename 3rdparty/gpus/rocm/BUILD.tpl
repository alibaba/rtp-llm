licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "rocm_headers",
    hdrs = [
        "rocm/rocm_config.h",
        %{rocm_headers}
    ],
    includes = [
        ".",
        "rocm/include",
        "rocm/include/rocrand",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip",
    srcs = ["rocm/lib/%{hip_lib}"],
    data = ["rocm/lib/%{hip_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocblas",
    srcs = ["rocm/lib/%{rocblas_lib}"],
    data = ["rocm/lib/%{rocblas_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hipblas",
    srcs = ["rocm/lib/%{hipblas_lib}"],
    data = ["rocm/lib/%{hipblas_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hipblaslt",
    srcs = ["rocm/lib/%{hipblaslt_lib}"],
    data = ["rocm/lib/%{hipblaslt_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hipfft",
    srcs = ["rocm/lib/%{hipfft_lib}"],
    data = ["rocm/lib/%{hipfft_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hiprand",
    srcs = ["rocm/lib/%{hiprand_lib}"],
    data = ["rocm/lib/%{hiprand_lib}"],
    includes = [
        ".",
        "rocm/include",
        "rocm/include/rocrand",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "miopen",
    srcs = ["rocm/lib/%{miopen_lib}"],
    data = ["rocm/lib/%{miopen_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rccl",
    srcs = ["rocm/lib/%{rccl_lib}"],
    data = ["rocm/lib/%{rccl_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hipblaslt",
    srcs = ["rocm/lib/%{hipblaslt_lib}"],
    data = ["rocm/lib/%{hipblaslt_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers",
        ":hip",
        ":rocblas",
        ":hipblas",
        ":hipblaslt",
        ":hipfft",
        ":hiprand",
        ":miopen",
    ],
)

filegroup(
    name = "rocprim_headers",
    srcs = glob([
        "rocm/include/hipcub/**",
        "rocm/include/rocprim/**",
    ]),
)

cc_library(
    name = "rocprim",
    srcs = [
        "rocm/include/hipcub/hipcub_version.hpp",
        "rocm/include/rocprim/rocprim_version.hpp",
    ],
    hdrs = [":rocprim_headers"],
    includes = [
        ".",
        "rocm/include/hipcub",
        "rocm/include/rocprim",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

%{copy_rules}
