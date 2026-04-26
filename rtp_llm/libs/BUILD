load("//bazel:arch_select.bzl", "copy_all_so")
load("//bazel:defs.bzl", "copy_files")

package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

# copy 3fs libs
copy_files(
    name = "copy_hf3fs_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/3fs:hf3fs_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_remote_kv_cache_manager_client_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_acclep_libs",
    srcs = select({
        "//rtp_llm/models_py/bindings/cuda/ops:use_accl_ep": ["//3rdparty/accl_ep:accl_ep_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "frontend_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so",
    ],
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":copy_acclep_libs",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
    ],
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":librtp_compute_ops_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
    ],
)
