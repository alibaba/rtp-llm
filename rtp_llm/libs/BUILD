load("@arch_config//:arch_select.bzl", "copy_all_so")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("//bazel:defs.bzl", "copy_files", "copy_so")

package(default_visibility = ["//rtp_llm:__subpackages__"])

selects.config_setting_group(
    name = "using_any_kvcm",
    match_any = [
        "//:using_kvcm_emb_storage",
        "//:using_remote_kv_cache",
    ],
)

copy_all_so()
copy_so("@rtp_llm//:mm_kvcm_writer")

filegroup(
    name = "optional_mm_kvcm_writer",
    srcs = select({
        "//:using_kvcm_emb_storage": [":libmm_kvcm_writer_so"],
        "//conditions:default": [],
    }),
)

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
        ":using_any_kvcm": ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared"],
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
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
        ":copy_remote_kv_cache_manager_client_libs",
        ":optional_mm_kvcm_writer",
    ],
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
        ":libth_grammar_tokenizer_info_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
        ":copy_remote_kv_cache_manager_client_libs",
        ":optional_mm_kvcm_writer",
    ],
)
