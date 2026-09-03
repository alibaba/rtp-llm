load("@arch_config//:arch_select.bzl", "copy_all_so")
load("//bazel:defs.bzl", "copy_files")

package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

# FlashInfer's AOT kernels are cc_shared_library targets. Bazel adds them to
# runfiles automatically, but py_wheel does not include transitive shared
# libraries unless they are explicit package data. CUDA 12 images get these
# DSOs from flashinfer-jit-cache; CUDA 13 currently has no matching cache wheel,
# so ship the exact libraries linked by librtp_compute_ops.so in the RTP-LLM
# wheel.
_FLASHINFER_RUNTIME_LIBS = [
    "flashinfer_batch_paged_decode",
    "flashinfer_batch_paged_decode_256",
    "flashinfer_batch_paged_prefill",
    "flashinfer_batch_paged_prefill_256",
    "flashinfer_batch_ragged_prefill",
    "flashinfer_batch_ragged_prefill_256",
    "flashinfer_single_decode",
    "flashinfer_single_decode_256",
    "flashinfer_single_prefill",
    "flashinfer_single_prefill_256",
    "flashinfer_sm90",
]

copy_files(
    name = "copy_flashinfer_runtime_libs",
    srcs = select({
        "//:using_cuda13_x86": ["@flashinfer_cpp_cu13//:%s" % name for name in _FLASHINFER_RUNTIME_LIBS],
        "//:using_cuda13_arm": ["@flashinfer_cpp_cu13//:%s" % name for name in _FLASHINFER_RUNTIME_LIBS],
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
        ":copy_flashinfer_runtime_libs",
        ":librtp_compute_ops_so",
        ":libth_grammar_tokenizer_info_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
    ],
)
