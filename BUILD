load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps", "flashinfer_deps", "select_py_bindings")
load("@bazel_skylib//lib:selects.bzl", "selects")
flashinfer_deps()

config_setting(
    name = "enable_triton",
    values = {"define": "enable_triton=true"},
)

config_setting(
    name = "using_cuda",
    define_values = {"using_cuda": "true"},
)

config_setting(
    name = "using_cuda12",
    values = {"define": "using_cuda12=true"},
)

config_setting(
    name = "using_cuda12_9_x86",
    define_values = {"using_cuda12": "true", "using_cuda12_9_x86": "true"},
)

config_setting(
    name = "cuda_pre_12_9",
    define_values = {"using_cuda12_9_x86": "false", "using_cuda12_arm": "false"},
)

config_setting(
    name = "using_cuda12_arm",
    values = {"define": "using_cuda12_arm=true"},
)

config_setting(
    name = "using_cuda12_x86",
    values = {"define": "using_cuda12_x86=true"},
)

config_setting(
    name = "using_rocm",
    values = {"define": "using_rocm=true"},
)

config_setting(
    name = "using_aiter_src",
    values = {"define": "using_aiter_src=true"},
)

config_setting(
    name = "using_rocm_with_aiter_src",
    define_values = {
        "using_rocm": "true",
        "using_aiter_src": "true",
    },
)

config_setting(
    name = "using_arm",
    values = {"define": "using_arm=true"},
)

config_setting(
    name = "using_cpu",
    values = {"define": "using_cpu=true"},
)

selects.config_setting_group(
    name = "using_cuda12_9",
    match_any = [
        ":using_cuda12_9_x86",
        ":using_cuda12_arm",
    ],
)

config_setting(
    name = "xft_use_icx",
    values = {"define": "xft_use_icx=true"},
)

config_setting(
    name = "using_3fs",
    define_values = {"use_3fs": "true",},
)

cc_binary(
    name = "th_transformer_config",
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_config_lib",
    ],
    copts = copts(),
    linkshared = 1,
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "rtp_compute_ops",
    deps = [
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + select({
        "@//:using_cuda12": [
            "//rtp_llm/cpp/pybind:th_transformer_gpu",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    linkshared = 1,
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
        "-Wl,-rpath=$(NVSHMEM_DIR)/lib",
        "-L$(NVSHMEM_DIR)/lib",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "th_transformer",
    srcs = [
        ":rtp_compute_ops",
    ],
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_lib",
    ],
    copts = copts(),
    linkshared = 1,
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
        # "-Wl,--exclude-libs,ALL",  # 添加这行，隐藏静态库符号
    ],
    visibility = ["//visibility:public"],
)


py_runtime(
    name = "python310",
    interpreter_path = "/opt/conda310/bin/python",
    visibility = ["//visibility:public"],
    python_version = "PY3",
    stub_shebang = "#!/opt/conda310/bin/python"
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compdb",
    targets = {
        "//rtp_llm/cpp/model_rpc:model_rpc_server": "--config=cuda12_6 --config=debug --sandbox_base=/mnt/ram/",
    },
)
