load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("//:def.bzl", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps", "flashinfer_deps", "select_py_bindings")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_python//python:defs.bzl", "py_runtime_pair")
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
    define_values = {
        "using_cuda12": "true",
        "using_cuda12_9_x86": "true",
    },
)

config_setting(
    name = "cuda_pre_12_9",
    define_values = {
        "using_cuda12_9_x86": "false",
        "using_cuda12_arm": "false",
        "using_cuda13_x86": "false",
    },
)

config_setting(
    name = "using_cuda12_arm",
    values = {"define": "using_cuda12_arm=true"},
)

# using_cuda13_arm is a stricter subset of using_cuda12_arm, so C++/toolchain
# selects that only carry an ARM branch keep working on cuda13_arm. The pip supply
# does not piggyback that way: requirement() gives cuda13_arm its own branch
# pointing at the @arch_config absence stub, because this tree has no cuda13 pip
# supply. This setting additionally enables the code paths that need to
# differentiate CUDA 13 from CUDA 12.
config_setting(
    name = "using_cuda13_arm",
    # Lists every define the cuda13_arm config sets so this setting is a strict
    # specialization of using_cuda / using_cuda12 / using_cuda12_arm: selects
    # that carry those branches alongside this one stay unambiguous (Bazel
    # picks the stricter match).
    define_values = {
        "using_cuda": "true",
        "using_cuda12": "true",
        "using_cuda12_arm": "true",
        "using_cuda13_arm": "true",
    },
)

# x86_64 counterpart of using_cuda13_arm — same CUDA-13-vs-12 differentiation,
# applied on x86 builds.  The CUDA-13 variants of the C++ deps (cutlass /
# flashinfer) are not declared in this tree, so the branches guarded by this
# setting point at the @arch_config absence stubs instead of at a CUDA-12 build.
# define_values lists ALL flags this config requires so Bazel can detect
# specialization: a select() with both `using_cuda` and `using_cuda13_x86`
# keys picks `using_cuda13_x86` for cuda13 builds (it's the strict superset).
# Without listing using_cuda + using_cuda12 here, Bazel would report
# "multiple matching configs" for those selects.
config_setting(
    name = "using_cuda13_x86",
    define_values = {
        "using_cuda": "true",
        "using_cuda12": "true",
        "using_cuda13_x86": "true",
    },
)

config_setting(
    name = "using_rocm",
    values = {"define": "using_rocm=true"},
)

config_setting(
    name = "rocm_gfx950",
    define_values = {
        "using_rocm": "true",
        "gfx950": "true",
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

# Selects whose cuda13_x86 behavior is identical to cuda12_9_x86 use this
# group so we don't have to add a parallel branch to every select().
selects.config_setting_group(
    name = "using_cu12_9_or_13_x86",
    match_any = [
        ":using_cuda12_9_x86",
        ":using_cuda13_x86",
    ],
)

config_setting(
    name = "xft_use_icx",
    values = {"define": "xft_use_icx=true"},
)

config_setting(
    name = "using_ppu",
    define_values = {"use_ppu": "true"},
)

config_setting(
    name = "using_remote_kv_cache",
    define_values = {"use_remote_kv_cache": "true"},
)

cc_binary(
    name = "th_transformer_config",
    copts = copts(),
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_config_lib",
    ],
)

cc_binary(
    name = "th_grammar_tokenizer_info",
    copts = copts(),
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/engine_base/grammar:grammar_tokenizer_info_python",
    ],
)

cc_binary(
    name = "rtp_compute_ops",
    copts = copts(),
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
        "-Wl,-rpath=$(NVSHMEM_DIR)/lib",
        "-L$(NVSHMEM_DIR)/lib",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + select({
        "@//:using_cuda12": [
            "//rtp_llm/cpp/pybind:th_transformer_gpu",
        ],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "th_transformer",
    srcs = [
        ":rtp_compute_ops",
    ],
    copts = copts(),
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
        # "-Wl,--exclude-libs,ALL",  # add this line to hide static-library symbols
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_lib",
    ],
)

exports_files(["cc_test_wrapper.sh"])

py_runtime(
    name = "python310",
    interpreter_path = "/opt/conda310/bin/python",
    python_version = "PY3",
    stub_shebang = "#!/opt/conda310/bin/python",
    visibility = ["//visibility:public"],
)

# conda310 registered as the official python toolchain.
# Under bzlmod, pip hubs and py_* rules go through toolchain resolution
# (@bazel_tools//tools/python:toolchain_type); the old --python_top cannot feed it — on 7.7.1
# analysis, a missing registration means "No matching toolchains". target and exec share the
# same runtime (conda310 is this project's only interpreter; build tools and runtime share the same source).
py_runtime_pair(
    name = "python310_pair",
    py2_runtime = None,
    py3_runtime = ":python310",
)

toolchain(
    name = "python310_toolchain",
    toolchain = ":python310_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
    visibility = ["//visibility:public"],
)

refresh_compile_commands(
    name = "refresh_compdb",
    targets = {
        "//rtp_llm/cpp/model_rpc:model_rpc_server": "--config=cuda12_6 --config=debug --sandbox_base=/mnt/ram/",
    },
)
