"""cc_toolchain_config rule for configuring Intel XPU (SYCL) toolchain on Linux."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
)
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "CC_FLAGS_MAKE_VARIABLE_ACTION_NAME",
    "CPP_COMPILE_ACTION_NAME",
    "CPP_HEADER_PARSING_ACTION_NAME",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_EXECUTABLE_ACTION_NAME",
    "CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_STATIC_LIBRARY_ACTION_NAME",
    "CPP_MODULE_CODEGEN_ACTION_NAME",
    "CPP_MODULE_COMPILE_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
    "LINKSTAMP_COMPILE_ACTION_NAME",
    "LTO_BACKEND_ACTION_NAME",
    "LTO_INDEXING_ACTION_NAME",
    "PREPROCESS_ASSEMBLE_ACTION_NAME",
    "STRIP_ACTION_NAME",
)

ACTION_NAMES = struct(
    c_compile = C_COMPILE_ACTION_NAME,
    cpp_compile = CPP_COMPILE_ACTION_NAME,
    linkstamp_compile = LINKSTAMP_COMPILE_ACTION_NAME,
    cc_flags_make_variable = CC_FLAGS_MAKE_VARIABLE_ACTION_NAME,
    cpp_module_codegen = CPP_MODULE_CODEGEN_ACTION_NAME,
    cpp_header_parsing = CPP_HEADER_PARSING_ACTION_NAME,
    cpp_module_compile = CPP_MODULE_COMPILE_ACTION_NAME,
    preprocess_assemble = PREPROCESS_ASSEMBLE_ACTION_NAME,
    lto_indexing = LTO_INDEXING_ACTION_NAME,
    lto_backend = LTO_BACKEND_ACTION_NAME,
    cpp_link_executable = CPP_LINK_EXECUTABLE_ACTION_NAME,
    cpp_link_dynamic_library = CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_nodeps_dynamic_library = CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_static_library = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
    strip = STRIP_ACTION_NAME,
)

def _impl(ctx):
    # Forced configuration paths optimized for Linux execution engines
    toolchain_identifier = "local_linux"
    host_system_name = "local"
    target_system_name = "local"
    target_cpu = "local"
    target_libc = "local"
    compiler = "compiler"
    abi_version = "local"
    abi_libc_version = "local"

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    action_configs = []

    # Foundational compilation settings
    supports_pic_feature = feature(name = "supports_pic", enabled = True)
    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)
    supports_interface_shared_libraries_feature = feature(name = "supports_interface_shared_libraries", enabled = True)

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                ],
            ),
        ],
    )

    # Core XPU and Ahead-of-Time SYCL Compilation Features
    xpu_link_flags_feature = feature(
        name = "xpu_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                # -L points the linker at the directory where libze_loader.so was
                # probed (it may live in oneAPI's lib dir, not a default search
                # path); see xpu_configure.bzl ze_loader probe.
                flag_groups = [flag_group(flags = ["-fsycl", "-L%{ze_loader_lib_dir}", "-lze_loader"])],
            ),
        ],
    )

    # Not enabled globally — only needed for SYCL kernel sources.
    # Enable per-target: cc_library(features = ["xpu_sycl_compile"])
    xpu_sycl_compile_feature = feature(
        name = "xpu_sycl_compile",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fsycl", "-fsycl-targets=%{xpu_sycl_target}"])],
            ),
        ],
    )

    hardening_feature = feature(name = "hardening", flag_sets = [])

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-g0", "-O2", "-ffunction-sections", "-fdata-sections"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
            ),
        ],
        implies = ["disable-assertions"],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-g"])],
            ),
        ],
    )

    disable_assertions_feature = feature(
        name = "disable-assertions",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            ),
        ],
    )

    fastbuild_feature = feature(name = "fastbuild")

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    unfiltered_flag_sets = []
    if ctx.attr.host_unfiltered_compile_flags:
        unfiltered_flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [flag_group(flags = ctx.attr.host_unfiltered_compile_flags)],
            ),
        ]

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = unfiltered_flag_sets,
    )

    warnings_feature = feature(
        name = "warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-Wall"] + ctx.attr.host_compiler_warnings)],
            ),
        ],
    )

    determinism_feature = feature(
        name = "determinism",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                        ],
                    ),
                ],
            ),
        ],
    )

    features = [
        supports_pic_feature,
        supports_dynamic_linker_feature,
        supports_interface_shared_libraries_feature,
        pic_feature,
        xpu_link_flags_feature,
        xpu_sycl_compile_feature,
        hardening_feature,
        opt_feature,
        dbg_feature,
        disable_assertions_feature,
        fastbuild_feature,
        user_compile_flags_feature,
        unfiltered_compile_flags_feature,
        warnings_feature,
        determinism_feature,
    ]

    # Resolve compiler tool path directly using workspace discovery parameters
    final_compiler_path = "/usr/bin/gcc"
    if ctx.attr.host_compiler_path:
        final_compiler_path = ctx.attr.host_compiler_path

    # Fallback guard clause to handle blank/missing compiler prefixes safely
    prefix = "/usr/bin"
    if ctx.attr.host_compiler_prefix:
        prefix = ctx.attr.host_compiler_prefix

    tool_paths = [
        tool_path(name = "gcc", path = final_compiler_path),
        tool_path(name = "ar", path = prefix + "/ar"),
        tool_path(name = "compat-ld", path = prefix + "/ld"),
        tool_path(name = "cpp", path = prefix + "/cpp"),
        tool_path(name = "dwp", path = prefix + "/dwp"),
        tool_path(name = "gcov", path = prefix + "/gcov"),
        tool_path(name = "ld", path = prefix + "/ld"),
        tool_path(name = "nm", path = prefix + "/nm"),
        tool_path(name = "objcopy", path = prefix + "/objcopy"),
        tool_path(name = "objdump", path = prefix + "/objdump"),
        tool_path(name = "strip", path = prefix + "/strip"),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        toolchain_identifier = toolchain_identifier,
        host_system_name = host_system_name,
        target_system_name = target_system_name,
        target_cpu = target_cpu,
        target_libc = target_libc,
        compiler = compiler,
        abi_version = abi_version,
        abi_libc_version = abi_libc_version,
        cxx_builtin_include_directories = ctx.attr.builtin_include_directories,
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["local", "darwin", "x64_windows"]),
        "host_compiler_warnings": attr.string_list(),
        "host_unfiltered_compile_flags": attr.string_list(),
        "builtin_include_directories": attr.string_list(),
        "extra_no_canonical_prefixes_flags": attr.string_list(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "linker_bin_path": attr.string(default = "/usr/bin"),

        # Stubs for deprecated parameters to stop the configuration engine from crashing
        "msvc_cl_path": attr.string(default = ""),
        "msvc_lib_path": attr.string(default = ""),
        "msvc_link_path": attr.string(default = ""),
        "msvc_ml_path": attr.string(default = ""),
        "msvc_env_path": attr.string(default = ""),
        "msvc_env_include": attr.string(default = ""),
        "msvc_env_lib": attr.string(default = ""),
        "msvc_env_tmp": attr.string(default = ""),
    },
    provides = [CcToolchainConfigInfo],
)
