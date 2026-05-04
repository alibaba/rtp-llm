"""Repository rule for Ascend CANN autoconfiguration.

`ascend_configure` depends on the following environment variables:

  * `TF_NEED_ASCEND`: Whether to enable building with Ascend CANN.
  * `ASCEND_TOOLKIT_PATH`: The path to the Ascend CANN toolkit. Default is
    `/usr/local/Ascend/ascend-toolkit/latest`.
  * `TF_ASCEND_VERSION`: The version of the CANN toolkit.
"""

load(
    ":cuda_configure.bzl",
    "make_copy_dir_rule",
    "make_copy_files_rule",
    "to_list_of_strings",
    "verify_build_defines",
)

_ASCEND_TOOLKIT_PATH = "ASCEND_TOOLKIT_PATH"
_TF_ASCEND_VERSION = "TF_ASCEND_VERSION"
_TF_NEED_ASCEND = "TF_NEED_ASCEND"
_DEFAULT_ASCEND_TOOLKIT_PATH = "/usr/local/Ascend/ascend-toolkit/latest"

def _ascend_autoconf_impl(repository_ctx):
    """Implementation of the ascend_configure repository rule."""

    # 1. Check TF_NEED_ASCEND environment variable
    tf_need_ascend = repository_ctx.os.environ.get(_TF_NEED_ASCEND, "0")
    if tf_need_ascend != "1":
        _create_dummy_repo(repository_ctx)
        return

    # 2. Determine CANN installation path
    ascend_toolkit_path = repository_ctx.os.environ.get(
        _ASCEND_TOOLKIT_PATH, _DEFAULT_ASCEND_TOOLKIT_PATH
    )

    # 3. Probe CANN headers and libraries
    include_path = ascend_toolkit_path + "/include"
    lib_path = ascend_toolkit_path + "/lib64"

    _check_file_exists(repository_ctx, include_path + "/acl/acl.h", "CANN header")
    _check_file_exists(repository_ctx, lib_path + "/libascendcl.so", "CANN library")

    # 4. Get version number
    ascend_version = repository_ctx.os.environ.get(_TF_ASCEND_VERSION, "")

    # 5. Generate template files
    _ascend_create(repository_ctx, ascend_toolkit_path, include_path,
                   lib_path, ascend_version)


def _check_file_exists(repository_ctx, path, label):
    if not repository_ctx.path(path).exists:
        fail("Cannot find %s at %s" % (label, path))


def _create_dummy_repo(repository_ctx):
    """Create empty BUILD and build_defs.bzl when Ascend is not configured."""
    repository_ctx.file("ascend/BUILD", """
package(default_visibility = ["//visibility:public"])
config_setting(name = "using_ascend", values = {"define": "using_ascend=true"})
""")
    repository_ctx.file("ascend/build_defs.bzl", """
def if_ascend(if_true, if_false = []):
    return select({
        "@local_config_ascend//ascend:using_ascend": if_true,
        "//conditions:default": if_false,
    })

def ascend_default_copts():
    return if_ascend([])

def ascend_is_configured():
    return False
""")


def _ascend_create(repository_ctx, toolkit_path, include_path, lib_path, version):
    """Generate the @local_config_ascend repository from CANN installation."""

    # --- Copy headers ---
    copy_rules = []

    copy_rules.append(
        make_copy_dir_rule(
            repository_ctx,
            name = "ascend-include",
            src_dir = include_path,
            out_dir = "ascend/include",
            exceptions = None,
        )
    )

    # --- Copy library files ---
    lib_files = {
        "libascendcl.so": "ascend",
        "libaclblas.so":  "aclblas",
        "libhccl.so":     "hccl",
    }

    lib_srcs = []
    lib_outs = []
    for lib_name, target_name in lib_files.items():
        src = lib_path + "/" + lib_name
        if repository_ctx.path(src).exists:
            lib_srcs.append(src)
            lib_outs.append("ascend/lib/" + lib_name)

    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "ascend-lib",
            srcs = lib_srcs,
            outs = lib_outs,
        )
    )

    # --- Generate BUILD content ---
    build_content = """package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_ascend",
    values = {{"define": "using_ascend=true"}},
)

cc_library(
    name = "ascend_headers",
    hdrs = glob(["ascend/include/**/*.h"]),
    includes = [".", "ascend/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend",
    srcs = ["ascend/lib/libascendcl.so"],
    data = ["ascend/lib/libascendcl.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aclblas",
    srcs = ["ascend/lib/libaclblas.so"],
    data = ["ascend/lib/libaclblas.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hccl",
    srcs = ["ascend/lib/libhccl.so"],
    data = ["ascend/lib/libhccl.so"],
    includes = [".", "ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

""" + "\n".join(copy_rules)

    repository_ctx.file("ascend/BUILD", build_content)

    # --- Generate build_defs.bzl ---
    repository_ctx.file("ascend/build_defs.bzl", """def if_ascend(if_true, if_false = []):
    return select({{
        "@local_config_ascend//ascend:using_ascend": if_true,
        "//conditions:default": if_false,
    }})

def ascend_default_copts():
    return if_ascend([])

def ascend_is_configured():
    return True
""")

    # --- Generate ascend_config.h ---
    repository_ctx.file("ascend/ascend_config.h", """#ifndef ASCEND_ASCEND_CONFIG_H_
#define ASCEND_ASCEND_CONFIG_H_
#define TF_ASCEND_TOOLKIT_PATH "%s"
#define TF_ASCEND_VERSION "%s"
#endif
""" % (toolkit_path, version))


ascend_configure = repository_rule(
    implementation = _ascend_autoconf_impl,
    environ = [
        _TF_NEED_ASCEND,
        _ASCEND_TOOLKIT_PATH,
        _TF_ASCEND_VERSION,
    ],
)
