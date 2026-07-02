"""Repository rule for Intel XPU autoconfiguration.

`xpu_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: The python binary path. Used to detect site-packages
    for the torch_xpu repository.
  * `ONEAPI_ROOT`: Path to Intel oneAPI installation. Default: /opt/intel/oneapi
  * `SYCL_TARGET`: SYCL ahead-of-time compilation target. Default: spir64
    (JIT). Examples: intel_gpu_pvc, intel_gpu_bmg, spir64.
"""

load("//3rdparty/gpus:xpu_python_utils.bzl", "resolve_venv_python")

_ONEAPI_ROOT = "ONEAPI_ROOT"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_SYCL_TARGET = "SYCL_TARGET"
_DEFAULT_SYCL_TARGET = "spir64"

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//3rdparty/gpus/%s.tpl" % tpl),
        substitutions,
    )

def to_list_of_strings(elements):
    result = ""
    for element in elements:
        result += ("\"" + element + "\",")
    return result

def verify_build_defines(params):
    """Verify all variables substituted into crosstool/BUILD are present."""
    missing = []
    pattern = [
        "%{cxx_builtin_include_directories}",
        "%{extra_no_canonical_prefixes_flags}",
        "%{host_compiler_path}",
        "%{host_compiler_prefix}",
        "%{host_compiler_warnings}",
        "%{unfiltered_compile_flags}",
        "%{linker_bin_path}",
        "%{compiler_deps}",
        "%{linker_files}",
        "%{win_linker_files}",
        "%{msvc_cl_path}",
        "%{msvc_env_include}",
        "%{msvc_env_lib}",
        "%{msvc_env_path}",
        "%{msvc_env_tmp}",
        "%{msvc_lib_path}",
        "%{msvc_link_path}",
        "%{msvc_ml_path}",
    ]
    for p in pattern:
        if p not in params:
            missing.append(p)
    if missing:
        auto_configure_fail(
            "crosstool/BUILD.tpl template is missing these variables: " +
            str(missing) +
            ". Are you using a modified BUILD.tpl? Please update it.")

def auto_configure_fail(msg):
    """Output failure message for auto configuration."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def _cxx_inc_convert(path):
    """Convert path returned by the compiler to its true path."""
    path = path.strip()
    if path.startswith("("):
        path = path.strip("()")
    return path

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Get built-in include directories from compiler."""
    lang = "c++" if lang_is_cpp else "c"
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"])
    stderr = result.stderr
    index1 = stderr.find("#include <...>")
    if index1 == -1:
        return []
    index1 = stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = stderr.find("\n ", index1 + 1)
    if index2 == -1:
        return []
    index3 = stderr.find("End of search list", index2)
    if index3 == -1:
        return []
    inc_dirs = stderr[index2:index3]
    return [
        repository_ctx.path(_cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
        if len(p.strip()) > 0
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)
    includes_cpp_set = {str(d): None for d in includes_cpp}
    return includes_cpp + [
        inc for inc in includes_c if str(inc) not in includes_cpp_set
    ]

def _oneapi_root(repository_ctx):
    """Return the oneAPI root path."""
    oneapi_root = repository_ctx.os.environ.get(_ONEAPI_ROOT, "")
    if oneapi_root:
        # A stale/invalid ONEAPI_ROOT must not silently shadow a working default
        # install. Validate the env-provided path before trusting it.
        if not repository_ctx.path(oneapi_root).exists:
            auto_configure_fail(
                "ONEAPI_ROOT is set to '" + oneapi_root + "' but that path does " +
                "not exist. Unset it to use a default install, or point it at a " +
                "valid Intel oneAPI root.")
        return oneapi_root
    # Try common install locations
    for path in ["/opt/intel/oneapi", "/opt/oneapi"]:
        if repository_ctx.path(path).exists:
            return path
    auto_configure_fail(
        "Cannot find Intel oneAPI. Set ONEAPI_ROOT environment variable " +
        "or install to /opt/intel/oneapi.")

def _find_icx(repository_ctx, oneapi_root):
    """Find icx compiler path."""
    candidate = oneapi_root + "/compiler/latest/bin/icx"
    if repository_ctx.path(candidate).exists:
        return candidate
    icx = repository_ctx.which("icx")
    if icx != None:
        return str(icx)
    auto_configure_fail("Cannot find icx compiler. Ensure Intel oneAPI is installed.")

def _find_icpx(repository_ctx, oneapi_root):
    """Find icpx compiler path."""
    candidate = oneapi_root + "/compiler/latest/bin/icpx"
    if repository_ctx.path(candidate).exists:
        return candidate
    icpx = repository_ctx.which("icpx")
    if icpx != None:
        return str(icpx)
    auto_configure_fail("Cannot find icpx compiler. Ensure Intel oneAPI is installed.")

def _get_sycl_target(repository_ctx):
    """Get the SYCL ahead-of-time compilation target."""
    return repository_ctx.os.environ.get(_SYCL_TARGET, _DEFAULT_SYCL_TARGET)

def _enable_xpu(repository_ctx):
    """Check if XPU build is requested via TF_NEED_XPU env var and oneAPI is available."""
    if repository_ctx.os.environ.get("TF_NEED_XPU", "0") != "1":
        return False
    oneapi_root = repository_ctx.os.environ.get(_ONEAPI_ROOT, "")
    if oneapi_root and repository_ctx.path(oneapi_root).exists:
        return True
    for path in ["/opt/intel/oneapi", "/opt/oneapi"]:
        if repository_ctx.path(path).exists:
            return True
    auto_configure_fail(
        "TF_NEED_XPU=1 but cannot find oneAPI SDK. " +
        "Set ONEAPI_ROOT or install to /opt/intel/oneapi.")

def _get_python_bin(repository_ctx):
    """Get the python binary path, or fail with a clear message."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH, "")
    if python_bin:
        return python_bin
    python_bin = repository_ctx.which("python3")
    if python_bin != None:
        return str(python_bin)
    auto_configure_fail(
        "Cannot find python3 in PATH. Please set PYTHON_BIN_PATH " +
        "environment variable or ensure python3 is installed.")

def _get_python_include(repository_ctx, python_bin):
    """Get the Python C include directory via sysconfig."""
    result = repository_ctx.execute([
        python_bin, "-c",
        "import sysconfig; print(sysconfig.get_path('include'))",
    ])
    if result.return_code != 0:
        auto_configure_fail("Cannot get Python include path: " + result.stderr)
    return result.stdout.strip()

def _get_python_lib(repository_ctx, python_bin):
    """Get the Python shared library path (libpython3.x.so)."""
    result = repository_ctx.execute([
        python_bin, "-c",
        "import sysconfig, os; d=sysconfig.get_config_var('LIBDIR'); " +
        "v=sysconfig.get_config_var('LDVERSION'); " +
        "print(os.path.join(d, 'libpython'+v+'.so'))",
    ])
    if result.return_code != 0:
        auto_configure_fail("Cannot get Python lib path: " + result.stderr)
    return result.stdout.strip()

def _create_dummy_repository(repository_ctx):
    """Create a minimal stub repository when XPU SDK is not available."""
    repository_ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "crosstool",
)

# Stub py_runtime so --python_top=@local_config_xpu//:python_runtime resolves.
py_runtime(
    name = "python_runtime",
    interpreter_path = "/usr/bin/python3",
    python_version = "PY3",
    visibility = ["//visibility:public"],
)

# Stub python_headers/python_lib so @local_config_xpu targets resolve on non-XPU builds.
cc_library(name = "python_headers")
cc_library(name = "python_lib")
""")
    repository_ctx.file("crosstool/BUILD", """
package(default_visibility = ["//visibility:public"])

filegroup(name = "empty", srcs = [])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {},
)
""")

    # Stub xpu/BUILD so @local_config_xpu//xpu: targets resolve on non-XPU builds.
    repository_ctx.file("xpu/BUILD", """
package(default_visibility = ["//visibility:public"])

cc_library(name = "xpu_headers")
cc_library(name = "sycl_runtime")
cc_library(name = "ze_loader")
cc_library(name = "xpu")
""")

def _xpu_configure_impl(repository_ctx):
    """Implementation of the xpu_configure repository rule."""
    if not _enable_xpu(repository_ctx):
        _create_dummy_repository(repository_ctx)
        return
    oneapi_root = _oneapi_root(repository_ctx)
    icx_path = _find_icx(repository_ctx, oneapi_root)
    icpx_path = _find_icpx(repository_ctx, oneapi_root)

    # Resolve symlinks so paths match what the compiler reports to Bazel
    icx_path = str(repository_ctx.path(icx_path).realpath)
    icpx_path = str(repository_ctx.path(icpx_path).realpath)
    oneapi_compiler_dir = str(repository_ctx.path(oneapi_root + "/compiler/latest").realpath)
    oneapi_include = oneapi_compiler_dir + "/include"

    # Use icpx as the host compiler for getting include directories
    host_compiler_includes = get_cxx_inc_directories(repository_ctx, icpx_path)
    if not host_compiler_includes:
        auto_configure_fail(
            "TF_NEED_XPU=1 but failed to detect include directories from " +
            icpx_path + ". Verify icpx is installed and executes correctly.")

    host_compiler_prefix = "/usr/bin"

    # --- Generate the crosstool wrapper script ---
    _tpl(
        repository_ctx,
        "crosstool:clang/bin/crosstool_wrapper_driver_xpu",
        {
            "%{icx_path}": icx_path,
            "%{icpx_path}": icpx_path,
            "%{oneapi_include_path}": oneapi_include,
        },
        out = "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
    )

    # --- Generate crosstool BUILD and cc_toolchain_config ---
    xpu_defines = {}
    xpu_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    xpu_defines["%{linker_bin_path}"] = "/usr/bin"
    xpu_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    xpu_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DUSING_XPU=1",
    ])
    xpu_defines["%{host_compiler_warnings}"] = to_list_of_strings([
        "-Wno-error",
    ])
    xpu_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        [str(d) for d in host_compiler_includes] + [oneapi_include],
    )
    xpu_defines["%{compiler_deps}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{linker_files}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{win_linker_files}"] = ":empty"

    # Dummy Windows defines (required by verify_build_defines)
    xpu_defines["%{msvc_cl_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_include}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_lib}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_tmp}"] = "msvc_not_used"
    xpu_defines["%{msvc_lib_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_link_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_ml_path}"] = "msvc_not_used"

    verify_build_defines(xpu_defines)

    _tpl(repository_ctx, "crosstool:BUILD", xpu_defines)

    # Probe for the Level Zero loader BEFORE generating the toolchain config so
    # its directory can be handed to the linker as -L (the toolchain links
    # -lze_loader unconditionally, and the loader may live in oneAPI's lib dir
    # rather than a default linker search path). Fail fast if it is missing.
    ze_loader_lib = ""
    for path in ["/usr/lib/x86_64-linux-gnu/libze_loader.so",
                 "/usr/lib64/libze_loader.so",
                 oneapi_compiler_dir + "/lib/libze_loader.so"]:
        if repository_ctx.path(path).exists:
            ze_loader_lib = path
            break
    if not ze_loader_lib:
        auto_configure_fail(
            "TF_NEED_XPU=1 but libze_loader.so not found. " +
            "The XPU toolchain unconditionally links -lze_loader. " +
            "Install the Level Zero loader (e.g. level-zero-devel) or " +
            "ensure it is in /usr/lib/x86_64-linux-gnu/, /usr/lib64/, " +
            "or " + oneapi_compiler_dir + "/lib/.")
    ze_loader_lib_dir = ze_loader_lib.rsplit("/", 1)[0]

    # Substitute SYCL target and ze_loader search dir into toolchain config template
    sycl_target = _get_sycl_target(repository_ctx)
    _tpl(
        repository_ctx,
        "crosstool:xpu_cc_toolchain_config.bzl",
        {
            "%{xpu_sycl_target}": sycl_target,
            "%{ze_loader_lib_dir}": ze_loader_lib_dir,
        },
        out = "crosstool/cc_toolchain_config.bzl",
    )

    # --- Generate xpu/BUILD with SYCL runtime libraries ---
    sycl_lib = oneapi_compiler_dir + "/lib/libsycl.so"

    xpu_build_substitutions = {}
    if repository_ctx.path(sycl_lib).exists:
        repository_ctx.symlink(sycl_lib, "xpu/lib/libsycl.so")
        xpu_build_substitutions["%{sycl_runtime_srcs}"] = '["lib/libsycl.so"]'
    else:
        auto_configure_fail(
            ("TF_NEED_XPU=1 but libsycl.so not found at %s. " +
            "Is the oneAPI DPC++ compiler installed correctly?") % sycl_lib)

    repository_ctx.symlink(ze_loader_lib, "xpu/lib/libze_loader.so")
    xpu_build_substitutions["%{ze_loader_srcs}"] = '["lib/libze_loader.so"]'

    xpu_build_substitutions["%{copy_rules}"] = ""

    # Symlink SYCL headers
    sycl_include = oneapi_compiler_dir + "/include"
    if repository_ctx.path(sycl_include + "/sycl").exists:
        repository_ctx.symlink(sycl_include, "xpu/include")
    else:
        auto_configure_fail(
            ("TF_NEED_XPU=1 but SYCL headers not found at %s/sycl. " +
            "Is the oneAPI DPC++ compiler installed correctly?") % sycl_include)

    _tpl(repository_ctx, "xpu:BUILD", xpu_build_substitutions)

    # --- Generate py_runtime so .bazelrc can set --python_top=@local_config_xpu//:python_runtime ---
    python_bin = _get_python_bin(repository_ctx)
    # Resolve symlinked python to venv python so site-packages is correct
    python_bin = resolve_venv_python(repository_ctx, python_bin)

    # Validate Python version: XPU builds require Python 3.12 (PyTorch XPU
    # ships only cp312 wheels).  In XPU containers /opt/conda310/bin/python3
    # is a symlink to a Python 3.12 venv; fail early if the resolved
    # interpreter is something else.
    _py_ver = repository_ctx.execute([python_bin, "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
    if _py_ver.return_code == 0:
        _ver = _py_ver.stdout.strip()
        if _ver != "3.12":
            auto_configure_fail(
                "XPU build requires Python 3.12 but PYTHON_BIN_PATH (%s) " % python_bin +
                "resolves to Python %s. " % _ver +
                "In the XPU Docker image, /opt/conda310/bin/python3 should be a symlink " +
                "to a Python 3.12 venv. Check your container setup.")
    else:
        auto_configure_fail(
            "Failed to detect Python version from %s (exit code %d).\n" % (python_bin, _py_ver.return_code) +
            "stdout: %s\nstderr: %s" % (_py_ver.stdout.strip(), _py_ver.stderr.strip()))

    python_include = _get_python_include(repository_ctx, python_bin)
    python_lib = _get_python_lib(repository_ctx, python_bin)

    # Symlink Python include dir and lib .so into the repo so Bazel glob() can
    # use package-relative paths (glob() rejects absolute paths).
    if not repository_ctx.path(python_include).exists:
        auto_configure_fail(
            "Python include directory not found at " + python_include + ". " +
            "Install python3-dev / python3-devel or ensure PYTHON_BIN_PATH " +
            "points to a Python with development headers.")
    repository_ctx.symlink(python_include, "python_include")
    if not repository_ctx.path(python_lib).exists:
        auto_configure_fail(
            "Python shared library not found at " + python_lib + ". " +
            "Ensure PYTHON_BIN_PATH points to a Python built with --enable-shared " +
            "or install the python3-dev / libpython3-dev package.")
    python_lib_basename = repository_ctx.path(python_lib).basename
    repository_ctx.symlink(python_lib, "python_lib/" + python_lib_basename)

    repository_ctx.file("BUILD.bazel", content = """
py_runtime(
    name = "python_runtime",
    interpreter_path = "{python_bin}",
    python_version = "PY3",
    stub_shebang = "#!{python_bin}",
    visibility = ["//visibility:public"],
)

# Python headers/lib resolved from the XPU venv python (Python 3.12),
# ensuring C++ targets that link torch_xpu use the same ABI as the runtime.
# Absolute paths are symlinked into the repo as python_include/ and python_lib/
# because Bazel glob() does not accept absolute paths.
cc_library(
    name = "python_headers",
    hdrs = glob(["python_include/**/*.h"]),
    includes = ["python_include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "python_lib",
    srcs = ["python_lib/{python_lib_basename}"],
    visibility = ["//visibility:public"],
)
""".format(
        python_bin          = python_bin,
        python_lib_basename = python_lib_basename,
    ))

    # --- Torch XPU site-packages setup ---
    result = repository_ctx.execute([
        python_bin, "-c",
        "import site; print(site.getsitepackages()[0])",
    ])
    if result.return_code == 0:
        site_packages = result.stdout.strip()
        repository_ctx.file(
            "xpu/site_packages.bzl",
            'XPU_SITE_PACKAGES = "%s"\n' % site_packages,
        )
    else:
        # buildifier: disable=print
        print("WARNING: site-packages detection failed for XPU: " + result.stderr)

xpu_configure = repository_rule(
    implementation = _xpu_configure_impl,
    environ = [
        "TF_NEED_XPU",
        _ONEAPI_ROOT,
        _PYTHON_BIN_PATH,
        _SYCL_TARGET,
    ],
    doc = """Configures the Intel XPU (icx/icpx) C/C++ toolchain.

Add the following to your WORKSPACE:

```python
xpu_configure(name = "local_config_xpu")
```
""",
)
