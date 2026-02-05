load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    _if_cuda = "if_cuda",
)

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

if_rocm = _if_rocm
if_cuda = _if_cuda

def rpm_library(
        name,
        hdrs,
        include_path=None,
        lib_path=None,
        rpms=None,
        static_lib=None,
        static_libs=[], # multi static libs, do not add to cc_library, provide .a filegroup
        shared_lib=None,
        shared_libs=[],
        bins=[],
        include_prefix=None,
        static_link=False,
        deps=[],
        header_only=False,
        tags={},
        **kwargs):
    hdrs = [ "include/" + hdr for hdr in hdrs ]
    outs = [] + hdrs
    if static_lib:
        outs.append(static_lib)
    if shared_lib :
        outs.append(shared_lib)
    if not rpms:
        rpms = ["@" + name + "//file:file"]
    bash_cmd = "mkdir " + name + " && cd " + name
    bash_cmd += " && for e in $(SRCS); do rpm2cpio ../$$e | cpio -idm; done"
    if include_path != None:
        if header_only:
            bash_cmd += "&& cp -rf " + include_path + "/* ../$(@D)/"
        else:
            bash_cmd += "&& cp -rf " + include_path + "/* ../$(@D)/include"
    if len(static_libs) > 0:
        # extract all .a files to its own directory in case .o file conflict, and ar them together to target .a file.
        bash_cmd += "&& for a in " + " ".join(static_libs) + "; do d=$${a%.a} && mkdir $$d && cd $$d && ar x ../" + lib_path + "$$a && cd -; done && ar rc ../$(@D)/" + static_lib + " */*.o"
    elif static_lib:
        bash_cmd += "&& cp -L " + lib_path + "/*.a" + " ../$(@D)/"
    if shared_lib:
        bash_cmd += "&& cp -L " + lib_path + "/" + shared_lib + " ../$(@D) && patchelf --set-soname " + shared_lib + " ../$(@D)/" + shared_lib
    for share_lib in shared_libs:
        outs.append(share_lib)
        bash_cmd += "&& cp -L " + lib_path + "/" + share_lib + " ../$(@D) && patchelf --set-soname " + share_lib + " ../$(@D)/" + share_lib
    for path in bins:
        outs.append(path)
        bash_cmd += "&& cp -rL " + path + " ../$(@D)"
    bash_cmd += " && cd -"

    native.genrule(
        name = name + "_files",
        srcs = rpms,
        outs = outs,
        cmd = bash_cmd,
        visibility = ["//visibility:public"],
        tags=tags,
    )
    hdrs_fg_target = name + "_hdrs_fg"
    native.filegroup(
        name = hdrs_fg_target,
        srcs = hdrs,
    )
    if static_lib:
        native.filegroup(
            name = name + "_static",
            srcs = [static_lib],
            visibility = ["//visibility:public"],
        )
    srcs = []
    shared_files = shared_libs + (shared_lib and [shared_lib] or [])
    if shared_files:
        shared_filegroup = name + "_shared"
        native.filegroup(
            name = shared_filegroup,
            srcs = shared_files,
            visibility = ["//visibility:public"],
        )
        if shared_libs:
            srcs.append(shared_filegroup)

    if bins:
        bins_filegroup = name + "_bins"
        native.filegroup(
            name = bins_filegroup,
            srcs = bins,
            visibility = ["//visibility:public"],
            tags=tags,
        )

    if static_lib == None:
        native.cc_library(
            name = name,
            hdrs = [hdrs_fg_target],
            srcs = shared_files,
            deps = deps,
            strip_include_prefix = "include",
            include_prefix = include_prefix,
            visibility = ["//visibility:public"],
            **kwargs
        )
    else:
        import_target = name + "_import"
        alwayslink = static_lib!=None
        native.cc_import(
            name = import_target,
            static_library = static_lib,
            shared_library = shared_lib,
            alwayslink=alwayslink,
            visibility = ["//visibility:public"],
        )
        native.cc_library(
            name = name,
            hdrs = [hdrs_fg_target],
            srcs = srcs,
            deps = deps + [import_target],
            visibility = ["//visibility:public"],
            strip_include_prefix = "include",
            include_prefix = include_prefix,
            **kwargs
        )

def copts():
    return [
        "-DTORCH_CUDA",
    ] + if_cuda([
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-DUSE_C10D_NCCL",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
    ]) + if_rocm([
        "-x", "rocm",
    ])

def host_copts():
    """Copts for host-only C++ code that must not be compiled as HIP/CUDA (avoids undeclared inclusion errors)."""
    return ["-DTORCH_CUDA"]

def cuda_copts():
    # add --objdir-as-tempdir to rm tmp file after build
    return copts() + cuda_default_copts() + if_cuda(["-nvcc_options=objdir-as-tempdir"])

def rocm_copts():
    return copts() + rocm_default_copts() + if_rocm(["-Wc++17-extensions"])

def any_cuda_copts():
    return copts() + cuda_default_copts() + if_cuda(["-nvcc_options=objdir-as-tempdir"]) + rocm_default_copts() + if_rocm(["-Wc++17-extensions"])

def gen_cpp_code(name, elements_list, template_header, template, template_tail,
                 element_per_file = 1, suffix=".cpp"):
    bases = []
    base = 1

    for i in range(len(elements_list)):
        base = len(elements_list[i]) * base

    base_tmp = base
    for i in range(len(elements_list)):
        base_tmp = base_tmp // len(elements_list[i])
        bases.append(base_tmp)

    files = []
    current = 0
    count = 0
    current_str = template_header
    for i in range(base):
        replace_elements_list = []
        num = i
        for j in range(len(bases)):
            this_element = elements_list[j][num // bases[j]]
            if type(this_element) == 'tuple':
                replace_elements_list.extend(this_element)
            else:
                replace_elements_list.append(this_element)
            num %= bases[j]
        # for all permutations here

        if type(replace_elements_list[0]) == "tuple":
            replace_elements_list = replace_elements_list[0]
        else:
            replace_elements_list = tuple(replace_elements_list)
        current_str += template.format(*replace_elements_list)
        current += 1
        if current == element_per_file or i == base - 1:
            cpp_name = name + "_" + str(count)
            count += 1
            file_name = cpp_name + suffix
            content = current_str + template_tail
            native.genrule(
                name = cpp_name,
                srcs = [],
                outs = [file_name],
                cmd = "cat > $@  << 'EOF'\n" + content + "EOF",
            )
            current = 0
            current_str = template_header
            files.append(cpp_name)

    native.filegroup(
        name = name,
        srcs = files
    )


def _read_release_version_impl(repository_ctx):
    # Read the release_version.py file
    release_version_content = repository_ctx.read(repository_ctx.path(Label("//rtp_llm:release_version.py")))
    # Extract the RELEASE_VERSION value from the Python file
    # We assume the format is RELEASE_VERSION = "x.x.x"
    release_version = "0.0.1"  # fallback version

    # Look for the pattern RELEASE_VERSION = "x.x.x"
    pattern = 'RELEASE_VERSION = "'
    start_index = release_version_content.find(pattern)
    if start_index != -1:
        # Find the start of the version string
        start_index += len(pattern)
        # Find the end of the version string
        end_index = release_version_content.find('"', start_index)
        if end_index != -1:
            release_version = release_version_content[start_index:end_index]

    repository_ctx.file("BUILD", "")
    repository_ctx.file("defs.bzl", "RELEASE_VERSION = '{}'".format(release_version))

read_release_version = repository_rule(
    implementation = _read_release_version_impl,
    attrs = {},
)