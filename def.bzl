load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    "if_cuda",
)

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    "if_rocm",
)

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

def torch_deps():
    torch_version = "2.1_py310"
    return [
        "@torch_" + torch_version + "_cpu//:torch_api",
        "@torch_" + torch_version + "_cpu//:torch",
    ]

def copts():
    return [
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-DTORCH_CUDA",
        "-DUSE_C10D_NCCL",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
    ]

def cuda_copts():
    # add --objdir-as-tempdir to rm tmp file after build
    return copts() + cuda_default_copts() + if_cuda(["-nvcc_options=objdir-as-tempdir"])

def rocm_copts():
    return copts() + rocm_default_copts() + [
        "-Wc++17-extensions",
    ]
