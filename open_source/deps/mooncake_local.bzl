def _mooncake_transfer_engine_local_repo_impl(ctx):
    install_root = ctx.path(ctx.attr.install_path)
    jsoncpp_include_root = ctx.path(ctx.attr.jsoncpp_include_path)
    glog_lib_root = ctx.path(ctx.attr.glog_lib_path)
    gflags_lib_root = ctx.path(ctx.attr.gflags_lib_path)

    if install_root.exists:
        ctx.symlink(str(install_root) + "/include", "include")
        ctx.symlink(str(install_root) + "/lib", "lib")
        if jsoncpp_include_root.exists:
            ctx.symlink(str(jsoncpp_include_root), "jsoncpp_include")
            includes = '[\"include\", \"jsoncpp_include\"]'
        else:
            includes = '[\"include\"]'
        if glog_lib_root.exists:
            ctx.symlink(str(glog_lib_root), "glog_lib")
        if gflags_lib_root.exists:
            ctx.symlink(str(gflags_lib_root), "gflags_lib")
        ctx.file(
            "BUILD.bazel",
            """package(default_visibility = [\"//visibility:public\"])

cc_import(
    name = \"asio_shared\",
    shared_library = \"lib/libasio.so\",
)

cc_import(
    name = \"transfer_engine_shared\",
    shared_library = \"lib/libtransfer_engine.so\",
)

cc_import(
    name = \"glog_shared\",
    shared_library = \"glog_lib/libglog.so.0\",
)

cc_import(
    name = \"gflags_shared\",
    shared_library = \"gflags_lib/libgflags.so.2.2\",
)

cc_library(
    name = \"glog\",
    deps = [\":glog_shared\"],
)

cc_library(
    name = \"gflags\",
    deps = [\":gflags_shared\"],
)

cc_library(
    name = \"transfer_engine\",
    hdrs = glob([\"include/**/*.h\"]) + glob([\"jsoncpp_include/**/*.h\"]),
    includes = {includes},
    linkopts = [
        \"-L/usr/local/lib64\",
        \"-Wl,-rpath,/usr/local/lib64\",
        \"-libverbs\",
        \"-lnuma\",
    ],
    deps = [
        \":asio_shared\",
        \":transfer_engine_shared\",
        \":glog_shared\",
        \":gflags_shared\",
    ],
)
""".format(includes = includes),
        )
    else:
        ctx.file(
            "BUILD.bazel",
            """package(default_visibility = [\"//visibility:public\"])

cc_library(
    name = \"transfer_engine\",
)

cc_library(
    name = \"glog\",
)

cc_library(
    name = \"gflags\",
)
""",
        )

_mooncake_transfer_engine_local_repo = repository_rule(
    implementation = _mooncake_transfer_engine_local_repo_impl,
    attrs = {
        "install_path": attr.string(mandatory = True),
        "jsoncpp_include_path": attr.string(mandatory = True),
        "glog_lib_path": attr.string(mandatory = True),
        "gflags_lib_path": attr.string(mandatory = True),
    },
    local = True,
)

def mooncake_transfer_engine_local(
        name,
        install_path = "/data0/qiongshi.gb/mooncake/Mooncake/build-rtp/install",
        jsoncpp_include_path = "/data0/qiongshi.gb/tools/jsoncpp-install/include",
        glog_lib_path = "/data0/qiongshi.gb/tools/glog-0.3.3-install/lib",
        gflags_lib_path = "/data0/qiongshi.gb/tools/gflags-install/lib"):
    _mooncake_transfer_engine_local_repo(
        name = name,
        install_path = install_path,
        jsoncpp_include_path = jsoncpp_include_path,
        glog_lib_path = glog_lib_path,
        gflags_lib_path = gflags_lib_path,
    )
