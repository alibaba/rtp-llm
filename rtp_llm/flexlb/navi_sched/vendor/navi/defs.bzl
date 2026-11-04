# Platform-specific build configurations.
load("@com_google_protobuf//:protobuf.bzl", "proto_gen")
load("//rtp_llm/flexlb/navi_sched/vendor/bazel:tf_proto.bzl", "cc_proto_library", "tf_proto_library_py", "tf_deps", "clean_dep")

def tf_proto_library_cc(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        cc_api_version = 2,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        make_default_target_header_only = False,
        include = None):
    js_codegen = js_codegen  # unused argument
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
        testonly = testonly,
        visibility = visibility,
    )
    _ignore = (create_service, create_java_proto)

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    protolib_deps = tf_deps(protodeps, "")
    cc_deps = tf_deps(protodeps, "_cc")
    cc_name = name + "_cc"
    if not srcs:
        # This is a collection of sub-libraries. Build header-only and impl
        # libraries containing all the sources.
        proto_gen(
            name = name + "_genproto",
            protoc = "@com_google_protobuf//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in protolib_deps],
        )

        native.alias(
            name = cc_name + "_genproto",
            actual = name + "_genproto",
            testonly = testonly,
            visibility = visibility,
        )

        native.alias(
            name = cc_name + "_headers_only",
            actual = cc_name,
            testonly = testonly,
            visibility = visibility,
        )

        native.cc_library(
            name = cc_name,
            deps = cc_deps + ["@com_google_protobuf//:protobuf_headers"] + [name + "_cc_impl"],
            testonly = testonly,
            visibility = visibility,
        )
        native.cc_library(
            name = cc_name + "_impl",
            deps = [s + "_impl" for s in cc_deps] + ["@com_google_protobuf//:cc_wkt_protos"],
            visibility = visibility,
        )

        return

    # all this file is copied from tf, to add -Dfinal= to remove final and override final virtual function, trick but work(hope so)
    cc_proto_library(
        name = cc_name,
        protolib_name = name,
        testonly = testonly,
        srcs = srcs,
        cc_libs = cc_libs + ["@com_google_protobuf//:protobuf"],
        copts = [
            "-Wno-unknown-warning-option",
            "-Wno-unused-but-set-variable",
            "-Wno-sign-compare",
            "-Dfinal=",
        ],
        make_default_target_header_only = make_default_target_header_only,
        protoc = "@com_google_protobuf//:protoc",
        use_grpc_plugin = use_grpc_plugin,
        use_grpc_namespace = use_grpc_namespace,
        visibility = visibility,
        deps = cc_deps + ["@com_google_protobuf//:cc_wkt_protos"],
        protolib_deps = protolib_deps,
        include = include,
    )

def tf_proto_library(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        include = None,
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_api_version = 2,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        create_go_proto = False,
        create_grpc_library = False,
        make_default_target_header_only = False,
        exports = [],
        tags = []):
    """Make a proto library, possibly depending on other proto libraries."""

    # TODO(b/145545130): Add docstring explaining what rules this creates and how
    # opensource projects importing TF in bazel can use them safely (i.e. w/o ODR or
    # ABI violations).
    _ignore = (
        js_codegen,
        exports,
        create_service,
        create_java_proto,
        create_grpc_library,
        cc_stubby_versions,
        create_go_proto,
    )

    tf_proto_library_cc(
        name = name,
        testonly = testonly,
        srcs = srcs,
        cc_grpc_version = cc_grpc_version,
        use_grpc_namespace = use_grpc_namespace,
        cc_libs = cc_libs,
        make_default_target_header_only = make_default_target_header_only,
        protodeps = protodeps,
        visibility = visibility,
        include = include,
    )

    tf_proto_library_py(
        name = name,
        testonly = testonly,
        srcs = srcs,
        protodeps = protodeps,
        srcs_version = "PY2AND3",
        use_grpc_plugin = has_services,
        visibility = visibility,
        include = include,
    )

def cc_proto_custom(
        name,
        srcs,
        deps = None,
        include = ".",
        import_prefix = None,
        strip_import_prefix = None,
        visibility = None,
        use_grpc_plugin = False,
        cc_compile_grpc = True,
        py_compile_grpc = True):
    cc_compile_grpc = cc_compile_grpc if use_grpc_plugin else False
    py_compile_grpc = py_compile_grpc if use_grpc_plugin else False
    tf_proto_library(
        name = name,
        srcs = srcs,
        protodeps = deps,
        visibility = visibility,
        include = include,
        cc_grpc_version = cc_compile_grpc,
        has_services = py_compile_grpc,
    )
    native.cc_library(
        name = name + "_cc_proto",
        hdrs = [s[:-len(".proto")] + ".pb.h" for s in srcs],
        deps = [name + "_cc_impl"],
        include_prefix = import_prefix,
        strip_include_prefix = strip_import_prefix,
        visibility = visibility,
    )
    header_deps = []
    for dep in deps:
        if str(dep).startswith("//"):
            header_deps.append(dep + "_cc_proto_headers")
        else:
            # tensorflow proto library
            header_deps.append(dep + "_cc_headers_only")
    native.cc_library(
        name = name + "_cc_proto_headers",
        hdrs = [s[:-len(".proto")] + ".pb.h" for s in srcs],
        deps = header_deps,
        include_prefix = import_prefix,
        strip_include_prefix = strip_import_prefix,
        visibility = visibility,
        copts=["-Dfinal='abc'"]
    )
    return name + "_cc_proto"
