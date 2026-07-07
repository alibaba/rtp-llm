package(default_visibility = ["//visibility:public"])

cc_import(
    name = "v6d_kvs_client_lib",
    alwayslink = True,
    static_library = "lib/libv6d_kvs_client.a",
)

cc_import(
    name = "v6d_kvs_transfer_client_lib",
    alwayslink = True,
    static_library = "lib/libv6d_kvs_transfer_client.a",
)

cc_import(
    name = "vineyard_transfer_engine_lib",
    alwayslink = True,
    static_library = "lib/libvineyard_transfer_engine.a",
)

cc_import(
    name = "srpc_stream_engine",
    shared_library = "lib/libsrpc_stream_engine.so",
)

cc_import(
    name = "srpc_barex_bridge",
    shared_library = "lib/libsrpc_barex_bridge.so",
)

cc_import(
    name = "glog",
    alwayslink = True,
    static_library = "lib/libglog.a",
)

cc_import(
    name = "gflags",
    alwayslink = True,
    static_library = "lib/libgflags_nothreads.a",
)

cc_library(
    name = "kvs_client",
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/single_include",
    ],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    deps = [
        ":gflags",
        ":glog",
        ":srpc_stream_engine",
        ":v6d_kvs_client_lib",
        ":v6d_kvs_transfer_client_lib",
        ":vineyard_transfer_engine_lib",
    ],
)
