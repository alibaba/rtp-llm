load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

_JSONCPP_HEADERS = [
    "include/json/allocator.h",
    "include/json/assertions.h",
    "include/json/config.h",
    "include/json/forwards.h",
    "include/json/json.h",
    "include/json/json_features.h",
    "include/json/reader.h",
    "include/json/value.h",
    "include/json/version.h",
    "include/json/writer.h",
]

genrule(
    name = "jsoncpp_headers",
    outs = _JSONCPP_HEADERS,
    cmd = """
      for i in $(OUTS); do
        i=$${i##*/}
        ln -sf /usr/include/jsoncpp/json/$$i $(@D)/include/json/$$i
      done
    """,
)

cc_library(
    name = "jsoncpp",
    hdrs = _JSONCPP_HEADERS,
    includes = ["include"],
    linkopts = ["-ljsoncpp"],
)

cc_library(
    name = "yaml_cpp",
    linkopts = ["-lyaml-cpp"],
)

cc_library(
    name = "glog",
    linkopts = ["-lglog"],
)

cc_library(
    name = "gflags",
    linkopts = ["-lgflags"],
)

cc_library(
    name = "ibverbs",
    linkopts = ["-libverbs"],
)

cc_library(
    name = "numa",
    linkopts = ["-lnuma"],
)

cc_library(
    name = "pybind11_headers",
    hdrs = glob(["extern/pybind11/include/**/*.h"]),
    includes = ["extern/pybind11/include"],
)

cc_library(
    name = "yalantinglibs",
    defines = ["YLT_ENABLE_IBV"],
    hdrs = glob([
        "thirdparties/yalantinglibs-0.5.7/include/**/*.h",
        "thirdparties/yalantinglibs-0.5.7/include/**/*.hpp",
        "thirdparties/yalantinglibs-0.5.7/include/**/*.ipp",
        "thirdparties/yalantinglibs-0.5.7/src/include/*.h",
    ]),
    includes = [
        "thirdparties/yalantinglibs-0.5.7/include",
        "thirdparties/yalantinglibs-0.5.7/include/ylt",
        "thirdparties/yalantinglibs-0.5.7/include/ylt/thirdparty",
        "thirdparties/yalantinglibs-0.5.7/include/ylt/standalone",
        "thirdparties/yalantinglibs-0.5.7/src/include",
    ],
    linkopts = ["-lpthread"],
)

cc_library(
    name = "mooncake_common",
    srcs = [
        "mooncake-common/src/default_config.cpp",
        "mooncake-common/src/environ.cpp",
    ],
    hdrs = glob(["mooncake-common/include/**/*.h"]),
    copts = ["-std=c++20"],
    includes = ["mooncake-common/include"],
    deps = [
        ":jsoncpp",
        ":yalantinglibs",
        ":yaml_cpp",
    ],
)

cc_library(
    name = "asio_shared",
    srcs = ["mooncake-asio/asio_impl.cpp"],
    copts = ["-std=c++20"],
    defines = ["ASIO_DYN_LINK"],
    local_defines = ["ASIO_SEPARATE_COMPILATION"],
    linkopts = ["-lpthread"],
)

cc_library(
    name = "transfer_engine",
    srcs = [
        "mooncake-transfer-engine/src/common/base/status.cpp",
        "mooncake-transfer-engine/src/config.cpp",
        "mooncake-transfer-engine/src/memory_location.cpp",
        "mooncake-transfer-engine/src/multi_transport.cpp",
        "mooncake-transfer-engine/src/topology.cpp",
        "mooncake-transfer-engine/src/transfer_engine.cpp",
        "mooncake-transfer-engine/src/transfer_engine_impl.cpp",
        "mooncake-transfer-engine/src/transfer_metadata.cpp",
        "mooncake-transfer-engine/src/transfer_metadata_dump.cpp",
        "mooncake-transfer-engine/src/transfer_metadata_plugin.cpp",
        "mooncake-transfer-engine/src/transport/transport.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp",
        "mooncake-transfer-engine/src/transport/rpc_communicator/rpc_communicator.cpp",
        "mooncake-transfer-engine/src/transport/rpc_communicator/rpc_interface.cpp",
        "mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp",
    ],
    hdrs = glob(["mooncake-transfer-engine/include/**/*.h"]) + glob(["mooncake-transfer-engine/src/**/*.h"]),
    copts = ["-std=c++20"],
    defines = ["USE_TCP"],
    includes = [
        "mooncake-common/include",
        "mooncake-transfer-engine/include",
        "mooncake-transfer-engine/src",
    ],
    deps = [
        ":asio_shared",
        ":gflags",
        ":glog",
        ":ibverbs",
        ":jsoncpp",
        ":mooncake_common",
        ":numa",
        ":pybind11_headers",
        ":yalantinglibs",
        ":yaml_cpp",
        "@local_config_python//:python_headers",
        "@local_config_python//:python_lib",
    ],
)
