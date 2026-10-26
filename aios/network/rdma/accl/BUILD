package(default_visibility = ["//aios/network/rdma:__subpackages__"])

# Open-sourced from the closed @alibaba_rdma repository. accl-barex is a
# permanently closed library: the @alibaba_rdma repository is defined only by
# the internal deps overlay, so the accl-barex dep appears only in the
# if_rdma ON branch and the accl-carrying targets keep empty srcs/hdrs when
# //:using_rdma is OFF (no public platform config enables it). A default
# public `bazel build //...` therefore never resolves @alibaba_rdma; building
# with --define=use_rdma=true requires the internal overlay (see
# rtp_llm/cpp/rdma_transport/barex/README.md).

load("//bazel:defs.bzl", "if_rdma")

cc_library(
    name = "accl_impl",
    srcs = if_rdma(glob(["*.cpp"])),
    hdrs = if_rdma(glob(["*.h"])),
    copts = ["-DUSE_RDMA=1"],
    include_prefix = ".",
    deps = [
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock",
        "@havenask//aios/autil:net",
        "@havenask//aios/network/arpc",
        "@havenask//aios/network/arpc/arpc/metric:arpc_metric",
    ] + if_rdma([
        "@alibaba_rdma//aios/network/accl-barex",
    ]),
)

# The no-accl variant is the counterpart of :accl_impl for the OFF
# configuration only: the root rdma_mocks aggregator selects it just
# when //:using_rdma is OFF, and its sources are exactly the subset that
# compiles without the accl-barex include path. The internal overlay defines
# USE_RDMA=1 globally for build:cuda, which would make these sources take the
# #ifdef USE_RDMA paths (RdmaMempool.h -> RdmaMempoolImpl.h ->
# aios/network/accl-barex/.../barex_types.h) that this target deliberately
# carries no dep for, so srcs/hdrs stay empty in the ON branch — mirroring how
# the accl-carrying targets keep empty srcs/hdrs when OFF.
cc_library(
    name = "no_accl_impl",
    srcs = if_rdma([], [
        "RdmaMempoolConfig.cpp",
        "RdmaMempoolFactory.cpp",
    ]),
    hdrs = if_rdma([], [
        "RdmaMempool.h",
        "RdmaMempoolConfig.h",
        "RdmaMempoolFactory.h",
    ]),
    include_prefix = ".",
    deps = [
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:log",
        "@havenask//aios/network/arpc",
    ],
)

cc_library(
    name = "mocks_accl_impl",
    srcs = if_rdma(glob(["unittest/mock/*.cpp"])),
    hdrs = if_rdma(glob(["unittest/mock/*.h"])),
    include_prefix = ".",
    deps = [
        ":accl_impl",
        "@havenask//aios/unittest_framework",
    ],
)

cc_library(
    name = "mocks_no_accl_impl",
    srcs = if_rdma([], glob(["unittest/mock/MockXSimplemempool.cpp"])),
    hdrs = if_rdma([], glob(["unittest/mock/*.h"])),
    include_prefix = ".",
    deps = [
        ":no_accl_impl",
        "@havenask//aios/unittest_framework",
    ],
)

# Needs the closed accl-barex (via :accl_impl) to link and real RDMA NICs to
# run: manual, so public wildcard lanes never build or schedule it; run
# explicitly on internal RDMA workers.
cc_test(
    name = "rdma_accl_unittest",
    srcs = [
        "unittest/AcclClientWrapperTestOnMock.cpp",
        "unittest/AcclServerWrapperTestOnMock.cpp",
        "unittest/FakeRdmaRequest.cpp",
        "unittest/FakeRdmaRequest.h",
        "unittest/MessageConverterTestOnMock.cpp",
        "unittest/RdmaChannelClientCallbackTestOnMock.cpp",
        "unittest/RdmaChannelClientTestOnMock.cpp",
        "unittest/RdmaChannelContextTestOnMock.cpp",
        "unittest/RdmaChannelServerCallbackTestOnMock.cpp",
        "unittest/RdmaChannelServerTestOnMock.cpp",
        "unittest/RdmaClientTestOnMock.cpp",
        "unittest/RdmaMempoolConfigTest.cpp",
        "unittest/RdmaMempoolTestOnMock.cpp",
        "unittest/RdmaMessageTestOnMock.cpp",
        "unittest/RdmaRequestTestOnMock.cpp",
        "unittest/SimpleMessageConverterTestOnMock.cpp",
        "unittest/TimerTest.cpp",
        "unittest/CudaUtilTest.cpp",
    ],
    copts = [
        "-DUSE_RDMA=1",
        "-fno-access-control",
    ],
    deps = [
        ":accl_impl",
        ":mocks_accl_impl",
        "//aios/network/arpc/arpc/metric/mock:arpc_metric_mock",
        "//aios/network/arpc/arpc/proto/test:arpc_test_proto_cc_proto",
        "@havenask//aios/unittest_framework",
    ],
    tags = ["manual"],
)

cc_test(
    name = "rdma_accl_smoketest",
    srcs = if_rdma([
        "unittest/FakeRdmaChannelClosure.h",
        "unittest/FakeRdmaRequest.cpp",
        "unittest/FakeRdmaRequest.h",
        "unittest/FakeRdmaServer.cpp",
        "unittest/FakeRdmaServer.h",
        "unittest/FakeRdmaServerAdapter.cpp",
        "unittest/FakeRdmaServerAdapter.h",
        "unittest/MultiThreadPostTest.cpp",
        "unittest/RdmaChannelClientTest.cpp",
        "unittest/RdmaChannelServerTest.cpp",
        "unittest/RdmaMempoolFactoryTest.cpp",
    ]),
    copts = [
        "-DUSE_RDMA=1",
        "-Wno-deprecated-declarations",
        "-fno-access-control",
    ],
    deps = [
        ":accl_impl",
        "//aios/network/arpc/arpc/proto/test:arpc_test_proto_cc_proto",
        "@havenask//aios/unittest_framework",
    ],
    tags = ["manual"],
)
