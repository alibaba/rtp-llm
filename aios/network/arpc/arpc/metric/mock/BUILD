package(default_visibility = ["//visibility:public"])

# Test-support mocks for the open-sourced RDMA tests
# (//aios/network/rdma/{accl,arpc}/unittest): the moved tests include
# "aios/network/arpc/arpc/metric/mock/MockClientMetricReporter.h"
# (RdmaRPCChannelTestOnMock.cpp, RdmaChannelClientTestOnMock.cpp) and
# "aios/network/arpc/arpc/metric/mock/MockServerMetricReporter.h"
# (RdmaRPCServerClosureTestOnMock.cpp, RdmaChannelServerTestOnMock.cpp,
# RdmaChannelServerCallbackTestOnMock.cpp) — paths the closed @alibaba_rdma
# world expected @havenask's
# aios/network/arpc/arpc/metric:arpc_metric_mock target to provide. Neither
# the public github havenask pin nor the internal gitlab fork declares that
# target or carries the files, so the mocks are provided locally at the
# include-compatible paths; the moved-tree test BUILDs depend on this label
# instead of @havenask.
cc_library(
    name = "arpc_metric_mock",
    hdrs = [
        "MockClientMetricReporter.h",
        "MockServerMetricReporter.h",
    ],
    deps = [
        "@havenask//aios/network/arpc/arpc/metric:arpc_metric",
        "@havenask//aios/unittest_framework:unittest_framework",
    ],
    visibility = ["//visibility:public"],
)
