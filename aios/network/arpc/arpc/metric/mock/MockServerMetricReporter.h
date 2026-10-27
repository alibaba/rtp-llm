#pragma once

#include "aios/network/arpc/arpc/metric/ServerMetricReporter.h"
#include "aios/network/arpc/arpc/metric/ServerRPCStats.h"
#include "unittest/unittest.h"

namespace arpc {

class MockServerMetricReporter : public ServerMetricReporter {
public:
    MockServerMetricReporter() : ServerMetricReporter(MetricReporterConfig()) {}
    ~MockServerMetricReporter() {}

private:
    MOCK_METHOD1(doReportMetric, void(ServerRPCStats*));
};

} // namespace arpc
