#pragma once

#include "aios/network/arpc/arpc/metric/ClientMetricReporter.h"
#include "aios/network/arpc/arpc/metric/ClientRPCStats.h"
#include "unittest/unittest.h"

namespace arpc {

class MockClientMetricReporter : public ClientMetricReporter {
public:
    MockClientMetricReporter() : ClientMetricReporter(MetricReporterConfig()) {}
    ~MockClientMetricReporter() {}

private:
    MOCK_METHOD1(doReportMetric, void(ClientRPCStats*));
};

} // namespace arpc
