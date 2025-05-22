#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"

namespace rtp_llm {

class MockApiServerMetricReporter: public ApiServerMetricReporter {
public:
    MockApiServerMetricReporter()           = default;
    ~MockApiServerMetricReporter() override = default;

public:
    MOCK_METHOD1(reportQpsMetric, void(const std::string&));
    MOCK_METHOD1(reportCancelQpsMetric, void(const std::string&));
    MOCK_METHOD1(reportSuccessQpsMetric, void(const std::string&));
    MOCK_METHOD2(reportErrorQpsMetric, void(const std::string&, int));
    MOCK_METHOD0(reportConflictQpsMetric, void());
    MOCK_METHOD0(reportResponseIterateQpsMetric, void());

    MOCK_METHOD1(reportResponseLatencyMs, void(double));
    MOCK_METHOD1(reportResponseFirstTokenLatencyMs, void(double));
    MOCK_METHOD1(reportResponseIterateLatencyMs, void(double));
    MOCK_METHOD1(reportResponseIterateCountMetric, void(int32_t));

    MOCK_METHOD0(reportUpdateQpsMetric, void());
    MOCK_METHOD0(reportErrorUpdateTargetQpsMetric, void());
    MOCK_METHOD1(reportUpdateLatencyMs, void(double));

    MOCK_METHOD1(reportFTIterateCountMetric, void(double));
    MOCK_METHOD1(reportFTInputTokenLengthMetric, void(double));
    MOCK_METHOD1(reportFTOutputTokenLengthMetric, void(double));
    MOCK_METHOD1(reportFTPreTokenProcessorRtMetric, void(double));
    MOCK_METHOD1(reportFTPostTokenProcessorRtMetric, void(double));
    MOCK_METHOD1(reportFTNumBeansMetric, void(double));
};

}  // namespace rtp_llm
