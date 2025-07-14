#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/GenerateStreamWrapper.h"

namespace rtp_llm {

class MockGenerateStreamWrapper: public GenerateStreamWrapper {
public:
    MockGenerateStreamWrapper(const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                              const std::shared_ptr<TokenProcessor>&          token_processor):
        GenerateStreamWrapper(metric_reporter, token_processor) {}
    ~MockGenerateStreamWrapper() override = default;

public:
    MOCK_METHOD0(generateResponse, std::pair<MultiSeqsResponse, bool>());
};

}  // namespace rtp_llm
