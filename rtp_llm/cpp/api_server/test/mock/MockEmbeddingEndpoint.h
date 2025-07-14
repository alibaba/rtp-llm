#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/EmbeddingEndpoint.h"

namespace rtp_llm {

class MockEmbeddingEndpoint: public EmbeddingEndpoint {
public:
    MockEmbeddingEndpoint(): EmbeddingEndpoint(nullptr, nullptr, py::none()) {}
    ~MockEmbeddingEndpoint() override = default;

public:
    MOCK_METHOD((std::pair<std::string, std::optional<std::string>>),
                handle,
                (const std::string&                              body,
                 std::optional<EmbeddingEndpoint::EmbeddingType> type,
                 const kmonitor::MetricsReporterPtr&             metrics_reporter,
                 int64_t                                         start_time_us),
                (override));
};

}  // namespace rtp_llm
