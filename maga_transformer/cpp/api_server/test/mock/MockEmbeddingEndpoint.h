#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "maga_transformer/cpp/api_server/EmbeddingEndpoint.h"

namespace rtp_llm {

class MockEmbeddingEndpoint: public EmbeddingEndpoint {
public:
    MockEmbeddingEndpoint(): EmbeddingEndpoint(nullptr, nullptr, py::none()) {}
    ~MockEmbeddingEndpoint() override = default;

public:
    MOCK_METHOD2(handle,
            std::pair<std::string, std::optional<std::string>>(const std::string&,
                std::optional<EmbeddingEndpoint::EmbeddingType>));
};

}  // namespace rtp_llm
