#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "maga_transformer/cpp/api_server/TokenProcessor.h"

namespace rtp_llm {

class MockTokenProcessor: public TokenProcessor {
public:
    MockTokenProcessor(): TokenProcessor(py::none()) {}
    ~MockTokenProcessor() override = default;

public:
    MOCK_METHOD1(decode, std::string(const std::vector<int>&));
    MOCK_METHOD1(encode, std::vector<int>(const std::string&));
    MOCK_METHOD1(tokenizer, std::shared_ptr<TokenizerEncodeResponse>(const std::string&));
};

}  // namespace rtp_llm