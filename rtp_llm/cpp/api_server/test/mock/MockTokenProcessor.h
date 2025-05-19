#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/TokenProcessor.h"

namespace rtp_llm {

class MockTokenProcessor: public TokenProcessor {
public:
    MockTokenProcessor(): TokenProcessor(py::none()) {}
    ~MockTokenProcessor() override = default;

public:
    MOCK_METHOD1(decode, std::string(const std::vector<int>&));
    MOCK_METHOD1(encode, std::vector<int>(const std::string&));
    MOCK_METHOD1(tokenizer, std::shared_ptr<TokenizerEncodeResponse>(const std::string&));
    MOCK_METHOD4(decodeTokens,
                 std::vector<std::string>(std::shared_ptr<TokenProcessorPerStream>,
                                          GenerateOutputs&,
                                          std::vector<int>&,
                                          std::shared_ptr<GenerateConfig>));
    MOCK_METHOD3(getTokenProcessorCtx,
                 std::shared_ptr<TokenProcessorPerStream>(bool, int, const std::shared_ptr<TokenProcessor>&));
};

}  // namespace rtp_llm
