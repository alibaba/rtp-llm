#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/tokenizer/Tokenizer.h"

namespace rtp_llm {

class MockTokenizer: public Tokenizer {
public:
    MockTokenizer(): Tokenizer(py::none()) {}
    ~MockTokenizer() override = default;

public:
    MOCK_METHOD0(isPreTrainedTokenizer, bool());
    MOCK_METHOD0(getEosTokenId, std::optional<int>());
    MOCK_METHOD1(decode, std::string(const std::vector<int>&));
    MOCK_METHOD1(encode, std::vector<int>(const std::string&));
};

}  // namespace rtp_llm
