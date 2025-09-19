#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

class MockGenerateStream: public GenerateStream {
public:
    MockGenerateStream(const std::shared_ptr<GenerateInput>& input, const rtp_llm::GptInitParameter& param):
        GenerateStream(input, param, ResourceContext{}, nullptr) {}
    ~MockGenerateStream() override = default;

public:
    MOCK_METHOD0(cancel, void());
    MOCK_METHOD0(stopped, bool());
    MOCK_METHOD0(finished, bool());
    MOCK_METHOD0(nextOutput, ErrorResult<GenerateOutputs>());
    MOCK_METHOD1(updateOutput, void(const StreamUpdateInfo&));
};

}  // namespace rtp_llm
