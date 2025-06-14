#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/EngineBase.h"

namespace rtp_llm {

class MockEngineBase: public EngineBase {
public:
    MockEngineBase(): EngineBase(EngineInitParams()) {}
    ~MockEngineBase() override = default;

public:
    MOCK_METHOD1(enqueue, std::shared_ptr<GenerateStream>(const std::shared_ptr<GenerateInput>&));
    MOCK_METHOD1(enqueue, void(std::shared_ptr<GenerateStream>&));
    MOCK_METHOD0(stop, absl::Status());
    MOCK_METHOD2(preRun, absl::StatusOr<GenerateStreamPtr>(const std::shared_ptr<GenerateInput>&, preRunMode));
    MOCK_METHOD0(getLoadBalanceInfo, LoadBalanceInfo());
};

}  // namespace rtp_llm