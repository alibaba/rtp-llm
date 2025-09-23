#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/models/lora/LoraManager.h"

namespace rtp_llm::lora {

class MockLoraManager: public LoraManager {
public:
    MockLoraManager(): LoraManager() {}
    ~MockLoraManager() override = default;

public:
    MOCK_METHOD3(addLora,
                 void(const std::string&                        adapter_name,
                      const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                      const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights));
    MOCK_METHOD1(removeLora, void(const std::string&));
};

}  // namespace rtp_llm::lora
