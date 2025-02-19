#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "maga_transformer/cpp/lora/LoraManager.h"

namespace rtp_llm::lora {

class MockLoraManager: public LoraManager {
public:
    MockLoraManager(): LoraManager() {}
    ~MockLoraManager() override = default;

public:
    MOCK_METHOD3(addLora, void(const std::string& adapter_name,
                               const ft::lora::loraLayerWeightsMap& lora_a_weights,
                               const ft::lora::loraLayerWeightsMap& lora_b_weights));
    MOCK_METHOD1(removeLora, void(const std::string&));
};

}  // namespace rtp_llm::lora
