#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/WeightsLoader.h"

namespace rtp_llm {

class MockWeightsLoader: public WeightsLoader {
public:
    MockWeightsLoader(): WeightsLoader(py::none()) {}
    ~MockWeightsLoader() override = default;

public:
    MOCK_METHOD((std::pair<std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>,
                           std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>>),
                loadLoraWeights,
                (const std::string& adapter_name, const std::string& lora_path),
                (override));
};

}  // namespace rtp_llm
