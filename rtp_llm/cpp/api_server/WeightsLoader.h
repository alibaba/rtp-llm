#pragma once

#include "rtp_llm/cpp/devices/LoraWeights.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

class WeightsLoader {
public:
    WeightsLoader(py::object model_weights_loader);
    virtual ~WeightsLoader() = default;

    virtual std::pair<std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>,
                      std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>>
    loadLoraWeights(const std::string& adapter_name, const std::string& lora_path);

private:
    py::object model_weights_loader_;
};

}  // namespace rtp_llm
