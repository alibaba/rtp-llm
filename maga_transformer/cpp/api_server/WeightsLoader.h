#pragma once

#include "src/fastertransformer/devices/LoraWeights.h"
#include "maga_transformer/cpp/utils/PyUtils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class WeightsLoader {
public:
    WeightsLoader(py::object model_weights_loader);
    virtual ~WeightsLoader() = default;

    virtual std::pair<std::unique_ptr<ft::lora::loraLayerWeightsMap>, std::unique_ptr<ft::lora::loraLayerWeightsMap>>
    loadLoraWeights(const std::string& adapter_name, const std::string& lora_path);

private:
    py::object model_weights_loader_;
};

}
