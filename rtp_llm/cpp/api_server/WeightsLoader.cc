
#include "rtp_llm/cpp/api_server/WeightsLoader.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"

namespace rtp_llm {

WeightsLoader::WeightsLoader(py::object model_weights_loader): model_weights_loader_(model_weights_loader) {}

std::pair<std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>, std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap>>
WeightsLoader::loadLoraWeights(const std::string& adapter_name, const std::string& lora_path) {
    std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap> lora_a_weights, lora_b_weights;
    {
        py::gil_scoped_acquire acquire;
        auto                   res =
            model_weights_loader_.attr("load_lora_weights")(py::str(adapter_name), py::str(lora_path), py::str("cpu"));
        auto py_lora_a_weights = res.attr("lora_a_weights");
        auto py_lora_b_weights = res.attr("lora_b_weights");
        auto convert           = rtp_llm::WeightsConverter(true);
        lora_a_weights         = convert.convertLayerWeights_(py_lora_a_weights);
        lora_b_weights         = convert.convertLayerWeights_(py_lora_b_weights);
    }
    return std::make_pair(std::move(lora_a_weights), std::move(lora_b_weights));
}

}  // namespace rtp_llm
