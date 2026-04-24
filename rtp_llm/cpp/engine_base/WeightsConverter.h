#pragma once
#include <cstddef>
#include <pybind11/pytypes.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace th = torch;

namespace rtp_llm {

using TensorMap  = std::unordered_map<std::string, th::Tensor>;
using TensorMaps = std::vector<TensorMap>;

class WeightsConverter {
public:
    WeightsConverter(bool need_copy, rtp_llm::QuantAlgo quant_alog = rtp_llm::QuantAlgo());

    // Here include two interfaces to create gpt weights:
    // 1st. interface is to convert py::object to rtp_llm::weights,
    // this is used for python lib.
    // 2nd. interface is to convert torch::Tensor to rtp_llm::weights,
    // this is used for cpp unittest.
    std::unique_ptr<rtp_llm::Weights> createGptWeights(py::object layer_weights, py::object global_weight);

    std::unique_ptr<rtp_llm::Weights> createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                                                       std::unique_ptr<TensorMap>  global_weight);

private:
    std::unique_ptr<TensorMaps> convertLayerWeights(py::object py_layer_weights);
    std::unique_ptr<TensorMap>  convertGlobalWeight(py::object py_global_weight);

    // helper function

    torch::Tensor CopyTensorToGPU(const torch::Tensor& tensor);

    torch::Tensor mayFindTensor(const TensorMap& map, const std::string& key);

    rtp_llm::DenseWeightsPtr mayCreateDenseWeights(const TensorMap&   map,
                                                   const std::string& kernel_key,
                                                   const std::string& bias_key   = "",
                                                   const std::string& scales_key = "",
                                                   const std::string& zeros_key  = "");

    rtp_llm::LayerNormWeightsPtr mayCreateLayerNormWeights(const TensorMap&   map,
                                                           const std::string& gamma_key,
                                                           const std::string& beta_key             = "",
                                                           const std::string& scale_key            = "",
                                                           const std::string& scale_reciprocal_key = "");

    rtp_llm::FfnLayerWeights createFfnWeights(const TensorMap& map);

    rtp_llm::AttentionLayerWeights createAttentionWeights(const TensorMap& map);

private:
    bool               need_copy_;
    rtp_llm::QuantAlgo quant_algo_;
    bool               use_linear_bias_slopes_;
};

}  // namespace rtp_llm
