#pragma once
#include <cstddef>
#include <pybind11/pytypes.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace th = torch;

namespace rtp_llm {

using TensorMap          = std::unordered_map<std::string, th::Tensor>;
using TensorMaps         = std::vector<TensorMap>;
using ConstBufferPtrMap  = std::unordered_map<std::string, rtp_llm::ConstBufferPtr>;
using ConstBufferPtrMaps = std::vector<ConstBufferPtrMap>;

class WeightsConverter {
public:
    WeightsConverter(bool need_copy, rtp_llm::QuantAlgo quant_alog = rtp_llm::QuantAlgo()):
        need_copy_(need_copy), quant_algo_(quant_alog) {
        if (need_copy_) {
            device_ = rtp_llm::DeviceFactory::getDefaultDevice();
        }
    };

    // Here include three interface to create gpt weights,
    // 1st. interface is to convert py::object to rtp_llm::weights,
    // this used for python lib.
    // 2ed. interface is to convert torch::Tensor to rtp_llm::weights,
    // this used for cpp unittest.
    // 3rd. interface is to convert BufferPtr to rtp_llm::weights,
    // this is the core impl, above 2 interface invoke this interface.
    std::unique_ptr<rtp_llm::Weights> createGptWeights(py::object layer_weights, py::object global_weight);

    std::unique_ptr<rtp_llm::Weights> createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                                                       std::unique_ptr<TensorMap>  global_weight);

    std::unique_ptr<rtp_llm::Weights> createGptWeights(std::unique_ptr<ConstBufferPtrMaps> layer_weights,
                                                       std::unique_ptr<ConstBufferPtrMap>  global_weight);

    // TODO(): rm old impl init
    std::unique_ptr<ConstBufferPtrMaps> convertLayerWeights_(py::object py_layer_weights);

    std::unique_ptr<ConstBufferPtrMap> convertGlobalWeight_(py::object py_global_weight);

private:
    std::unique_ptr<TensorMaps> convertLayerWeights(py::object py_layer_weights);
    std::unique_ptr<TensorMap>  convertGlobalWeight(py::object py_global_weight);

    std::unique_ptr<ConstBufferPtrMaps> convertLayerWeights(std::unique_ptr<TensorMaps> tensor_layer_weights);

    std::unique_ptr<ConstBufferPtrMap> convertGlobalWeight(std::unique_ptr<TensorMap> tensor_global_weight);

    // helper function

    rtp_llm::ConstBufferPtr mayFindBuffer(const ConstBufferPtrMap& map, const std::string& key);

    rtp_llm::DenseWeightsPtr mayCreateDenseWeights(const ConstBufferPtrMap& map,
                                                   const std::string&       kernel_key,
                                                   const std::string&       bias_key   = "",
                                                   const std::string&       scales_key = "",
                                                   const std::string&       zeros_key  = "");

    rtp_llm::LayerNormWeightsPtr mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                                                           const std::string&       gamma_key,
                                                           const std::string&       beta_key             = "",
                                                           const std::string&       scale_key            = "",
                                                           const std::string&       scale_reciprocal_key = "");

    rtp_llm::FfnLayerWeights createFfnWeights(const ConstBufferPtrMap& map);

    rtp_llm::AttentionLayerWeights createAttentionWeights(const ConstBufferPtrMap& map);

    rtp_llm::ConstBufferPtr CopyTensorToBufferPtr(const torch::Tensor& tensor);

private:
    bool                 need_copy_;
    rtp_llm::QuantAlgo   quant_algo_;
    rtp_llm::DeviceBase* device_;
    bool                 use_linear_bias_slopes_;
};

}  // namespace rtp_llm
