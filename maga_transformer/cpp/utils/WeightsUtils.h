#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class WeightsConverter {
public:
    WeightsConverter(bool need_copy): need_copy_(need_copy) {
        if (need_copy_) {
            device_ = ft::DeviceFactory::getDefaultDevice();
        }
    };

    std::unique_ptr<const ft::Weights> convertPythonWeights(const PyModelWeights& weights);

private:

    ft::ConstBufferPtr mayFindTensor2Buffer(std::unordered_map<std::string, th::Tensor> tensor_map,
                                            const std::string& key);

    ft::LayerNormWeightsPtr mayCreateLayerNormWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                                      const std::string& gamma_key, const std::string& beta_key);
    ft::DenseWeightsPtr mayCreateDenseWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                              const std::string& kernel_key, const std::string& bias_key = "");
    ft::DenseWeightsPtr createDenseWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                           const std::string& kernel_key, const std::string& bias_key = "");

private:
    bool need_copy_;
    ft::DeviceBase* device_;
};

}
