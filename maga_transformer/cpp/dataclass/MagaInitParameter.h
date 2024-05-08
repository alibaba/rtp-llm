#pragma once

#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace rtp_llm {

class PyModelWeights: public th::jit::CustomClassHolder {
public:
    std::unordered_map<std::string, th::Tensor>              model_global_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_int8_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_int8_scales_;
};

class MagaInitParams: public th::jit::CustomClassHolder {
public:
    th::intrusive_ptr<GptInitParameter>  gpt_init_parameter;
    th::intrusive_ptr<PyModelWeights>    model_weights;
};

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
                                            const std::string&                          key);

    ft::LayerNormWeightsPtr mayCreateLayerNormWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                                      const std::string&                          gamma_key,
                                                      const std::string&                          beta_key);
    ft::DenseWeightsPtr     mayCreateDenseWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                                  const std::string&                          kernel_key,
                                                  const std::string&                          bias_key = "");
    ft::DenseWeightsPtr     createDenseWeights(std::unordered_map<std::string, th::Tensor> tensor_map,
                                               const std::string&                          kernel_key,
                                               const std::string&                          bias_key = "");

private:
    bool            need_copy_;
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
