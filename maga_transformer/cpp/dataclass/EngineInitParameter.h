#pragma once

#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "kmonitor/client/MetricsReporter.h"

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

class EngineInitParams: public th::jit::CustomClassHolder {
public:
    EngineInitParams(const ft::GptInitParameter&                                             gpt_init_parameter,
                     const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layers_weights,
                     const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights);

public:
    ft::GptInitParameter         gpt_init_parameter;
    ft::Weights                  gpt_weights;
    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;

    // TODO(): rm old impl init
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layers_weights;
    std::unordered_map<std::string, ft::ConstBufferPtr>              global_weights;
};

class WeightsConverter {
public:
    WeightsConverter(bool need_copy): need_copy_(need_copy) {
        if (need_copy_) {
            device_ = ft::DeviceFactory::getDefaultDevice();
        }
    };

    std::unique_ptr<const ft::Weights> convertPythonWeights(const PyModelWeights& weights);

    static ft::Weights
    createGptWeights(const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layers_weights,
                     const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    static std::unordered_map<std::string, ft::ConstBufferPtr> convertPyWeightsMap(py::object py_global_weights);
    static std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> convertPyWeightsMapVec(py::object py_global_weights);

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
