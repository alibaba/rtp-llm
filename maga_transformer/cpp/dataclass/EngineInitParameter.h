#pragma once

#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "kmonitor/client/MetricsReporter.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

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
using TensorMap  = std::unordered_map<std::string, th::Tensor>;
using TensorMaps = std::vector<TensorMap>;
using ConstBufferPtrMap  = std::unordered_map<std::string, ft::ConstBufferPtr>;
using ConstBufferPtrMaps = std::vector<ConstBufferPtrMap>;

class EngineInitParams: public th::jit::CustomClassHolder {
public:
   // This class is the only one that holds gpt_weights object globally.
    EngineInitParams(const ft::GptInitParameter&    gpt_init_parameter,
                     ConstBufferPtrMaps             layers_weights,
                     ConstBufferPtrMap              global_weights,
                     ft::Weights&&                  gpt_weights) :
                     gpt_init_parameter(gpt_init_parameter),
                     layers_weights(layers_weights),
                     global_weights(global_weights),
                     gpt_weights(std::move(gpt_weights)) {}

public:
    ft::GptInitParameter         gpt_init_parameter;
    ft::Weights                  gpt_weights;
    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;

    // TODO(): rm old impl init
    ConstBufferPtrMaps layers_weights;
    ConstBufferPtrMap global_weights;
};

class WeightsConverter {
public:
    WeightsConverter(bool need_copy): need_copy_(need_copy) {
        if (need_copy_) {
            device_ = ft::DeviceFactory::getDefaultDevice();
        }
    };

    // Here include three interface to create gpt weights,
    // 1st. interface is to convert py::object to ft::weights,
    // this used for python lib.
    // 2ed. interface is to convert torch::Tensor to ft::weights,
    // this used for cpp unittest.
    // 3rd. interface is to convert BufferPtr to ft::weights,
    // this is the core impl, above 2 interface invoke this interface.
    std::unique_ptr<ft::Weights>
    createGptWeights(py::object layer_weights,
                     py::object  global_weight);

    std::unique_ptr<ft::Weights>
    createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                     std::unique_ptr<TensorMap>  global_weight);

    std::unique_ptr<ft::Weights>
    createGptWeights(std::unique_ptr<ConstBufferPtrMaps> layer_weights,
                     std::unique_ptr<ConstBufferPtrMap>  global_weight);
    

    // TODO(): rm old impl init
    std::unique_ptr<ConstBufferPtrMaps> convertLayerWeights_(py::object py_layer_weights);
    
    std::unique_ptr<ConstBufferPtrMap>  convertGlobalWeight_(py::object py_global_weight);

private:

    std::unique_ptr<TensorMaps> convertLayerWeights(py::object py_layer_weights);
    std::unique_ptr<TensorMap>  convertGlobalWeight(py::object py_global_weight);

    std::unique_ptr<ConstBufferPtrMaps>
    convertLayerWeights(std::unique_ptr<TensorMaps> tensor_layer_weights);
    
    std::unique_ptr<ConstBufferPtrMap> 
    convertGlobalWeight(std::unique_ptr<TensorMap> tensor_global_weight);

    // helper function

    ft::ConstBufferPtr mayFindBuffer(const ConstBufferPtrMap& map,
                                     const std::string& key);

    ft::DenseWeightsPtr mayCreateDenseWeights(const ConstBufferPtrMap& map,
                                              const std::string& kernel_key,
                                              const std::string& bias_key = "",
                                              const std::string& scales_key = "");

    ft::LayerNormWeightsPtr
    mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                              const std::string& gamma_key,
                              const std::string& beta_key = "");
    
    ft::FfnLayerWeights
    createFfnWeights(const ConstBufferPtrMap& map);

    ft::AttentionLayerWeights
    createAttentionWeights(const ConstBufferPtrMap& map);

    ft::ConstBufferPtr CopyTensorToBufferPtr(const torch::Tensor& tensor);

private:
    bool            need_copy_;
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
