#pragma once
#include <cstddef>
#include <pybind11/pytypes.h>
#include <tuple>

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "kmonitor/client/MetricsReporter.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace rtp_llm {

using TensorMap  = std::unordered_map<std::string, th::Tensor>;
using TensorMaps = std::vector<TensorMap>;
using ConstBufferPtrMap  = std::unordered_map<std::string, ft::ConstBufferPtr>;
using ConstBufferPtrMaps = std::vector<ConstBufferPtrMap>;

struct EngineInitParams: public th::jit::CustomClassHolder {
    EngineInitParams() {};
   // This class is the only one that holds gpt_weights object globally.
    EngineInitParams(const ft::GptInitParameter&    gpt_init_parameter,
                     ft::Weights&&                  gpt_weights) :
                     gpt_init_parameter(gpt_init_parameter),
                     gpt_weights(std::move(gpt_weights)) {}


    ft::GptInitParameter         gpt_init_parameter;
    ft::Weights                  gpt_weights;

    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;
};

struct ProposeModelEngineInitParams: public th::jit::CustomClassHolder {
    ProposeModelEngineInitParams() {};

    // Constructor for vanilla propose model
    ProposeModelEngineInitParams(std::string sp_type,
                     size_t gen_num_per_circle,
                     const ft::GptInitParameter&    gpt_init_parameter,
                     ft::Weights&&                  gpt_weights) :
                     sp_type(sp_type),
                     gen_num_per_circle(gen_num_per_circle),
                     vanilla_model_params(new EngineInitParams(gpt_init_parameter, std::move(gpt_weights))) {}

    // Consturctor for deterministic propose model
    ProposeModelEngineInitParams(std::string sp_type, size_t gen_num_per_circle) :
                    sp_type(sp_type), gen_num_per_circle(gen_num_per_circle) {}

    // Consturctor for mtp propose model
    ProposeModelEngineInitParams(std::string sp_type,
                                 size_t gen_num_per_circle,
                                 std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params) :
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(nullptr),
        mtp_model_params_(std::move(mtp_model_params)) {};

    bool gpt_model() {
        return sp_type == "vanilla" || sp_type == "mtp";
    }

    bool isVanilla() {
        return sp_type == "vanilla";
    }

    bool isMTP() {
        return sp_type == "mtp";
    }

    const ft::GptInitParameter& getGptInitParameter() {
        if (sp_type == "vanilla") {
            return vanilla_model_params->gpt_init_parameter;
        } else if (sp_type == "mtp") {
            FT_CHECK(!mtp_model_params_->empty());
            FT_CHECK(mtp_model_params_->at(0) != nullptr);
            return mtp_model_params_->at(0)->gpt_init_parameter;
        } else {
            FT_FAIL("error sp type[%s] do not have GptInitParameter", sp_type.c_str());
        }
    }


    std::string                       sp_type;
    size_t                            gen_num_per_circle = 0;
    std::unique_ptr<EngineInitParams> vanilla_model_params = nullptr;

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params_;
    py::object                        eagle_model;
    py::object                        medusa_model;
    kmonitor::MetricsReporterPtr      metrics_reporter = nullptr;
};

struct WarmUpResult {
    size_t device_reserved_bytes  = 0;
    size_t max_used_memory        = 0;
};

class WeightsConverter {
public:
    WeightsConverter(bool need_copy,
                     ft::QuantAlgo quant_alog = ft::QuantAlgo()) :
                     need_copy_(need_copy),
                     quant_algo_(quant_alog)
    {
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
                                              const std::string& scales_key = "",
                                              const std::string& zeros_key = "");

    ft::LayerNormWeightsPtr
    mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                              const std::string& gamma_key,
                              const std::string& beta_key = "",
                              const std::string& scale_key = "",
                              const std::string& scale_reciprocal_key = "");

    ft::FfnLayerWeights
    createFfnWeights(const ConstBufferPtrMap& map);

    ft::AttentionLayerWeights
    createAttentionWeights(const ConstBufferPtrMap& map);

    ft::ConstBufferPtr CopyTensorToBufferPtr(const torch::Tensor& tensor);

private:
    bool            need_copy_;
    ft::QuantAlgo   quant_algo_;
    ft::DeviceBase* device_;
    bool use_linear_bias_slopes_;
};

std::tuple<ft::GptInitParameter, std::unique_ptr<ft::Weights>> prepareEngineInitParams(py::object model, bool sp_model = false);

// extract mtp model weights list from model in python world.
// Note: keep mtp sequence.
std::deque<std::unique_ptr<ft::Weights>> prepareMTPModelWeights(py::object model);

std::unique_ptr<ProposeModelEngineInitParams> prepareMTPEngineInitParams(py::object model);

}  // namespace rtp_llm
