#pragma once
#include <cstddef>
#include <pybind11/pytypes.h>
#include <tuple>

#include "rtp_llm/cpp/th_op/GlobalConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace th = torch;


namespace rtp_llm {

using TensorMap  = std::unordered_map<std::string, th::Tensor>;
using TensorMaps = std::vector<TensorMap>;
using ConstBufferPtrMap  = std::unordered_map<std::string, rtp_llm::ConstBufferPtr>;
using ConstBufferPtrMaps = std::vector<ConstBufferPtrMap>;

static void initialize(const rtp_llm::GptInitParameter& config) {
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        auto& global_config = GlobalConfig::get();
        global_config.parallelism_distributed_config = config.parallelism_distributed_config;
        global_config.scheduler_config = config.scheduler_config;
        global_config.concurrency_config = config.concurrency_config;
        global_config.fmha_config = config.fmha_config;
        global_config.kv_cache_config = config.kv_cache_config;
        global_config.profiling_debug_logging_config = config.profiling_debug_logging_config;
        global_config.hw_kernel_config = config.hw_kernel_config;
        global_config.device_resource_config = config.device_resource_config;
        global_config.model_specific_config = config.model_specific_config;
        global_config.service_discovery_config = config.service_discovery_config;
        global_config.misc_config = config.misc_config;
        global_config.sampler_config = config.sampler_config;
        global_config.moe_config = config.moe_config;
        global_config.sp_config = config.sp_config;
        global_config.cache_store_config = config.cache_store_config;
        global_config.batch_decode_scheduler_config = config.batch_decode_scheduler_config;
        global_config.fifo_scheduler_config = config.fifo_scheduler_config;
        RTP_LLM_LOG_INFO("\nGlobal GptInitParameter initialized.\n");
    });
}

struct EngineInitParams: public th::jit::CustomClassHolder {
    EngineInitParams() {};
   // This class is the only one that holds gpt_weights object globally.
    EngineInitParams(size_t                           model_id,
                     const rtp_llm::GptInitParameter& gpt_init_parameter,
                     rtp_llm::Weights&&               gpt_weights):
        model_id(model_id), gpt_init_parameter(gpt_init_parameter), gpt_weights(std::move(gpt_weights)) {
            initialize(gpt_init_parameter);
    }

    size_t model_id;
    rtp_llm::GptInitParameter         gpt_init_parameter;
    rtp_llm::Weights                  gpt_weights;

    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;
public:
    void showGptInitParameter() {
        gpt_init_parameter.showDebugInfo();
    }
};

struct ProposeModelEngineInitParams: public th::jit::CustomClassHolder {
    ProposeModelEngineInitParams() {};

    // Constructor for vanilla propose model
    ProposeModelEngineInitParams(size_t                           model_id,
                                 std::string                      sp_type,
                                 size_t                           gen_num_per_circle,
                                 const rtp_llm::GptInitParameter& gpt_init_parameter,
                                 rtp_llm::Weights&&               gpt_weights):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(new EngineInitParams(model_id, gpt_init_parameter, std::move(gpt_weights))) {}

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

    bool draftModel() {
        return sp_type == "vanilla" || sp_type == "mtp" || sp_type == "eagle3";
    }

    const rtp_llm::GptInitParameter& getGptInitParameter() {
        if (sp_type == "vanilla") {
            return vanilla_model_params->gpt_init_parameter;
        } else if (sp_type == "mtp" || sp_type == "eagle3") {
            RTP_LLM_CHECK(!mtp_model_params_->empty());
            RTP_LLM_CHECK(mtp_model_params_->at(0) != nullptr);
            return mtp_model_params_->at(0)->gpt_init_parameter;
        } else {
            RTP_LLM_FAIL("error sp type[%s] do not have GptInitParameter", sp_type.c_str());
        }
    }


    std::string                       sp_type;
    size_t                            gen_num_per_circle = 0;
    std::unique_ptr<EngineInitParams> vanilla_model_params = nullptr;

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params_;
    py::object                        eagle_model;
    kmonitor::MetricsReporterPtr      metrics_reporter = nullptr;
};

struct WarmUpResult {
    size_t device_reserved_bytes  = 0;
    size_t max_used_memory        = 0;
};

class WeightsConverter {
public:
    WeightsConverter(bool need_copy,
                     rtp_llm::QuantAlgo quant_alog = rtp_llm::QuantAlgo()) :
                     need_copy_(need_copy),
                     quant_algo_(quant_alog)
    {
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
    std::unique_ptr<rtp_llm::Weights>
    createGptWeights(py::object layer_weights,
                     py::object  global_weight);

    std::unique_ptr<rtp_llm::Weights>
    createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                     std::unique_ptr<TensorMap>  global_weight);

    std::unique_ptr<rtp_llm::Weights>
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

    rtp_llm::ConstBufferPtr mayFindBuffer(const ConstBufferPtrMap& map,
                                     const std::string& key);

    rtp_llm::DenseWeightsPtr mayCreateDenseWeights(const ConstBufferPtrMap& map,
                                              const std::string& kernel_key,
                                              const std::string& bias_key = "",
                                              const std::string& scales_key = "",
                                              const std::string& zeros_key = "");

    rtp_llm::LayerNormWeightsPtr
    mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                              const std::string& gamma_key,
                              const std::string& beta_key = "",
                              const std::string& scale_key = "",
                              const std::string& scale_reciprocal_key = "");

    rtp_llm::FfnLayerWeights
    createFfnWeights(const ConstBufferPtrMap& map);

    rtp_llm::AttentionLayerWeights
    createAttentionWeights(const ConstBufferPtrMap& map);

    rtp_llm::ConstBufferPtr CopyTensorToBufferPtr(const torch::Tensor& tensor);

private:
    bool            need_copy_;
    rtp_llm::QuantAlgo   quant_algo_;
    rtp_llm::DeviceBase* device_;
    bool use_linear_bias_slopes_;
};

std::tuple<rtp_llm::GptInitParameter, std::unique_ptr<rtp_llm::Weights>> prepareEngineInitParams(py::object model, bool sp_model = false);

std::unique_ptr<ProposeModelEngineInitParams> prepareMTPEngineInitParams(size_t model_id, py::object model);

}  // namespace rtp_llm
