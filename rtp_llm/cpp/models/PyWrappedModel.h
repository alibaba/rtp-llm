
#pragma once
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include <optional>
#include <string>
#include <mutex>
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#else
namespace rtp_llm {
class CudaGraphRunnerBase;
}
#endif

#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm {

class KVCacheManager;  // Forward declaration

class PyWrappedModel: public ModelBase {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params,
                   py::object                py_instance,
                   bool                      is_prefill_cuda_graph_mode = false,
                   bool                      use_spec_decoding          = false,
                   const std::vector<int>&   kv_cache_layer_to_group    = {});
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);
    void            releaseBuffers() override;

private:
    std::optional<PyCacheStoreInputs> prepareWriteCacheParams(const GptModelInputs& inputs);

private:
    // Helper functions to reduce code duplication
    torch_ext::PyAttentionInputs   buildPyAttentionInputs(const GptModelInputs& inputs);
    torch_ext::BertEmbeddingInputs buildBertEmbeddingInputs(const GptModelInputs& inputs);
    void setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs, const GptModelInputs& inputs);
    GptModelOutputs callForwardPostLayers(torch::Tensor         hidden_states,
                                          const GptModelInputs& inputs,
                                          bool                  skip_final_layernorm,
                                          size_t                num_valid_tokens = -1);
    torch::Tensor   tensorHoldHostAndToCuda(const torch::Tensor& tensor);

    // Methods absorbed from GptModel
    torch::Tensor   tpSyncEmbeddingOrLogits(const torch::Tensor& input);
    GptModelOutputs forwardPostLayers(torch::Tensor         hidden,
                                      const bool            has_context_request,
                                      const bool            need_all_logits,
                                      const torch::Tensor&  lm_output_indexes,
                                      bool                  enable_sp,
                                      size_t                token_num,
                                      const GptModelInputs& inputs,
                                      torch::Tensor         merged_eagle3_hidden,
                                      bool                  skip_final_layernorm = false);
    MicroBatchPlan  planMicroBatches(const GptModelInputs& inputs);
    std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
         splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan);
    void holdInputsHostBuffers(const GptModelInputs& inputs);

    py::object initializeCudaGraphCapture(py::object                             py_instance,
                                          const torch_ext::PyModelInitResources& init_resources,
                                          const GptModelInitParams&              params,
                                          const std::vector<int>&                kv_cache_layer_to_group);

    // Member variables (formerly inherited from GptModel)
    const rtp_llm::ExecProperties            device_props_;
    const rtp_llm::ExecInitParams            device_init_params_;
    const rtp_llm::MlaOpsType                mla_ops_type_;
    const size_t                             layer_num_;
    const GptModelDescription                description_;
    std::optional<rtp_llm::CacheLayerLayout> kv_cache_layer_layout_;
    std::shared_ptr<KVCacheManager>          cache_manager_;  // For cache_store access
    torch::Tensor                            residual_scale_fp32_;
    torch::Tensor                            residual_scale_;
    ModelBufferHolder                        buffer_holder_;

    CudaGraphRunnerBase* graph_runner_{nullptr};
    py::object           py_model_;
    py::object           held_attn_pyobj_;
    bool                 enable_cuda_graph_{false};
    bool                 is_prefill_cuda_graph_mode_{false};
    bool                 use_spec_decoding_{false};
    bool                 enable_device_perf_{false};

    std::unique_ptr<IContextParallelProcessor> context_parallel_processor_{nullptr};
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode,
                                      bool                      use_spec_decoding,
                                      const std::vector<int>&   kv_cache_layer_to_group):
    device_props_(buildExecProperties(params.exec_init_params)),
    device_init_params_(params.exec_init_params),
    mla_ops_type_(params.exec_init_params.mla_ops_type),
    layer_num_(params.weights.layers.size()),
    description_(params.description),
    cache_manager_(params.cache_manager),
    enable_cuda_graph_(device_init_params_.hw_kernel_config.enable_cuda_graph),
    is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
    use_spec_decoding_(use_spec_decoding),
    enable_device_perf_(device_init_params_.profile_debug_logging_config.enable_device_perf) {
    weights_               = params.weights;
    model_id_              = params.model_id;
    kv_cache_layer_layout_ = params.kv_cache_layer_layout;
    if (abs(description_.residual_scalar - 1.0) > 1e-6) {
        auto residual_tensor = torch::tensor({(float)description_.residual_scalar}, torch::kFloat32).cuda();
#if USING_CUDA
        c10::cuda::getCurrentCUDAStream().synchronize();
#endif
        residual_scale_fp32_ = residual_tensor;
        residual_scale_      = residual_tensor.to(dataTypeToTorchType(description_.data_type));
    }
    if (params.description.ffn_conf.moe_configs.has_value()) {
        auto moe_conf         = params.description.ffn_conf.moe_configs.value();
        overall_expert_stats_ = execCreateMoeExpertStates(
            {layer_num_, moe_conf.ep_size, moe_conf.expert_num, moe_conf.expert_num + moe_conf.extra_expert_num});
    }

    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }

    py::gil_scoped_acquire          gil;
    torch_ext::PyModelInitResources init_resources;

    if (params.kv_cache_layer_layout.has_value()) {
        torch_ext::KVCache kv_cache;
        kv_cache.seq_size_per_block        = params.description.attention_conf.tokens_per_block;
        kv_cache.kernel_seq_size_per_block = params.description.attention_conf.kernel_tokens_per_block;
        const auto& layout                 = params.kv_cache_layer_layout.value();
        kv_cache.kv_cache_base_by_layer.reserve(layout.layers_to_kv_buffer_ptrs.size());
        kv_cache.num_kv_heads  = params.description.attention_conf.kv_head_num;
        kv_cache.head_dim      = params.description.attention_conf.size_per_head;
        kv_cache.use_mla       = params.description.attention_conf.use_mla;
        kv_cache.kv_lora_rank  = params.description.attention_conf.kv_lora_rank;
        kv_cache.rope_head_dim = params.description.attention_conf.rope_head_dim;
        for (const auto& t : layout.layers_to_kv_buffer_ptrs) {
            kv_cache.kv_cache_base_by_layer.push_back(t);
        }
        kv_cache.kv_scale_base_by_layer.reserve(layout.layers_to_scale_buffer_ptrs.size());
        for (const auto& t : layout.layers_to_scale_buffer_ptrs) {
            kv_cache.kv_scale_base_by_layer.push_back(t);
        }

        kv_cache.layer_attn_types = layout.layer_attn_types;
        init_resources.kv_cache   = kv_cache;
    }

    py::object py_init_result;
    // Always initialize py_model_ so it can be used as fallback when CUDA graph cannot run
    py_model_                 = py_instance;
    auto py_initialize_method = py_model_.attr("initialize");
    py_init_result            = py_initialize_method(init_resources);
    if (enable_cuda_graph_) {
        py_init_result = initializeCudaGraphCapture(py_instance, init_resources, params, kv_cache_layer_to_group);
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    if (device_props_.enable_prefill_cp) {
        context_parallel_processor_ =
            ContextParallelProcessorFactory::create(ProcessorType::ZIG_ZAG, params.parallelism_config);
        RTP_LLM_LOG_INFO("Context parallel processor initialized with ZIG_ZAG strategy.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
