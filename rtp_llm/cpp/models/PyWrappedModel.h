
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
// cuda_graph_base.h is platform-agnostic (only defines GraphParams/CudaGraphState structs),
// safe to include unconditionally. cuda_graph_runner.h requires CUDA/ROCm runtime.
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#endif

#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/CacheStoreAsyncWriter.h"

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

    GraphBase* graph_runner_{nullptr};
    py::object py_model_;
    py::object held_attn_pyobj_;
    bool       enable_cuda_graph_{false};
    bool       is_prefill_cuda_graph_mode_{false};
    bool       use_spec_decoding_{false};
    bool       enable_device_perf_{false};

    std::unique_ptr<IContextParallelProcessor> context_parallel_processor_{nullptr};
    std::unique_ptr<CacheStoreAsyncWriter>     cache_store_async_writer_;
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
#if USING_CUDA || USING_ROCM
        c10::ScalarType dtype = dataTypeToTorchType(description_.data_type);

        // Create GraphParams from ExecInitParams
        const auto& device_params = device_init_params_;
        GraphParams graph_params;
        graph_params.enable_cuda_graph            = device_params.hw_kernel_config.enable_cuda_graph;
        graph_params.enable_cuda_graph_debug_mode = device_params.hw_kernel_config.enable_cuda_graph_debug_mode;
        graph_params.is_prefill_cuda_graph_mode   = is_prefill_cuda_graph_mode;
        graph_params.max_seq_len                  = device_params.max_seq_len;
        graph_params.tokens_per_block             = device_params.tokens_per_block;
        graph_params.kernel_tokens_per_block      = device_params.kernel_tokens_per_block;
        graph_params.hidden_size                  = device_params.hidden_size;
        graph_params.model_data_type              = dtype;
        graph_params.concurrency_limit            = device_params.concurrency_config.concurrency_limit;
        graph_params.prefill_capture_seq_lens     = device_params.hw_kernel_config.prefill_capture_seq_lens;
        graph_params.decode_capture_batch_sizes   = device_params.hw_kernel_config.decode_capture_batch_sizes;
        graph_params.kv_cache_group_num           = device_params.kv_cache_group_num;

        if (kv_cache_layer_to_group.size() > 0) {
            graph_params.kv_cache_layer_to_group = kv_cache_layer_to_group;
        } else {
            graph_params.kv_cache_layer_to_group = device_params.kv_cache_layer_to_group;
        }

        // clang-format off
        // Decision table for num_tokens_per_bs:
        // +---------------------------+--------------------------+----------------+----------+-------------------------+
        // | Model Type                | is_prefill_cuda_graph    | sp_config.type | model_id | num_tokens_per_bs       |
        // +---------------------------+--------------------------+----------------+----------+-------------------------+
        // | Embedding Model (prefill) | true                     | SP_TYPE_NONE   | -        | max_seq_len             |
        // | Draft Model (prefill)     | true                     | != SP_TYPE_NONE| 1        | gen_num_per_cycle + 1   |
        // | Normal Model (decode)     | false                    | SP_TYPE_NONE   | -        | 1 (default)             |
        // | Target Model (verify)     | false                    | != SP_TYPE_NONE| 0        | gen_num_per_cycle + 1   |
        // | Draft Model (decode)      | false                    | != SP_TYPE_NONE| 1        | 1 (default)             |
        // +---------------------------+--------------------------+----------------+----------+-------------------------+
        // clang-format on

        if (is_prefill_cuda_graph_mode && device_params.sp_config.type == SP_TYPE_NONE) {
            // for embedding model
            graph_params.num_tokens_per_bs = device_params.max_seq_len;
        } else if (device_params.sp_config.type != SP_TYPE_NONE && device_params.sp_config.gen_num_per_cycle > 1
                   && (!params.model_id || is_prefill_cuda_graph_mode)) {
            // for target model verify and draft model prefill
            // Only use multi-token capture when SP is actually enabled;
            // gen_num_per_cycle may be >1 from config even when SP is disabled.
            graph_params.num_tokens_per_bs = device_params.sp_config.gen_num_per_cycle + 1;
        } else {
            graph_params.num_tokens_per_bs = 1;
        }
        graph_params.is_target_verify = use_spec_decoding;
        if (device_params.sp_config.type != SP_TYPE_NONE) {
            graph_params.sp_steps = device_params.sp_config.gen_num_per_cycle;
        }

        graph_runner_ = new CudaGraphRunner(graph_params, py_instance);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be nullptr in PyWrapper");
        {
            void* nccl_comm = cuda_graph::getGraphCaptureTpNcclComm();
            cuda_graph::register_graph_capture_nccl_comm(
                nccl_comm, static_cast<int>(device_params.tp_size), static_cast<int>(device_params.tp_rank));
        }
#else
        RTP_LLM_CHECK_WITH_INFO(false, "CUDA/HIP Graph is only supported on CUDA/ROCm platform");
#endif
        if (weights_.position_encoding) {
            graph_runner_->setPositionEncoding(weights_.position_encoding->kernel.cuda());
        }
        if (weights_.token_type_embedding) {
            graph_runner_->setTokenTypeEmbedding(weights_.token_type_embedding->kernel.cuda());
        }
        graph_runner_->setInputEmbeddingScalar(description_.input_embedding_scalar);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be null");
        auto py_initialize_method = py_instance.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    cache_store_async_writer_ = std::make_unique<CacheStoreAsyncWriter>();

    if (device_props_.enable_prefill_cp) {
        context_parallel_processor_ =
            ContextParallelProcessorFactory::create(ProcessorType::ZIG_ZAG, params.parallelism_config);
        RTP_LLM_LOG_INFO("Context parallel processor initialized with ZIG_ZAG strategy.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
