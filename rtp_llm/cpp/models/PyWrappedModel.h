
#pragma once
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include <optional>
#include <string>
#include <atomic>
#include <mutex>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
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
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

namespace py = pybind11;

namespace rtp_llm {

inline void syncCudaGraphCaptureRanks(const ParallelismConfig& parallelism_config, const char* phase) {
    if (parallelism_config.world_size <= 1) {
        return;
    }

    py::gil_scoped_acquire gil;
    try {
        auto collective = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        auto group      = collective.attr("Group").attr("DP_AND_TP");
        collective.attr("barrier")(group);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("CUDA graph capture rank sync failed at %s:\n%s", phase, e.what());
        throw;
    }
}

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
    torch::Tensor   getMtpTargetHiddenStates(int64_t num_tokens) override;
    torch::Tensor   getMtpLastHiddenStates(int64_t num_tokens) override;
    void            prepareAttentionInputs(const GptModelInputs& inputs) override;
    void            prepareAttentionInputs(const GptModelInputs& inputs, bool skip_forward_event_sync);
    void            updateKVCacheKernelBlockId(const GptModelInputs& inputs) override;

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
    const rtp_llm::MlaOpsType                mla_ops_type_;
    const size_t                             layer_num_;
    const GptModelDescription                description_;
    std::optional<rtp_llm::CacheLayerLayout> kv_cache_layer_layout_;
    std::shared_ptr<KVCacheManager>          cache_manager_;  // For cache_store access
    torch::Tensor                            residual_scale_fp32_;
    torch::Tensor                            residual_scale_;
    TensorHolder                             buffer_holder_;

    GraphBase* graph_runner_{nullptr};
    py::object py_model_;
    py::object held_attn_pyobj_;
    bool       enable_cuda_graph_{false};
    bool       is_prefill_cuda_graph_mode_{false};
    bool       use_spec_decoding_{false};
    bool       enable_device_perf_{false};
    bool       check_nan_{false};

    std::unique_ptr<IContextParallelProcessor> context_parallel_processor_{nullptr};
    std::unique_ptr<CacheStoreAsyncWriter>     cache_store_async_writer_;

    // Accumulated H2D copies from tensorHoldHostAndToCuda(); flushed as one kernel per forward.
    FusedD2DCopyParams d2d_copies_;

    // is_pinned() is expensive on CPU; only assert during first N forwards as a sanity check.
    static constexpr int kPinnedCheckForwardCount = 3;
    int                  pinned_check_remaining_{kPinnedCheckForwardCount};

    std::atomic<bool>            prepared_attention_inputs_{false};
    torch_ext::PyAttentionInputs attention_inputs_;
    CudaGraphState               graph_state_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode,
                                      bool                      use_spec_decoding,
                                      const std::vector<int>&   kv_cache_layer_to_group):
    device_props_(buildExecProperties(params.parallelism_config, params.device_resource_config)),
    mla_ops_type_(params.mla_ops_type),
    layer_num_(params.weights.layers.size()),
    description_(params.description),
    cache_manager_(params.cache_manager),
    enable_cuda_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
    use_spec_decoding_(use_spec_decoding),
    enable_device_perf_(params.profile_debug_logging_config.enable_device_perf),
    check_nan_(params.profile_debug_logging_config.check_nan) {
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

        kv_cache.layer_group_types             = layout.layer_group_types;
        kv_cache.group_region_names            = layout.group_region_names;
        kv_cache.layer_region_to_group_id      = layout.layer_region_to_group_id;
        kv_cache.kv_cache_base_by_layer_region = layout.layers_to_kv_buffer_ptrs_by_attn;
        kv_cache.kv_scale_base_by_layer_region = layout.layers_to_scale_buffer_ptrs_by_attn;

        // Flatten by_attn into a 1D vector for pybind11 compatibility
        // Layout: [layer_0_type_0, ..., layer_0_type_7, layer_1_type_0, ...]
        {
            const size_t attn_count = static_cast<size_t>(rtp_llm::KVCacheRegionName::REGION_COUNT);
            const size_t num_layers = layout.layers_to_kv_buffer_ptrs_by_attn.size();
            kv_cache.kv_cache_base_by_layer_region_flat.resize(num_layers * attn_count);
            for (size_t l = 0; l < num_layers; ++l) {
                for (size_t a = 0; a < attn_count && a < layout.layers_to_kv_buffer_ptrs_by_attn[l].size(); ++a) {
                    auto& t = layout.layers_to_kv_buffer_ptrs_by_attn[l][a];
                    kv_cache.kv_cache_base_by_layer_region_flat[l * attn_count + a] =
                        t.defined() ? t : torch::empty({0});
                }
            }
        }

        init_resources.kv_cache = kv_cache;
    }
    init_resources.is_speculative = (params.sp_config.type != SP_TYPE_NONE);
    init_resources.is_decode_role = (params.parallelism_config.role_type == RoleType::DECODE);
    init_resources.max_context_batch_size = params.runtime_config.fifo_scheduler_config.max_context_batch_size;

    py::object py_init_result;
    // Always initialize py_model_ so it can be used as fallback when CUDA graph cannot run
    py_model_                 = py_instance;
    auto py_initialize_method = py_model_.attr("initialize");
    try {
        py_init_result = py_initialize_method(init_resources);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python model initialize failed:\n%s", e.what());
        throw;
    }
    const auto py_model_class_name = py::str(py_instance.attr("__class__").attr("__name__")).cast<std::string>();
    if (enable_cuda_graph_ && py_model_class_name == "DeepSeekV4Model" && !params.kv_cache_layer_layout.has_value()) {
        RTP_LLM_LOG_WARNING(
            "Disable CUDA graph for DeepSeekV4 warmup without kv_cache_layer_layout; real executor can capture after "
            "CacheManager is initialized.");
        enable_cuda_graph_ = false;
    }
    if (enable_cuda_graph_) {
#if USING_CUDA || USING_ROCM
        c10::ScalarType dtype = dataTypeToTorchType(description_.data_type);

        // Create GraphParams from individual config fields
        GraphParams graph_params;
        graph_params.enable_cuda_graph            = params.hw_kernel_config.enable_cuda_graph;
        graph_params.enable_cuda_graph_debug_mode = params.hw_kernel_config.enable_cuda_graph_debug_mode;
        graph_params.is_prefill_cuda_graph_mode   = is_prefill_cuda_graph_mode;
        graph_params.max_seq_len                  = params.max_seq_len;
        graph_params.tokens_per_block             = params.tokens_per_block;
        graph_params.kernel_tokens_per_block      = params.kernel_tokens_per_block;
        graph_params.hidden_size                  = params.hidden_size;
        graph_params.hc_mult                      = params.hc_mult;
        graph_params.model_data_type              = dtype;
        graph_params.max_context_batch_size       = params.concurrency_config.concurrency_limit;
        graph_params.prefill_capture_seq_lens     = params.hw_kernel_config.prefill_capture_seq_lens;
        graph_params.decode_capture_batch_sizes   = params.hw_kernel_config.decode_capture_batch_sizes;
        graph_params.kv_cache_group_num           = params.kv_cache_group_num;

        if (kv_cache_layer_to_group.size() > 0) {
            graph_params.kv_cache_layer_to_group = kv_cache_layer_to_group;
        } else {
            graph_params.kv_cache_layer_to_group = params.kv_cache_layer_to_group;
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

        if (is_prefill_cuda_graph_mode && params.sp_config.type == SP_TYPE_NONE) {
            // for embedding model
            graph_params.num_tokens_per_bs = params.max_seq_len;
        } else if (params.sp_config.type != SP_TYPE_NONE && params.sp_config.gen_num_per_cycle > 0
                   && (!params.model_id || is_prefill_cuda_graph_mode)) {
            // for target model verify and draft model prefill
            // Only use multi-token capture when SP is actually enabled;
            // gen_num_per_cycle may be >1 from config even when SP is disabled.
            graph_params.num_tokens_per_bs = params.sp_config.gen_num_per_cycle + 1;
        } else {
            graph_params.num_tokens_per_bs = 1;
        }
        // Target-model decode path with SP enabled (num_tokens_per_bs>1,
        // not prefill-graph, model_id==0) must set is_target_verify so the
        // Python dispatch routes through forward_decode.  NormalExecutor's
        // decodeWarmUp path defaults use_spec_decoding=false but still sees
        // sp_config enabled, so infer the flag from config instead of
        // relying solely on the constructor arg.
        const bool is_target_verify_decode =
            params.sp_config.type != SP_TYPE_NONE
            && params.sp_config.gen_num_per_cycle > 0
            && !params.model_id
            && !is_prefill_cuda_graph_mode;
        graph_params.is_target_verify = use_spec_decoding || is_target_verify_decode;
        if (params.sp_config.type != SP_TYPE_NONE) {
            graph_params.sp_steps = params.sp_config.gen_num_per_cycle;
        }

        graph_runner_ = new CudaGraphRunner(graph_params, py_instance);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be nullptr in PyWrapper");
        {
            void* nccl_comm = cuda_graph::getGraphCaptureTpNcclComm();
            cuda_graph::register_graph_capture_nccl_comm(nccl_comm,
                                                         static_cast<int>(params.parallelism_config.tp_size),
                                                         static_cast<int>(params.parallelism_config.tp_rank));
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
        try {
            syncCudaGraphCaptureRanks(params.parallelism_config, "before_initCapture");
            py_init_result = py_initialize_method(init_resources);
            graph_runner_->initCapture();
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_ERROR("Python model initialize failed (cuda_graph branch):\n%s", e.what());
            throw;
        }
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
