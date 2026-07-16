
#pragma once
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include <optional>
#include <string>
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

class KVCacheManager;  // Forward declaration

class PyWrappedModel: public ModelBase {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params,
                   py::object                py_instance,
                   bool                      is_prefill_cuda_graph_mode = false,
                   bool                      use_spec_decoding          = false);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);
    void            releaseBuffers() override;
    torch::Tensor   getMtpTargetHiddenStates(int64_t num_tokens) override;
    torch::Tensor   getMtpLastHiddenStates(int64_t num_tokens) override;

private:
    std::optional<PyCacheStoreInputs> prepareWriteCacheParams(const GptModelInputs& inputs);

private:
    // Helper functions to reduce code duplication
    torch_ext::PyAttentionInputs    buildPyAttentionInputs(const GptModelInputs& inputs);
    torch_ext::PyEmbeddingInputs    buildPyEmbeddingInputs(const GptModelInputs& inputs);
    torch_ext::PyMultimodalInputs   buildPyMultimodalInputs(const GptModelInputs& inputs);
    torch_ext::BertEmbeddingInputs  buildBertEmbeddingInputs(const GptModelInputs& inputs);
    torch_ext::AttentionInputsByTag setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                                   const GptModelInputs&         inputs);
    GptModelOutputs                 callForwardPostLayers(torch::Tensor         hidden_states,
                                                          const GptModelInputs& inputs,
                                                          bool                  skip_final_layernorm,
                                                          size_t                num_valid_tokens = -1);
    torch::Tensor                   tensorHoldHostAndToCuda(const torch::Tensor& tensor);

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
    GptModelOutputs forwardPostLayersLastHidden(torch::Tensor hidden, const GptModelInputs& inputs);
    MicroBatchPlan  planMicroBatches(const GptModelInputs& inputs);
    std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
         splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan);
    void holdInputsHostBuffers(const GptModelInputs& inputs);

    // Member variables (formerly inherited from GptModel)
    const rtp_llm::ExecProperties                   device_props_;
    const rtp_llm::MlaOpsType                       mla_ops_type_;
    const size_t                                    layer_num_;
    const GptModelDescription                       description_;
    std::optional<rtp_llm::GroupedCacheLayerLayout> kv_cache_layer_layout_;
    std::shared_ptr<KVCacheManager>                 cache_manager_;  // For cache_store access
    torch::Tensor                                   residual_scale_fp32_;
    torch::Tensor                                   residual_scale_;
    ModelBufferHolder                               buffer_holder_;

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
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode,
                                      bool                      use_spec_decoding):
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

    c10::InferenceMode inference_guard(true);

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
        init_resources.kv_cache.emplace(params.kv_cache_layer_layout.value());
    }
    init_resources.is_speculative         = (params.sp_config.type != SP_TYPE_NONE);
    init_resources.is_decode_role         = (params.parallelism_config.role_type == RoleType::DECODE);
    init_resources.max_context_batch_size = params.runtime_config.fifo_scheduler_config.max_context_batch_size;

    py::object py_init_result;
    // Always initialize py_model_ so it can be used as fallback when CUDA graph cannot run
    py_model_                 = py_instance;
    auto py_initialize_method = py_model_.attr("initialize");
    py_init_result            = py_initialize_method(init_resources);
    if (enable_cuda_graph_ && !params.kv_cache_layer_layout.has_value() && !is_prefill_cuda_graph_mode) {
        RTP_LLM_LOG_WARNING(
            "CUDA graph enabled but kv_cache_layer_layout not available (warmup?), skipping graph capture");
        enable_cuda_graph_ = false;
    } else if (enable_cuda_graph_) {
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
        if (params.kv_cache_layer_layout.has_value()) {
            graph_params.kv_cache_group_tags = params.kv_cache_layer_layout->topology().groupTagsSnapshot();
        }
        // Derive combo_position_ids capture-buffer factor from the C++ rope_config:
        // 0 = model has no combo_position_ids (no buffer allocated, capture skips it);
        // >0 = factor (Mrope models such as qwen3-vl / qwen35-moe set rope_config.style
        // = Mrope and rope_config.index_factor accordingly). No Python reflection — the
        // rope style is intrinsic to the model description and already populated here.
        graph_params.position_id_len_factor = (description_.attention_conf.rope_config.style == RopeStyle::Mrope) ?
                                                  description_.attention_conf.rope_config.index_factor :
                                                  0;

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
            graph_params.num_tokens_per_bs = params.sp_config.gen_num_per_cycle + 1;
        } else {
            graph_params.num_tokens_per_bs = 1;
        }
        // Target-model decode with SP enabled is the multi-token verify path.
        // NormalExecutor::decodeWarmUp does not set use_spec_decoding, so infer
        // this graph role from the model/config identity as well; otherwise the
        // Python model sees is_prefill=true and incorrectly enters prefill.
        const bool is_target_verify_decode = params.sp_config.type != SP_TYPE_NONE
                                             && params.sp_config.gen_num_per_cycle > 0 && !params.model_id
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
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    cache_store_async_writer_ = std::make_unique<CacheStoreAsyncWriter>(params.parallelism_config.local_rank);

    if (device_props_.enable_prefill_cp) {
        context_parallel_processor_ =
            ContextParallelProcessorFactory::create(ProcessorType::ZIG_ZAG, params.parallelism_config);
        RTP_LLM_LOG_INFO("Context parallel processor initialized with ZIG_ZAG strategy.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
