
#pragma once
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/GenerationPrefillCudaGraphEligibility.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include <limits>
#include <optional>
#include <string>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
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
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif
#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"

namespace py = pybind11;

namespace rtp_llm {

namespace test {
struct PyWrappedModelTestPeer;
}

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

// Fixed construction-time role of a DSpARK Python-model wrapper. This is not
// per-call phase metadata: propose and commit own different model wrappers and
// different CUDA-graph input widths for the lifetime of the executor.
enum class DSparkModelRole : uint8_t {
    NONE,
    PROPOSE,
    COMMIT,
};

class PyWrappedModel: public ModelBase {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params,
                   py::object                py_instance,
                   bool                      is_prefill_cuda_graph_mode   = false,
                   bool                      use_spec_decoding            = false,
                   DSparkModelRole           dspark_model_role            = DSparkModelRole::NONE,
                   bool                      allow_cuda_graph             = true,
                   bool                      track_cache_store_completion = false);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);
    void            releaseBuffers() override;
    torch::Tensor   getMtpTargetHiddenStates(int64_t num_tokens) override;
    torch::Tensor   getMtpLastHiddenStates(int64_t num_tokens) override;
    bool            hasMtpTargetHiddenBuffer() const override;
    void            prepareAttentionInputs(const GptModelInputs& inputs) override;
    void            prepareAttentionInputs(const GptModelInputs& inputs, bool skip_forward_event_sync);
    void            updateKVCacheKernelBlockId(const GptModelInputs& inputs) override;
    std::string     waitCacheStorePublication() override;

private:
    friend struct test::PyWrappedModelTestPeer;

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
    // CP gather-last-hidden exit: `hidden` is already the lm_output_indexes-selected,
    // post-final-layernorm rows produced by handleOutputsLastHidden, so this runs
    // lm_head directly (no index_select, no final layernorm — matching the existing
    // CP path's skip_final_layernorm=true) and returns the small [num_lm, hidden]
    // both as hidden_states and all_hidden_states.
    GptModelOutputs forwardPostLayersLastHidden(torch::Tensor hidden, const GptModelInputs& inputs);
    MicroBatchPlan  planMicroBatches(const GptModelInputs& inputs);
    std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
                    splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan);
    void            holdInputsHostBuffers(const GptModelInputs& inputs);
    GraphBase*      selectGraphRunner(const torch_ext::PyAttentionInputs& attention_inputs) const;
    CudaGraphState& selectGraphState(const torch_ext::PyAttentionInputs& attention_inputs);
    void            cleanupAfterConstructionFailure() noexcept;

    // Member variables (formerly inherited from GptModel)
    const rtp_llm::ExecProperties                   device_props_;
    const bool                                      enable_prefill_cp_;
    const DSparkModelRole                           dspark_model_role_;
    const bool                                      track_cache_store_completion_;
    const rtp_llm::MlaOpsType                       mla_ops_type_;
    const size_t                                    layer_num_;
    const GptModelDescription                       description_;
    std::optional<rtp_llm::GroupedCacheLayerLayout> kv_cache_layer_layout_;
    std::shared_ptr<KVCacheManager>                 cache_manager_;  // For cache_store access
    torch::Tensor                                   residual_scale_fp32_;
    torch::Tensor                                   residual_scale_;
    TensorHolder                                    buffer_holder_;

    std::unique_ptr<GraphBase> graph_runner_;
    std::unique_ptr<GraphBase> generation_prefill_graph_runner_;
    py::object                 py_model_;
    py::object                 py_forward_method_;
    py::object                 held_attn_pyobj_;
    // Per-wrapper ownership, not the process-wide configuration request. Only
    // the normal main-generation wrapper can own this secondary runner.
    const bool                       owns_generation_prefill_cuda_graph_{false};
    bool                             enable_cuda_graph_{false};
    bool                             is_prefill_cuda_graph_mode_{false};
    GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_init_status_{
        GenerationPrefillCudaGraphStatus::NOT_REQUESTED};
    bool use_spec_decoding_{false};
    bool has_mtp_hidden_buffer_{false};
    bool enable_device_perf_{false};
    bool check_nan_{false};

    std::unique_ptr<IContextParallelProcessor> context_parallel_processor_{nullptr};
    std::shared_ptr<CacheStoreAsyncWriter>     cache_store_async_writer_;

    // Accumulated H2D copies from tensorHoldHostAndToCuda(); flushed as one kernel per forward.
    FusedD2DCopyParams d2d_copies_;

    // is_pinned() is expensive on CPU; only assert during first N forwards as a sanity check.
    static constexpr int kPinnedCheckForwardCount = 3;
    int                  pinned_check_remaining_{kPinnedCheckForwardCount};

    std::atomic<bool>               prepared_attention_inputs_{false};
    torch_ext::PyAttentionInputs    attention_inputs_;
    torch_ext::AttentionInputsByTag attention_inputs_by_tag_;
    CudaGraphState                  graph_state_;
    CudaGraphState                  generation_prefill_cuda_graph_state_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode,
                                      bool                      use_spec_decoding,
                                      DSparkModelRole           dspark_model_role,
                                      bool                      allow_cuda_graph,
                                      bool                      track_cache_store_completion):
    device_props_(buildExecProperties(params.parallelism_config, params.device_resource_config)),
    // Every prefill-shaped forward of a CP-enabled model goes through the
    // standard split/gather path — including the DSpARK draft commit, whose
    // incremental-prefill geometry CP-splits like any prompt while its
    // already-rank-local hidden rows pass through untouched. The fixed-width
    // non-causal propose block never reaches a CP-enabled model: proposals
    // run only on decode roles, where prefill CP is off (colocated CP is
    // rejected at executor construction).
    enable_prefill_cp_(device_props_.enable_prefill_cp),
    dspark_model_role_(dspark_model_role),
    track_cache_store_completion_(track_cache_store_completion),
    mla_ops_type_(params.mla_ops_type),
    layer_num_(params.weights.layers.size()),
    description_(params.description),
    cache_manager_(params.cache_manager),
    owns_generation_prefill_cuda_graph_(shouldCreateGenerationPrefillCudaGraph(params.hw_kernel_config,
                                                                               allow_cuda_graph,
                                                                               is_prefill_cuda_graph_mode,
                                                                               params.parallelism_config.role_type,
                                                                               params.sp_config.type)),
    enable_cuda_graph_(params.hw_kernel_config.enable_cuda_graph && allow_cuda_graph),
    is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
    generation_prefill_cuda_graph_init_status_(owns_generation_prefill_cuda_graph_ ?
                                                   GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE :
                                                   GenerationPrefillCudaGraphStatus::NOT_REQUESTED),
    use_spec_decoding_(use_spec_decoding),
    enable_device_perf_(params.profile_debug_logging_config.enable_device_perf),
    check_nan_(params.profile_debug_logging_config.check_nan) {

    c10::InferenceMode inference_guard(true);

    const auto& generation_prefill_cuda_graph_buckets =
        params.hw_kernel_config.generation_prefill_capture_token_buckets;
    if (owns_generation_prefill_cuda_graph_) {
        const auto& parallelism = params.parallelism_config;
        RTP_LLM_CHECK_WITH_INFO(isSingleDeviceGenerationPrefillCudaGraphConfig(parallelism),
                                "Generation prefill CUDA graph currently supports only single-device execution: "
                                "world_size=%ld tp_size=%ld dp_size=%ld ep_size=%ld pp_size=%ld ffn_sp_size=%ld "
                                "ffn_tp_size=%ld enable_sp=%d prefill_cp=%d ffn_disaggregate=%d",
                                parallelism.world_size,
                                parallelism.tp_size,
                                parallelism.dp_size,
                                parallelism.ep_size,
                                parallelism.pp_size,
                                parallelism.ffn_sp_size,
                                parallelism.ffn_tp_size,
                                static_cast<int>(parallelism.enable_sp),
                                static_cast<int>(parallelism.prefill_cp_config.is_enabled()
                                                 || parallelism.prefill_cp_config.is_prefill_enabled()),
                                static_cast<int>(parallelism.ffn_disaggregate_config.enable_ffn_disaggregate));
        const auto& buckets = generation_prefill_cuda_graph_buckets;
        RTP_LLM_CHECK_WITH_INFO(!buckets.empty(), "GENERATION_PREFILL_CAPTURE_CONFIG must not be empty");
        RTP_LLM_CHECK_WITH_INFO(
            buckets.size() <= static_cast<size_t>(HWKernelConfig::kGenerationPrefillCudaGraphMaxCaptureBuckets),
            "GENERATION_PREFILL_CAPTURE_CONFIG must not exceed %d buckets, got %zu",
            HWKernelConfig::kGenerationPrefillCudaGraphMaxCaptureBuckets,
            buckets.size());
        RTP_LLM_CHECK_WITH_INFO(params.hw_kernel_config.generation_prefill_cuda_graph_max_requests > 0,
                                "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS must be positive, got %d",
                                params.hw_kernel_config.generation_prefill_cuda_graph_max_requests);
        RTP_LLM_CHECK_WITH_INFO(
            params.hw_kernel_config.generation_prefill_cuda_graph_max_requests
                <= HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests,
            "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS must not exceed the backend limit %d, got %d",
            HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests,
            params.hw_kernel_config.generation_prefill_cuda_graph_max_requests);
        const int64_t reachable_request_capacity = generationPrefillCudaGraphReachableRequestCapacity(
            params.runtime_config.fifo_scheduler_config.max_context_batch_size,
            params.concurrency_config.concurrency_limit);
        RTP_LLM_CHECK_WITH_INFO(
            generationPrefillCudaGraphMaxRequestsFitsCapacity(
                params.hw_kernel_config.generation_prefill_cuda_graph_max_requests,
                params.runtime_config.fifo_scheduler_config.max_context_batch_size,
                params.concurrency_config.concurrency_limit),
            "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS=%d must not exceed the reachable context batch capacity "
            "min(MAX_CONTEXT_BATCH_SIZE=%ld, CONCURRENCY_LIMIT=%d)=%ld",
            params.hw_kernel_config.generation_prefill_cuda_graph_max_requests,
            params.runtime_config.fifo_scheduler_config.max_context_batch_size,
            params.concurrency_config.concurrency_limit,
            reachable_request_capacity);
        const auto invalid_bucket = std::find_if(buckets.begin(), buckets.end(), [&](int bucket) {
            return bucket <= 0 || bucket > params.max_seq_len
                   || bucket > HWKernelConfig::kGenerationPrefillCudaGraphMaxCaptureTokens;
        });
        RTP_LLM_CHECK_WITH_INFO(invalid_bucket == buckets.end(),
                                "Generation prefill CUDA graph buckets must be in "
                                "[1, min(max_seq_len=%ld, limit=%d)], got %d",
                                params.max_seq_len,
                                HWKernelConfig::kGenerationPrefillCudaGraphMaxCaptureTokens,
                                invalid_bucket == buckets.end() ? 0 : *invalid_bucket);
        const int64_t max_bucket = *std::max_element(buckets.begin(), buckets.end());
        RTP_LLM_CHECK_WITH_INFO(
            params.hw_kernel_config.generation_prefill_cuda_graph_max_requests < std::numeric_limits<int>::max()
                && generationPrefillCudaGraphPaddedTokenIndexFitsInt32(
                    params.hw_kernel_config.generation_prefill_cuda_graph_max_requests, max_bucket),
            "Generation prefill CUDA graph padded token index exceeds int32 capacity: max_requests=%d "
            "max_bucket=%ld",
            params.hw_kernel_config.generation_prefill_cuda_graph_max_requests,
            max_bucket);
    }

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

    py::gil_scoped_acquire gil;
    // A failed C++ constructor does not call ~PyWrappedModel(). Keep a local
    // guard active until every late Python cast, writer allocation, attribute
    // probe, and CP processor construction has completed.
    auto cleanup_on_failure = [](PyWrappedModel* model) noexcept { model->cleanupAfterConstructionFailure(); };
    std::unique_ptr<PyWrappedModel, decltype(cleanup_on_failure)> construction_guard(this, cleanup_on_failure);
    torch_ext::PyModelInitResources                               init_resources;

    if (params.kv_cache_layer_layout.has_value()) {
        // Block geometry travels on GptModelInitParams (filled from
        // cache_manager->cacheConfig() in NormalExecutor/MtpExecutor) rather
        // than the model-static attention_conf — for DSV4 the cache manager
        // promotes seq_size_per_block to a 256-token physical block while
        // attention_conf still reflects the 64-token --seq_size_per_block
        // CLI flag, causing the fused compressor to index state block_table
        // with the wrong stride and trap on unallocated ring slots.
        RTP_LLM_CHECK_WITH_INFO(params.tokens_per_block > 0 && params.kernel_tokens_per_block > 0
                                    && params.tokens_per_block % params.kernel_tokens_per_block == 0,
                                "GptModelInitParams must carry valid tokens_per_block / kernel_tokens_per_block "
                                "from CacheConfig before constructing PyWrappedModel KVCache; got tokens_per_block=%zu "
                                "kernel_tokens_per_block=%zu",
                                params.tokens_per_block,
                                params.kernel_tokens_per_block);
        init_resources.kv_cache.emplace(params.kv_cache_layer_layout.value());
    }
    init_resources.is_speculative         = (params.sp_config.type != SP_TYPE_NONE);
    init_resources.is_decode_role         = (params.parallelism_config.role_type == RoleType::DECODE);
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
    const char* forward_method     = dspark_model_role_ == DSparkModelRole::PROPOSE ? "forward_propose" :
                                     dspark_model_role_ == DSparkModelRole::COMMIT  ? "forward_commit" :
                                                                                      "forward";
    py_forward_method_             = py_model_.attr(forward_method);
    const auto py_model_class_name = py::str(py_instance.attr("__class__").attr("__name__")).cast<std::string>();
    const bool is_deepseek_v4_python_model = py_model_class_name == "DeepSeekV4Model"
                                             || py_model_class_name == "DeepSeekV4MtpModel"
                                             || py_model_class_name == "DeepSeekV4DSparkModel";
    if (enable_cuda_graph_ && !params.kv_cache_layer_layout.has_value() && !is_prefill_cuda_graph_mode) {
        RTP_LLM_LOG_WARNING(
            "CUDA graph enabled but kv_cache_layer_layout not available (warmup?), skipping graph capture");
        enable_cuda_graph_ = false;
        if (owns_generation_prefill_cuda_graph_) {
            generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE;
        }
    } else if (enable_cuda_graph_ && is_deepseek_v4_python_model && !params.kv_cache_layer_layout.has_value()) {
        // DeepSeekV4 also refuses to capture prefill graphs during warmup: the
        // real executor captures once the CacheManager exists.
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
        // Default input_hiddens row width for MTP: hc_mult * hidden_size. DSpARK
        // consumes len(target_layer_ids) * hidden_size instead, which only the
        // Python model knows.
        graph_params.input_hidden_size = static_cast<size_t>(params.hidden_size) * static_cast<size_t>(params.hc_mult);
        graph_params.input_embedding_scalar = description_.input_embedding_scalar;
        if (weights_.position_encoding) {
            graph_params.position_encoding = weights_.position_encoding->kernel.cuda();
        }
        if (weights_.token_type_embedding) {
            graph_params.token_type_embedding = weights_.token_type_embedding->kernel.cuda();
        }
        if (dspark_model_role_ != DSparkModelRole::NONE) {
            auto width = py_instance.attr("cuda_graph_input_hidden_size")().cast<int64_t>();
            RTP_LLM_CHECK_WITH_INFO(width > 0, "DSpARK CUDA graph input hidden width must be positive, got %ld", width);
            graph_params.input_hidden_size = static_cast<size_t>(width);
        }
        graph_params.model_data_type            = dtype;
        graph_params.max_context_batch_size     = params.concurrency_config.concurrency_limit;
        graph_params.prefill_capture_seq_lens   = params.hw_kernel_config.prefill_capture_seq_lens;
        graph_params.decode_capture_batch_sizes = params.hw_kernel_config.decode_capture_batch_sizes;
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
        // | DSpARK proposal (decode)  | false                    | DSpARK         | 1        | gen_num_per_cycle       |
        // | DSpARK commit (decode)    | false                    | DSpARK         | 1        | gen_num_per_cycle + 1   |
        // | Draft commit (prefill)    | true                     | != SP_TYPE_NONE| 1        | gen_num_per_cycle + 1   |
        // | Normal Model (decode)     | false                    | SP_TYPE_NONE   | -        | 1 (default)             |
        // | Target Model (verify)     | false                    | != SP_TYPE_NONE| 0        | gen_num_per_cycle + 1   |
        // | Draft Model (decode)      | false                    | != SP_TYPE_NONE| 1        | 1 (default)             |
        // +---------------------------+--------------------------+----------------+----------+-------------------------+
        // clang-format on

        if (dspark_model_role_ == DSparkModelRole::PROPOSE) {
            graph_params.num_tokens_per_bs = params.sp_config.gen_num_per_cycle;
        } else if (dspark_model_role_ == DSparkModelRole::COMMIT) {
            graph_params.num_tokens_per_bs = params.sp_config.gen_num_per_cycle + 1;
        } else if (is_prefill_cuda_graph_mode && params.sp_config.type == SP_TYPE_NONE) {
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
        graph_params.is_target_verify =
            dspark_model_role_ != DSparkModelRole::NONE || use_spec_decoding || is_target_verify_decode;
        graph_params.role =
            graph_params.is_target_verify ? CudaGraphRole::TARGET_VERIFY :
            is_prefill_cuda_graph_mode    ? (params.sp_config.type == SP_TYPE_NONE ? CudaGraphRole::EMBEDDING_PREFILL :
                                                                                     CudaGraphRole::MTP_DRAFT_PREFILL) :
                                            CudaGraphRole::DECODE;
        if (params.sp_config.type != SP_TYPE_NONE) {
            graph_params.sp_steps = params.sp_config.gen_num_per_cycle;
        }

        auto graph_runner =
            std::make_unique<CudaGraphRunner>(graph_params, py_instance, forward_method, params.metrics_reporter);
        {
            void* nccl_comm = cuda_graph::getGraphCaptureTpNcclComm();
            cuda_graph::register_graph_capture_nccl_comm(nccl_comm,
                                                         static_cast<int>(params.parallelism_config.tp_size),
                                                         static_cast<int>(params.parallelism_config.tp_rank));
        }
        auto py_initialize_method = py_instance.attr("initialize");
        try {
            py_init_result = py_initialize_method(init_resources);
            // Python initialization/JIT can take a different amount of time on
            // each EP/TP rank. Synchronize immediately before capture so every
            // rank enters graph-held collectives in the same order.
            syncCudaGraphCaptureRanks(params.parallelism_config, "after_initialize_before_initCapture");
            graph_runner_.reset(CudaGraphRunner::initializeCapture(std::move(graph_runner)));
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_ERROR("Python model initialize failed (cuda_graph branch):\n%s", e.what());
            throw;
        }

        if (owns_generation_prefill_cuda_graph_) {
            bool masked_moe_backend_supported = false;
#if USING_CUDA
            masked_moe_backend_supported =
                supportsGenerationPrefillCudaGraphMaskedMoeBackend(getComputeCapabilityMajor());
#endif
            const bool supported_moe_config = supportsGenerationPrefillCudaGraphMoe(
                description_, params.parallelism_config, description_.moe_runtime_config, masked_moe_backend_supported);
            const bool model_supported = params.sp_config.type == SP_TYPE_NONE
                                         && description_.data_type == DataType::TYPE_BF16
                                         && !description_.attention_conf.use_mla && supported_moe_config
                                         && params.device_resource_config.enable_layer_micro_batch == 0;
            if (!model_supported) {
                generation_prefill_cuda_graph_init_status_ =
                    !supported_moe_config && description_.ffn_conf.moe_configs.has_value() ?
                        GenerationPrefillCudaGraphStatus::MOE_CONFIG_NOT_SUPPORTED :
                        GenerationPrefillCudaGraphStatus::MODEL_NOT_SUPPORTED;
                const char* reason = params.device_resource_config.enable_layer_micro_batch != 0 ?
                                         "layer_micro_batch_enabled" :
                                     !supported_moe_config ? "unsupported_moe_config" :
                                                             "unsupported_model";
                if (!supported_moe_config && description_.ffn_conf.moe_configs.has_value()) {
                    RTP_LLM_LOG_WARNING("generation prefill CUDA graph disabled reason=%s moe_strategy=%s "
                                        "use_all_gather=%d "
                                        "tp_size=%ld ep_size=%ld dp_size=%ld pp_size=%ld",
                                        reason,
                                        description_.moe_runtime_config.moe_strategy.c_str(),
                                        static_cast<int>(description_.moe_runtime_config.use_all_gather),
                                        params.parallelism_config.tp_size,
                                        params.parallelism_config.ep_size,
                                        params.parallelism_config.dp_size,
                                        params.parallelism_config.pp_size);
                } else {
                    RTP_LLM_LOG_WARNING("generation prefill CUDA graph disabled reason=%s", reason);
                }
            } else if (params.cache_manager == nullptr) {
                generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE;
                RTP_LLM_LOG_WARNING("generation prefill CUDA graph disabled reason=kv_cache_unavailable");
            } else if (!supportsGenerationPrefillCudaGraphCacheTopology(
                           params.cache_manager->cacheConfig().groupTypesSnapshot())) {
                generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::MODEL_NOT_SUPPORTED;
                RTP_LLM_LOG_WARNING("generation prefill CUDA graph disabled reason=unsupported_cache_topology; "
                                    "the first version requires "
                                    "exactly one FULL cache group");
            } else {
                GraphParams generation_prefill_cuda_graph_params                = graph_params;
                generation_prefill_cuda_graph_params.role                       = CudaGraphRole::GENERATION_PREFILL;
                generation_prefill_cuda_graph_params.is_prefill_cuda_graph_mode = true;
                generation_prefill_cuda_graph_params.is_target_verify           = false;
                generation_prefill_cuda_graph_params.num_tokens_per_bs          = 1;
                generation_prefill_cuda_graph_params.prefill_capture_seq_lens   = generation_prefill_cuda_graph_buckets;
                generation_prefill_cuda_graph_params.max_context_batch_size =
                    static_cast<size_t>(params.hw_kernel_config.generation_prefill_cuda_graph_max_requests + 1);
                generation_prefill_cuda_graph_params.generation_prefill_cuda_graph_max_requests =
                    params.hw_kernel_config.generation_prefill_cuda_graph_max_requests;
                generation_prefill_cuda_graph_params.generation_prefill_cuda_graph_pad_token_id = 0;
                try {
                    generation_prefill_graph_runner_.reset(CudaGraphRunner::initializeCapture(
                        std::make_unique<CudaGraphRunner>(generation_prefill_cuda_graph_params, py_instance)));
                    generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::NOT_REQUESTED;
                    RTP_LLM_LOG_INFO("generation prefill CUDA graph enabled: max_requests=%d moe_strategy=%s",
                                     params.hw_kernel_config.generation_prefill_cuda_graph_max_requests,
                                     description_.ffn_conf.moe_configs.has_value() ?
                                         description_.moe_runtime_config.moe_strategy.c_str() :
                                         "dense");
                } catch (const DirtyCudaGraphCaptureError& e) {
                    RTP_LLM_LOG_ERROR("generation prefill CUDA graph initialization failed after capture began; eager "
                                      "fallback is unsafe and model initialization will fail: %s",
                                      e.what());
                    // initializeCapture intentionally retains the failed
                    // prefill runner. The same dirty device capture state also
                    // makes teardown of the already-captured decode runner
                    // unsafe (ROCm aborts while draining it). Retain both until
                    // fail-fast process exit; neither owns a KV-cache request.
                    (void)graph_runner_.release();
                    throw;
                } catch (const GenerationPrefillCudaGraphUnsupportedBackendError& e) {
                    generation_prefill_cuda_graph_init_status_ =
                        GenerationPrefillCudaGraphStatus::ATTENTION_BACKEND_UNSUPPORTED;
                    RTP_LLM_LOG_ERROR(
                        "generation prefill CUDA graph initialization failed reason=unsupported_backend error=%s",
                        e.what());
                    throw;
                } catch (const std::exception& e) {
                    generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE;
                    RTP_LLM_LOG_ERROR(
                        "generation prefill CUDA graph initialization failed reason=capture_unavailable error=%s",
                        e.what());
                    throw;
                } catch (...) {
                    generation_prefill_cuda_graph_init_status_ = GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE;
                    RTP_LLM_LOG_ERROR("generation prefill CUDA graph initialization failed reason=unknown_exception");
                    throw;
                }
            }
        }
#else
        RTP_LLM_CHECK_WITH_INFO(false, "CUDA/HIP Graph is only supported on CUDA/ROCm platform");
#endif
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    cache_store_async_writer_ =
        std::make_shared<CacheStoreAsyncWriter>(static_cast<int>(params.parallelism_config.local_rank),
                                                cache_manager_,
                                                model_id_,
                                                params.mtp_cache_config_index);

    if (py::hasattr(py_model_, "has_mtp_hidden_buffer")) {
        has_mtp_hidden_buffer_ = py_model_.attr("has_mtp_hidden_buffer")().cast<bool>();
    }

    // Speculative prefill CP needs every target rank to retain the complete
    // rank-local hidden sequence for the following draft prefill. The normal
    // CP output contains only the selected last-token rows, so reject the
    // configuration while initializing the target model if that hand-off is
    // unavailable.
    if (device_props_.enable_prefill_cp && use_spec_decoding_ && !hasMtpTargetHiddenBuffer()) {
        throw std::runtime_error(
            "speculative prefill CP requires the target model to provide a rank-local MTP hidden buffer");
    }

    if (device_props_.enable_prefill_cp) {
        // Every prefill-shaped forward of a CP-enabled model goes through the
        // standard split/gather path — including the DSpARK draft commit, whose
        // incremental-prefill geometry CP-splits like any prompt while its
        // already-rank-local hidden rows pass through untouched. The fixed-width
        // non-causal propose block never reaches a CP-enabled model: proposals
        // run only on decode roles, where prefill CP is off (colocated CP is
        // rejected at executor construction).
        //
        // MTP hidden buffer is a DeepSeek-specific hand-off. It is written after
        // CP token splitting and already contains rank-local rows, so it must not
        // be split by the context-parallel processor again.
        const bool split_hidden_states = !has_mtp_hidden_buffer_;
        context_parallel_processor_    = ContextParallelProcessorFactory::create(
            ProcessorType::ZIG_ZAG, params.parallelism_config, split_hidden_states);
        RTP_LLM_LOG_INFO("Context parallel processor initialized with ZIG_ZAG strategy, split_hidden_states=%d.",
                         static_cast<int>(split_hidden_states));
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
    (void)construction_guard.release();
}

}  // namespace rtp_llm
