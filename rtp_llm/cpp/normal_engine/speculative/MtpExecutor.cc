#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>

namespace rtp_llm {

bool MtpExecutor::isTpRank0() const {
    return tp_rank_ == 0;
}

void MtpExecutor::maybeOverrideLastHiddenWithMtpBuffer(GptModelInputs& model_input,
                                                       ModelBase&       source,
                                                       bool             request_actual_rows) {
    if (!model_input.combo_tokens.defined() || model_input.combo_tokens.numel() == 0) {
        return;
    }
    const auto mtp_hidden_rows = request_actual_rows ? -1 : model_input.combo_tokens.numel();
    auto       pre_hc          = source.getMtpTargetHiddenStates(mtp_hidden_rows);
    if (!pre_hc.defined() || pre_hc.numel() == 0) {
        RTP_LLM_CHECK_WITH_INFO(!request_actual_rows,
                                "CP MTP hidden buffer must contain local rows before draft prefill");
        return;
    }
    model_input.last_hidden_states = pre_hc;
}

void MtpExecutor::maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output, ModelBase& source) {
    if (!model_output.all_hidden_states.defined() || model_output.all_hidden_states.size(0) == 0) {
        return;
    }
    auto pre_hc = source.getMtpTargetHiddenStates(model_output.all_hidden_states.size(0));
    if (!pre_hc.defined() || pre_hc.numel() == 0) {
        return;
    }
    model_output.all_hidden_states = pre_hc;
}


void MtpExecutor::maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const {
    bool force = tp_rank_ == 0 && enable_detail_log_;
    if (force) {
        RTP_LLM_LOG_INFO("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    } else {
        RTP_LLM_LOG_DEBUG("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    }
}

static std::shared_ptr<NormalGenerateStream> makeFakeStream(int                    max_new_tokens,
                                                            size_t                 reserved_blocks,
                                                            const ModelConfig&     model_config,
                                                            const RuntimeConfig&   runtime_config,
                                                            const ResourceContext& resource_context) {
    std::shared_ptr<GenerateInput> fake_input   = std::make_shared<GenerateInput>();
    fake_input->input_ids                       = torch::zeros({1}, torch::kInt32);
    fake_input->generate_config                 = std::make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->generate_config->top_k          = 1;
    fake_input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query                      = true;

    auto fake_stream = std::make_shared<NormalGenerateStream>(
        fake_input, model_config, runtime_config, resource_context, nullptr, max_new_tokens);
    fake_stream->setIsFakeStream(true);
    fake_stream->setMetricsReporter(nullptr);
    fake_stream->fakeInitKVBlock(reserved_blocks);

    return fake_stream;
}

static SpeculativeExecutorStreamOutputPtr
makeFakeSPOutputBuffer(DataType data_type, size_t hidden_size, size_t vocab_size, size_t propose_step) {
    auto sp_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();

    auto fake_hidden_states = torch::zeros(
        {1, (int64_t)hidden_size}, torch::TensorOptions().dtype(dataTypeToTorchType(data_type)).device(torch::kCUDA));
    auto fake_probs =
        torch::zeros({1, (int64_t)vocab_size}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    sp_buffer->propose_step  = propose_step;
    sp_buffer->all_probs     = fake_probs;
    sp_buffer->tokens        = torch::zeros({1, 2}, torch::kInt32);
    sp_buffer->hidden_states = fake_hidden_states;

    return sp_buffer;
}

GenerateStreamPtr MtpExecutor::createMinFakePrefillStream(int                    max_new_tokens,
                                                          const ModelConfig&     model_config,
                                                          const RuntimeConfig&   runtime_config,
                                                          const ResourceContext& resource_context) {
    return makeFakeStream(max_new_tokens, 1, model_config, runtime_config, resource_context);
}

GenerateStreamPtr MtpExecutor::createMinFakeDecodeStream(int                    max_new_tokens,
                                                         const ModelConfig&     model_config,
                                                         const RuntimeConfig&   runtime_config,
                                                         const ResourceContext& resource_context) {
    auto fake_stream =
        makeFakeStream(max_new_tokens, 1 + max_new_tokens, model_config, runtime_config, resource_context);

    // Fake SP buffer's hidden_states stands in for the target's pre-output
    // residual that the draft consumes (DSv4: [T, hc_mult*hidden_size];
    // non-DSv4 keeps hc_mult=1 so the shape is plain [T, hidden_size]).
    auto sp_buffer = makeFakeSPOutputBuffer(model_config.data_type,
                                            model_config.hidden_size * model_config.hc_mult,
                                            model_config.vocab_size,
                                            max_new_tokens);

    auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

    StreamUpdateInfo update_info{new_tokens,
                                 1,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 false};

    fake_stream->update(update_info);
    fake_stream->setSPOutputBuffer(sp_buffer);
    return fake_stream;
}

MtpExecutor::MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type,
                         int32_t                                        kv_cache_group_num,
                         const std::vector<int32_t>&                    kv_cache_layer_to_group,
                         bool                                           warm_up):
    Executor(),
    cache_manager_(cache_manager),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    speculative_sampler_(new speculative::SpeculativeSampler(propose_params->gen_num_per_circle)),
    fast_topk_sampler_(new speculative::FastTopKSampler()),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type) {
    data_type_          = params.model_config_.data_type;
    // Pre-hc residual width for the target → draft hand-off (DSv4: hc*dim;
    // non-DSv4 hc_mult=1 → plain hidden_size). makeFakeSPOutputBuffer in
    // the perf_test path (l. ~800) sizes the fake hidden_states tensor
    // from this; it must match getMtpTargetHiddenStates.
    hidden_size_        = params.model_config_.hidden_size * params.model_config_.hc_mult;
    propose_step_       = propose_params->gen_num_per_circle;
    vocab_size_         = params.model_config_.vocab_size;
    propose_vocab_size_ = propose_params->getEngineInitParams().model_config_.vocab_size;

    enable_detail_log_  = params.profiling_debug_logging_config.enable_detail_log;
    tp_rank_            = params.parallelism_config.tp_rank;
    parallelism_config_ = params.parallelism_config;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, tp_rank_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int         first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        auto        moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        bool        is_gated_activation = params.model_config_.isGatedActivation();
        auto        moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

        expert_balancer_ =
            std::make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                             params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                             params.model_config_.num_layers,
                                             moe_inter_size,
                                             params.model_config_.hidden_size,
                                             params.parallelism_config.ep_rank,
                                             params.parallelism_config.ep_size,
                                             params.py_eplb,
                                             moe_weight_type,
                                             params.model_config_.quant_algo,
                                             metrics_reporter_,
                                             params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{}));

    // Optional per-layer cache buffers from KVCacheManager::allLayerCacheBase().
    std::optional<CacheLayerLayout> kv_cache_layer_layout = std::nullopt;
    if (cache_manager && cache_manager->cacheConfig().groupNums() > 1) {
        kv_cache_layer_layout = cache_manager->allLayerCacheBase();
    }

    auto target_cache_layer_layout = cache_manager->getMainModelCacheLayerLayout();
    auto draft_cache_layer_layout  = cache_manager->getMTPModuleCacheLayerLayout(0);

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(target_cache_layer_layout) : std::nullopt,
         params.model_id,
         params.parallelism_config,
         params.hw_kernel_config,
         params.profiling_debug_logging_config,
         params.runtime_config,
         params.concurrency_config,
         params.sp_config,
         params.device_resource_config,
         mla_ops_type,
         params.model_config_.max_seq_len,
         params.model_config_.hidden_size,
         params.model_config_.attn_config.tokens_per_block,
         params.model_config_.attn_config.kernel_tokens_per_block,
         kv_cache_group_num,
         kv_cache_layer_to_group,
         cache_manager,
         params.model_config_.hc_mult});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(
            model_init_params, params.py_model, false, true, target_cache_layer_layout.layer_to_groups));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.model_config_,
                                                              params.pd_sep_config,
                                                              params.profiling_debug_logging_config,
                                                              cache_config,
                                                              params.sp_config,
                                                              warm_up_));

    LogitsProcessorFactory::init(params.model_config_.ckpt_path, params.sp_config.tree_decode_config);
    cudaProfilerBegin();

    for (auto& mtp_params : *propose_params->mtp_model_params_) {
        auto model_params =
            GptModelInitParams({mtp_params->gpt_weights,
                                Executor::genModelDescription(mtp_params->model_config_,
                                                              mtp_params->parallelism_config,
                                                              mtp_params->eplb_config,
                                                              mtp_params->moe_config),
                                cache_manager ? std::make_optional(draft_cache_layer_layout) : std::nullopt,
                                mtp_params->model_id,
                                mtp_params->parallelism_config,
                                params.hw_kernel_config,
                                params.profiling_debug_logging_config,
                                params.runtime_config,
                                params.concurrency_config,
                                params.sp_config,
                                params.device_resource_config,
                                mla_ops_type,
                                mtp_params->model_config_.max_seq_len,
                                mtp_params->model_config_.hidden_size,
                                mtp_params->model_config_.attn_config.tokens_per_block,
                                mtp_params->model_config_.attn_config.kernel_tokens_per_block,
                                kv_cache_group_num,
                                kv_cache_layer_to_group,
                                cache_manager,
                                mtp_params->model_config_.hc_mult});
        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("[speculative decoding] using py model");
            draft_model_.reset(new PyWrappedModel(
                model_params, params.py_sp_model, false, false, draft_cache_layer_layout.layer_to_groups));
            // Create separate model for speculative prefill with CUDA graph if enabled (from params)
            const bool enable_cuda_graph = params.hw_kernel_config.enable_cuda_graph;
            RTP_LLM_LOG_INFO(
                "[speculative decoding] enable_cuda_graph=%d (set ENABLE_CUDA_GRAPH=1 when starting server to enable sp_prefill_draft_model_)",
                static_cast<int>(enable_cuda_graph));
            if (enable_cuda_graph) {
                RTP_LLM_LOG_INFO(
                    "[speculative decoding] creating separate prefill draft model with CUDA graph support");
                sp_prefill_draft_model_.reset(new PyWrappedModel(
                    model_params, params.py_sp_model, true, false, draft_cache_layer_layout.layer_to_groups));
            }
        }
        break;  // NOTE: only support one mtp model now
    }

    RTP_LLM_CHECK_WITH_INFO(
        target_cache_layer_layout.layer_to_groups.size() == target_cache_layer_layout.layers_to_kv_buffer_ptrs.size(),
        "target_cache_layer_layout layer_to_groups size[%zu] != layers_to_kv_buffer_ptrs size[%zu]",
        target_cache_layer_layout.layer_to_groups.size(),
        target_cache_layer_layout.layers_to_kv_buffer_ptrs.size());
    RTP_LLM_CHECK_WITH_INFO(
        draft_cache_layer_layout.layer_to_groups.size() == draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size(),
        "draft_cache_layer_layout layer_to_groups size[%zu] != layers_to_kv_buffer_ptrs size[%zu]",
        draft_cache_layer_layout.layer_to_groups.size(),
        draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size());

    target_kv_cache_layer_to_group =
        torch::empty({(int64_t)target_cache_layer_layout.layer_to_groups.size()}, torch::kInt32);
    draft_kv_cache_layer_to_group =
        torch::empty({(int64_t)draft_cache_layer_layout.layer_to_groups.size()}, torch::kInt32);

    if (!target_cache_layer_layout.layer_to_groups.empty()) {
        memcpy(target_kv_cache_layer_to_group.data_ptr<int>(),
               target_cache_layer_layout.layer_to_groups.data(),
               target_cache_layer_layout.layer_to_groups.size() * sizeof(int));
    }
    if (!draft_cache_layer_layout.layer_to_groups.empty()) {
        memcpy(draft_kv_cache_layer_to_group.data_ptr<int>(),
               draft_cache_layer_layout.layer_to_groups.data(),
               draft_cache_layer_layout.layer_to_groups.size() * sizeof(int));
    }
}

/*
 * @brief mtp prefill step:
 *
 * +-----------------------------+
 * |     gather model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |    target model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     target model sample     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     update model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model sample      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |  dispatch output to streams |
 * +-----------------------------+
 *
 * @param streams
 * @return absl::Status
 */
absl::Status MtpExecutor::prefillStep(const std::list<GenerateStreamPtr>& streams,
                                      MtpMetricsCollector&                metrics_collector,
                                      int64_t                             schedule_time_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prefill_step(prefill_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&  tps_collector      = metrics_collector.tps_collector;

    StreamGroups    stream_groups(streams);
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    GptModelOutputs draft_model_output;
    SamplerOutput   draft_sampler_output;
    torch::Tensor   draft_last_hidden_states;

    // placeholder for some tensors
    torch::Tensor                      draft_probs;
    torch::Tensor                      draft_token_ids;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;
    int64_t                            model_forward_us = 0;

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(gather_model_input)");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(tp_sync_input)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // release model input before forward
    model_->releaseBuffers();
    draft_model_->releaseBuffers();

    // CP+MTP: PyWrappedModel's CP processor (handleInputs) MUTATES
    // ``model_input.combo_tokens`` and ``model_input.input_lengths`` in
    // place to the rank-local zigzag chunk layout for the target forward.
    // The post-target MTP pipeline (updatePrefillPostDraftModelInput +
    // draft re-CP-slice) needs the FULL/global view, so snapshot both
    // tensors here while they still hold the global sequence and restore
    // on rank 0 before the second tpSync (which then broadcasts the
    // restored full view to every rank for the draft pass).
    const bool    cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
    torch::Tensor saved_combo_tokens;
    torch::Tensor saved_input_lengths;
    if (cp_enabled) {
        // Pinned host buffers preserved via clone().pin_memory() — the
        // upstream tensors come from gather_model_input pinned, and
        // PyWrappedModel's fused d2d copy path asserts the host source
        // is pinned (PyWrappedModel.cc:67). A plain clone() drops the
        // pinned flag and would trip that assert when the draft pass
        // re-enters PyWrappedModel::forward.
        saved_combo_tokens  = model_input.combo_tokens.clone().pin_memory();
        saved_input_lengths = model_input.input_lengths.clone().pin_memory();
    }

    // target model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_forward)");
        maybePrintModelInput(model_input, "prefill target model");
        int64_t start_time_us              = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        model_output                        = std::move(model_->forward(model_input));
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // target model sample
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_sample)");
        if (model_input.is_fake_stream) {
            model_input.last_hidden_states = model_output.all_hidden_states;
        } else {
            CHECK_AND_RETURN_REF(sampler_input,
                                 batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
            sampler_output = std::move(sampler_->forward(sampler_input));
            // Restore the full combo_tokens / input_lengths before the MTP
            // shift logic — under CP both were mutated to rank-local by the
            // target forward's handleInputs and the shift formula assumes a
            // contiguous full sequence (offset += input_length, last token
            // overwrite at offset+input_length-1).
            if (cp_enabled) {
                model_input.combo_tokens  = saved_combo_tokens;
                model_input.input_lengths = saved_input_lengths;
            }
            batch_stream_processor_->updatePrefillPostDraftModelInput(model_input, model_output, sampler_output);
        }
    }

    // draft model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_forward)");
        // Under prefill CP the post-reduce hidden just copied by
        // updatePrefillPostDraftModelInput is not the tensor consumed by
        // DSV4 MTP.  Avoid broadcasting it; after sync each rank reloads the
        // full pre-hc residual from the Python MTP buffer, and the CP input
        // processor slices it with the same zigzag plan as combo_tokens.
        if (cp_enabled) {
            model_input.last_hidden_states = torch::Tensor();
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        maybePrintModelInput(model_input, "prefill post draft model");
        int64_t     start_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        // Source = main (just ran prefill; its pre-hc buffer is current).
        maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_, cp_enabled);
        draft_model_output = std::move(draft_model_->forward(model_input));
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        cudaSyncAndCheck();
        model_->releaseBuffers();
        draft_model_->releaseBuffers();
        return absl::OkStatus();
    }

    if (cp_enabled) {
        draft_last_hidden_states = draft_model_->getMtpLastHiddenStates(stream_groups.totalSamplerBatchSizeOut());
        RTP_LLM_CHECK_WITH_INFO(draft_last_hidden_states.defined() && draft_last_hidden_states.numel() > 0,
                                "CP MTP draft last-hidden buffer must contain per-request rows");
    } else {
        maybeOverrideLastHiddenWithMtpBuffer(draft_model_output, *draft_model_);
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_sample)");
        fast_topk_sampler_output       = fast_topk_sampler_->forward(draft_model_output.logits);
        draft_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
        draft_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(collect_metrics)");
        executor_collector.context_batch_size = stream_groups.totalContextBatchSize();
        executor_collector.execute_token_size = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len        = stream_groups.maxSeqLen();

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;
        executor_collector.model_forward_us += model_forward_us;
        int64_t tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = model_forward_us;
        }

        tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                   stream_groups.contextExecuteTokenSizeWithCache(),
                                   0,
                                   stream_groups.modelExecuteTokenSize(),
                                   tps_execute_time_us);
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(dispatch_output)");
        auto result =
            batch_stream_processor_->dispatchPrefill(stream_groups,
                                                     {std::move(model_output), std::move(sampler_output)},
                                                     {std::move(draft_model_output), std::move(draft_sampler_output)},
                                                     draft_last_hidden_states);
        RTP_LLM_LOG_DEBUG("dispatch done");

        model_->releaseBuffers();
        draft_model_->releaseBuffers();

        return result;
    }
}

/*
+-------------------------------+
|       gather model input      |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |<------------------+
+-------------------------------+                   |
        |                                           |
        v                              +------------------------+
+-------------------------------+      |    update model input  |
|     draft model sample        |      +------------------------+
+-------------------------------+                   |
        |                                           |
        |                                           |
        +---[if steps < propose_step-1] ------------+
        |
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|    target model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|     target model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|      rejection sample         |
+-------------------------------+
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|      draft model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|   dispatch output to streams  |
+-------------------------------+
*/

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                     MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(decode_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector&          executor_collector  = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&           tps_collector       = metrics_collector.tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector& sp_engine_collector = metrics_collector.sp_engine_collector;

    StreamGroups    stream_groups(streams);
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;

    GptModelOutputs                       draft_model_output;
    SamplerOutput                         draft_sampler_output;
    GptModelOutputs                       draft_prefill_model_output;
    SamplerOutput                         draft_prefill_sampler_output;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    // placeholder for some tensors
    torch::Tensor                      draft_token_probs_d_t;
    torch::Tensor                      hidden_states_d_t;
    torch::Tensor                      draft_probs_t;
    torch::Tensor                      draft_token_ids_t;
    torch::Tensor                      spec_token_ids_t;
    std::vector<torch::Tensor>         draft_probs_list;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;
    int64_t                            model_forward_us = 0;

    size_t total_accept_len = 0;

    // clone tensors from grpc
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(clone_sp_tensors,stream_count=%zu)", streams.size());
        for (auto& stream : streams) {
            auto        sp_output_buffer = stream->getSPOutputBuffer();
            auto const& tensors_holder   = sp_output_buffer->tensors_holder;
            if (!tensors_holder.empty()) {
                auto const& propose_probs       = tensors_holder[0];
                auto const& propose_hidden      = tensors_holder[1];
                sp_output_buffer->all_probs     = propose_probs.to(torch::kCUDA).clone();
                sp_output_buffer->hidden_states = propose_hidden.to(torch::kCUDA).clone();
            }
        }
    }

    size_t batch_size = streams.size();
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(gather_model_input)");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherDecodeModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input = std::move(model_input_status.value());
        executor_collector.gather_model_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_input_rank0)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        if (model_input.skip_run) {
            tpSyncModelInputs(model_input, parallelism_config_);
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TODO(yinzhi): consider beam search & lora

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_decode_input_and_tp_sync)");
        if (isTpRank0()) {
            if (propose_step_ == 1) {
                batch_stream_processor_->prepareOneStepSpecDecodeModelInput(stream_groups, model_input);
            } else {
                batch_stream_processor_->prepareDecodeDraftModelInput(stream_groups, model_input);
            }
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
    }

    // release hold buffers before draft model forward
    draft_model_->releaseBuffers();
    model_->releaseBuffers();
    if (sp_prefill_draft_model_) {
        sp_prefill_draft_model_->releaseBuffers();
    }

    if (propose_step_ > 1) {
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t, model_forward_us);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_model_verify)");
        maybePrintModelInput(model_input, "decode target model");
        int64_t start_time_us              = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.is_target_verify        = true;
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG(
            "[MTP decode] target model verify forward start, input_lengths_size=%ld, prefix_lengths_size=%ld, seq_lengths_size=%ld",
            model_input.input_lengths.size(0),
            model_input.prefix_lengths.size(0),
            model_input.sequence_lengths.size(0));
        model_output = std::move(model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
        model_input.is_target_verify = false;
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_draft_sampler_output)");
        if (!model_input.is_fake_stream) {
            if (propose_step_ == 1) {
                batch_stream_processor_->updateOneStepDraftSamplerOutput(
                    stream_groups, draft_sampler_output, draft_token_probs_d_t);
            } else {
                batch_stream_processor_->updateMultiStepDraftSamplerOutput(stream_groups,
                                                                           draft_sampler_output,
                                                                           draft_token_ids_t,
                                                                           spec_token_ids_t,
                                                                           draft_token_probs_d_t,
                                                                           draft_probs_list);
            }
        }
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(rejection_sampling)");

        if (model_input.is_fake_stream) {
            auto accept_tokens                       = torch::zeros({1, 1}, torch::kInt32);
            speculative_sampler_output.accept_len    = {1};
            speculative_sampler_output.accept_tokens = {std::move(accept_tokens)};
            cudaSyncAndCheck();
        } else {
            // target model sample
            CHECK_AND_RETURN_REF(
                sampler_input,
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output));
            sampler_output           = std::move(sampler_->forward(sampler_input));
            sampler_output.all_probs = sampler_output.all_probs.reshape(
                {(int64_t)batch_size, (int64_t)(propose_step_ + 1), (int64_t)vocab_size_});

            // rejection sampling
            speculative_sampler_output = speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);
        }
        // Replace target's post-reduce hidden with pre-hc MTP buffer so that
        // updateDecodePostDraftModelInput gathers the correct residual for the
        // draft model.
        maybeOverrideLastHiddenWithMtpBuffer(model_output, *model_);
        // NOTE: here will have cuda device sync before update model input
        batch_stream_processor_->updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size,
            hidden_states_d_t, total_accept_len);
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
        tpSyncModelInputs(model_input, parallelism_config_);
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_draft_prefill_input)");
        maybePrintModelInput(model_input, "decode post draft model");
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        // last_hidden_states already set to pre-hc by updateDecodePostDraftModelInput
        ModelBase* draft_prefill_source = draft_model_.get();
        if (sp_prefill_draft_model_) {
            draft_prefill_model_output = std::move(sp_prefill_draft_model_->forward(model_input));
            draft_prefill_source       = sp_prefill_draft_model_.get();
        } else {
            draft_prefill_model_output = std::move(draft_model_->forward(model_input));
        }
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        maybeOverrideLastHiddenWithMtpBuffer(draft_prefill_model_output, *draft_prefill_source);
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        cudaSyncAndCheck();
        draft_model_->releaseBuffers();
        model_->releaseBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_sample)");
        fast_topk_sampler_output               = fast_topk_sampler_->forward(draft_prefill_model_output.logits);
        draft_prefill_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
        draft_prefill_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
        executor_collector.generate_batch_size = stream_groups.totalModelBatchSize();
        executor_collector.execute_token_size += total_accept_len;
        executor_collector.max_seq_len = stream_groups.maxSeqLen();
        executor_collector.model_forward_us += model_forward_us;

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

        tps_collector.addTokenSize(0, 0, total_accept_len, total_accept_len, model_forward_us);

        sp_engine_collector.total_accepted_token_num = total_accept_len;
        sp_engine_collector.total_stream_num         = stream_groups.size();
        sp_engine_collector.total_propose_token_num  = stream_groups.size() * propose_step_;
        sp_engine_collector.spec_steps               = propose_step_;
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
        auto result = batch_stream_processor_->dispatchDecode(
            stream_groups,
            speculative_sampler_output,
            {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)});
        // clean holder tensors from grpc
        for (auto& stream : streams) {
            stream->getSPOutputBuffer()->tensors_holder.clear();
        }

        draft_model_->releaseBuffers();
        model_->releaseBuffers();

        return result;
    }
}

void MtpExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams,
                                 std::list<GenerateStreamPtr>&       prefill_streams,
                                 std::list<GenerateStreamPtr>&       decode_streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prepare_streams(stream_size=%zu)", streams.size());

    for (auto& stream : streams) {
        // split streams into prefill and decode
        if (stream->isContextStream()) {
            prefill_streams.push_back(stream);
        } else {
            stream->setScoreLen(propose_step_ + 1);
            if (stream->getSPOutputBuffer() == nullptr && stream->isPerfTest()) {
                auto sp_output_buffer = makeFakeSPOutputBuffer(data_type_, hidden_size_, vocab_size_, propose_step_);
                stream->setSPOutputBuffer(sp_output_buffer);
            }
            decode_streams.push_back(stream);
        }

        // init sp output buffer if not exist
        stream->setReturnAllProbs(true);
        if (stream->getSPOutputBuffer() == nullptr) {
            auto sp_output_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->tokens = torch::zeros({1, 2}, torch::kInt32);

            stream->setSPOutputBuffer(sp_output_buffer);
        }

        // set propose_step
        auto sp_output_buffer          = stream->getSPOutputBuffer();
        sp_output_buffer->propose_step = propose_step_;
    }
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.process(stream_size=%zu,mtp_step=%zu)", streams.size(), propose_step_);

    const int64_t process_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    if (schedule_time_us <= 0) {
        schedule_time_us = process_start_time_us;
    }
    MtpMetricsCollector metrics_collector;
    auto tps_active_guard =
        tps_reporter_.makeActiveGuard(metrics_reporter_ && isTpRank0() && !warm_up_ && !streams.empty());

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    // prepare streams
    prepareStreams(streams, prefill_streams, decode_streams);

    // step forward
    int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(prefillStep(prefill_streams, metrics_collector, schedule_time_us));
    }

    if (role_type_ == RoleType::DECODE || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(decodeStep(decode_streams, metrics_collector));
    }

    metrics_collector.sp_engine_collector.step_latency_us =
        autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;

    // report metrics
    if (isTpRank0() && metrics_reporter_ && metrics_collector.not_skip) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.process(report_metrics)");
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(
            nullptr, &metrics_collector.executor_collector);
        tps_reporter_.report(&metrics_collector.tps_collector);
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(
            nullptr, &metrics_collector.sp_engine_collector);
    }

    return absl::OkStatus();
}

bool MtpExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

void MtpExecutor::draftModelDecode(GptModelInputs&             model_input,
                                   const StreamGroups&         stream_groups,
                                   std::vector<torch::Tensor>& draft_probs_list,
                                   torch::Tensor&              draft_token_ids_t,
                                   int64_t&                    model_forward_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(batch_size=%zu)", model_input.combo_tokens.size(0));

    // clear host buffers holder
    buffer_holder_.release();

    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes = mtp_cache_cfg.kv_scale_stride_bytes;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_ids_list;
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t batch_size   = model_input.combo_tokens.size(0);
    spec_prefix_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();

    auto pre_propose_token_t_raw = model_input.combo_tokens.to(torch::kCUDA).clone();

    auto pre_target_token = torch::empty({(int64_t)batch_size}, torch::kInt32);
    int  batch_idx        = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        int* propose_tokens                         = stream->getSPOutputBuffer()->tokens.data_ptr<int>();
        pre_target_token.data_ptr<int>()[batch_idx] = propose_tokens[0];
        batch_idx++;
    }

    auto pre_target_token_t         = pre_target_token.to(torch::kCUDA);
    auto pre_target_token_t_reshape = pre_target_token_t.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_target_token_t_reshape);

    auto pre_propose_token_t_reshape = pre_propose_token_t_raw.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_propose_token_t_reshape);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(loop_iter=%d)", i);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d/%d start, batch_size %zu", i, propose_step_ - 1, batch_size);
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        draft_decode_model_output = std::move(draft_model_->forward(model_input));
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        maybeOverrideLastHiddenWithMtpBuffer(draft_decode_model_output, *draft_model_);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d forward done", i);

        // sample
        auto fast_topk_sampler_output = fast_topk_sampler_->forward(draft_decode_model_output.logits, 1);
        auto draft_probs              = fast_topk_sampler_output.all_probs;
        auto draft_probs_reshape      = draft_probs.reshape({(int)batch_size, 1, -1});
        auto draft_token_ids          = fast_topk_sampler_output.token_ids;

        if (model_input.is_fake_stream) {
            draft_token_ids.zero_();
            draft_decode_model_output.all_hidden_states.zero_();
        }

        draft_token_ids = draft_token_ids.to(torch::kInt32).to(torch::kCUDA);
        draft_token_ids_list.push_back(draft_token_ids);
        draft_probs_list.push_back(draft_probs_reshape);

        // update model input
        if (i != propose_step_ - 2) {
            batch_stream_processor_->updateDecodeDraftModelInput(
                model_input, draft_decode_model_output, draft_token_ids);
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_decode_input)");
        // prepare spec decode input
        draft_token_ids_t =
            torch::cat(draft_token_ids_list, 1).reshape({(int)batch_size, (int)(propose_step_ + 1)}).contiguous();

        auto lm_output_indexes =
            torch::empty({(int64_t)(batch_size * (propose_step_ + 1))},
                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        auto input_lengths = torch::empty({(int64_t)batch_size},
                                          torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));

        for (int i = 0; i < batch_size; i++) {
            input_lengths.data_ptr<int>()[i] = propose_step_ + 1;
        }
        for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
            lm_output_indexes.data_ptr<int>()[i] = i;
        }

        model_input.input_lengths     = std::move(input_lengths);
        model_input.lm_output_indexes = std::move(lm_output_indexes);
        model_input.prefix_lengths    = spec_prefix_lengths;
        model_input.combo_tokens      = draft_token_ids_t.reshape({(int64_t)(batch_size * (propose_step_ + 1))});
        model_input.sequence_lengths =
            torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        model_input.last_hidden_states = torch::Tensor();

        // Since other tp ranks don't have streams, its combo_tokens' first token is not correct.
        // Thus, we need to broadcast the combo_tokens to other tp ranks.
        if (parallelism_config_.tp_size > 1) {
            execBroadcast({{model_input.combo_tokens}, 0});
        }

        const auto& cache_cfg             = cache_manager_->cacheConfig();
        model_input.kv_block_stride_bytes = cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes = cache_cfg.kv_scale_stride_bytes;
    }
}

}  // namespace rtp_llm
