#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
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
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>
#include <vector>

namespace rtp_llm {

bool MtpExecutor::isTpRank0() const {
    return tp_rank_ == 0;
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
                                                         const ResourceContext& resource_context,
                                                         int                    vocab_size) {
    auto fake_stream =
        makeFakeStream(max_new_tokens, 1 + max_new_tokens, model_config, runtime_config, resource_context);

    auto sp_buffer =
        makeFakeSPOutputBuffer(model_config.data_type, model_config.hidden_size, vocab_size, max_new_tokens);

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
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type),
    collect_metrics_stream_(cuda_graph::graphGetStreamFromPool(true)),
    target_verify_prepare_runner_(cuda_graph::graphGetStreamFromPool(true)),
    draft_prefill_prepare_runner_(cuda_graph::graphGetStreamFromPool(true)),
    spec_bookkeeping_runner_(cuda_graph::graphGetStreamFromPool(true)) {
    data_type_        = params.model_config_.data_type;
    hidden_size_      = params.model_config_.hidden_size;
    propose_step_     = propose_params->gen_num_per_circle;
    vocab_size_       = params.model_config_.vocab_size;
    draft_vocab_size_ = propose_params->getEngineInitParams().model_config_.vocab_size;

    RTP_LLM_LOG_INFO("[speculative decoding] vocab_size_ = %d, draft_vocab_size_ = %d", vocab_size_, draft_vocab_size_);

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
                                             params.parallelism_config.world_size,
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
         cache_manager});

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
                                cache_manager});
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

    target_kv_cache_layer_to_group =
        torch::empty({(int64_t)target_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32);
    draft_kv_cache_layer_to_group =
        torch::empty({(int64_t)draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32);

    memcpy(target_kv_cache_layer_to_group.data_ptr<int>(),
           target_cache_layer_layout.layer_to_groups.data(),
           target_cache_layer_layout.layer_to_groups.size() * sizeof(int));
    memcpy(draft_kv_cache_layer_to_group.data_ptr<int>(),
           draft_cache_layer_layout.layer_to_groups.data(),
           draft_cache_layer_layout.layer_to_groups.size() * sizeof(int));

    const auto& draft_weights = propose_params->getEngineInitParams().gpt_weights;
    d2t_map_                  = draft_model_ ? draft_model_->weights_.d2t_map : draft_weights.d2t_map;
    speculative_sampler_.reset(new speculative::SpeculativeSampler(d2t_map_, propose_step_));
    fast_topk_sampler_.reset(new speculative::FastTopKSampler(d2t_map_));

    RTP_LLM_LOG_INFO("[speculative decoding] d2t_map size: %ld", d2t_map_.defined() ? d2t_map_.numel() : 0);
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
                                      MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prefill_step(prefill_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&  tps_collector      = metrics_collector.tps_collector;

    StreamGroups    stream_groups(streams);
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    GptModelOutputs draft_model_output;
    SamplerOutput   draft_sampler_output;

    // placeholder for some tensors
    torch::Tensor                      draft_probs;
    torch::Tensor                      draft_token_ids;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;

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

    // target model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_forward)");
        maybePrintModelInput(model_input, "prefill target model");
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        model_output                        = std::move(model_->forward(model_input));
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
            batch_stream_processor_->updatePrefillPostDraftModelInput(model_input, model_output, sampler_output);
        }
    }

    // draft model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_forward)");
        tpSyncModelInputs(model_input, parallelism_config_);
        maybePrintModelInput(model_input, "prefill post draft model");
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        draft_model_output                  = std::move(draft_model_->forward(model_input));
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        cudaSyncAndCheck();
        model_->releaseBuffers();
        draft_model_->releaseBuffers();
        return absl::OkStatus();
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

        tps_collector.context_tps = stream_groups.modelExecuteTokenSize();
        tps_collector.total_tps   = tps_collector.context_tps;
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(dispatch_output)");
        auto result =
            batch_stream_processor_->dispatchPrefill(stream_groups,
                                                     {std::move(model_output), std::move(sampler_output)},
                                                     {std::move(draft_model_output), std::move(draft_sampler_output)});
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
    torch::Event                       accept_len_ready_event = cuda_graph::makeGraphEvent();

    // Stream-async: events recorded on the main stream at the
    // earliest point each downstream consumer needs. The bookkeeping worker
    // stream cudaStreamWaitEvent's both before reading the corresponding
    // tensors. Recording early (not at the end of decodeStep) shaves the
    // worker's GPU wait by avoiding queue tail (broadcast / draft_forward /
    // draft_sample). They stay null when stream-async is off.
    //   - rejection_event: signals after speculative_sampler_->forward, so
    //     accept_len/accept_tokens are valid for the worker's D2H.
    //   - draft_event: signals after draft_model_sample, so all_probs is
    //     valid for prepareDecodeSpecUpdateInfo's clone.
    std::shared_ptr<torch::Event> rejection_event;
    std::shared_ptr<torch::Event> draft_event;

    const auto& cache_cfg     = cache_manager_->cacheConfig();
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);

    // Stream-async: 1-step bookkeeping buffer. Cap outstanding
    // worker tasks at 1 step by blocking CPU here on the previous step's
    // worker (specUpdate + KV release + GPU swapLinearBlocks). Without this,
    // step N+2 would launch while step N's bookkeeping is still in flight,
    // leading to KV exhaustion (conservative reservation never released) and
    // race-y reads of stream state.
    //
    // Ordering guarantee: this sync runs BEFORE wait_pending_linear_attn_swaps
    // and gatherDecodeModelInput so:
    //  - the swap-done event has been recorded by the worker (no race on
    //    pending_swap_done_event_ shared_ptr read);
    //  - gatherDecodeModelInput's host-side block-id read sees the post-swap
    //    KV block mapping (target verify will then operate on the correct
    //    blocks).
    //
    // No-op when stream-async is off (useStreamAsync() == false;
    // AsyncRunner.sync
    // is also a no-op on the very first decodeStep (task_done_ starts true).
    if (useStreamAsync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(wait_prev_bookkeeping,stream_count=%zu)",
                                      streams.size());
        spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
    }

    // linear-attention KV swap synchronisation. On the
    // fully async path, the previous step's specUpdate runs on the result
    // thread and rewrites KV-block mappings via swapLinearBlocks. This step's
    // target verify forward must wait on that swap to complete before
    // reading KV. The producer side records a torch::Event on the worker
    // stream after dispatchDecode (see MtpExecutor::dispatchDecodeAsync) and
    // hands it to the stream via setPendingSwapDoneEvent. Consumers here
    // make the main stream wait via cudaStreamWaitEvent (= torch::Event::block),
    // then clear the handle. nullptr is the common path (no pending swap;
    // e.g., stream-async is OFF) and incurs only a per-stream pointer load.
    //
    // After the wait_prev_bookkeeping above, the event is already
    // signalled by the time we get here (worker has CPU-synced its stream),
    // so this is effectively a no-op stream dependency. Kept as a separate
    // step so future the fully async path (where wait_prev_bookkeeping moves
    // off the hot path) keeps the verify-side correctness guarantee.
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(wait_pending_linear_attn_swaps,stream_count=%zu)",
                                      streams.size());
        for (auto& stream : streams) {
            auto event_handle = stream->getPendingSwapDoneEvent();
            if (event_handle) {
                auto event = std::static_pointer_cast<torch::Event>(event_handle);
                event->block(cuda_graph::graphGetCurrentStream());
                stream->clearPendingSwapDoneEvent();
            }
        }
    }

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
    size_t batch_size = model_input.input_lengths.size(0);

    // release hold buffers before draft model forward
    draft_model_->releaseBuffers();
    model_->releaseBuffers();

    if (propose_step_ > 1) {
        {
            // NOTE: combo_tokens never used in prepare stage, so it is safe to use shallow copy
            auto model_input_copy                    = model_input;
            model_input_copy.kv_block_stride_bytes   = cache_cfg.kv_block_stride_bytes;
            model_input_copy.kv_scale_stride_bytes   = cache_cfg.kv_scale_stride_bytes;
            model_input_copy.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
            model_input_copy.input_lengths =
                torch::full({(long)batch_size},
                            (long)(propose_step_ + 1),
                            torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            model_input_copy.lm_output_indexes =
                torch::arange(0,
                              batch_size * (propose_step_ + 1),
                              propose_step_ + 1,
                              torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            model_input_copy.prefix_lengths     = model_input.sequence_lengths.clone().pin_memory();
            model_input_copy.last_hidden_states = torch::Tensor();
            model_input_copy.sequence_lengths   = torch::empty({0}, torch::kInt32);
            model_input_copy.is_target_verify   = true;

            target_verify_prepare_runner_.launch([this, model_input_copy = std::move(model_input_copy)]() mutable {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_attention_inputs)");
                model_->prepareAttentionInputs(model_input_copy);
            });
        }

        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
    }

    // Launch the draft-prefill prepareAttentionInputs *before* the
    // target verify forward so it overlaps with the target verify GPU work
    // instead of running serially after it. The runner uses an independent
    // cudaStream + thread (AsyncRunner), and operates on a value-captured copy
    // of model_input, so subsequent in-place mutations of the main-stream
    // model_input (rejection sampling, updateDecodePostDraftModelInput, ...)
    // don't affect what the runner sees. Sync still happens at the same place
    // (just before draft_model_forward) so the consumer order is preserved.
    {
        auto* prefill_model    = sp_prefill_draft_model_ ? sp_prefill_draft_model_.get() : draft_model_.get();
        auto  model_input_copy = model_input;
        model_input_copy.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input_copy.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input_copy.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        draft_prefill_prepare_runner_.launch([prefill_model, model_input_copy = std::move(model_input_copy)]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_draft_prefill_input)");
            prefill_model->prepareAttentionInputs(model_input_copy);
        });
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_model_verify)");
        maybePrintModelInput(model_input, "decode target model");
        model_input.is_target_verify        = true;
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG(
            "[MTP decode] target model verify forward start, input_lengths_size=%ld, prefix_lengths_size=%ld, seq_lengths_size=%ld",
            model_input.input_lengths.size(0),
            model_input.prefix_lengths.size(0),
            model_input.sequence_lengths.size(0));
        target_verify_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
        model_output = std::move(model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
        model_input.is_target_verify = false;
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
            speculative_sampler_output.accept_len =
                torch::full({1},
                            (int64_t)(propose_step_ + 1),
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            speculative_sampler_output.accept_tokens =
                torch::zeros({1, (int64_t)(propose_step_ + 1)},
                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
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
        batch_stream_processor_->updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t);
        if (metrics_reporter_) {
            accept_len_ready_event.record(cuda_graph::graphGetCurrentStream());
        }
    } else {
        model_input.lm_output_indexes =
            torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        model_input.last_hidden_states = model_output.all_hidden_states;
    }

    // Stream-async: record rejection event on the main stream.
    // Earliest valid point — accept_len/accept_tokens have already been
    // produced by speculative_sampler_->forward above (and consumed by
    // updateDecodePostDraftModelInput, which only adds derivative ops). The
    // worker stream waits on this so it can issue D2H of accept_* without
    // blocking on later kernels (broadcast, draft forward/sample).
    if (useStreamAsync()) {
        rejection_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        rejection_event->record(cuda_graph::graphGetCurrentStream());
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
        // only broadcast combo_tokens, last_hidden_states, and lm_output_indexes
        // because these are the only fields updated by updateDecodePostDraftModelInput after rejection sampling
        if (parallelism_config_.tp_size > 1) {
            execBroadcast({{model_input.combo_tokens}, 0});
            execBroadcast({{model_input.last_hidden_states}, 0});
            execBroadcast({{model_input.lm_output_indexes}, 0});
        }
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
    }

    draft_prefill_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_forward)");
        maybePrintModelInput(model_input, "decode post draft model");
        // Use sp_prefill_draft_model_ if CUDA graph is enabled, otherwise use draft_model_
        if (sp_prefill_draft_model_) {
            draft_prefill_model_output = std::move(sp_prefill_draft_model_->forward(model_input));
        } else {
            draft_prefill_model_output = std::move(draft_model_->forward(model_input));
        }
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

    // Stream-async: record draft event on the main stream after
    // draft_model_sample so the bookkeeping worker can wait on it before
    // cloning all_probs / reading token_ids inside prepareDecodeSpecUpdateInfo.
    // Recorded here (not at the end of the function) so the worker doesn't
    // also wait for metrics collection / dispatch_output's per-stream slicing.
    if (useStreamAsync()) {
        draft_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        draft_event->record(cuda_graph::graphGetCurrentStream());
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
        size_t total_accept_len = 0;
        {
            c10::StreamGuard stream_guard(collect_metrics_stream_);
            // launch when data is ready
            accept_len_ready_event.block(collect_metrics_stream_);
            // implicitly stream sync here
            total_accept_len = speculative_sampler_output.accept_len.sum().item<int>();
        }
        executor_collector.generate_batch_size = stream_groups.totalModelBatchSize();
        executor_collector.execute_token_size += total_accept_len;
        executor_collector.max_seq_len = stream_groups.maxSeqLen();

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

        tps_collector.generate_tps = total_accept_len;
        tps_collector.total_tps += total_accept_len;

        sp_engine_collector.total_accepted_token_num = total_accept_len;
        sp_engine_collector.total_stream_num         = stream_groups.size();
        sp_engine_collector.total_propose_token_num  = stream_groups.size() * propose_step_;
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
        absl::Status result;
        if (useStreamAsync()) {
            // Stream-async path: hand off to the bookkeeping
            // worker with the rejection/draft events recorded earlier on
            // the main stream. The worker stream waits via
            // cudaStreamWaitEvent (no main-thread CPU sync); main thread
            // returns immediately.
            result =
                dispatchDecodeAsync(stream_groups,
                                    speculative_sampler_output,
                                    {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)},
                                    std::move(rejection_event),
                                    std::move(draft_event));
        } else {
            result = batch_stream_processor_->dispatchDecode(
                stream_groups,
                speculative_sampler_output,
                {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)});
        }
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
                auto sp_output_buffer =
                    makeFakeSPOutputBuffer(data_type_, hidden_size_, draft_vocab_size_, propose_step_);
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

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.process(stream_size=%zu,mtp_step=%zu)", streams.size(), propose_step_);

    MtpMetricsCollector metrics_collector;

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    // prepare streams
    prepareStreams(streams, prefill_streams, decode_streams);

    // step forward
    int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(prefillStep(prefill_streams, metrics_collector));
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
        metrics_reporter_->report<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
            nullptr, &metrics_collector.tps_collector);
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
                                   torch::Tensor&              draft_token_ids_t) {
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

    auto pre_propose_token_t_raw = model_input.combo_tokens.to(torch::kCUDA, true).clone();

    auto pre_target_token =
        torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    int batch_idx = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        int* propose_tokens                         = stream->getSPOutputBuffer()->tokens.data_ptr<int>();
        pre_target_token.data_ptr<int>()[batch_idx] = propose_tokens[0];
        batch_idx++;
    }

    buffer_holder_.hold(pre_target_token);
    auto pre_target_token_t         = pre_target_token.to(torch::kCUDA, true);
    auto pre_target_token_t_reshape = pre_target_token_t.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_target_token_t_reshape);

    auto pre_propose_token_t_reshape = pre_propose_token_t_raw.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_propose_token_t_reshape);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(loop_iter=%d)", i);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d/%d start, batch_size %zu", i, propose_step_ - 1, batch_size);
        draft_decode_model_output = std::move(draft_model_->forward(model_input));
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

bool MtpExecutor::useStreamAsync() const {
    // Env-gated stream-async. Read once and cache. Default off so
    // production behaviour is unchanged unless RTP_LLM_MTP_STREAM_ASYNC=1
    // is exported on server start. Worth recording: per-instance state isn't
    // needed since the flag is process-wide and the AsyncRunner objects (and
    // the underlying CUDA streams / worker threads) are eagerly constructed
    // in the MtpExecutor ctor whether or not the path is taken.
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_MTP_STREAM_ASYNC");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO("[stream-async] RTP_LLM_MTP_STREAM_ASYNC=%s -> useStreamAsync=%d",
                         env ? env : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

absl::Status MtpExecutor::dispatchDecodeAsync(const StreamGroups&                          stream_groups,
                                              const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                              MergedOutput                                 draft_prefill_output,
                                              std::shared_ptr<torch::Event>                rejection_event,
                                              std::shared_ptr<torch::Event>                draft_event) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output_async)");

    const auto& accept_len_gpu_all    = spec_decode_output.accept_len;
    const auto& accept_tokens_gpu_all = spec_decode_output.accept_tokens;
    // draft_prefill_output.sampler_output.token_ids is the per-stream propose
    // token output of this step's draft model sample. Shape [batch_size,
    // token_stride] on CUDA. Attached per-stream so the next step's prepare
    // can build combo_tokens without waiting for the worker's specUpdate to
    // populate sp_output_buffer->propose_tokens_gpu.
    const auto& propose_tokens_gpu_all = draft_prefill_output.sampler_output.token_ids;

    auto all_streams = stream_groups.allStreams();

    // Snapshot host-side seqLength per stream BEFORE the conservative bump.
    // The bookkeeping worker rolls back to this value before specUpdate so
    // GenerateStream::specUpdate (which increments seqLength by num_new_tokens)
    // lands on the correct base.
    std::vector<int> old_seq_lens;
    old_seq_lens.reserve(all_streams.size());
    for (const auto& stream : all_streams) {
        old_seq_lens.push_back(stream->seqLength());
    }

    // (events are recorded by the caller in decodeStep — see the
    // rejection_event/draft_event recording sites above. Recording in the
    // caller lets us hit the earliest point each tensor becomes valid on
    // the main stream, instead of waiting for the queue tail at this point.)

    // Compute per-stream device-resident state and attach via setter.
    // accept_len has shape [batch_size]; accept_tokens has shape [batch_size,
    // propose_step + 1]. next_seq_len_gpu is computed on GPU as
    // (cur_seq_len_t + accept_len_slice) so the next decode step's prepare
    // can build sequence_lengths without waiting on a CPU value.
    int64_t idx = 0;
    for (auto& stream : all_streams) {
        torch::Tensor accept_len_slice =
            accept_len_gpu_all.defined() ? accept_len_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor accept_tokens_slice =
            accept_tokens_gpu_all.defined() ? accept_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor propose_tokens_slice =
            propose_tokens_gpu_all.defined() ? propose_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();

        torch::Tensor next_seq_len_gpu;
        if (accept_len_slice.defined()) {
            auto cur_seq_len_t = torch::full({1},
                                             static_cast<int64_t>(old_seq_lens[idx]),
                                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            next_seq_len_gpu   = (cur_seq_len_t + accept_len_slice).to(torch::kInt32);
        }
        stream->setSpecDecodeDeviceState(std::move(accept_len_slice),
                                         std::move(accept_tokens_slice),
                                         std::move(next_seq_len_gpu),
                                         std::move(propose_tokens_slice));
        ++idx;
    }

    // Step 3: bump host seqLength to the conservative upper bound. The
    // scheduler / KV manager reads seqLength to size the next step's KV
    // reservation; reserve_step_ in NormalEngine.cc:104 already pre-reserves
    // propose_step + 1 blocks, so this bump just makes the reservation
    // visible. Worker rolls back to old_seq_lens[i] and lets specUpdate set
    // the actual value (old + accept_len).
    //
    // TODO(phase32): the rollback in the worker is not atomic with respect
    // to scheduler reads of seqLength; under the env switch (+) we
    // need a per-stream lock or a separate "conservative seqLength" field
    // that the scheduler reads. For  the path is dead so the race
    // cannot fire.
    idx = 0;
    for (auto& stream : all_streams) {
        stream->setSeqLength(old_seq_lens[idx] + static_cast<int>(propose_step_) + 1);
        ++idx;
    }

    // fork the bookkeeping worker. The worker's stream guard switches
    // the current torch stream to the worker stream, so .cpu() inside
    // prepareDecodeSpecUpdateInfo issues its D2H on the worker stream and
    // only blocks the worker thread.
    auto* processor          = batch_stream_processor_.get();
    auto  spec_decode_copy   = spec_decode_output;
    auto  draft_prefill_copy = std::move(draft_prefill_output);
    auto  stream_groups_copy = stream_groups;
    spec_bookkeeping_runner_.launch([processor,
                                     stream_groups_copy = std::move(stream_groups_copy),
                                     spec_decode_copy   = std::move(spec_decode_copy),
                                     draft_prefill_copy = std::move(draft_prefill_copy),
                                     rejection_event,
                                     draft_event,
                                     old_seq_lens = std::move(old_seq_lens)]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_bookkeeping_worker)");

        // Wait via cudaStreamWaitEvent (NOT cudaEventSynchronize) — the
        // worker stream queues a wait, the worker thread continues to issue
        // more work without blocking. The .cpu() inside
        // prepareDecodeSpecUpdateInfo will eventually CPU-sync the worker
        // stream, which is the only intended block point on this thread.
        //
        //   - rejection_event: produced after speculative_sampler_->forward
        //     on the main stream → guards accept_len/accept_tokens.
        //   - draft_event: produced after draft_model_sample on the main
        //     stream → guards draft_prefill_output.sampler_output.all_probs
        //     (cloned in prepareDecodeSpecUpdateInfo).
        //
        // Both being null is legal (caller skipped recording when stream-
        // async is off; this lambda only runs when useStreamAsync() == true,
        // but defensive null-check keeps the path obviously safe).
        if (rejection_event) {
            rejection_event->block(cuda_graph::graphGetCurrentStream());
        }
        if (draft_event) {
            draft_event->block(cuda_graph::graphGetCurrentStream());
        }

        // Roll back the conservative seqLength bump and clear device-resident
        // handles. specUpdate (called from updateStreams below) will
        // increment seqLength by the actual num_new_tokens (= accept_len).
        auto worker_streams = stream_groups_copy.allStreams();
        int  i              = 0;
        for (auto& stream : worker_streams) {
            stream->setSeqLength(old_seq_lens[i]);
            stream->clearSpecDecodeDeviceState();
            ++i;
        }

        // Reuse the original synchronous dispatch logic. .cpu() inside
        // prepareDecodeSpecUpdateInfo lands on the worker stream (current
        // stream guard is set by AsyncRunner::workerLoop).
        auto status = processor->dispatchDecode(stream_groups_copy, spec_decode_copy, draft_prefill_copy);
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[stream-async] dispatchDecode (worker) failed: %s", status.ToString().c_str());
        }

        // Record a swap-done event for each stream so the next verify can wait
        // via cudaStreamWaitEvent. Recording on all streams is cheap and keeps
        // the consumer path uniform even when a stream did not actually swap.
        for (auto& stream : worker_streams) {
            auto event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
            event->record(cuda_graph::graphGetCurrentStream());
            stream->setPendingSwapDoneEvent(std::static_pointer_cast<void>(event));
        }
    });

    // Main thread returns immediately. The next step can be dispatched while
    // this step's bookkeeping is still in flight on the worker.
    return absl::OkStatus();
}

}  // namespace rtp_llm
