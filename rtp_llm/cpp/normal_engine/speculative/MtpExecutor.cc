#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
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
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <algorithm>
#include <memory>
#include <random>

namespace rtp_llm {

namespace {

#if USING_CUDA
void recordSpecTensorUseOnCurrentStream(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
                                                      at::cuda::getCurrentCUDAStream(tensor.device().index()));
    }
}
#else
void recordSpecTensorUseOnCurrentStream(const torch::Tensor& tensor) {
    (void)tensor;
}
#endif

void applySpecLogitsAcceptLenCap(const SamplerInputs&                   sampler_input,
                                 const SamplerOutput&                   target_sampler_output,
                                 speculative::SpeculativeSamplerOutput& output,
                                 int64_t                                batch_size,
                                 int64_t                                propose_step) {
    if (!sampler_input.spec_cap_gpu.defined()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(output.accept_len.defined() && output.accept_len.is_cuda(),
                            "spec logits cap requires CUDA accept_len");

    recordSpecTensorUseOnCurrentStream(sampler_input.spec_cap_gpu);
    auto cap_gpu      = sampler_input.spec_cap_gpu.to(output.accept_len.options());
    auto cap_plus_one = cap_gpu + 1;
    output.accept_len = torch::minimum(output.accept_len, cap_plus_one);

    RTP_LLM_CHECK_WITH_INFO(output.accept_tokens.defined() && output.accept_tokens.is_cuda(),
                            "spec logits cap requires CUDA accept_tokens");
    RTP_LLM_CHECK_WITH_INFO(target_sampler_output.token_ids.defined(),
                            "spec logits cap requires target sampler token_ids");
    auto target_token_ids = target_sampler_output.token_ids;
    if (!target_token_ids.is_cuda()) {
        target_token_ids = target_token_ids.to(output.accept_tokens.device(), /*non_blocking=*/true);
    }
    const int64_t token_stride  = target_token_ids.size(1);
    auto          target_tokens = target_token_ids.reshape({batch_size, propose_step + 1, token_stride})
                             .select(2, token_stride - 1)
                             .to(output.accept_tokens.options());
    auto cap_index =
        sampler_input.spec_cap_gpu.to(torch::TensorOptions().device(output.accept_tokens.device()).dtype(torch::kLong));
    auto replacement = target_tokens.gather(1, cap_index.unsqueeze(1));

    auto cols = torch::arange(propose_step + 1,
                              torch::TensorOptions().device(output.accept_tokens.device()).dtype(torch::kLong))
                    .unsqueeze(0)
                    .expand({batch_size, propose_step + 1});
    auto replace_mask = (cap_gpu < propose_step).unsqueeze(1) & (output.accept_len > cap_gpu).unsqueeze(1)
                        & (cols == cap_index.unsqueeze(1));
    output.accept_tokens =
        torch::where(replace_mask, replacement.expand({batch_size, propose_step + 1}), output.accept_tokens);

    output.accept_tokens_cpu = output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
    output.accept_len_cpu    = output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

}  // namespace

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
                                                         const ResourceContext& resource_context) {
    auto fake_stream =
        makeFakeStream(max_new_tokens, 1 + max_new_tokens, model_config, runtime_config, resource_context);

    auto sp_buffer = makeFakeSPOutputBuffer(
        model_config.data_type, model_config.hidden_size, model_config.vocab_size, max_new_tokens);

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
    speculative_sampler_(new speculative::SpeculativeSampler(propose_params->gen_num_per_circle)),
    fast_topk_sampler_(new speculative::FastTopKSampler()),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type) {
    data_type_          = params.model_config_.data_type;
    hidden_size_        = params.model_config_.hidden_size;
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

    size_t total_accept_len = 0;

    MtpSpecLogitsVerifyResult spec_logits_result;

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

    if (propose_step_ > 1) {
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
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

    if (isTpRank0() && !warm_up_ && !model_input.is_fake_stream) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_logits_verify)");
        torch::Tensor draft_tokens;
        if (propose_step_ == 1) {
            if (draft_sampler_output.token_ids.defined() && draft_sampler_output.token_ids.numel() > 0) {
                draft_tokens = draft_sampler_output.token_ids;
            } else {
                draft_tokens = model_input.combo_tokens.reshape(
                    {static_cast<int64_t>(streams.size()), static_cast<int64_t>(propose_step_ + 1)});
            }
        } else {
            draft_tokens = draft_token_ids_t;
        }
        spec_logits_result = buildSpecLogitsVerifyInline(streams, draft_tokens);
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
            const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            speculative_sampler_output.accept_tokens =
                torch::zeros({1, 1}, cuda_i32);
            speculative_sampler_output.accept_len = torch::ones({1}, cuda_i32);
            speculative_sampler_output.accept_tokens_cpu =
                speculative_sampler_output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
            speculative_sampler_output.accept_len_cpu =
                speculative_sampler_output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
            speculative_sampler_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
            cudaSyncAndCheck();
        } else {
            // target model sample
            CHECK_AND_RETURN_REF(
                sampler_input,
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output,
                                                                spec_logits_result));

            sampler_output           = std::move(sampler_->forward(sampler_input));
            sampler_output.all_probs = sampler_output.all_probs.reshape(
                {(int64_t)batch_size, (int64_t)(propose_step_ + 1), (int64_t)vocab_size_});

            speculative_sampler_output =
                speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);
            applySpecLogitsAcceptLenCap(sampler_input,
                                        sampler_output,
                                        speculative_sampler_output,
                                        static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(propose_step_));
        }
        // Commit accepted tokens to each stream via GenerateStream::specUpdate
        // -> updateStatus (the Token Commit Invariant), which advances each
        // attached token-constraint processor exactly once per committed token
        // and emits the opt-in RTP_SP_ACCEPT_TRACE=1 trace. Skipped for fake
        // streams (their accept_len is synthetic).
        // NOTE: here will have cuda device sync before update model input
        batch_stream_processor_->updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t, total_accept_len);
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

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
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
        auto result = batch_stream_processor_->dispatchDecode(
            stream_groups,
            speculative_sampler_output,
            {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)});

        // Per-stream constraint commits + observability happen via
        // GenerateStream::specUpdate → notifyCommit → BaseLogitsProcessor::
        // updateStatus, which fires uniformly for prefill (T0) and decode
        // (verified suffix). MtpExecutor must not dispatch a per-constraint
        // commit hook here — doing so used to double-commit on decode and
        // miss prefill T0 entirely.

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

        // set base properties
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
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
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t batch_size             = model_input.combo_tokens.size(0);
    const auto draft_token_cols   = static_cast<int64_t>(propose_step_ + 1);
    draft_token_ids_t =
        torch::empty({static_cast<int64_t>(batch_size), draft_token_cols},
                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    spec_prefix_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();

    auto int32_cuda_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto pre_propose_token_t_raw =
        model_input.combo_tokens.to(int32_cuda_options).contiguous().clone();

    auto pre_target_token = torch::empty({(int64_t)batch_size}, torch::kInt32);
    int  batch_idx        = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_buf = stream->getSPOutputBuffer();
        RTP_LLM_CHECK(sp_buf != nullptr);
        RTP_LLM_CHECK(sp_buf->tokens.defined() && sp_buf->tokens.is_contiguous()
                      && sp_buf->tokens.device().is_cpu()
                      && sp_buf->tokens.scalar_type() == torch::kInt32
                      && sp_buf->tokens.numel() >= 1);
        const auto* propose_tokens                           = sp_buf->tokens.data_ptr<int32_t>();
        pre_target_token.data_ptr<int32_t>()[batch_idx]      = propose_tokens[0];
        batch_idx++;
    }

    auto pre_target_token_t = pre_target_token.to(int32_cuda_options).contiguous();
    auto pre_target_token_t_reshape = pre_target_token_t.reshape({static_cast<int64_t>(batch_size), 1});
    draft_token_ids_t.narrow(1, 0, 1).copy_(pre_target_token_t_reshape);

    auto pre_propose_token_t_reshape = pre_propose_token_t_raw.reshape({static_cast<int64_t>(batch_size), 1});
    draft_token_ids_t.narrow(1, 1, 1).copy_(pre_propose_token_t_reshape);

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

        draft_token_ids = draft_token_ids.to(int32_cuda_options).contiguous();
        draft_token_ids_t.narrow(1, i + 2, 1)
            .copy_(draft_token_ids.reshape({static_cast<int64_t>(batch_size), 1}));
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
        draft_token_ids_t = draft_token_ids_t.contiguous();

        auto lm_output_indexes =
            torch::empty({(int64_t)(batch_size * (propose_step_ + 1))},
                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        auto input_lengths = torch::empty({(int64_t)batch_size},
                                          torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));

        for (int i = 0; i < batch_size; i++) {
            input_lengths.data_ptr<int32_t>()[i] = propose_step_ + 1;
        }
        for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
            lm_output_indexes.data_ptr<int32_t>()[i] = i;
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

namespace {

void fillAllAllowBitmask(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessor::kBitmaskAllowAll);
    }
}

void bitwiseAndBitmaskInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

}  // namespace

void MtpExecutor::ensureSpecLogitsBuffersFit(size_t total_streams,
                                           int    propose_step,
                                           size_t vocab_size,
                                           size_t bitmask_words) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = B * (P + 1);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    (void)vocab_size;

    auto cpu_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32 = cpu_i32.pinned_memory(true);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    // Pinned: merged_bitmask_cpu_ is the source of mask_gpu.copy_(non_blocking=true)
    // every spec step. Pageable source forces PyTorch to internally pin+copy
    // synchronously, which silently strips the non_blocking property.
    // processor_bitmask_cpu_ is local-scratch (only AND'd into merged_bitmask_cpu_,
    // never directly H2D'd) so pinning it isn't strictly required, but we pin
    // it too for symmetry — the buffer is small and only allocated once.
    if (!processor_bitmask_cpu_.defined() || processor_bitmask_cpu_.numel() < (P + 1) * W) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, pinned_i32);
    }
    if (!merged_bitmask_cpu_.defined() || merged_bitmask_cpu_.numel() < rows * W) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, pinned_i32);
        // Whole buffer starts allow-all so per-call code only has to reset
        // last-call's active rows + fill this-call's active rows.
        std::fill_n(merged_bitmask_cpu_.data_ptr<int32_t>(),
                    merged_bitmask_cpu_.numel(),
                    SpecLogitsProcessor::kBitmaskAllowAll);
        last_active_stream_rows_.clear();
    }
    if (!spec_cap_cpu_.defined() || spec_cap_cpu_.numel() < B) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
#if USING_CUDA
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!merged_bitmask_gpu_.defined() || merged_bitmask_gpu_.numel() < rows * W) {
        merged_bitmask_gpu_ = torch::empty({rows, W}, cuda_i32);
        // Mirror CPU side: GPU buffer is allow-all, and we incrementally
        // overwrite only active rows. The first H2D below seeds it from CPU
        // pinned memory in one shot.
        merged_bitmask_gpu_.copy_(merged_bitmask_cpu_.narrow(0, 0, rows).narrow(1, 0, W));
        last_active_stream_rows_.clear();
    }
    if (!spec_cap_gpu_.defined() || spec_cap_gpu_.numel() < B) {
        spec_cap_gpu_ = torch::empty({B}, cuda_i32);
    }
#else
    (void)rows;
#endif
}

void MtpExecutor::materializeDraftTokensToCpu(size_t               total_streams,
                                              int                  propose_step,
                                              const torch::Tensor& draft_tokens) {
    const int64_t B = static_cast<int64_t>(total_streams);
    const int64_t P = static_cast<int64_t>(propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(draft_tokens.defined(), "MTP spec logits verify requires draft tokens");
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.numel() >= B * P && draft_tokens.numel() % B == 0,
                            "MTP spec logits verify draft token shape mismatch");
    const int64_t draft_cols   = draft_tokens.numel() / B;
    const int64_t draft_offset = draft_cols > P ? 1 : 0;
    RTP_LLM_CHECK_WITH_INFO(draft_cols >= draft_offset + P, "MTP spec logits verify draft token columns mismatch");
    auto draft     = draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
    auto dst       = draft_tokens_cpu_.narrow(0, 0, B).narrow(1, 0, P);
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32);
}

MtpSpecLogitsVerifyResult
MtpExecutor::buildSpecLogitsVerifyInline(const std::list<GenerateStreamPtr>& streams,
                                         const torch::Tensor&                draft_tokens) {
    RTP_LLM_PROFILE_SCOPE("mtp_executor.spec_logits_verify_inline");
    MtpSpecLogitsVerifyResult result;

    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx      = 0;
        size_t                 processor_idx   = 0;
        uint64_t               stream_id       = 0;
        int64_t                base_seq_len    = 0;
        int64_t                base_output_len = 0;
    };
    std::vector<ActiveProcessor> active;
    size_t                     stream_idx = 0;
    for (const auto& stream : streams) {
        size_t processor_idx = 0;
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            auto spec_processor = std::dynamic_pointer_cast<SpecLogitsProcessor>(processor);
            if (spec_processor) {
                active.push_back({spec_processor,
                                  stream_idx,
                                  processor_idx,
                                  static_cast<uint64_t>(stream->streamId()),
                                  static_cast<int64_t>(stream->seqLength()),
                                  static_cast<int64_t>(stream->outputTokenLen())});
            }
            ++processor_idx;
        }
        ++stream_idx;
    }
    if (active.empty()) {
        return result;
    }

    const size_t B    = streams.size();
    const int    P    = static_cast<int>(propose_step_);
    const size_t V    = vocab_size_;
    const size_t W    = SpecLogitsProcessor::bitmaskWordCount(V);
    const size_t rows = B * static_cast<size_t>(P + 1);
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid MTP spec logits verify task");

    ensureSpecLogitsBuffersFit(B, P, V, W);
    // materializeDraftTokensToCpu does an O(B*P) device->host copy that's only
    // useful if at least one processor is eligible. Defer it to the first
    // eligible-hit; if every active processor is filtered out, we skip the
    // copy entirely and short-circuit returning {} below.
    bool draft_tokens_materialized = false;

    // Sparse-row strategy:
    //   * merged_bitmask_cpu_ / _gpu_ are kept allow-all on un-touched rows.
    //   * On each call we reset rows that were narrowed by the *previous* call
    //     back to allow-all (small CPU memset + targeted H2D), then fill this
    //     call's active rows.
    //   * Streams without a SpecLogitsProcessor never pay the fill / H2D cost,
    //     so a mixed batch with a single grammar request no longer blows up
    //     into B*(P+1)*W bits of CPU work and PCIe traffic.
    auto merged = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(W));
    auto* merged_base = merged_bitmask_cpu_.data_ptr<int32_t>();
    const size_t row_words   = (P + 1) * W;
    const size_t buffer_rows = static_cast<size_t>(merged_bitmask_cpu_.size(0)) / static_cast<size_t>(P + 1);

    // Clear any rows narrowed by a previous call so they go back to allow-all
    // for the consumer. Skip rows that fall outside the current B in case the
    // batch shrank — the leftover allow-all state is still correct for them.
    std::vector<size_t> rows_to_reset;
    rows_to_reset.reserve(last_active_stream_rows_.size());
    for (size_t prev : last_active_stream_rows_) {
        if (prev < buffer_rows) {
            std::fill_n(merged_base + prev * row_words, row_words, SpecLogitsProcessor::kBitmaskAllowAll);
            rows_to_reset.push_back(prev);
        }
    }
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    auto proc_mask = processor_bitmask_cpu_.narrow(0, 0, P + 1).narrow(1, 0, static_cast<int64_t>(W));
    std::vector<size_t> this_active_rows;
    this_active_rows.reserve(active.size());
    bool applied_processor = false;
    for (const auto& item : active) {
        if (!item.processor || !item.processor->isSpecVerifyEligible()) {
            continue;
        }
        applied_processor = true;
        if (!draft_tokens_materialized) {
            materializeDraftTokensToCpu(B, P, draft_tokens);
            draft_tokens_materialized = true;
        }

        fillAllAllowBitmask(proc_mask);
        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * P;
        request.propose_step       = P;
        request.bitmask_cpu_out    = proc_mask.data_ptr<int32_t>();
        request.bitmask_size_int32 = W;
        request.vocab_size         = V;
        request.stream_id          = item.stream_id;
        request.base_seq_len       = item.base_seq_len;
        request.base_output_len    = item.base_output_len;

        int cap = item.processor->tryAcceptAndFillBitmask(request);
        cap     = std::max(0, std::min(cap, P));

        auto* merged_row = merged_base + item.stream_idx * row_words;
        bitwiseAndBitmaskInplace(merged_row, proc_mask.data_ptr<int32_t>(), row_words);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
        this_active_rows.push_back(item.stream_idx);
    }

    if (!applied_processor) {
        // Nothing to upload, but make sure any rows narrowed last call get
        // their allow-all state synced back to GPU before we early-return so
        // the next caller sees a clean buffer.
        last_active_stream_rows_.clear();
#if USING_CUDA
        if (!rows_to_reset.empty()) {
            for (size_t row : rows_to_reset) {
                auto cpu_slice = merged_bitmask_cpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
                auto gpu_slice = merged_bitmask_gpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
                gpu_slice.copy_(cpu_slice, /*non_blocking=*/true);
            }
        }
#endif
        return {};
    }

    auto cap_cpu = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B));
#if USING_CUDA
    auto mask_gpu = merged_bitmask_gpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(W));
    auto cap_gpu  = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(B));

    // Upload only the rows that changed — both rows we reset to allow-all and
    // rows we just narrowed. This is a per-stream copy_(non_blocking) instead
    // of one big mask_gpu.copy_(merged), keeping H2D traffic O(active streams)
    // rather than O(B*(P+1)).
    auto upload_row = [&](size_t row) {
        auto cpu_slice = merged_bitmask_cpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
        auto gpu_slice = merged_bitmask_gpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
        gpu_slice.copy_(cpu_slice, /*non_blocking=*/true);
    };
    for (size_t row : rows_to_reset) {
        upload_row(row);
    }
    for (size_t row : this_active_rows) {
        upload_row(row);
    }
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);

    last_active_stream_rows_ = std::move(this_active_rows);

    result.spec_vocab_mask_gpu       = mask_gpu;
    result.spec_cap_gpu              = cap_gpu;
    result.has_active_processor      = true;
    result.spec_vocab_mask_cpu_owner = merged;
    result.spec_cap_cpu_owner        = cap_cpu;
#else
    RTP_LLM_FAIL("MTP spec logits verify requires CUDA for packed bitmask upload");
#endif
    return result;
}

}  // namespace rtp_llm
