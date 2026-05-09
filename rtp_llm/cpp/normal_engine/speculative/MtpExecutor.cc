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
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#endif
#include "autil/TimeUtility.h"
#include <cstdlib>
#include <memory>
#include <thread>
#include <random>
#include <string>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

bool readEnvFlagOnce(const char* env_name, const char* log_tag, const char* label) {
    const char* env = std::getenv(env_name);
    const bool  on  = (env != nullptr && std::string(env) == "1");
    RTP_LLM_LOG_INFO("[%s] %s=%s -> %s=%d", log_tag, env_name, env ? env : "(unset)", label, static_cast<int>(on));
    return on;
}

void holdSamplerInputHostBuffers(TensorHolder& holder, const SamplerInputs& inputs) {
    holder.hold_host(inputs.token_ids);
    holder.hold_host(inputs.input_lengths);
    holder.hold_host(inputs.sequence_lengths);
    holder.hold_host(inputs.num_beams_in);
    holder.hold_host(inputs.num_beams_out);
    holder.hold_host(inputs.top_k);
    holder.hold_host(inputs.top_p);
    holder.hold_host(inputs.temperature);
    holder.hold_host(inputs.repetition_penalty);
    holder.hold_host(inputs.presence_penalty);
    holder.hold_host(inputs.frequency_penalty);
    holder.hold_host(inputs.no_repeat_ngram_size);
    holder.hold_host(inputs.do_sample);
    holder.hold_host(inputs.finished_mask);
    holder.hold_host(inputs.cum_log_probs);
}

}  // namespace

bool MtpExecutor::isTpRank0() const {
    return tp_rank_ == 0;
}

bool MtpExecutor::useDeviceInput() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DEVICE_INPUT", "mtp-device-input", "enabled");
    }();
    return enabled;
}

bool MtpExecutor::checkDeviceInput() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DEVICE_INPUT_CHECK", "mtp-device-input", "enabled");
    }();
    return enabled;
}

void MtpExecutor::ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag) {
    if (!useDeviceInput()) {
        return;
    }

    auto to_cuda = [this, tag](torch::Tensor& tensor, const char* name) {
        if (!tensor.defined() || tensor.is_cuda()) {
            return;
        }
        if (tensor.numel() == 0) {
            tensor = torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
            return;
        }
        if (!tensor.is_pinned()) {
            RTP_LLM_LOG_WARNING(
                "[mtp-device-input] %s.%s is CPU but not pinned; H2D falls back to blocking copy", tag, name);
            tensor = tensor.to(torch::kCUDA);
            return;
        }
        buffer_holder_.hold_host(tensor);
        tensor = tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };

    to_cuda(model_input.combo_tokens, "combo_tokens");
    to_cuda(model_input.input_lengths, "input_lengths");
    to_cuda(model_input.sequence_lengths, "sequence_lengths");
    to_cuda(model_input.prefix_lengths, "prefix_lengths");
    to_cuda(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    to_cuda(model_input.lm_output_indexes, "lm_output_indexes");
    checkModelInputsOnCuda(model_input, tag);
}

void MtpExecutor::checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const {
    if (!checkDeviceInput()) {
        return;
    }
    auto check = [tag](const torch::Tensor& tensor, const char* name) {
        if (!tensor.defined()) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(),
                                "[mtp-device-input] %s.%s expected CUDA tensor, got device=%s numel=%ld",
                                tag,
                                name,
                                tensor.device().str().c_str(),
                                tensor.numel());
    };
    check(model_input.combo_tokens, "combo_tokens");
    check(model_input.input_lengths, "input_lengths");
    check(model_input.sequence_lengths, "sequence_lengths");
    check(model_input.prefix_lengths, "prefix_lengths");
    check(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    check(model_input.lm_output_indexes, "lm_output_indexes");
    RTP_LLM_LOG_DEBUG("[mtp-device-input] %s metadata tensors are CUDA", tag);
}

MtpExecutor::AcceptLenMetricsSnapshot MtpExecutor::consumePendingAcceptLenMetrics() {
    AcceptLenMetricsSnapshot snapshot;
    if (!metrics_accept_len_sum_cpu_.defined()) {
        return snapshot;
    }

    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(consume_accept_len_metrics)");
    if (metrics_accept_len_ready_event_) {
        // This waits for the previous decode step's tiny D2H only; the current
        // step's accept_len is staged below and reported on the next iteration.
        metrics_accept_len_ready_event_->synchronize();
    }

    snapshot.total_accept_len        = metrics_accept_len_sum_cpu_.item<int64_t>();
    snapshot.total_stream_num        = metrics_accept_len_stream_num_;
    snapshot.total_propose_token_num = metrics_accept_len_propose_token_num_;
    snapshot.valid                   = true;

    metrics_accept_len_sum_gpu_ = torch::Tensor();
    metrics_accept_len_sum_cpu_ = torch::Tensor();
    metrics_accept_len_ready_event_.reset();
    metrics_accept_len_stream_num_        = 0;
    metrics_accept_len_propose_token_num_ = 0;
    return snapshot;
}

void MtpExecutor::stageAcceptLenMetrics(const torch::Tensor& accept_len,
                                        torch::Event&        accept_len_ready_event,
                                        size_t               stream_count) {
    if (!accept_len.defined()) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(stage_accept_len_metrics)");
    metrics_accept_len_stream_num_        = static_cast<int64_t>(stream_count);
    metrics_accept_len_propose_token_num_ = static_cast<int64_t>(stream_count * propose_step_);

    if (!accept_len.is_cuda()) {
        metrics_accept_len_sum_gpu_ = torch::Tensor();
        metrics_accept_len_sum_cpu_ = accept_len.to(torch::kInt64).sum().reshape({1}).pin_memory();
        metrics_accept_len_ready_event_.reset();
        return;
    }

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(collect_metrics_stream_));
    accept_len_ready_event.block(collect_metrics_stream_);
    metrics_accept_len_sum_gpu_ = accept_len.to(torch::kInt64).sum().reshape({1});
    metrics_accept_len_sum_cpu_ =
        torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    metrics_accept_len_sum_cpu_.copy_(metrics_accept_len_sum_gpu_, /*non_blocking=*/true);
    metrics_accept_len_ready_event_ = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    metrics_accept_len_ready_event_->record(collect_metrics_stream_);
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
    const auto& cache_config   = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    is_linear_attention_model_ = cache_config.linear_group_num > 0;
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
        torch::empty({(int64_t)target_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32).pin_memory();
    draft_kv_cache_layer_to_group =
        torch::empty({(int64_t)draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32).pin_memory();

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
    releaseAllModelBuffers();

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
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
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

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;

    StreamGroups   stream_groups(streams);
    GptModelInputs model_input;

    SamplerOutput                         draft_sampler_output;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    // Placeholders shared across draftModelDecode and the post-rejection update.
    torch::Tensor              draft_token_probs_d_t;
    torch::Tensor              hidden_states_d_t;
    torch::Tensor              draft_token_ids_t;
    torch::Tensor              spec_token_ids_t;
    std::vector<torch::Tensor> draft_probs_list;
    torch::Event               accept_len_ready_event = cuda_graph::makeGraphEvent();

    // Stream-async events recorded on the main stream at the earliest point
    // each downstream consumer needs them. The bookkeeping worker stream
    // cudaStreamWaitEvent's both before reading the corresponding tensors.
    // They stay null when stream-async is off. See dispatchDecodeAsync.
    //   - rejection_event: signals after speculative_sampler_->forward, so
    //     accept_len/accept_tokens are valid for the worker's D2H.
    //   - draft_event: signals after draft_model_sample, so all_probs is
    //     valid for prepareDecodeSpecUpdateInfo's clone.
    std::shared_ptr<torch::Event> rejection_event;
    std::shared_ptr<torch::Event> draft_event;

    if (useDeviceInput()) {
        // TensorHolder release point (MtpExecutor device-input path):
        // advances the one-extra-round hold window for decode-step model-input
        // H2D sources. draftModelDecode must not release again in this path
        // because it may still be holding sources queued before TP sync.
        buffer_holder_.release();
    }

    waitPreviousBookkeepingAndKvSwaps(streams);
    rebuildAsyncDeviceStateFromHolder(streams);

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
            ensureModelInputsOnCuda(model_input, "decode.prepare_decode_input");
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        ensureModelInputsOnCuda(model_input, "decode.after_tp_sync");
    }
    size_t batch_size = model_input.input_lengths.size(0);

    // release hold buffers before draft model forward
    releaseAllModelBuffers();

    if (propose_step_ > 1) {
        launchTargetVerifyPrepareAsync(model_input, batch_size);

        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
    }

    // Launch draft-prefill prepare BEFORE target verify forward so it overlaps
    // with target verify GPU work instead of running serially after it. Sync
    // on this prepare happens just before draft_model_forward below.
    launchDraftPrefillPrepareAsync(model_input);

    GptModelOutputs model_output = runTargetVerifyForward(model_input, stream_groups);

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

    SamplerOutput sampler_output;
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(rejection_sampling)");

        if (model_input.is_fake_stream) {
            speculative_sampler_output.accept_len = torch::full(
                {1}, (int64_t)(propose_step_ + 1), torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            speculative_sampler_output.accept_tokens = torch::zeros(
                {1, (int64_t)(propose_step_ + 1)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        } else {
            // target model sample
            CHECK_AND_RETURN_REF(
                sampler_input,
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output));
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
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

    // Stream-async: record rejection event on the main stream so the
    // bookkeeping worker can wait via cudaStreamWaitEvent before D2H of
    // accept_len/accept_tokens. Recording here (not at the function tail)
    // shaves the worker's GPU wait by avoiding queue tail (broadcast,
    // draft_forward, draft_sample). See dispatchDecodeAsync.
    if (useStreamAsync()) {
        rejection_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        rejection_event->record(cuda_graph::graphGetCurrentStream());
    }

    broadcastPostRejectionInputs(model_input);

    draft_prefill_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());

    GptModelOutputs draft_prefill_model_output = runDraftPrefillForward(model_input);

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        releaseAllModelBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    SamplerOutput draft_prefill_sampler_output;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_sample)");
        auto fast_topk_sampler_output          = fast_topk_sampler_->forward(draft_prefill_model_output.logits);
        draft_prefill_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
        draft_prefill_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
    }

    // Stream-async: record draft event after draft_model_sample so the
    // bookkeeping worker can wait before cloning all_probs / reading token_ids
    // inside prepareDecodeSpecUpdateInfo. Recorded here (not at the function
    // tail) so the worker doesn't also wait for metrics collection /
    // dispatch_output's per-stream slicing.
    if (useStreamAsync()) {
        draft_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        draft_event->record(cuda_graph::graphGetCurrentStream());
    }

    if (metrics_reporter_) {
        collectDecodeMetrics(stream_groups, accept_len_ready_event, speculative_sampler_output, metrics_collector);
    }

    return dispatchDecodeOutput(stream_groups,
                                streams,
                                speculative_sampler_output,
                                std::move(draft_prefill_model_output),
                                std::move(draft_prefill_sampler_output),
                                std::move(rejection_event),
                                std::move(draft_event));
}

void MtpExecutor::waitPreviousBookkeepingAndKvSwaps(const std::list<GenerateStreamPtr>& streams) {
    // Cap outstanding stream-async bookkeeping to one step unless
    // RTP_LLM_DROP_BROAD_SYNC explicitly drops the front-loaded CPU sync.
    // Device-state tensors cover stale host reads, per-stream swap events
    // protect linear KV remapping, and AsyncRunner::launch still single-slots
    // bookkeeping workers.
    if (useStreamAsync() && !useDropBroadSync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(wait_prev_bookkeeping,stream_count=%zu)",
                                      streams.size());
        spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
    }

    // Linear-attention KV swap synchronisation. The previous step may rewrite
    // KV-block mappings via swapLinearBlocks while this step is preparing
    // target verify; wait on the producer event before reading KV. This loop
    // remains load-bearing when the broad bookkeeping sync is disabled.
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
}

void MtpExecutor::rebuildAsyncDeviceStateFromHolder(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(clone_sp_tensors,stream_count=%zu)", streams.size());
    for (auto& stream : streams) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (!sp_output_buffer) {
            continue;
        }
        auto const& tensors_holder = sp_output_buffer->tensors_holder;
        if (tensors_holder.empty()) {
            continue;
        }

        auto const& propose_probs_cpu   = tensors_holder[0];
        auto const& propose_hidden_cpu  = tensors_holder[1];
        sp_output_buffer->all_probs     = propose_probs_cpu.to(torch::kCUDA);
        sp_output_buffer->hidden_states = propose_hidden_cpu.to(torch::kCUDA);

        auto propose_tokens_gpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        auto accept_len    = torch::ones({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto accept_tokens = torch::zeros({1, (long)propose_step_ + 1},
                                          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        accept_tokens[0][0]   = sp_output_buffer->tokens[0][0];
        propose_tokens_gpu[0] = sp_output_buffer->tokens[0][1];

        auto next_seq_len = torch::ones({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        next_seq_len[0]   = stream->seqLength();

        stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{
            .epoch                  = 0,
            .accept_len_gpu         = accept_len,
            .accept_tokens_gpu      = accept_tokens,
            .next_seq_len_gpu       = next_seq_len,
            .propose_tokens_gpu     = propose_tokens_gpu,
            .last_hidden_states_gpu = sp_output_buffer->hidden_states,
            .draft_all_probs_gpu    = sp_output_buffer->all_probs,
            .next_real_seq_len      = stream->seqLength(),
        });
    }
}

void MtpExecutor::launchTargetVerifyPrepareAsync(const GptModelInputs& model_input, size_t batch_size) {
    const auto& cache_cfg = cache_manager_->cacheConfig();
    // NOTE: combo_tokens never used in prepare stage, so it is safe to use shallow copy
    auto model_input_copy                    = model_input;
    model_input_copy.kv_block_stride_bytes   = cache_cfg.kv_block_stride_bytes;
    model_input_copy.kv_scale_stride_bytes   = cache_cfg.kv_scale_stride_bytes;
    model_input_copy.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input)");
        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        // Async prepare only needs the token count for metadata/cudagraph sizing.
        // The actual target-verify token ids are produced by draftModelDecode below.
        model_input_copy.combo_tokens =
            torch::empty({static_cast<int64_t>(batch_size * (propose_step_ + 1))}, cuda_i32);
#if USING_CUDA
        const bool can_fuse_target_prepare = model_input.sequence_lengths.defined()
                                             && model_input.sequence_lengths.is_cuda()
                                             && model_input.sequence_lengths.scalar_type() == torch::kInt32
                                             && model_input.sequence_lengths.is_contiguous();
        if (can_fuse_target_prepare) {
            model_input_copy.input_lengths           = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.prefix_lengths          = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.sequence_lengths_plus_1 = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.lm_output_indexes       = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input_fused)");
            invokeMtpTargetVerifyPrepare(model_input.sequence_lengths,
                                         model_input_copy.input_lengths,
                                         model_input_copy.prefix_lengths,
                                         model_input_copy.sequence_lengths_plus_1,
                                         model_input_copy.lm_output_indexes,
                                         static_cast<int32_t>(propose_step_ + 1),
                                         cuda_graph::graphGetCurrentStream().stream());
        } else
#endif
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input_fallback)");
            model_input_copy.input_lengths =
                torch::full({static_cast<int64_t>(batch_size)}, static_cast<int64_t>(propose_step_ + 1), cuda_i32);
            model_input_copy.lm_output_indexes       = torch::arange(0,
                                                               static_cast<int64_t>(batch_size * (propose_step_ + 1)),
                                                               static_cast<int64_t>(propose_step_ + 1),
                                                               cuda_i32);
            model_input_copy.prefix_lengths          = model_input.sequence_lengths.is_cuda() ?
                                                           model_input.sequence_lengths.clone() :
                                                           model_input.sequence_lengths.to(cuda_i32);
            model_input_copy.sequence_lengths_plus_1 = model_input_copy.prefix_lengths + 1;
        }
    }
    model_input_copy.last_hidden_states = torch::Tensor();
    model_input_copy.sequence_lengths =
        torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input_copy.is_target_verify = true;
    ensureModelInputsOnCuda(model_input_copy, "decode.target_prepare");

    // Device-first inputs are produced on the main stream; the async prepare
    // stream must wait before it materializes CPU mirrors.
    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(cuda_graph::graphGetCurrentStream());
    target_verify_prepare_runner_.launch(
        [this, input_ready_event, model_input_copy = std::move(model_input_copy)]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_attention_inputs)");
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_wait_input)");
                input_ready_event->block(cuda_graph::graphGetCurrentStream());
            }
            checkModelInputsOnCuda(model_input_copy, "decode.target_prepare.forwarded");
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_model_inputs)");
                model_->prepareAttentionInputs(model_input_copy);
            }
        });
}

void MtpExecutor::launchDraftPrefillPrepareAsync(const GptModelInputs& model_input) {
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);
    // The runner uses an independent cudaStream + thread (AsyncRunner) and
    // operates on a value-captured copy of model_input, so subsequent in-place
    // mutations of the main-stream model_input (rejection sampling,
    // updateDecodePostDraftModelInput, ...) don't affect what the runner sees.
    auto* prefill_model    = sp_prefill_draft_model_ ? sp_prefill_draft_model_.get() : draft_model_.get();
    auto  model_input_copy = model_input;
    model_input_copy.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
    model_input_copy.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input_copy.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
    ensureModelInputsOnCuda(model_input_copy, "decode.draft_prefill_prepare");
    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(cuda_graph::graphGetCurrentStream());
    draft_prefill_prepare_runner_.launch(
        [this, prefill_model, input_ready_event, model_input_copy = std::move(model_input_copy)]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_draft_prefill_input)");
            input_ready_event->block(cuda_graph::graphGetCurrentStream());
            checkModelInputsOnCuda(model_input_copy, "decode.draft_prefill_prepare.forwarded");
            prefill_model->prepareAttentionInputs(model_input_copy);
        });
}

GptModelOutputs MtpExecutor::runTargetVerifyForward(GptModelInputs& model_input, const StreamGroups& stream_groups) {
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

    // Linear-attention only: page table advances every token. Standard paged
    // attention (MHA/MLA) page table rarely changes within a propose+verify
    // cycle, so the re-gather is skipped there.
    if (is_linear_attention_model_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_kv_cache_kernel_block_id)");
        spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());

        if (tp_rank_ == 0) {
            model_input.kv_cache_kernel_block_id =
                batch_stream_processor_->gatherKvCacheKernelBlockId(stream_groups).value();
        }

        if (parallelism_config_.tp_size > 1) {
            execBroadcast({{model_input.kv_cache_kernel_block_id}, 0});
        }

        // Focused refresh: re-alias attention_inputs_._device(_by_group) and
        // (if cuda-graph) re-mirror held buffers + re-fill FlashInfer Plan
        // against the new tensor. Skips all the unrelated work that a full
        // prepareAttentionInputs would redo (combo_tokens H2D, padding_offset,
        // sequence_lengths copies, ...).
        model_->updateKVCacheKernelBlockId(model_input);

        // Pre-kernel safety check: causal_conv1d_update reads block_map at
        // (sequence_length - 2) // SBP. If that slot is NULL_BLOCK_IDX the
        // kernel hits IMA. Catch it here on host with full diagnostics so we
        // don't have to dig through GPU coredumps.
        // debugCheckLinearBlockMapAtKernelRead(model_input, stream_groups);
    }

    ensureModelInputsOnCuda(model_input, "decode.target_verify_forward");
    GptModelOutputs model_output = model_->forward(model_input);
    RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
    model_input.is_target_verify = false;
    return model_output;
}

void MtpExecutor::debugCheckLinearBlockMapAtKernelRead(const GptModelInputs& model_input,
                                                       const StreamGroups&   stream_groups) const {
    // Diagnose causal_conv1d_update IMA: the kernel reads block_map at offset
    // (sequence_length - 2) // SBP. If that slot is NULL_BLOCK_IDX the read
    // hits OOB. Verify here on host before forward so we capture the offending
    // iter without going through GPU coredumps.
    static const bool always_print = std::getenv("RTP_LLM_DEBUG_TARGET_VERIFY_INPUT") != nullptr;

    if (!model_input.kv_cache_kernel_block_id.defined() || !model_input.sequence_lengths.defined()) {
        return;
    }
    if (cache_manager_ == nullptr) {
        return;
    }
    const int sbp = static_cast<int>(cache_manager_->cacheConfig().seq_size_per_block);
    if (sbp <= 0) {
        return;
    }

    auto seq_len_cpu  = model_input.sequence_lengths.to(torch::kCPU);
    auto block_id_cpu = model_input.kv_cache_kernel_block_id.to(torch::kCPU);
    if (seq_len_cpu.scalar_type() != torch::kInt32 || block_id_cpu.scalar_type() != torch::kInt32) {
        return;
    }
    if (block_id_cpu.dim() != 3) {
        return;
    }

    const auto    all_streams = stream_groups.allStreams();
    const int64_t group_dim   = block_id_cpu.size(0);
    const int64_t batch_dim   = block_id_cpu.size(1);
    const int64_t max_blocks  = block_id_cpu.size(2);
    const int     batch       = static_cast<int>(seq_len_cpu.numel());

    auto dump_row = [&](int64_t g, int64_t b) {
        std::ostringstream oss;
        oss << "[";
        auto row = block_id_cpu.select(0, g).select(0, b);
        for (int64_t i = 0; i < row.size(0); ++i) {
            oss << row[i].item<int32_t>();
            if (i + 1 < row.size(0))
                oss << ",";
        }
        oss << "]";
        return oss.str();
    };

    const auto*        sl         = seq_len_cpu.data_ptr<int32_t>();
    bool               found_null = false;
    std::ostringstream summary;
    summary << "[debug-target-verify] batch=" << batch << " sbp=" << sbp << " group_dim=" << group_dim
            << " batch_dim=" << batch_dim << " max_blocks=" << max_blocks;
    for (int b = 0; b < batch && b < batch_dim; ++b) {
        const int seq_len   = sl[b];
        const int read_off  = (seq_len - 2) / sbp;
        int64_t   stream_id = -1;
        if (b < static_cast<int>(all_streams.size())) {
            auto it = all_streams.begin();
            std::advance(it, b);
            if (*it) {
                stream_id = (*it)->streamId();
            }
        }
        for (int64_t g = 0; g < group_dim; ++g) {
            std::string row_dump;
            if (always_print || (read_off >= 0 && read_off < max_blocks)) {
                row_dump = dump_row(g, b);
            }
            if (read_off < 0 || read_off >= max_blocks) {
                RTP_LLM_LOG_ERROR(
                    "[debug-target-verify] OOB read_off batch=%d stream=%ld group=%ld seq_len=%d read_off=%d max_blocks=%ld row=%s",
                    b,
                    stream_id,
                    g,
                    seq_len,
                    read_off,
                    max_blocks,
                    dump_row(g, b).c_str());
                found_null = true;
                continue;
            }
            const int32_t bid = block_id_cpu.select(0, g).select(0, b).index({read_off}).item<int32_t>();
            if (always_print) {
                summary << "\n  batch=" << b << " stream=" << stream_id << " group=" << g << " seq_len=" << seq_len
                        << " read_off=" << read_off << " bid=" << bid << " row=" << row_dump;
            }
            if (bid == -1) {
                RTP_LLM_LOG_ERROR(
                    "[debug-target-verify] NULL block_id at kernel read batch=%d stream=%ld group=%ld seq_len=%d read_off=%d row=%s",
                    b,
                    stream_id,
                    g,
                    seq_len,
                    read_off,
                    row_dump.c_str());
                found_null = true;
            }
        }
    }
    if (always_print) {
        RTP_LLM_LOG_INFO("%s", summary.str().c_str());
    }
    RTP_LLM_CHECK_WITH_INFO(!found_null,
                            "linear cache NULL at kernel read position — see [debug-target-verify] log lines above");
}

void MtpExecutor::broadcastPostRejectionInputs(GptModelInputs& model_input) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);
    // Only broadcast combo_tokens, last_hidden_states, and lm_output_indexes —
    // these are the only fields updated by updateDecodePostDraftModelInput
    // after rejection sampling.
    //
    // All three tensors are device-resident at this point.
    // - combo_tokens: produced on GPU by toCudaInt32(accept_tokens.reshape(...))
    //   inside updateDecodePostDraftModelInput.
    // - last_hidden_states: aliased from model_output.all_hidden_states (target
    //   verify forward output, never round-trips to host).
    // - lm_output_indexes: produced on GPU by torch::arange + accept_len_d
    //   inside updateDecodePostDraftModelInput.
    // The broadcast therefore stays NCCL-only with no implicit D2H/H2D.
    // Non-root TP ranks fill last_hidden_states from their local target verify
    // output; this NCCL broadcast lets rank 0's rejection-sampled view replace
    // it across ranks.
    if (parallelism_config_.tp_size > 1) {
        execBroadcast({{model_input.combo_tokens}, 0});
        execBroadcast({{model_input.last_hidden_states}, 0});
        execBroadcast({{model_input.lm_output_indexes}, 0});
    }
    model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
}

GptModelOutputs MtpExecutor::runDraftPrefillForward(GptModelInputs& model_input) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_forward)");
    maybePrintModelInput(model_input, "decode post draft model");
    ensureModelInputsOnCuda(model_input, "decode.draft_prefill_forward");
    // Use sp_prefill_draft_model_ if CUDA graph is enabled, otherwise use draft_model_.
    if (sp_prefill_draft_model_) {
        return sp_prefill_draft_model_->forward(model_input);
    }
    return draft_model_->forward(model_input);
}

void MtpExecutor::collectDecodeMetrics(const StreamGroups&                          stream_groups,
                                       torch::Event&                                accept_len_ready_event,
                                       const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                       MtpMetricsCollector&                         metrics_collector) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
    auto& executor_collector  = metrics_collector.executor_collector;
    auto& tps_collector       = metrics_collector.tps_collector;
    auto& sp_engine_collector = metrics_collector.sp_engine_collector;

    const auto accept_len_metrics = consumePendingAcceptLenMetrics();
    stageAcceptLenMetrics(speculative_sampler_output.accept_len, accept_len_ready_event, stream_groups.size());
    const int64_t total_accept_len         = accept_len_metrics.total_accept_len;
    executor_collector.generate_batch_size = stream_groups.totalModelBatchSize();
    executor_collector.execute_token_size += total_accept_len;
    executor_collector.max_seq_len = stream_groups.maxSeqLen();

    executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
    executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
    executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

    tps_collector.generate_tps = total_accept_len;
    tps_collector.total_tps += total_accept_len;

    sp_engine_collector.total_accepted_token_num = total_accept_len;
    sp_engine_collector.total_stream_num         = accept_len_metrics.total_stream_num;
    sp_engine_collector.total_propose_token_num  = accept_len_metrics.total_propose_token_num;
}

absl::Status MtpExecutor::dispatchDecodeOutput(const StreamGroups&                          stream_groups,
                                               const std::list<GenerateStreamPtr>&          streams,
                                               const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                               GptModelOutputs                              draft_prefill_model_output,
                                               SamplerOutput                 draft_prefill_sampler_output,
                                               std::shared_ptr<torch::Event> rejection_event,
                                               std::shared_ptr<torch::Event> draft_event) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
    absl::Status result;
    if (useStreamAsync()) {
        // Stream-async path: hand off to the bookkeeping worker with the
        // rejection/draft events recorded earlier on the main stream. The
        // worker stream waits via cudaStreamWaitEvent (no main-thread CPU
        // sync); main thread returns immediately.
        result = dispatchDecodeAsync(stream_groups,
                                     speculative_sampler_output,
                                     {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)},
                                     std::move(rejection_event),
                                     std::move(draft_event));
    } else {
        MergedOutput draft_prefill_output{std::move(draft_prefill_model_output),
                                          std::move(draft_prefill_sampler_output)};
        result =
            batch_stream_processor_->dispatchDecode(stream_groups, speculative_sampler_output, draft_prefill_output);
        if (result.ok()) {
            publishSyncMtpDeviceState(stream_groups, speculative_sampler_output, draft_prefill_output);
        }
    }
    // clean holder tensors from grpc
    for (auto& stream : streams) {
        stream->getSPOutputBuffer()->tensors_holder.clear();
    }

    return result;
}

void MtpExecutor::releaseAllModelBuffers() {
    // TensorHolder release point (MtpExecutor phase boundary): after the current
    // TP sync/model-input preparation has consumed staged H2D sources, advance
    // the hold window for executor-owned model/sampler staging tensors.
    buffer_holder_.release();
    // PyWrappedModel TensorHolder release points for target/draft model-internal
    // staging buffers.
    model_->releaseBuffers();
    draft_model_->releaseBuffers();
    if (sp_prefill_draft_model_) {
        sp_prefill_draft_model_->releaseBuffers();
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

    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes = mtp_cache_cfg.kv_scale_stride_bytes;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_columns;
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t     batch_size       = model_input.combo_tokens.size(0);
    const auto cuda_i32         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       to_cuda_i32_flat = [&cuda_i32, batch_size](const torch::Tensor& tensor) -> torch::Tensor {
        auto tensor_d = (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) ?
                                  tensor :
                                  tensor.to(cuda_i32, /*non_blocking=*/true);
        tensor_d      = tensor_d.reshape({static_cast<int64_t>(batch_size)});
        return tensor_d.is_contiguous() ? tensor_d : tensor_d.contiguous();
    };
    spec_prefix_lengths =
        model_input.sequence_lengths.defined() ?
            (model_input.sequence_lengths.is_cuda() ?
                 model_input.sequence_lengths.clone() :
                 model_input.sequence_lengths.to(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))) :
            torch::Tensor();

    torch::Tensor pre_propose_token_t_raw;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_propose_token)");
        // Keep the original propose token tensor alive without cloning; later
        // model_input.combo_tokens assignments do not mutate this storage.
        pre_propose_token_t_raw = to_cuda_i32_flat(model_input.combo_tokens);
    }
    const auto all_streams = stream_groups.allStreams();

    torch::Tensor pre_target_token_t;
    // Always prefer the device-state path: MtpAsyncDeviceState.accept_tokens_gpu /
    // accept_len_gpu are published by dispatchDecodeAsync on the main thread
    // BEFORE the bookkeeping worker is launched, so reading them here is race-
    // free regardless of RTP_LLM_DEVICE_INPUT. The legacy host fallback below
    // reads sp_output_buffer->tokens which is written by the worker thread in
    // specUpdate (line 791-792 of GenerateStream.cc) and therefore RACES the
    // previous step's worker when DROP_BROAD_SYNC=1 — producing repeated /
    // gibberish output even at <30 tokens (verified by bisecting narrow syncs).
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_device_gather)");
        std::vector<torch::Tensor> pre_target_slices_gpu;
        pre_target_slices_gpu.reserve(batch_size);
        bool all_device_state = !all_streams.empty();
        for (const auto& stream : all_streams) {
            const auto& accept_tokens = stream->getAcceptTokensGpu();
            const auto& accept_len    = stream->getAcceptLenGpu();
            if (!accept_tokens.defined() || !accept_tokens.is_cuda() || !accept_len.defined()
                || !accept_len.is_cuda()) {
                all_device_state = false;
                break;
            }
            auto idx_t = (accept_len - 1).to(torch::kLong);
            pre_target_slices_gpu.push_back(accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t));
        }
        if (all_device_state && pre_target_slices_gpu.size() == batch_size && !pre_target_slices_gpu.empty()) {
            pre_target_token_t = torch::cat(pre_target_slices_gpu, 0).to(torch::kInt32);
        } else if (all_streams.empty()) {
            // Non-root TP ranks have no GenerateStream objects here; rank 0
            // broadcasts the assembled combo_tokens later, so this placeholder
            // only needs to be device-resident and shape-correct.
            pre_target_token_t =
                torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }
    }

    if (!pre_target_token_t.defined()) {
        // Legacy racy fallback retained only for streams that have not yet
        // published MtpAsyncDeviceState (e.g., PD-disaggregate decode side
        // before rebuildAsyncDeviceStateFromHolder has ever fired). When this
        // path is hit while a previous worker is still in flight (DROP_BROAD_
        // SYNC=1), output corruption is expected.
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_host_fallback)");
        auto pre_target_token =
            torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        int batch_idx = 0;
        for (const auto& stream : all_streams) {
            int* propose_tokens                         = stream->getSPOutputBuffer()->tokens.data_ptr<int>();
            pre_target_token.data_ptr<int>()[batch_idx] = propose_tokens[0];
            batch_idx++;
        }
        buffer_holder_.hold_host(pre_target_token);
        pre_target_token_t = pre_target_token.to(torch::kCUDA, true);
    }
    draft_token_columns.push_back(to_cuda_i32_flat(pre_target_token_t));
    draft_token_columns.push_back(pre_propose_token_t_raw);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(loop_iter=%d)", i);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d/%d start, batch_size %zu", i, propose_step_ - 1, batch_size);
        ensureModelInputsOnCuda(model_input, "draft_decode.loop_forward");
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

        draft_token_ids = to_cuda_i32_flat(draft_token_ids);
        draft_token_columns.push_back(draft_token_ids);
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
        const auto    tokens_per_batch = static_cast<int32_t>(propose_step_ + 1);
        torch::Tensor input_lengths;
#if USING_CUDA
        if (tokens_per_batch <= 8) {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_tokens_metadata_fused)");
            draft_token_ids_t =
                torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(tokens_per_batch)}, cuda_i32);
            input_lengths = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input.lm_output_indexes =
                torch::empty({static_cast<int64_t>(batch_size * tokens_per_batch)}, cuda_i32);
            invokeMtpSpecDecodeTokensMetadataPrepare(draft_token_columns,
                                                     draft_token_ids_t,
                                                     input_lengths,
                                                     model_input.lm_output_indexes,
                                                     tokens_per_batch,
                                                     at::cuda::getCurrentCUDAStream().stream());
        } else {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_cat_tokens)");
            draft_token_ids_t = torch::stack(draft_token_columns, 1).contiguous();
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_metadata_fused)");
                input_lengths = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
                model_input.lm_output_indexes =
                    torch::empty({static_cast<int64_t>(batch_size * tokens_per_batch)}, cuda_i32);
                invokeMtpSpecDecodeMetadataPrepare(input_lengths,
                                                   model_input.lm_output_indexes,
                                                   tokens_per_batch,
                                                   at::cuda::getCurrentCUDAStream().stream());
            }
        }
#else
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_lengths_indexes)");
            draft_token_ids_t = torch::stack(draft_token_columns, 1).contiguous();
            input_lengths     = torch::full({(int64_t)batch_size}, static_cast<int64_t>(propose_step_ + 1), cuda_i32);
            model_input.lm_output_indexes =
                torch::arange(0, static_cast<int64_t>(batch_size * (propose_step_ + 1)), cuda_i32);
        }
#endif

        model_input.input_lengths      = std::move(input_lengths);
        model_input.prefix_lengths     = spec_prefix_lengths;
        model_input.combo_tokens       = draft_token_ids_t.reshape({(int64_t)(batch_size * (propose_step_ + 1))});
        model_input.sequence_lengths   = torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        model_input.last_hidden_states = torch::Tensor();
        ensureModelInputsOnCuda(model_input, "draft_decode.build_spec_decode_input");

        // Since other tp ranks don't have streams, its combo_tokens' first token is not correct.
        // Thus, we need to broadcast the combo_tokens to other tp ranks.
        if (parallelism_config_.tp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_tp_broadcast)");
            execBroadcast({{model_input.combo_tokens}, 0});
        }

        const auto& cache_cfg             = cache_manager_->cacheConfig();
        model_input.kv_block_stride_bytes = cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes = cache_cfg.kv_scale_stride_bytes;
    }
}

bool MtpExecutor::useStreamAsync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_STREAM_ASYNC", "stream-async", "useStreamAsync");
    }();
    return enabled;
}

bool MtpExecutor::useAsyncDeviceState() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "async-device-state", "enabled");
    }();
    return enabled;
}

bool MtpExecutor::useAsyncHostMirror() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_MTP_ASYNC_HOST_MIRROR", "async-host-mirror", "enabled");
    }();
    return enabled;
}

bool MtpExecutor::useDropBroadSync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DROP_BROAD_SYNC", "drop-broad-sync", "enabled");
    }();
    return enabled;
}

bool MtpExecutor::useAsyncStopExtra() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_MTP_ASYNC_STOP_EXTRA", "async-stop-extra", "enabled");
    }();
    return enabled;
}

void MtpExecutor::publishSyncMtpDeviceState(const StreamGroups&                          stream_groups,
                                            const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                            const MergedOutput&                          draft_prefill_output) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(publish_sync_mtp_device_state)");

    auto all_streams = stream_groups.allStreams();
    if (all_streams.empty()) {
        return;
    }

    const auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       to_cuda_i32 = [&cuda_i32](const torch::Tensor& tensor) -> torch::Tensor {
        if (!tensor.defined()) {
            return torch::Tensor();
        }
        return (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) ? tensor : tensor.to(cuda_i32);
    };

    torch::Tensor accept_len_all       = to_cuda_i32(spec_decode_output.accept_len);
    torch::Tensor accept_tokens_all    = to_cuda_i32(spec_decode_output.accept_tokens);
    torch::Tensor propose_tokens_all   = to_cuda_i32(draft_prefill_output.sampler_output.token_ids);
    torch::Tensor draft_all_probs_full = draft_prefill_output.sampler_output.all_probs.defined() ?
                                             (draft_prefill_output.sampler_output.all_probs.is_cuda() ?
                                                  draft_prefill_output.sampler_output.all_probs :
                                                  draft_prefill_output.sampler_output.all_probs.to(torch::kCUDA)) :
                                             torch::Tensor();
    torch::Tensor draft_all_hidden_full =
        draft_prefill_output.model_output.all_hidden_states.defined() ?
            (draft_prefill_output.model_output.all_hidden_states.is_cuda() ?
                 draft_prefill_output.model_output.all_hidden_states :
                 draft_prefill_output.model_output.all_hidden_states.to(torch::kCUDA)) :
            torch::Tensor();

    if (!accept_len_all.defined() || !accept_tokens_all.defined()) {
        RTP_LLM_LOG_WARNING(
            "[mtp-device-state] skip sync publish: accept_len/accept_tokens undefined, stream_count=%zu",
            all_streams.size());
        return;
    }

    int64_t idx              = 0;
    int64_t hidden_token_off = 0;
    int64_t probs_batch_off  = 0;
    for (auto& stream : all_streams) {
        torch::Tensor accept_len_slice    = accept_len_all.narrow(0, idx, 1);
        torch::Tensor accept_tokens_slice = accept_tokens_all.narrow(0, idx, 1);
        torch::Tensor propose_tokens_slice =
            propose_tokens_all.defined() ? propose_tokens_all.narrow(0, idx, 1) : torch::Tensor();

        // Synchronous dispatch has already run GenerateStream::specUpdate, so
        // host seqLength is the authoritative committed length for this stream.
        torch::Tensor next_seq_len_gpu = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32);

        torch::Tensor last_hidden_states_gpu;
        torch::Tensor draft_all_probs_slice_gpu;
        const auto    next_batch_size = stream->nextBatchSize();
        if (propose_step_ > 1 && draft_all_hidden_full.defined()) {
            auto stream_hidden     = draft_all_hidden_full.narrow(0, hidden_token_off, propose_step_ + 1);
            auto idx_t             = (accept_len_slice - 1).to(torch::kLong);
            last_hidden_states_gpu = stream_hidden.index_select(/*dim=*/0, idx_t);
        }
        if (draft_all_probs_full.defined() && next_batch_size > 0) {
            draft_all_probs_slice_gpu = draft_all_probs_full.narrow(0, probs_batch_off, next_batch_size).clone();
        }

        GenerateStream::MtpAsyncDeviceState state;
        state.accept_len_gpu         = std::move(accept_len_slice);
        state.accept_tokens_gpu      = std::move(accept_tokens_slice);
        state.next_seq_len_gpu       = std::move(next_seq_len_gpu);
        state.propose_tokens_gpu     = std::move(propose_tokens_slice);
        state.last_hidden_states_gpu = std::move(last_hidden_states_gpu);
        state.draft_all_probs_gpu    = std::move(draft_all_probs_slice_gpu);
        // Sync dispatch already ran specUpdate; host seqLength is authoritative.
        state.next_real_seq_len = stream->seqLength();
        stream->setMtpAsyncDeviceState(std::move(state));

        hidden_token_off += static_cast<int64_t>(propose_step_ + 1);
        probs_batch_off += next_batch_size;
        ++idx;
    }
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
    // Also publish per-stream draft hidden states (for next step's
    // gatherHiddenStates, propose_step > 1 only) and per-stream draft
    // all_probs (for next step's update*DraftSamplerOutput) here on the main
    // thread. Without this, those readers race the worker's specUpdate writes
    // to sp_output_buffer->hidden_states / all_probs and silently see the
    // PREVIOUS-PREVIOUS step's values when DROP_BROAD_SYNC=1.
    const auto& draft_all_hidden_full = draft_prefill_output.model_output.all_hidden_states;
    const auto& draft_all_probs_full  = draft_prefill_output.sampler_output.all_probs;

    auto all_streams = stream_groups.allStreams();

    // (events are recorded by the caller in decodeStep — see the
    // rejection_event/draft_event recording sites above. Recording in the
    // caller lets us hit the earliest point each tensor becomes valid on
    // the main stream, instead of waiting for the queue tail at this point.)

    // Compute per-stream device-resident state and attach via the
    // setMtpAsyncDeviceState API.
    //
    // Race fix: with DROP_BROAD_SYNC the previous step's worker may still
    // be in flight (specUpdate hasn't advanced host seqLength yet). Reading
    // stream->seqLength() here would give a stale base, producing a wrong
    // next_seq_len_gpu that propagates as wrong sequence_lengths into the
    // next step's model input — causing the model to "forget" recently
    // accepted tokens and regenerate them (duplicate output).
    //
    // Fix: chain next_seq_len_gpu from the PREVIOUS step's device-resident
    // value (getNextSeqLenGpu). That tensor was set by the previous
    // dispatchDecodeAsync on the main thread and is always the correct
    // post-step length regardless of worker progress. On the first decode
    // step (no prior device state) we fall back to host seqLength, which
    // is correct because no worker is running yet.
    int64_t idx              = 0;
    int64_t hidden_token_off = 0;  // offset into draft_all_hidden_full per stream
    int64_t probs_batch_off  = 0;  // offset into draft_all_probs_full per stream
    for (auto& stream : all_streams) {
        torch::Tensor accept_len_slice =
            accept_len_gpu_all.defined() ? accept_len_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor accept_tokens_slice =
            accept_tokens_gpu_all.defined() ? accept_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor propose_tokens_slice =
            propose_tokens_gpu_all.defined() ? propose_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();

        torch::Tensor next_seq_len_gpu;
        if (accept_len_slice.defined()) {
            const auto&   prev_next_seq_len = stream->getNextSeqLenGpu();
            torch::Tensor cur_seq_len_t;
            if (prev_next_seq_len.defined() && prev_next_seq_len.is_cuda()) {
                cur_seq_len_t = prev_next_seq_len;
            } else {
                cur_seq_len_t = torch::full({1},
                                            static_cast<int64_t>(stream->seqLength()),
                                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            }
            next_seq_len_gpu = (cur_seq_len_t + accept_len_slice).to(torch::kInt32);
        }

        // Per-stream draft hidden states (last accepted position) and
        // draft all_probs slice. Both ops stay on the main stream — no D2H,
        // no .item(), no synchronize. Ordered after speculative_sampler->
        // forward via main-stream enqueue order (rejection_event records the
        // same point on the main stream above).
        torch::Tensor last_hidden_states_gpu;
        torch::Tensor draft_all_probs_slice_gpu;
        const auto    next_batch_size = stream->nextBatchSize();
        if (propose_step_ > 1 && draft_all_hidden_full.defined() && accept_len_slice.defined()) {
            // Per-stream rows [hidden_token_off, hidden_token_off + propose+1)
            // contain the draft prefill hidden states. The last accepted
            // position's hidden state is at offset (accept_len - 1).
            auto stream_hidden     = draft_all_hidden_full.narrow(0, hidden_token_off, propose_step_ + 1);
            auto idx_t             = (accept_len_slice - 1).to(torch::kLong);
            last_hidden_states_gpu = stream_hidden.index_select(/*dim=*/0, idx_t);
        }
        if (draft_all_probs_full.defined() && next_batch_size > 0) {
            // Clone to break the alias to draft_prefill_output's storage —
            // that storage is held by the worker lambda's `draft_prefill_copy`
            // and may be released before the next step reads via this view.
            // Match the lifetime semantics of the legacy worker write at
            // MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo:834.
            draft_all_probs_slice_gpu = draft_all_probs_full.narrow(0, probs_batch_off, next_batch_size).clone();
        }

        GenerateStream::MtpAsyncDeviceState state;
        state.accept_len_gpu         = std::move(accept_len_slice);
        state.accept_tokens_gpu      = std::move(accept_tokens_slice);
        state.next_seq_len_gpu       = std::move(next_seq_len_gpu);
        state.propose_tokens_gpu     = std::move(propose_tokens_slice);
        state.last_hidden_states_gpu = std::move(last_hidden_states_gpu);
        state.draft_all_probs_gpu    = std::move(draft_all_probs_slice_gpu);
        // Upper-bound chain so handleRunning's incrKVBlock doesn't race the
        // worker's specUpdate. Fall back to host seqLength only on first publish.
        const auto& prev_state  = stream->getMtpAsyncDeviceState();
        const int   cur_real    = prev_state.next_real_seq_len > 0 ? prev_state.next_real_seq_len : stream->seqLength();
        state.next_real_seq_len = cur_real + static_cast<int>(propose_step_ + 1);
        stream->setMtpAsyncDeviceState(std::move(state));

        hidden_token_off += static_cast<int64_t>(propose_step_ + 1);
        probs_batch_off += next_batch_size;
        ++idx;
    }

    // no host seqLength bump. The previous "bump to
    // base + propose_step + 1 then have the worker roll back" pattern was
    // (a) racy with concurrent scheduler/finishCheck reads of seqLength
    // before the worker's rollback fired, and (b) double-counted KV reserve
    // because StreamCacheResource::incrKVBlock already passes
    // reserve_step_ = propose_step + 1 to the allocator, and the allocator
    // computes blocks needed as (seq_len + reserve_step + ...). Combining
    // a bumped seq_len with a non-zero reserve_step over-reserved by an
    // entire propose_step + 1 worth of blocks. Removing the bump leaves
    // committed seqLength as the single source of truth on the host;
    // reserve_step_ continues to provide the speculative KV budget; the
    // device-resident next_seq_len_gpu provides the accurate post-step
    // length for the next prepare without a host roundtrip.
    //
    // fork the bookkeeping worker. The worker's stream guard switches
    // the current torch stream to the worker stream, so .cpu() inside
    // prepareDecodeSpecUpdateInfo issues its D2H on the worker stream and
    // only blocks the worker thread.
    auto* processor          = batch_stream_processor_.get();
    auto  spec_decode_copy   = spec_decode_output;
    auto  draft_prefill_copy = std::move(draft_prefill_output);
    auto  stream_groups_copy = stream_groups;

    // Inc on the main thread BEFORE handing the task to the runner. This
    // claims a logical hold on each stream's KV resource so that
    // GenerateStateMachine::releaseResource will defer instead of returning
    // blocks to the cache_manager pool while we still race to read them.
    auto streams_for_inc = stream_groups_copy.allStreams();
    for (auto& s : streams_for_inc) {
        s->incPendingAsyncBookkeeping();
    }

    spec_bookkeeping_runner_.launch([processor,
                                     stream_groups_copy = std::move(stream_groups_copy),
                                     spec_decode_copy   = std::move(spec_decode_copy),
                                     draft_prefill_copy = std::move(draft_prefill_copy),
                                     rejection_event,
                                     draft_event]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_bookkeeping_worker)");

        auto worker_streams = stream_groups_copy.allStreams();

        // RAII: even if dispatchDecode throws, every captured stream gets a dec
        // exactly once. Capturing worker_streams by value into the deleter keeps
        // the GenerateStreamPtr refcount alive until after dec runs.
        auto dec_guard = std::shared_ptr<void>(nullptr, [worker_streams](void*) {
            for (auto& s : worker_streams) {
                s->decPendingAsyncBookkeepingAndMaybeRelease();
            }
        });

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

        // Reuse the original synchronous dispatch logic. .cpu() inside
        // prepareDecodeSpecUpdateInfo lands on the worker stream (current
        // stream guard is set by AsyncRunner::workerLoop).
        //
        // Run the original synchronous bookkeeping on the worker. The
        // per-stream MtpAsyncDeviceState was already published on the main
        // thread and remains alive for the next prepare; dispatchDecode only
        // commits host-side bookkeeping / specUpdate.
        auto status = processor->dispatchDecode(stream_groups_copy, spec_decode_copy, draft_prefill_copy);
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[stream-async] dispatchDecode (worker) failed: %s", status.ToString().c_str());
        }

        // Keep MtpAsyncDeviceState alive after specUpdate. Both sync and async
        // decode paths use it as the canonical next-step GPU state; the next
        // dispatchDecodeAsync overwrites it with a newer epoch.

        // Record a swap-done event for each stream so the next verify can wait
        // via cudaStreamWaitEvent. Recording on all streams is cheap and keeps
        // the consumer path uniform even when a stream did not actually swap.
        for (auto& stream : worker_streams) {
            auto event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
            event->record(cuda_graph::graphGetCurrentStream());
            stream->setPendingSwapDoneEvent(std::static_pointer_cast<void>(event));
        }
        // dec_guard destructs here, dec'ing each stream's pending count.
    });

    // Main thread returns immediately. The next step can be dispatched while
    // this step's bookkeeping is still in flight on the worker.
    return absl::OkStatus();
}

}  // namespace rtp_llm
