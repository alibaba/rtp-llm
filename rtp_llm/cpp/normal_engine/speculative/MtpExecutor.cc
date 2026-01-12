#include "rtp_llm/cpp/models/MTPModel.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
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
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>

namespace rtp_llm {

bool MtpExecutor::isTpRank0() const {
    return device_->getDeviceProperties().tp_rank == 0;
}

void MtpExecutor::maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const {
    bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
    if (force) {
        RTP_LLM_LOG_INFO("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    } else {
        RTP_LLM_LOG_DEBUG("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    }
}

static std::shared_ptr<NormalGenerateStream> makeFakeStream(int                    max_new_tokens,
                                                            const ModelConfig&     model_config,
                                                            const RuntimeConfig&   runtime_config,
                                                            const ResourceContext& resource_context,
                                                            DeviceBase*            device) {
    std::shared_ptr<GenerateInput> fake_input = std::make_shared<GenerateInput>();
    fake_input->input_ids =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1}, rtp_llm::AllocationType::HOST});
    device->bufMemset(*fake_input->input_ids, 0);
    fake_input->generate_config                 = std::make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->generate_config->top_k          = 1;
    fake_input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query                      = true;

    auto fake_stream = std::make_shared<NormalGenerateStream>(
        fake_input, model_config, runtime_config, resource_context, nullptr, max_new_tokens);
    fake_stream->setIsFakeStream(true);
    fake_stream->setMetricsReporter(nullptr);
    fake_stream->fakeInitKVBlock();

    return fake_stream;
}

static SpeculativeExecutorStreamOutputPtr makeFakeSPOutputBuffer(
    DataType data_type, size_t hidden_size, size_t vocab_size, size_t propose_step, DeviceBase* device) {
    auto sp_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();

    auto fake_hidden_states = device->allocateBuffer({data_type, {1, hidden_size}, AllocationType::DEVICE});
    auto fake_probs         = device->allocateBuffer({DataType::TYPE_FP32, {1, vocab_size}, AllocationType::DEVICE});
    auto fake_tokens = device->allocateBuffer({DataType::TYPE_INT32, {1, 2}, AllocationType::HOST}, {"spec_tokens"});

    device->bufMemset(*fake_hidden_states, 0);
    device->bufMemset(*fake_probs, 0);
    device->bufMemset(*fake_tokens, 0);

    sp_buffer->propose_step  = propose_step;
    sp_buffer->all_probs     = fake_probs;
    sp_buffer->tokens        = fake_tokens;
    sp_buffer->hidden_states = fake_hidden_states;

    return sp_buffer;
}

GenerateStreamPtr MtpExecutor::createMinFakePrefillStream(int                    max_new_tokens,
                                                          const ModelConfig&     model_config,
                                                          const RuntimeConfig&   runtime_config,
                                                          const ResourceContext& resource_context,
                                                          DeviceBase*            device) {
    return makeFakeStream(max_new_tokens, model_config, runtime_config, resource_context, device);
}

GenerateStreamPtr MtpExecutor::createMinFakeDecodeStream(int                    max_new_tokens,
                                                         const ModelConfig&     model_config,
                                                         const RuntimeConfig&   runtime_config,
                                                         const ResourceContext& resource_context,
                                                         DeviceBase*            device) {
    auto fake_stream = makeFakeStream(max_new_tokens, model_config, runtime_config, resource_context, device);

    auto sp_buffer = makeFakeSPOutputBuffer(
        model_config.data_type, model_config.hidden_size, model_config.vocab_size, max_new_tokens, device);

    BufferPtr new_tokens =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1, 1}, rtp_llm::AllocationType::HOST});
    *new_tokens->dataWithOffset<int32_t>(0) = 0;

    StreamUpdateInfo update_info{
        new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

    fake_stream->update(update_info);
    fake_stream->setSPOutputBuffer(sp_buffer);
    return fake_stream;
}

MtpExecutor::MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         rtp_llm::DeviceBase*                           device,
                         const std::shared_ptr<lora::LoraManager>&      lora_manager,
                         bool                                           warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    metrics_reporter_(params.metrics_reporter),
    speculative_sampler_(new speculative::SpeculativeSampler(device, propose_params->gen_num_per_circle)),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type) {
    data_type_          = params.model_config_.data_type;
    hidden_size_        = params.model_config_.hidden_size;
    propose_step_       = propose_params->gen_num_per_circle;
    vocab_size_         = params.model_config_.vocab_size;
    propose_vocab_size_ = propose_params->getEngineInitParams().model_config_.vocab_size;

    enable_detail_log_ = params.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = params.model_config_.moe_layer_index.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();
        bool is_gated_activation = params.model_config_.isGatedActivation();
        auto moe_inter_size =
            is_gated_activation ?
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1] / 2 :
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1];

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
                                             device_,
                                             params.model_config_.quant_algo,
                                             metrics_reporter_,
                                             params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{device_}));

    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(cache_manager->kvCacheBuffer()) : std::nullopt,
         params.model_id});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
        model_.reset(new NativeDeviceGraphModel(model_init_params));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.model_config_,
                                                              params.pd_sep_config,
                                                              params.profiling_debug_logging_config,
                                                              cache_config,
                                                              params.sp_config,
                                                              warm_up_));

    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(params.model_config_.ckpt_path,
                                                                    params.sp_config.tree_decode_config);

    size_t index = 0;
    for (auto& mtp_params : *propose_params->mtp_model_params_) {
        auto model_params = GptModelInitParams(
            {device_,
             mtp_params->gpt_weights,
             Executor::genModelDescription(mtp_params->model_config_,
                                           mtp_params->parallelism_config,
                                           mtp_params->eplb_config,
                                           mtp_params->moe_config),
             cache_manager ? std::make_optional(cache_manager->getMTPModuleKVCacheBuffer(static_cast<int>(index))) :
                             std::nullopt,
             mtp_params->model_id});
        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("[speculative decoding] using py model");
            draft_model_.reset(new PyWrappedModel(model_params, params.py_sp_model));
        } else {
            RTP_LLM_LOG_INFO("[speculative decoding] legacy c++ gpt model");
            draft_model_.reset(new MTPModel(model_params));
        }
        break;  // NOTE: only support one mtp model now
    }

    device_->profileStart();
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
    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&  tps_collector      = metrics_collector.tps_collector;

    StreamGroups    stream_groups(streams);
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    GptModelOutputs draft_model_output;
    SamplerOutput   draft_sampler_output;

    // placeholder for some tensors
    torch::Tensor draft_probs;
    torch::Tensor draft_token_ids;

    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TODO(yinzhi): consider beam search & lora

    // release model input before forward
    model_->releaseBuffers();
    draft_model_->releaseBuffers();

    // target model prefill
    {
        maybePrintModelInput(model_input, "prefill target model");
        model_output = std::move(model_->forward(model_input));
    }

    // eplb
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // target model sample
    if (isTpRank0()) {
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
        tpSyncModelInputs(model_input, device_);
        maybePrintModelInput(model_input, "prefill post draft model");
        const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;
        draft_model_output = std::move(draft_model_->forward(model_input));
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        device_->syncAndCheck();
        model_->releaseBuffers();
        draft_model_->releaseBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    draftModelSample(draft_model_output.logits, draft_sampler_output, draft_probs, draft_token_ids);

    // collect metrics
    if (metrics_reporter_) {
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
|    target model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|     target model sample       |
+-------------------------------+
*/

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                     MtpMetricsCollector&                metrics_collector) {
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
    torch::Tensor              draft_token_probs_d_t;
    torch::Tensor              hidden_states_d_t;
    torch::Tensor              draft_probs_t;
    torch::Tensor              draft_token_ids_t;
    torch::Tensor              spec_token_ids_t;
    std::vector<torch::Tensor> draft_probs_list;

    size_t total_accept_len = 0;

    // clone tensors from grpc
    for (auto& stream : streams) {
        auto        sp_output_buffer = stream->getSPOutputBuffer();
        auto const& tensors_holder   = sp_output_buffer->tensors_holder;
        if (!tensors_holder.empty()) {
            auto const& propose_probs   = tensors_holder[0];
            auto const& propose_hidden  = tensors_holder[1];
            sp_output_buffer->all_probs = device_->clone({*torchTensor2Buffer(propose_probs), AllocationType::DEVICE});
            sp_output_buffer->hidden_states =
                device_->clone({*torchTensor2Buffer(propose_hidden), AllocationType::DEVICE});
        }
    }

    size_t batch_size = streams.size();
    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherDecodeModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input = std::move(model_input_status.value());
        executor_collector.gather_model_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        if (model_input.skip_run) {
            tpSyncModelInputs(model_input, device_);
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TODO(yinzhi): consider beam search & lora

    if (isTpRank0()) {
        if (propose_step_ == 1) {
            batch_stream_processor_->prepareOneStepSpecDecodeModelInput(stream_groups, model_input);
        } else {
            batch_stream_processor_->prepareDecodeDraftModelInput(stream_groups, model_input);
        }
    }

    tpSyncModelInputs(model_input, device_);
    if (model_input.skip_run) {
        return absl::OkStatus();
    }

    // release hold buffers before draft model forward
    draft_model_->releaseBuffers();
    model_->releaseBuffers();

    if (propose_step_ > 1) {
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t);
    }

    maybePrintModelInput(model_input, "decode target model");
    model_output = std::move(model_->forward(model_input));

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
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
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        if (model_input.is_fake_stream) {
            BufferPtr accept_tokens = device_->allocateBuffer({DataType::TYPE_INT32, {1, 1}, AllocationType::HOST});
            *accept_tokens->dataWithOffset<int32_t>(0) = 0;
            speculative_sampler_output.accept_len      = {1};
            speculative_sampler_output.accept_tokens   = {std::move(accept_tokens)};
            device_->syncAndCheck();
        } else {
            // target model sample
            CHECK_AND_RETURN_REF(
                sampler_input,
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output));
            sampler_output = std::move(sampler_->forward(sampler_input));
            sampler_output.all_probs->updateShape({batch_size, propose_step_ + 1, vocab_size_});

            // rejection sampling
            speculative_sampler_output = speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);
        }
        // NOTE: here will have cuda device sync before update model input
        batch_stream_processor_->updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t, total_accept_len);
    }

    tpSyncModelInputs(model_input, device_);

    maybePrintModelInput(model_input, "decode post draft model");
    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;

    draft_prefill_model_output = std::move(draft_model_->forward(model_input));

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        device_->syncAndCheck();
        draft_model_->releaseBuffers();
        model_->releaseBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    draftModelSample(draft_prefill_model_output.logits, draft_prefill_sampler_output, draft_probs_t, draft_token_ids_t);

    // collect metrics
    if (metrics_reporter_) {
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

void MtpExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams,
                                 std::list<GenerateStreamPtr>&       prefill_streams,
                                 std::list<GenerateStreamPtr>&       decode_streams) {
    for (auto& stream : streams) {
        // split streams into prefill and decode
        if (stream->isContextStream()) {
            prefill_streams.push_back(stream);
        } else {
            stream->setScoreLen(propose_step_ + 1);
            if (stream->getSPOutputBuffer() == nullptr && stream->isPerfTest()) {
                auto sp_output_buffer =
                    makeFakeSPOutputBuffer(data_type_, hidden_size_, vocab_size_, propose_step_, device_);
                stream->setSPOutputBuffer(sp_output_buffer);
            }
            decode_streams.push_back(stream);
        }

        // set base properties
        stream->setReturnAllProbs(true);
        if (stream->getSPOutputBuffer() == nullptr) {
            auto sp_output_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->propose_step = propose_step_;
            sp_output_buffer->tokens       = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, 2}, rtp_llm::AllocationType::HOST}, {"spec_tokens"});

            stream->setSPOutputBuffer(sp_output_buffer);
        }
    }
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams) {
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

std::tuple<torch::Tensor, torch::Tensor> MtpExecutor::fastTopK(const torch::Tensor& probs, int top_k, int dim) {
    if (top_k == 1) {
        return torch::max(probs, dim, true);
    } else {
        return torch::topk(probs, top_k, dim);
    }
}

void MtpExecutor::draftModelSample(const BufferPtr& logits,
                                   SamplerOutput&   sampler_output,
                                   torch::Tensor&   draft_probs,
                                   torch::Tensor&   draft_token_ids) {
    // hold draft_probs and draft_token_ids to avoid tensor destruction
    draft_probs           = torch::softmax(Buffer2torchTensor(*logits, false), -1);
    auto draft_sample_res = fastTopK(draft_probs, 1, -1);
    draft_token_ids       = std::get<1>(draft_sample_res);

    sampler_output.all_probs = torchTensor2Buffer(draft_probs);
    sampler_output.token_ids = torchTensor2Buffer(draft_token_ids);
}

void MtpExecutor::draftModelDecode(GptModelInputs&             model_input,
                                   const StreamGroups&         stream_groups,
                                   std::vector<torch::Tensor>& draft_probs_list,
                                   torch::Tensor&              draft_token_ids_t) {
    // clear host buffers holder
    buffer_holder_.release();

    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_ids_list;
    BufferPtr                  spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t batch_size   = model_input.combo_tokens->shape()[0];
    spec_prefix_lengths = device_->clone({*model_input.sequence_lengths, AllocationType::HOST});

    buffer_holder_.hold(model_input.combo_tokens);
    auto pre_propose_token = device_->clone({*model_input.combo_tokens, AllocationType::DEVICE});

    auto pre_target_token = device_->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST});
    int  batch_idx        = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        int* propose_tokens                      = stream->getSPOutputBuffer()->tokens->data<int>();
        pre_target_token->data<int>()[batch_idx] = propose_tokens[0];
        batch_idx++;
    }

    buffer_holder_.hold(pre_target_token);
    auto pre_target_token_d         = device_->clone({*pre_target_token, AllocationType::DEVICE});
    auto pre_target_token_t         = Buffer2torchTensor(pre_target_token_d, false);
    auto pre_target_token_t_reshape = pre_target_token_t.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_target_token_t_reshape);

    auto pre_propose_token_t         = Buffer2torchTensor(pre_propose_token, false);
    auto pre_propose_token_t_reshape = pre_propose_token_t.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_propose_token_t_reshape);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_LOG_DEBUG("draft model decode step %d batch_size %d", i, batch_size);
        draft_decode_model_output = std::move(draft_model_->forward(model_input));

        // sample
        auto draft_probs         = torch::softmax(Buffer2torchTensor(*draft_decode_model_output.logits, false), -1);
        auto draft_probs_reshape = draft_probs.reshape({(int)batch_size, 1, -1});
        auto [draft_token_probs, draft_token_ids] = fastTopK(draft_probs, 1, -1);

        if (model_input.is_fake_stream) {
            draft_token_ids.zero_();
            device_->bufMemset(*draft_decode_model_output.all_hidden_states, 0);
        }

        draft_token_ids = draft_token_ids.to(torch::kInt32);
        draft_token_ids_list.push_back(draft_token_ids);
        draft_probs_list.push_back(draft_probs_reshape);

        // update model input
        if (i != propose_step_ - 2) {
            batch_stream_processor_->updateDecodeDraftModelInput(
                model_input, draft_decode_model_output, draft_token_ids);
        }
    }

    // prepare spec decode input
    if (isTpRank0()) {
        draft_token_ids_t =
            torch::cat(draft_token_ids_list, 1).reshape({(int)batch_size, (int)(propose_step_ + 1)}).contiguous();

        auto lm_output_indexes = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size * (propose_step_ + 1)}, rtp_llm::AllocationType::HOST}, {});
        auto input_lengths = device_->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST});

        for (int i = 0; i < batch_size; i++) {
            input_lengths->data<int>()[i] = propose_step_ + 1;
        }
        for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
            lm_output_indexes->data<int>()[i] = i;
        }

        model_input.input_lengths     = input_lengths;
        model_input.lm_output_indexes = lm_output_indexes;
        model_input.prefix_lengths    = spec_prefix_lengths;
        model_input.combo_tokens      = torchTensor2Buffer(draft_token_ids_t);
        model_input.combo_tokens->updateShape({batch_size * (propose_step_ + 1)});
        model_input.sequence_lengths   = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
        model_input.last_hidden_states = nullptr;
    }

    tpSyncModelInputs(model_input, device_);
    const auto& cache_cfg             = cache_manager_->cacheConfig();
    model_input.kv_block_stride_bytes = cache_cfg.kv_block_stride_bytes;

    // TODO(yinzhi): if no sync here, maybe cause cuda error, need to find a better way to avoid this.
    device_->syncDeviceStream(DeviceStream::DEFAULT);
}

}  // namespace rtp_llm
