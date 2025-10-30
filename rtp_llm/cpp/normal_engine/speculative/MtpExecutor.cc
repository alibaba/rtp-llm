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
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>

namespace rtp_llm {

bool MtpExecutor::isTpRank0() const {
    return device_->getDeviceProperties().tp_rank == 0;
}

MtpExecutor::MtpExecutor(const EngineInitParams&                           params,
                         std::unique_ptr<ProposeModelEngineInitParams>&    propose_params,
                         const std::shared_ptr<CacheManager>&              cache_manager,
                         const std::vector<std::shared_ptr<CacheManager>>& mtp_cache_managers,
                         rtp_llm::DeviceBase*                              device,
                         const std::shared_ptr<lora::LoraManager>&         lora_manager,
                         bool                                              warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    mtp_cache_managers_(mtp_cache_managers),
    speculative_sampler_(new speculative::SpeculativeSampler(device, propose_params->gen_num_per_circle)),
    warm_up_(warm_up) {
    propose_step_       = propose_params->gen_num_per_circle;
    vocab_size_         = params.gpt_init_parameter.vocab_size_;
    propose_vocab_size_ = propose_params->getGptInitParameter().vocab_size_;

    auto& gpt_param    = params.gpt_init_parameter;
    enable_detail_log_ = gpt_param.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    // if (gpt_param.enable_eplb_ && gpt_param.moe_style_ != 0) {
    //     // use first moe layer weight as moe weight type
    //     int  first_moe_layer = gpt_param.moe_layer_index_.front();
    //     auto moe_weight_type =
    //     params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();

    //     expert_balancer_ = std::make_shared<ExpertBalancer>(gpt_param.expert_num_,
    //                                                    gpt_param.phy_exp_num_,
    //                                                    gpt_param.num_layers_,
    //                                                    gpt_param.moe_inter_padding_size_,
    //                                                    gpt_param.hidden_size_,
    //                                                    gpt_param.eplb_update_time_,
    //                                                    gpt_param.ep_rank_,
    //                                                    gpt_param.ep_size_,
    //                                                    gpt_param.py_eplb_,
    //                                                    moe_weight_type,
    //                                                    device_,
    //                                                    gpt_param.eplb_mode_,
    //                                                    gpt_param.quant_algo_,
    //                                                    metrics_reporter_);
    // }

    int eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;

    SamplerInitParams sampler_params{
        device_,
        eos_id,
        device->initParams().max_batch_size};  // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.gpt_init_parameter),
         cache_manager ? ((std::optional<KVCacheAllocator::KVCacheBuffer>)cache_manager->kvCacheBuffer()) :
                         std::nullopt,
         params.model_id});

    if (params.gpt_init_parameter.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }

    // if (!params.py_model.is_none()) {
    //     RTP_LLM_LOG_INFO("init executor with python model");
    //     model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    // } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
    //     RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
    //     model_.reset(new NativeDeviceGraphModel(model_init_params));
    // } else {
    //     RTP_LLM_LOG_INFO("init legacy c++ gpt model");
    // }

    // TODO(yinzhi): support py model for mtp
    model_.reset(new GptModel(model_init_params));

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.gpt_init_parameter, cache_config, warm_up_));

    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(
        params.gpt_init_parameter.ckpt_path_, params.gpt_init_parameter.sp_config.tree_decode_config);

    size_t index = 0;
    for (auto& mtp_params : *propose_params->mtp_model_params_) {
        auto mtp_cache_manager = (index < mtp_cache_managers.size()) ? mtp_cache_managers[index] : nullptr;
        auto model_params      = GptModelInitParams(
            {device_,
                  mtp_params->gpt_weights,
                  Executor::genModelDescription(mtp_params->gpt_init_parameter),
             mtp_cache_manager ? ((std::optional<KVCacheAllocator::KVCacheBuffer>)mtp_cache_manager->kvCacheBuffer()) :
                                      std::nullopt,
                  mtp_params->model_id});
        setDraftModel(std::make_unique<MTPModel>(model_params));
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
absl::Status MtpExecutor::prefillStep(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    GptModelOutputs                draft_model_output;
    SamplerOutput                  draft_sampler_output;

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

    // TODO(yinzhi): consider beam search & lora

    // target model prefill
    {
        // bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
        // if (force) {
        //     RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        // } else {
        //     RTP_LLM_LOG_DEBUG("model_input: %s", model_input.debugString(force).c_str());
        // }
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
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        updatePrefillPostDraftModelInput(model_input, model_output, sampler_output);
    }

    // draft model prefill
    {
        tpSyncModelInputs(model_input, device_);
        draft_model_output = std::move(draft_model_->forward(model_input));
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0) {
        return absl::OkStatus();
    }

    // draft model sample
    draftModelSample(draft_model_output.logits, draft_sampler_output, draft_probs, draft_token_ids);

    // dispatch
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    result =
            batch_stream_processor_->dispatchPrefill(stream_groups,
                                                     {std::move(model_output), std::move(sampler_output)},
                                                     {std::move(draft_model_output), std::move(draft_sampler_output)});
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
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
|    target model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|     target model sample       |
+-------------------------------+
*/

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;

    GptModelOutputs                       draft_model_output;
    SamplerOutput                         draft_sampler_output;
    GptModelOutputs                       draft_prefill_model_output;
    SamplerOutput                         draft_prefill_sampler_output;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    // placeholder for some tensors
    torch::Tensor draft_token_probs_d_t;
    torch::Tensor hidden_states_d_t;
    torch::Tensor draft_probs_t;
    torch::Tensor draft_token_ids_t;

    const size_t batch_size = streams.size();
    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherDecodeModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        model_input.skip_run = streams.empty() && !enable_ffn_disaggregate_;
        if (model_input.skip_run) {
            tpSyncModelInputs(model_input, device_);
            return absl::OkStatus();
        }
    }

    // TODO(yinzhi): consider beam search & lora

    // TODO(yinzhi): support multi step MTP
    if (propose_step_ == 1) {
        if (isTpRank0()) {
            prepareOneStepSpecDecodeModelInput(stream_groups, model_input);
        }
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
    } else {
        // TODO(yinzhi): prepare draft model input
        // TODO(yinzhi): draft model decode
        // TODO(yinzhi): prepare spec decode input
    }

    model_output = std::move(model_->forward(model_input));

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
        if (propose_step_ == 1) {
            updateOneStepDraftSamplerOutput(stream_groups, draft_sampler_output, draft_token_probs_d_t);
        } else {
            // TODO(yinzhi): support multi step MTP
        }
    }

    // eplb
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        // target model sample
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        sampler_output.all_probs->updateShape({batch_size, propose_step_ + 1, vocab_size_});

        // rejection sampling
        speculative_sampler_output = speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);

        // update model_input
        updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t);
    }

    tpSyncModelInputs(model_input, device_);

    draft_prefill_model_output = std::move(draft_model_->forward(model_input));

    if (!isTpRank0() || warm_up_ || streams.size() == 0) {
        return absl::OkStatus();
    }

    // draft model sample
    draftModelSample(draft_prefill_model_output.logits, draft_prefill_sampler_output, draft_probs_t, draft_token_ids_t);

    // dispatch
    auto result = batch_stream_processor_->dispatchDecode(
        stream_groups, speculative_sampler_output, {std::move(model_output), std::move(draft_prefill_sampler_output)});
    return result;
}

void MtpExecutor::updatePrefillPostDraftModelInput(GptModelInputs&  model_input,
                                                   GptModelOutputs& model_output,
                                                   SamplerOutput&   sampler_output) {
    model_input.last_hidden_states = model_output.all_hidden_states;
    const auto& new_all_token_ids  = sampler_output.token_ids;

    // set model_input.combo_tokens
    const size_t batch_size   = new_all_token_ids->shape()[0];
    const size_t token_stride = new_all_token_ids->shape()[1];

    int* input_lengths = (int*)model_input.input_lengths->data();
    int* combo_tokens  = (int*)model_input.combo_tokens->data();

    int offset = 0;
    for (int i = 0; i < batch_size; i++) {
        // should shift one token for combo_tokens
        int input_length = input_lengths[i];
        memcpy(combo_tokens + offset, combo_tokens + offset + 1, (input_length - 1) * sizeof(int));

        // set new token id
        int new_token_id                        = new_all_token_ids->data<int>()[i * token_stride + token_stride - 1];
        combo_tokens[offset + input_length - 1] = new_token_id;

        offset += input_length;
    }
}

void MtpExecutor::updateDecodePostDraftModelInput(GptModelInputs&                        model_input,
                                                  GptModelOutputs&                       model_output,
                                                  speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                                  size_t                                 batch_size,
                                                  torch::Tensor&                         hidden_states_d_t) {
    auto&  accept_lens      = speculative_sampler_output.accept_len;
    size_t total_accept_len = std::accumulate(accept_lens.begin(), accept_lens.end(), 0);

    auto last_hidden_states = device_->allocateBuffer({model_output.all_hidden_states->type(),
                                                       {total_accept_len, model_output.all_hidden_states->shape()[1]},
                                                       AllocationType::DEVICE});

    model_input.combo_tokens =
        device_->allocateBuffer({DataType::TYPE_INT32, {total_accept_len}, AllocationType::HOST});

    int  token_offset = 0;
    auto lm_output_indexes =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {batch_size}, rtp_llm::AllocationType::HOST});

    std::vector<torch::Tensor> hidden_states_list;
    for (int i = 0; i < batch_size; i++) {
        RTP_LLM_CHECK_WITH_INFO(accept_lens[i] == speculative_sampler_output.accept_tokens[i]->size(),
                                "accept_lens[%d] = %d, speculative_sampler_output.accept_tokens[%d]->size() = %d",
                                i,
                                accept_lens[i],
                                i,
                                speculative_sampler_output.accept_tokens[i]->size());

        memcpy(model_input.combo_tokens->dataWithOffset<int>(token_offset),
               speculative_sampler_output.accept_tokens[i]->data<int>(),
               accept_lens[i] * sizeof(int));

        auto hidden_slice = model_output.all_hidden_states->view(i * (propose_step_ + 1), accept_lens[i]);
        hidden_states_list.push_back(Buffer2torchTensor(hidden_slice, false));

        model_input.input_lengths->data<int>()[i] = accept_lens[i];
        token_offset += accept_lens[i];
        lm_output_indexes->data<int>()[i] = token_offset - 1;
    }

    hidden_states_d_t              = torch::cat(hidden_states_list).contiguous();
    model_input.last_hidden_states = torchTensor2Buffer(hidden_states_d_t);
    model_input.lm_output_indexes  = lm_output_indexes;
}

void MtpExecutor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                  SamplerOutput&      draft_sampler_output,
                                                  torch::Tensor&      draft_token_probs_d_t) {
    const size_t batch_size = stream_groups.size();
    BufferPtr    draft_token_ids =
        device_->allocateBuffer({DataType::TYPE_INT32, {batch_size, propose_step_}, AllocationType::HOST});

    std::vector<torch::Tensor> draft_token_probs_list;
    int                        batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer                                   = stream->getSPOutputBuffer();
        auto propose_tokens                                     = stream->getProposeToken();
        draft_token_ids->data<int>()[batch_idx * propose_step_] = propose_tokens[1];
        draft_token_probs_list.push_back(Buffer2torchTensor(sp_output_buffer->all_probs, false));
        batch_idx++;
    }

    draft_token_probs_d_t          = torch::stack(draft_token_probs_list, 0).contiguous();
    draft_sampler_output.all_probs = torchTensor2Buffer(draft_token_probs_d_t);
    draft_sampler_output.token_ids = draft_token_ids;
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    for (auto& stream : streams) {
        if (stream->isSpDecodeStream()) {
            decode_streams.push_back(stream);
        } else {
            prefill_streams.push_back(stream);
        }
        stream->setReturnAllProbs(true);

        if (stream->getSPOutputBuffer() == nullptr) {
            auto sp_output_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->propose_step = propose_step_;
            sp_output_buffer->tokens       = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, propose_step_}, rtp_llm::AllocationType::HOST}, {});

            stream->setSPOutputBuffer(sp_output_buffer);
        }
    }

    THROW_IF_STATUS_ERROR(prefillStep(prefill_streams));
    THROW_IF_STATUS_ERROR(decodeStep(decode_streams));

    return absl::OkStatus();
}

bool MtpExecutor::updateEplbConfig(const EplbConfig& config) {
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

void MtpExecutor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups, GptModelInputs& model_input) {
    size_t batch_size = stream_groups.size();

    BufferPtr draft_token_probs = device_->allocateBuffer(
        {DataType::TYPE_FP32, {(size_t)batch_size, propose_step_, vocab_size_}, AllocationType::DEVICE});

    device_->bufMemset(*draft_token_probs, 0);

    // prepare target model input buffer
    auto target_prefix_lengths = device_->clone({*model_input.sequence_lengths, AllocationType::HOST});

    // allocate target_combo_tokens shape [batch_size, propose_step_ + 1]
    auto target_combo_tokens = device_->allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)stream_groups.size() * (propose_step_ + 1)}, AllocationType::HOST});

    // copy propose tokens to target_combo_tokens
    int batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto& propose_tokens   = stream->getProposeToken();
        auto  sp_output_buffer = stream->getSPOutputBuffer();
        // print vector string
        RTP_LLM_LOG_DEBUG("propose_tokens = [%s]", vectorToString(propose_tokens).c_str());

        memcpy(target_combo_tokens->dataWithOffset<int>(batch_idx * (propose_step_ + 1)),
               propose_tokens.data(),
               sizeof(int) * propose_tokens.size());

        batch_idx++;
    }

    // update model_input
    model_input.combo_tokens       = target_combo_tokens;
    model_input.prefix_lengths     = target_prefix_lengths;
    model_input.sequence_lengths   = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
    model_input.last_hidden_states = nullptr;

    for (int i = 0; i < model_input.input_lengths->shape()[0]; i++) {
        model_input.input_lengths->data<int>()[i] = propose_step_ + 1;
    }

    // set lm_output_indexes
    auto lm_output_indexes = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {batch_size * (propose_step_ + 1)}, rtp_llm::AllocationType::HOST}, {});
    for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
        lm_output_indexes->data<int>()[i] = i;
    }
    model_input.lm_output_indexes = lm_output_indexes;
}

}  // namespace rtp_llm