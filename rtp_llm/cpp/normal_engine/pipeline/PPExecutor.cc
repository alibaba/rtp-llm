#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPExecutor::ModelFactory PPExecutor::test_model_factory = nullptr;

void PPExecutor::InflightBatch::reset() {
    skip_run         = true;
    stream_groups    = StreamGroups();
    schedule_time_us = 0;
}

void PPExecutor::sendObject(const torch::Tensor& object, PPTickets& tickets) {
    auto object_size = torch::tensor({object.numel()}, torch::kInt64);
    tickets.push_back(transport_->asyncSend(object_size));
    tickets.push_back(transport_->asyncSend(object));
}

torch::Tensor PPExecutor::receiveObject() {
    auto object_size  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
    auto size_receive = transport_->asyncReceive(object_size);
    size_receive->wait();

    auto object         = torch::empty({object_size.item<int64_t>()}, torch::TensorOptions().dtype(torch::kUInt8));
    auto object_receive = transport_->asyncReceive(object);
    object_receive->wait();
    return object;
}

void PPExecutor::asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets) {
    sendObject(pp_serialization::serializePlan(plan, empty_plan), tickets);
}

PPExecutionPlan PPExecutor::receivePlan() {
    return pp_serialization::deserializePlan(receiveObject());
}

void PPExecutor::asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets) {
    sendObject(pp_serialization::serializeExecutionResult(result), tickets);
}

void PPExecutor::asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets) {
    sendObject(pp_serialization::serializeTensorsMetadata(tensors), tickets);
    for (const auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncSend(tensor_entry.second));
        }
    }
}

PPIntermediateTensors PPExecutor::receiveTensors(PPTickets& tickets) {
    auto tensors = pp_serialization::deserializeTensorsMetadata(receiveObject());
    for (auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncReceive(tensor_entry.second));
        }
    }
    return tensors;
}

void PPExecutor::waitAll(PPTickets& tickets) {
    for (auto& ticket : tickets) {
        ticket->wait();
    }
    tickets.clear();
}

absl::Status PPExecutor::processExecutionResult(InflightBatch& batch) {
    auto result = pp_serialization::deserializeExecutionResult(receiveObject());
    return batch_stream_processor_->dispatchExecutionResult(batch.stream_groups, result);
}

PPExecutor::PPExecutor(const EngineInitParams&                params,
                       const std::shared_ptr<KVCacheManager>& cache_manager,
                       bool                                   warm_up,
                       MlaOpsType                             mla_ops_type,
                       std::function<void()>                  profile_step_start,
                       std::function<void()>                  profile_step_finish,
                       ProposeModelEngineInitParams*          propose_params):
    Executor(),
    warm_up_(warm_up),
    cache_manager_(cache_manager),
    mtp_enabled_(params.sp_config.type == SP_TYPE_MTP),
    propose_step_(params.sp_config.gen_num_per_cycle),
    position_id_len_factor_(params.model_config_.attn_config.rope_config.index_factor),
    parallelism_config_(params.parallelism_config),
    pp_layout_(PPLayout::fromParallelismConfig(parallelism_config_, params.model_config_.num_layers)),
    slots_(parallelism_config_.pp_size + 1),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.world_rank == 0 && !warm_up_ ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.world_rank == 0 && !warm_up_ ? metrics_reporter_ : nullptr)) {
    RTP_LLM_CHECK_WITH_INFO(!mtp_enabled_ || params.sp_config.gen_num_per_cycle > 0,
                            "PP MTP requires a positive gen_num_per_cycle, got %zu",
                            propose_step_);

    if (!warm_up_) {
        transport_ = std::make_unique<NcclPPTransport>(pp_layout_.prevRank(), pp_layout_.nextRank());
    }

    enable_detail_log_ = params.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, parallelism_config_.tp_rank);
    if (params.profiling_debug_logging_config.enable_model_inputs_log) {
        model_inputs_logger_ =
            std::make_shared<ModelInputsLogger>(params.parallelism_config.world_rank,
                                                params.profiling_debug_logging_config.log_file_backup_count,
                                                metrics_reporter_);
    }

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
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

    if (!warm_up_ && isLastStage() && isStageRoot()) {
        const auto initial_sampler_batch_size =
            static_cast<size_t>(std::max<int64_t>(1, params.runtime_config.max_generate_batch_size));
        sampler_ = std::make_unique<Sampler>(SamplerInitParams{initial_sampler_batch_size, false});
    }

    const size_t runtime_tokens_per_block        = cache_manager_ ? cache_manager_->cacheConfig().seq_size_per_block :
                                                                    params.model_config_.attn_config.tokens_per_block;
    const size_t runtime_kernel_tokens_per_block = cache_manager_ ?
                                                       cache_manager_->cacheConfig().kernel_seq_size_per_block :
                                                       params.model_config_.attn_config.kernel_tokens_per_block;

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager_ ? std::make_optional(cache_manager_->getMainModelGroupedCacheLayerLayout()) : std::nullopt,
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
         runtime_tokens_per_block,
         runtime_kernel_tokens_per_block,
         cache_manager_,
         std::nullopt,
         params.model_config_.hc_mult});

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_ = std::make_unique<PyWrappedModel>(model_init_params, params.py_model);
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    if (isLastStage() && propose_params) {
        const auto&                            draft_params = propose_params->getEngineInitParams();
        std::optional<GroupedCacheLayerLayout> draft_cache_layer_layout;
        size_t draft_tokens_per_block        = draft_params.model_config_.attn_config.tokens_per_block;
        size_t draft_kernel_tokens_per_block = draft_params.model_config_.attn_config.kernel_tokens_per_block;
        if (cache_manager_) {
            draft_cache_layer_layout       = cache_manager_->getMTPModuleGroupedCacheLayerLayout(0);
            const auto& draft_cache_config = *cache_manager_->cacheConfig().mtp_sub_configs[0];
            draft_tokens_per_block         = draft_cache_config.seq_size_per_block;
            draft_kernel_tokens_per_block  = draft_cache_config.kernel_seq_size_per_block;
        }

        GptModelInitParams draft_init_params({draft_params.gpt_weights,
                                              genModelDescription(draft_params.model_config_,
                                                                  draft_params.parallelism_config,
                                                                  draft_params.eplb_config,
                                                                  draft_params.moe_config),
                                              draft_cache_layer_layout,
                                              draft_params.model_id,
                                              draft_params.parallelism_config,
                                              params.hw_kernel_config,
                                              params.profiling_debug_logging_config,
                                              params.runtime_config,
                                              params.concurrency_config,
                                              params.sp_config,
                                              params.device_resource_config,
                                              mla_ops_type,
                                              draft_params.model_config_.max_seq_len,
                                              draft_params.model_config_.hidden_size,
                                              draft_tokens_per_block,
                                              draft_kernel_tokens_per_block,
                                              cache_manager_,
                                              std::make_optional(0),
                                              draft_params.model_config_.hc_mult});

        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("init PP executor with python draft model");
            draft_model_ = std::make_unique<PyWrappedModel>(draft_init_params, params.py_sp_model);
        } else if (test_model_factory) {
            draft_model_ = test_model_factory(draft_init_params);
        } else {
            RTP_LLM_LOG_WARNING("py_sp_model is None — draft model will not be initialized (test mode)");
        }
        if (mtp_enabled_ && isStageRoot()) {
            const auto& d2t_map  = draft_model_ ? draft_model_->weights_.d2t_map : draft_params.gpt_weights.d2t_map;
            fast_topk_sampler_   = std::make_unique<speculative::FastTopKSampler>(d2t_map);
            speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(d2t_map, propose_step_);
            spec_logits_verify_runner_ = std::make_unique<SpecLogitsVerifyRunner>();
        }
    }

    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : CacheConfig();
    batch_stream_processor_  = std::make_unique<PPBatchStreamProcessor>(params.model_config_,
                                                                       params.pd_sep_config,
                                                                       params.profiling_debug_logging_config,
                                                                       cache_config,
                                                                       warm_up_,
                                                                       mtp_enabled_);
    LogitsProcessorFactory::init(params.model_config_, params.grammar_config, params.sp_config.tree_decode_config);
    cudaProfilerBegin();
}

PPExecutor::~PPExecutor() {
    for (auto& slot : slots_) {
        waitAll(slot.plan_sends);
        waitAll(slot.activation_sends);
        waitAll(slot.execution_result_sends);
    }
    cudaProfilerEnd();
}

absl::Status PPExecutor::warmUp(const ScheduleOutput& schedule_output) {
    RTP_LLM_CHECK_WITH_INFO(model_ != nullptr, "model is not initialized for PP warmup");

    StreamGroups stream_groups(schedule_output.streams);
    auto         model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto model_input = std::move(model_input_status.value());

    /* Each stage warms up the same fake request locally and only requires TP synchronization. */
    tpSyncModelInputs(model_input, parallelism_config_);

    buffer_holder_.release();
    model_->releaseBuffers();
    if (cache_manager_ && model_input.kv_cache_update_mapping.defined()) {
        cache_manager_->blockBatchCopy(model_input.kv_cache_update_mapping);
    }

    PPIntermediateTensors input_tensors;
    PPIntermediateTensors output_tensors;
    if (!isFirstStage()) {
        input_tensors = model_->makePPWarmUpInputTensors(model_input);
    }

    auto model_output = model_->forwardPP(
        model_input, isFirstStage() ? nullptr : &input_tensors, isLastStage() ? nullptr : &output_tensors);
    if (expert_balancer_) {
        RtpLLMExecutorMetricsCollector collector;
        expert_balancer_->stepForward(*model_, collector);
    }

    /* Keep model tensors alive until lazy initialization kernels finish. */
    cudaSyncAndCheck();
    (void)model_output;
    model_->releaseBuffers();
    return absl::OkStatus();
}

void PPExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams) const {
    if (!mtp_enabled_) {
        return;
    }

    const auto token_count = static_cast<int64_t>(propose_step_ + 1);
    for (const auto& stream : streams) {
        auto       sp_output_buffer = stream->getSPOutputBuffer();
        const bool is_fake_stream   = stream->isFakeStream() || stream->isPerfTest();
        if (!sp_output_buffer) {
            RTP_LLM_CHECK_WITH_INFO(stream->isContextStream() || is_fake_stream,
                                    "PP MTP decode requires proposal tokens, request_id=%ld",
                                    stream->streamId());
            sp_output_buffer         = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->tokens = torch::zeros({1, token_count}, torch::kInt32);
            stream->setSPOutputBuffer(sp_output_buffer);
        } else if (is_fake_stream
                   && (!sp_output_buffer->tokens.defined() || sp_output_buffer->tokens.numel() != token_count)) {
            sp_output_buffer->tokens = torch::zeros({1, token_count}, torch::kInt32);
        }
        RTP_LLM_CHECK_WITH_INFO(
            sp_output_buffer->tokens.defined() && sp_output_buffer->tokens.device().is_cpu()
                && sp_output_buffer->tokens.scalar_type() == torch::kInt32 && sp_output_buffer->tokens.is_contiguous()
                && sp_output_buffer->tokens.numel() == token_count,
            "PP MTP token buffer must contain one target token and %zu draft tokens, request_id=%ld",
            propose_step_,
            stream->streamId());
        sp_output_buffer->propose_step = propose_step_;
    }
}

absl::StatusOr<PPExecutionPlan> PPExecutor::buildPlan(const StreamGroups&         stream_groups,
                                                      const std::vector<int64_t>& finished_request_ids) {
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    PPExecutionPlan plan;
    plan.finished_request_ids = finished_request_ids;

    const auto streams = stream_groups.allStreams();
    plan.is_decode     = mtp_enabled_ && !streams.empty() && !streams.front()->isContextStream();

    auto model_input_status =
        plan.is_decode ?
            batch_stream_processor_->gatherTargetVerifyModelInput(stream_groups, propose_step_, buffer_holder_) :
            batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    plan.model_input          = std::move(model_input_status.value());
    plan.model_input.skip_run = stream_groups.empty();

    if (!plan.model_input.skip_run) {
        plan.sampling_plan = batch_stream_processor_->gatherSamplingPlan(stream_groups);
        plan.output_config = batch_stream_processor_->gatherOutputConfig(stream_groups);
        plan.draft_next_position_ids =
            batch_stream_processor_->gatherDraftNextPositionIds(stream_groups, plan.model_input);
    }

    return plan;
}

void PPExecutor::advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result) {
    const auto stream_count = sampling_plan.request_ids.size(0);

    const auto*   request_ids      = sampling_plan.request_ids.data_ptr<int64_t>();
    const auto*   input_lengths    = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto*   sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    const auto&   cum_log_probs    = result.cum_log_probs;
    const auto* success   = result.sample_success.data_ptr<bool>();
    int64_t     batch_idx = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const int64_t stream_batch_size = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);

        bool stream_succeeded = true;
        for (int64_t sequence_idx = 0; sequence_idx < stream_batch_size; ++sequence_idx) {
            const int64_t row = batch_idx + sequence_idx;
            stream_succeeded  = stream_succeeded && success[row] && !result.processor_errors[row].has_value();
        }
        if (!stream_succeeded) {
            batch_idx += stream_batch_size;
            continue;
        }
        auto& state = sampling_states_.at(request_ids[stream_idx]);

        std::optional<ErrorInfo> error;
        const auto num_new_tokens = result.accept_len.defined() ? result.accept_len.data_ptr<int32_t>()[batch_idx] : 1;
        RTP_LLM_CHECK(num_new_tokens > 0 && num_new_tokens <= result.new_token_ids.size(1));
        const auto new_tokens =
            result.new_token_ids.narrow(0, batch_idx, stream_batch_size).narrow(1, 0, num_new_tokens);
        for (const auto& processor : state.logits_processors) {
            error = processor->updateStatus(new_tokens, num_new_tokens);
            if (error.has_value()) {
                break;
            }
        }
        const int64_t expected_output_len = sequence_lengths[batch_idx] - input_lengths[batch_idx] + num_new_tokens;
        if (!error.has_value()) {
            for (size_t processor_index = 0; processor_index < state.logits_processors.size(); ++processor_index) {
                const auto processor_output_len = state.logits_processors[processor_index]->committedOutputLen();
                if (processor_output_len.has_value() && processor_output_len.value() != expected_output_len) {
                    error = ErrorInfo(ErrorCode::UNKNOWN_ERROR,
                                      "logits processor committed output length mismatch: processor_index="
                                          + std::to_string(processor_index)
                                          + ", processor=" + std::to_string(processor_output_len.value())
                                          + ", expected=" + std::to_string(expected_output_len));
                    break;
                }
            }
        }
        if (error.has_value()) {
            for (int64_t sequence_idx = 0; sequence_idx < stream_batch_size; ++sequence_idx) {
                result.processor_errors[batch_idx + sequence_idx] = error;
            }
        } else if (cum_log_probs.defined()) {
            state.cum_log_probs.copy_(cum_log_probs.narrow(0, batch_idx, stream_batch_size));
        }
        batch_idx += stream_batch_size;
    }
}

absl::StatusOr<PPExecutionResult> PPExecutor::verifyDraftTokens(const PPExecutionPlan& plan,
                                                                const torch::Tensor&   target_logits) {
    const auto     batch_size         = plan.sampling_plan.request_ids.size(0);
    const auto     verify_token_count = static_cast<int64_t>(propose_step_ + 1);
    const auto     vocab_size         = target_logits.size(1);
    PPOutputConfig output_config;
    output_config.return_all_probs = ReturnAllProbsMode::DEFAULT;
    auto inputs_status             = batch_stream_processor_->gatherSamplerInputs(
        plan.sampling_plan, output_config, target_logits, sampling_states_, true, propose_step_);
    RETURN_IF_STATUS_OR_ERROR(inputs_status);
    auto inputs = std::move(inputs_status.value());
    inputs.logits_processor_states_ptr.reset();

    SamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = plan.model_input.combo_tokens.reshape({batch_size, verify_token_count})
                                         .narrow(1, 1, propose_step_)
                                         .contiguous();
    draft_sampler_output.token_ids_are_point_mass = true;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = batch_size;
    task.propose_step  = propose_step_;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = draft_sampler_output.token_ids;
    speculative::SpeculativeSamplingParams params;
    params.do_sample                                  = plan.sampling_plan.spec_do_sample;
    params.force_accept                               = plan.sampling_plan.force_sp_accept;
    const auto*                           request_ids = plan.sampling_plan.request_ids.data_ptr<int64_t>();
    std::vector<std::optional<ErrorInfo>> compatibility_errors(batch_size);
    for (int64_t row = 0; row < batch_size; ++row) {
        const auto& state = sampling_states_.at(request_ids[row]);
        params.generators.push_back(state.generator);
        for (const auto& processor : state.logits_processors) {
            const auto capability = processor->mtpCapability();
            if (capability.mode == MtpProcessorMode::UNSUPPORTED) {
                compatibility_errors[row] =
                    ErrorInfo(ErrorCode::INVALID_PARAMS,
                              "MTP decode is incompatible with logits processor: " + std::string(capability.reason));
                break;
            }
            if (capability.mode == MtpProcessorMode::SPEC_VERIFY) {
                task.active.push_back({processor, static_cast<size_t>(row)});
            }
        }
    }
    SpecLogitsVerifyRunner::LaunchResult verify_result;
    if (!task.active.empty()) {
        verify_result = spec_logits_verify_runner_->run(task);
        if (verify_result.ready_event) {
            verify_result.ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        SpecLogitsVerifyRunner::applyMaskToLogits(inputs.logits, verify_result, vocab_size);
    }

    auto target_sampler_output = sampler_->forward(inputs);
    target_sampler_output.all_probs =
        target_sampler_output.all_probs.reshape({batch_size, verify_token_count, vocab_size});
    speculative::SpeculativeSamplerOutput accepted;
    mtp::runRejectionSampling(
        *speculative_sampler_, params, draft_sampler_output, target_sampler_output, verify_result, accepted);
    accepted.transfer_done_event->synchronize();

    PPExecutionResult result;
    result.request_ids      = plan.sampling_plan.request_ids;
    result.new_token_ids    = std::move(accepted.accept_tokens_cpu);
    result.accept_len       = std::move(accepted.accept_len_cpu);
    result.sample_success   = torch::ones({batch_size}, torch::kBool);
    result.processor_errors = std::move(accepted.processor_errors);
    result.processor_errors.resize(batch_size);
    result.prompt_logits.resize(batch_size);
    for (int64_t row = 0; row < batch_size; ++row) {
        if (compatibility_errors[row].has_value()) {
            result.processor_errors[row] = std::move(compatibility_errors[row]);
        }
    }
    return result;
}

GptModelInputs PPExecutor::prepareDraftInputForPrefill(const GptModelInputs&  target_input,
                                                       const GptModelOutputs& target_output,
                                                       const torch::Tensor&   sampled_token_ids,
                                                       const torch::Tensor&   next_position_ids) {
    auto draft_input          = target_input;
    auto target_hidden_states = model_->getMtpTargetHiddenStates(target_input.combo_tokens.numel());
    if (!target_hidden_states.defined() || target_hidden_states.numel() == 0) {
        target_hidden_states = target_output.all_hidden_states;
    }
    mtp::prepareDraftInputForPrefill(draft_input,
                                    target_hidden_states,
                                    sampled_token_ids,
                                    next_position_ids,
                                    position_id_len_factor_,
                                    buffer_holder_);
    return draft_input;
}

GptModelInputs PPExecutor::prepareDraftInputForDecode(const GptModelInputs&  target_input,
                                                      const GptModelOutputs& target_output,
                                                      const torch::Tensor&   accepted_token_ids,
                                                      const torch::Tensor&   accepted_lengths) {
    auto draft_input          = target_input;
    auto target_hidden_states = model_->getMtpTargetHiddenStates(target_input.combo_tokens.numel());
    if (!target_hidden_states.defined() || target_hidden_states.numel() == 0) {
        target_hidden_states = target_output.all_hidden_states;
    }

    mtp::prepareDraftInputForDecode(draft_input,
                                   target_hidden_states,
                                   accepted_token_ids,
                                   accepted_lengths,
                                   mtp::DraftInputLayout::COMPACT,
                                   position_id_len_factor_,
                                   buffer_holder_);
    return draft_input;
}

torch::Tensor PPExecutor::proposeDraftTokens(GptModelInputs draft_input, size_t num_draft_tokens) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.propose_draft_tokens");
    torch::Tensor proposed_tokens;
    for (size_t step = 0; step < num_draft_tokens; ++step) {
        tpSyncModelInputs(draft_input, parallelism_config_);
        draft_model_->releaseBuffers();
        if (cache_manager_) {
            const auto& draft_cache_config    = cache_manager_->getMTPModuleCacheConfig(0);
            draft_input.kv_block_stride_bytes = draft_cache_config.kv_block_stride_bytes;
            draft_input.kv_scale_stride_bytes = draft_cache_config.kv_scale_stride_bytes;
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(draft_input, ModelInputsModelRole::DRAFT, draft_model_->model_id_);
        }
        auto draft_output = draft_model_->forward(draft_input);

        if (isStageRoot()) {
            auto draft_tokens = fast_topk_sampler_->forward(draft_output.logits).token_ids.to(torch::kInt32);
            if (step == 0) {
                proposed_tokens = torch::empty({draft_tokens.size(0), static_cast<int64_t>(num_draft_tokens)},
                                               draft_tokens.options());
            }
            proposed_tokens.select(1, step).copy_(draft_tokens.flatten());
            if (step + 1 < num_draft_tokens) {
                auto draft_hidden_states = draft_model_->getMtpTargetHiddenStates(draft_input.combo_tokens.numel());
                if (draft_hidden_states.defined() && draft_hidden_states.numel() > 0) {
                    draft_output.all_hidden_states = draft_hidden_states;
                }
                if (step == 0) {
                    const auto batch_size     = draft_input.input_lengths.numel();
                    const auto output_indexes = draft_input.lm_output_indexes.to(torch::kLong);

                    draft_input.combo_tokens = draft_tokens.reshape({batch_size});
                    draft_input.last_hidden_states = draft_output.all_hidden_states.index_select(
                        0, output_indexes.to(draft_output.all_hidden_states.device()));
                    draft_input.sequence_lengths =
                        draft_input.input_lengths + draft_input.prefix_lengths.to(draft_input.input_lengths.device())
                        + 1;
                    draft_input.prefix_lengths          = torch::empty({0}, draft_input.prefix_lengths.options());
                    draft_input.sequence_lengths_plus_1 = torch::Tensor();
                    draft_input.lm_output_indexes = torch::arange(batch_size, draft_input.lm_output_indexes.options());
                    draft_input.request_id              = torch::Tensor();
                    draft_input.request_pd_separation   = torch::Tensor();
                    draft_input.cache_keys              = torch::Tensor();
                    if (draft_input.combo_position_ids.defined()) {
                        const auto positions = draft_input.combo_position_ids.reshape(
                            {-1, static_cast<int64_t>(position_id_len_factor_)});
                        draft_input.combo_position_ids =
                            (positions.index_select(0, output_indexes.to(positions.device())) + 1).flatten().pin_memory();
                    }
                } else {
                    mtp::advanceDraftInput(
                        draft_input, draft_output.all_hidden_states, draft_tokens, position_id_len_factor_, buffer_holder_);
                }
            }
        }
    }
    if (isStageRoot()) {
        proposed_tokens = proposed_tokens.to(torch::kCPU);
    }
    cudaSyncAndCheck();
    draft_model_->releaseBuffers();
    return proposed_tokens;
}

absl::StatusOr<PPExecutionResult> PPExecutor::sampleTokens(const PPExecutionPlan& plan,
                                                           const GptModelOutputs& model_output) {
    if (mtp_enabled_ && plan.is_decode) {
        return verifyDraftTokens(plan, model_output.logits);
    }

    auto inputs = batch_stream_processor_->gatherSamplerInputs(
        plan.sampling_plan, plan.output_config, model_output.logits, sampling_states_);
    RETURN_IF_STATUS_OR_ERROR(inputs);
    auto sampler_output = sampler_->forward(inputs.value());
    auto result         = batch_stream_processor_->makeExecutionResult(plan, model_output, sampler_output);
    RETURN_IF_STATUS_OR_ERROR(result);
    if (mtp_enabled_) {
        result->accept_len = torch::ones({result->new_token_ids.size(0)}, torch::kInt32);
    }
    return result;
}

void PPExecutor::draftSampleAndPropose(const PPExecutionPlan& plan,
                                       const GptModelOutputs& model_output,
                                       PPExecutionResult&     execution_result) {
    auto draft_input = plan.model_input;
    if (isStageRoot()) {
        if (plan.is_decode) {
            draft_input = prepareDraftInputForDecode(
                plan.model_input, model_output, execution_result.new_token_ids, execution_result.accept_len);
        } else {
            draft_input = prepareDraftInputForPrefill(
                plan.model_input, model_output, execution_result.new_token_ids, plan.draft_next_position_ids);
        }
    }
    execution_result.propose_token_ids = proposeDraftTokens(std::move(draft_input), propose_step_);
}

absl::Status PPExecutor::process(const ScheduleOutput& schedule_output, int64_t schedule_time_us) {
    if (warm_up_) {
        return warmUp(schedule_output);
    }

    schedule_time_us = (schedule_time_us <= 0) ? autil::TimeUtility::currentTimeInMicroSeconds() : schedule_time_us;

    auto tps_active_guard      = tps_reporter_.makeActiveGuard(metrics_reporter_ && isFirstStage() && isStageRoot()
                                                          && !schedule_output.streams.empty());
    auto wall_tps_active_guard = wall_tps_reporter_.makeActiveGuard(metrics_reporter_ && isFirstStage() && isStageRoot()
                                                                    && !schedule_output.streams.empty());
    RTP_LLM_PROFILE_FUNCTION();

    /** 0. Prepare SP Buffer if need.  */
    if (isFirstStage() && isStageRoot()) {
        prepareStreams(schedule_output.streams);
    }

    /** 1. recv the plan from the previous stage */
    PPExecutionPlan plan;
    StreamGroups    scheduled_stream_groups;
    if (isFirstStage()) {
        scheduled_stream_groups = StreamGroups(schedule_output.streams);
        auto plan_status        = buildPlan(scheduled_stream_groups, schedule_output.finished_request_ids);
        RETURN_IF_STATUS_OR_ERROR(plan_status);
        plan = std::move(plan_status.value());
    } else {
        plan = receivePlan();
    }

    if (isLastStage() && isStageRoot()) {
        for (const auto request_id : plan.finished_request_ids) {
            sampling_states_.erase(request_id);
        }
    }

    /** 2. do the sync across the all ranks in the same stage. */
    tpSyncModelInputs(plan.model_input, parallelism_config_);

    /** 3. make sure the current slot is ready. */
    auto& inflight = slots_[current_slot_];
    waitAll(inflight.plan_sends);
    waitAll(inflight.activation_sends);
    waitAll(inflight.execution_result_sends);
    inflight.reset();
    inflight.skip_run = plan.model_input.skip_run;
    if (isFirstStage() && isStageRoot()) {
        inflight.stream_groups    = std::move(scheduled_stream_groups);
        inflight.schedule_time_us = schedule_time_us;
    }

    /** 4. send the plan to next stage  */
    if (!isLastStage()) {
        asyncSendPlan(plan, !isStageRoot(), inflight.plan_sends);
    }

    /** 5. run the batch. */
    if (!plan.model_input.skip_run) {
        PPTickets             tensor_receives;
        PPIntermediateTensors input_tensors;
        PPIntermediateTensors output_tensors;

        if (!isFirstStage()) {
            input_tensors = receiveTensors(tensor_receives);
            waitAll(tensor_receives);
        }

        if (profile_step_start_) {
            profile_step_start_();
        }

        GptModelInputs& local_model_input = plan.model_input;
        buffer_holder_.release();
        model_->releaseBuffers();
        if (cache_manager_ && local_model_input.kv_cache_update_mapping.defined()) {
            cache_manager_->blockBatchCopy(local_model_input.kv_cache_update_mapping);
        }

        const bool force = isStageRoot() && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", local_model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", local_model_input.debugString(force).c_str());
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(local_model_input, ModelInputsModelRole::NORMAL, model_->model_id_);
        }
        auto model_output = model_->forwardPP(
            local_model_input, isFirstStage() ? nullptr : &input_tensors, isLastStage() ? nullptr : &output_tensors);

        if (expert_balancer_) {
            RtpLLMExecutorMetricsCollector collector;
            expert_balancer_->stepForward(*model_, collector);
        }

        auto forward_done = cuda_graph::makeGraphEvent();
        forward_done.record(cuda_graph::graphGetCurrentStream());
        forward_done.synchronize();

        if (!isLastStage()) {
            asyncSendTensors(output_tensors, inflight.activation_sends);
        } else {
            PPExecutionResult execution_result;
            if (isStageRoot()) {
                auto res = sampleTokens(plan, model_output);
                RETURN_IF_STATUS_OR_ERROR(res);
                execution_result = std::move(res.value());
            }

            if (mtp_enabled_) {
                draftSampleAndPropose(plan, model_output, execution_result);
            }

            if (isStageRoot()) {
                advanceSamplingStates(plan.sampling_plan, execution_result);
                asyncSendExecutionResult(execution_result, inflight.execution_result_sends);
            }
        }

        if (profile_step_finish_) {
            profile_step_finish_();
        }
    }

    current_slot_ = (current_slot_ + 1) % slots_.size();

    /** 6. recv the execution result of next batch and process it. */
    auto& next_batch = slots_[current_slot_];
    if (isFirstStage() && isStageRoot() && !next_batch.skip_run) {

        const auto& stream_groups            = next_batch.stream_groups;
        auto        token_counts_by_priority = stream_groups.tokenCountsByPriority();
        RETURN_IF_STATUS_ERROR(processExecutionResult(next_batch));

        const int64_t tps_execute_time_us =
            autil::TimeUtility::currentTimeInMicroSeconds() - next_batch.schedule_time_us;
        if (metrics_reporter_ && tps_execute_time_us > 0) {
            RtpLLMTokenPSMetricsCollector tps_collector;
            tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                       stream_groups.contextExecuteTokenSizeWithCache(),
                                       stream_groups.totalDecodeBatchSize(),
                                       stream_groups.modelExecuteTokenSize(),
                                       tps_execute_time_us);
            tps_collector.addTokenSizeByPriority(token_counts_by_priority, tps_execute_time_us);
            tps_reporter_.report(&tps_collector);
            wall_tps_reporter_.report(&tps_collector);
        }
    }
    return absl::OkStatus();
}

bool PPExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
