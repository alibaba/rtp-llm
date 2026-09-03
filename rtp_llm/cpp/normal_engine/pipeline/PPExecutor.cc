#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <utility>

#include <ATen/Generator.h>
#if defined(USING_CUDA) || defined(USING_ROCM)
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPExecutor::ModelFactory PPExecutor::test_model_factory = nullptr;

static int64_t getProcessorEosTokenId(const ModelConfig& model_config) {
    const auto  eos_token_id     = model_config.special_tokens.eos_token_id;
    const auto& output_vocab_ids = model_config.output_vocab_ids;
    if (output_vocab_ids.empty()) {
        return eos_token_id;
    }

    const auto eos_it = std::lower_bound(output_vocab_ids.begin(), output_vocab_ids.end(), eos_token_id);
    RTP_LLM_CHECK_WITH_INFO(eos_it != output_vocab_ids.end() && *eos_it == eos_token_id,
                            "primary EOS token is absent from the configured output vocabulary");
    return std::distance(output_vocab_ids.begin(), eos_it);
}

void PPExecutor::InflightBatch::reset() {
    skip_run         = true;
    stream_groups    = StreamGroups();
    schedule_time_us = 0;
}

void PPExecutor::sendObject(const torch::Tensor& object, PPTickets& tickets) {
    auto object_size      = torch::tensor({object.numel()}, torch::kInt64).to(torch::kCUDA);
    auto object_on_device = object.to(torch::kCUDA);
    tickets.push_back(transport_->asyncSend(object_size));
    tickets.push_back(transport_->asyncSend(object_on_device));
}

torch::Tensor PPExecutor::receiveObject() {
    auto object_size  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto size_receive = transport_->asyncReceive(object_size);
    size_receive->wait();

    auto object =
        torch::empty({object_size.item<int64_t>()}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto object_receive = transport_->asyncReceive(object);
    object_receive->wait();
    return object.cpu();
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
                       std::function<void()>                  profile_step_finish):
    Executor(),
    cache_manager_(cache_manager),
    processor_eos_token_id_(getProcessorEosTokenId(params.model_config_)),
    warm_up_(warm_up),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.world_rank == 0 && !warm_up_ ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.world_rank == 0 && !warm_up_ ? metrics_reporter_ : nullptr)),
    parallelism_config_(params.parallelism_config),
    // Stage-role flags and materialized partition, shared with cache creation and the Python side.
    pp_layout_(PPLayout::fromParallelismConfig(parallelism_config_, params.model_config_.num_layers)),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    slots_(parallelism_config_.pp_size + 1) {

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

    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : CacheConfig();
    batch_stream_processor_  = std::make_unique<PPBatchStreamProcessor>(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, cache_config, warm_up_);
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

absl::StatusOr<PPExecutionPlan> PPExecutor::buildPlan(const StreamGroups&         stream_groups,
                                                      const std::vector<int64_t>& finished_request_ids) {
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    PPExecutionPlan plan;
    plan.finished_request_ids = finished_request_ids;

    auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    plan.model_input          = std::move(model_input_status.value());
    plan.model_input.skip_run = stream_groups.empty();

    if (!plan.model_input.skip_run) {
        plan.sampling_plan = batch_stream_processor_->gatherSamplingPlan(stream_groups);
        plan.output_config = batch_stream_processor_->gatherOutputConfig(stream_groups);
    }

    return plan;
}

absl::StatusOr<SamplerInputs> PPExecutor::makeSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                            const PPOutputConfig& output_config,
                                                            const torch::Tensor&  logits) {
    const auto batch_size = sampling_plan.logits_processor_configs.size();
    RTP_LLM_CHECK(logits.dim() == 2 && logits.size(0) == sampling_plan.request_ids.size(0));

    SamplerInputs inputs;
    inputs.logits        = output_config.return_logits || output_config.return_softmax_probs ? logits.clone() : logits;
    inputs.token_ids     = sampling_plan.token_ids;
    inputs.input_lengths = sampling_plan.input_lengths;
    inputs.sequence_lengths     = sampling_plan.sequence_lengths;
    inputs.num_beams_in         = torch::ones(sampling_plan.request_ids.sizes(), torch::kLong);
    inputs.num_beams_out        = torch::ones(sampling_plan.request_ids.sizes(), torch::kLong);
    inputs.top_k                = sampling_plan.top_k.clone();
    inputs.top_p                = sampling_plan.top_p;
    inputs.temperature          = sampling_plan.temperature;
    inputs.repetition_penalty   = sampling_plan.repetition_penalty;
    inputs.presence_penalty     = sampling_plan.presence_penalty;
    inputs.frequency_penalty    = sampling_plan.frequency_penalty;
    inputs.no_repeat_ngram_size = sampling_plan.no_repeat_ngram_size;
    inputs.do_sample            = sampling_plan.do_sample;
    inputs.finished_mask        = sampling_plan.finished_mask;
    if (output_config.return_cum_log_probs) {
        inputs.cum_log_probs = torch::empty(sampling_plan.request_ids.sizes(), torch::kFloat32);
    }
    if (output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        inputs.all_probs =
            torch::zeros({sampling_plan.request_ids.size(0), logits.size(1)}, logits.options().dtype(torch::kFloat32));
        inputs.return_original_all_probs = output_config.return_all_probs == ReturnAllProbsMode::ORIGINAL;
    }

    inputs.batch_size     = batch_size;
    inputs.batch_size_out = batch_size;
    inputs.step           = sampling_plan.token_ids.size(1) - 1;
    inputs.vocab_size     = logits.size(-1);
    inputs.generator.resize(batch_size);
    auto processor_states = std::make_shared<LogitsProcessorStates>();

    auto*       top_k         = inputs.top_k.data_ptr<int32_t>();
    auto*       cum_log_probs = inputs.cum_log_probs.defined() ? inputs.cum_log_probs.data_ptr<float>() : nullptr;
    const auto* request_ids   = sampling_plan.request_ids.data_ptr<int64_t>();
    const auto* input_lengths = sampling_plan.input_lengths.data_ptr<int32_t>();
    for (size_t index = 0; index < batch_size; ++index) {
        if (top_k[index] > 0) {
            top_k[index] = std::min(top_k[index], static_cast<int32_t>(logits.size(-1)));
        }

        auto state_it = sampling_states_.find(request_ids[index]);
        if (state_it == sampling_states_.end()) {
            const auto& processor_config = sampling_plan.logits_processor_configs[index];
            auto        config           = std::make_shared<GenerateConfig>();
            if (processor_config.grammar_type == "json") {
                config->json_schema = processor_config.grammar_value;
            } else if (processor_config.grammar_type == "regex") {
                config->regex = processor_config.grammar_value;
            } else if (processor_config.grammar_type == "ebnf") {
                config->ebnf = processor_config.grammar_value;
            } else if (processor_config.grammar_type == "structural_tag") {
                config->structural_tag = processor_config.grammar_value;
            } else if (!processor_config.grammar_type.empty()) {
                return absl::InvalidArgumentError("unsupported grammar type: " + processor_config.grammar_type);
            }
            config->combo_token_size       = processor_config.combo_token_size;
            config->banned_combo_token_ids = processor_config.banned_combo_token_ids;
            config->end_think_token_ids    = processor_config.end_think_token_ids;

            auto generate_input             = std::make_shared<GenerateInput>();
            generate_input->generate_config = config;
            generate_input->input_ids       = sampling_plan.token_ids[index].narrow(0, 0, input_lengths[index]).clone();
            auto processors_result          = LogitsProcessorFactory::createLogitsProcessors(
                std::move(generate_input), 1, 1, processor_eos_token_id_);
            if (!processors_result.ok()) {
                return absl::InvalidArgumentError("failed to initialize sampling state for request_id="
                                                  + std::to_string(request_ids[index]) + ": "
                                                  + processors_result.status().ToString());
            }

            SamplingState state;
            state.logits_processors = std::move(processors_result.value());
            state.cum_log_probs     = torch::zeros({1}, torch::kFloat32);
            if (sampling_plan.random_seeds[index].has_value()) {
#if defined(USING_CUDA) || defined(USING_ROCM)
                state.generator = torch::make_generator<torch::CUDAGeneratorImpl>();
#else
                state.generator = torch::make_generator<torch::CPUGeneratorImpl>();
#endif
                state.generator.set_current_seed(sampling_plan.random_seeds[index].value());
            }
            state_it = sampling_states_.emplace(request_ids[index], std::move(state)).first;
        }

        inputs.generator[index] = state_it->second.generator;
        if (cum_log_probs != nullptr) {
            cum_log_probs[index] = state_it->second.cum_log_probs.data_ptr<float>()[0];
        }
        for (const auto& processor : state_it->second.logits_processors) {
            processor_states->insert(processor, index, index + 1);
        }
    }
    inputs.logits_processor_states_ptr = std::move(processor_states);
    return std::move(inputs);
}

void PPExecutor::advanceSamplingStates(const PPSamplingPlan& sampling_plan,
                                       const SamplerOutput&  sampler_output,
                                       PPExecutionResult&    result) {
    const auto batch_size = sampling_plan.request_ids.size(0);

    const auto*   request_ids      = sampling_plan.request_ids.data_ptr<int64_t>();
    const auto*   input_lengths    = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto*   sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    torch::Tensor cum_log_probs;
    if (sampler_output.cum_log_probs.defined()) {
        RTP_LLM_CHECK(sampler_output.cum_log_probs.dim() == 1 && sampler_output.cum_log_probs.size(0) == batch_size);
        cum_log_probs = sampler_output.cum_log_probs.to(torch::kCPU).contiguous();
    }
    const auto* success = result.sample_success.data_ptr<bool>();
    for (int64_t index = 0; index < batch_size; ++index) {
        if (!success[index] || result.processor_errors[index].has_value()) {
            continue;
        }
        auto& state = sampling_states_.at(request_ids[index]);

        std::optional<ErrorInfo> error;
        const auto               new_token = result.new_token_ids.narrow(0, index, 1);
        for (const auto& processor : state.logits_processors) {
            error = processor->updateStatus(new_token, 1);
            if (error.has_value()) {
                break;
            }
        }
        const int64_t expected_output_len = sequence_lengths[index] - input_lengths[index] + 1;
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
            result.processor_errors[index] = std::move(error);
        } else if (cum_log_probs.defined()) {
            state.cum_log_probs.copy_(cum_log_probs.narrow(0, index, 1));
        }
    }
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
        } else if (isStageRoot()) {
            auto sampler_inputs_status = makeSamplerInputs(plan.sampling_plan, plan.output_config, model_output.logits);
            RETURN_IF_STATUS_OR_ERROR(sampler_inputs_status);
            auto sampler_inputs = std::move(sampler_inputs_status.value());
            auto sampler_output = sampler_->forward(sampler_inputs);
            auto execution_result_status =
                batch_stream_processor_->makeExecutionResult(plan, model_output, sampler_output);
            RETURN_IF_STATUS_OR_ERROR(execution_result_status);
            auto execution_result = std::move(execution_result_status.value());
            advanceSamplingStates(plan.sampling_plan, sampler_output, execution_result);
            asyncSendExecutionResult(execution_result, inflight.execution_result_sends);
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
