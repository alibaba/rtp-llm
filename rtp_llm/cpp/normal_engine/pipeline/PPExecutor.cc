#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <cstring>
#include <iterator>
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
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

PPSamplingData buildSamplingData(const StreamGroups& stream_groups) {
    RTP_LLM_CHECK(!stream_groups.empty());

    const auto   all_streams = stream_groups.allStreams();
    const size_t batch_size  = all_streams.size();
    const size_t token_width = stream_groups.maxSeqLen() + 1;

    for (const auto& stream : all_streams) {
        const auto& config = *stream->generateConfig();
        RTP_LLM_CHECK_WITH_INFO(!stream->hasNumBeams() && stream->numReturnSequences() == 1,
                                "initial PP sampling does not support beam search or multiple return sequences, "
                                "request_id=%ld",
                                stream->streamId());
        RTP_LLM_CHECK_WITH_INFO(config.return_all_probs == ReturnAllProbsMode::NONE,
                                "PP does not support return_all_probs, request_id=%ld",
                                stream->streamId());
        RTP_LLM_CHECK_WITH_INFO(!config.return_softmax_probs,
                                "PP does not support return_softmax_probs, request_id=%ld",
                                stream->streamId());
        RTP_LLM_CHECK_WITH_INFO(
            !config.return_logits, "PP does not support return_logits, request_id=%ld", stream->streamId());
        const auto grammar_count = config.json_schema.has_value() + config.regex.has_value() + config.ebnf.has_value()
                                   + config.structural_tag.has_value();
        RTP_LLM_CHECK_WITH_INFO(
            grammar_count <= 1, "only one grammar constraint may be set, request_id=%ld", stream->streamId());
    }

    static const auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

    PPSamplingData data;
    data.random_seeds.reserve(batch_size);
    data.logits_processor_configs.reserve(batch_size);
    data.need_cum_log_probs   = stream_groups.needReturnCumLogProbs();
    data.request_ids          = torch::empty({static_cast<int64_t>(batch_size)}, pinned_i64);
    data.token_ids            = torch::empty({data.request_ids.size(0), static_cast<int64_t>(token_width)}, pinned_i32);
    data.input_lengths        = torch::empty(data.request_ids.sizes(), pinned_i32);
    data.sequence_lengths     = torch::empty(data.request_ids.sizes(), pinned_i32);
    data.top_k                = torch::empty(data.request_ids.sizes(), pinned_i32);
    data.top_p                = torch::empty(data.request_ids.sizes(), pinned_f32);
    data.temperature          = torch::empty(data.request_ids.sizes(), pinned_f32);
    data.repetition_penalty   = torch::empty(data.request_ids.sizes(), pinned_f32);
    data.presence_penalty     = torch::empty(data.request_ids.sizes(), pinned_f32);
    data.frequency_penalty    = torch::empty(data.request_ids.sizes(), pinned_f32);
    data.no_repeat_ngram_size = torch::empty(data.request_ids.sizes(), pinned_i32);
    data.do_sample            = torch::empty(data.request_ids.sizes(), pinned_bool);
    data.finished_mask        = torch::empty(data.request_ids.sizes(), pinned_bool);

    auto* request_ids          = data.request_ids.data_ptr<int64_t>();
    auto* input_lengths        = data.input_lengths.data_ptr<int32_t>();
    auto* sequence_lengths     = data.sequence_lengths.data_ptr<int32_t>();
    auto* top_k                = data.top_k.data_ptr<int32_t>();
    auto* top_p                = data.top_p.data_ptr<float>();
    auto* temperature          = data.temperature.data_ptr<float>();
    auto* repetition_penalty   = data.repetition_penalty.data_ptr<float>();
    auto* presence_penalty     = data.presence_penalty.data_ptr<float>();
    auto* frequency_penalty    = data.frequency_penalty.data_ptr<float>();
    auto* no_repeat_ngram_size = data.no_repeat_ngram_size.data_ptr<int32_t>();
    auto* do_sample            = data.do_sample.data_ptr<bool>();
    auto* finished_mask        = data.finished_mask.data_ptr<bool>();

    size_t batch_idx = 0;
    for (const auto& stream : all_streams) {
        const auto& config = *stream->generateConfig();
        data.random_seeds.push_back(config.random_seed);

        RequestLogitsProcessorConfig processor_config;
        if (config.json_schema.has_value()) {
            processor_config.grammar_type  = "json";
            processor_config.grammar_value = config.json_schema.value();
        } else if (config.regex.has_value()) {
            processor_config.grammar_type  = "regex";
            processor_config.grammar_value = config.regex.value();
        } else if (config.ebnf.has_value()) {
            processor_config.grammar_type  = "ebnf";
            processor_config.grammar_value = config.ebnf.value();
        } else if (config.structural_tag.has_value()) {
            processor_config.grammar_type  = "structural_tag";
            processor_config.grammar_value = config.structural_tag.value();
        }
        processor_config.combo_token_size       = config.combo_token_size;
        processor_config.banned_combo_token_ids = config.banned_combo_token_ids;
        processor_config.end_think_token_ids    = config.end_think_token_ids;
        data.logits_processor_configs.push_back(std::move(processor_config));

        const auto complete_token_ids   = stream->completeTokenIds();
        const auto seq_len              = stream->seqLength();
        request_ids[batch_idx]          = stream->streamId();
        input_lengths[batch_idx]        = stream->inputLength();
        sequence_lengths[batch_idx]     = seq_len;
        top_k[batch_idx]                = config.top_k;
        top_p[batch_idx]                = config.top_p;
        temperature[batch_idx]          = config.temperature;
        repetition_penalty[batch_idx]   = config.repetition_penalty;
        presence_penalty[batch_idx]     = config.presence_penalty;
        frequency_penalty[batch_idx]    = config.frequency_penalty;
        no_repeat_ngram_size[batch_idx] = config.no_repeat_ngram_size.value_or(0);
        do_sample[batch_idx]            = config.do_sample;
        if (!do_sample[batch_idx]) {
            top_k[batch_idx]       = 1;
            top_p[batch_idx]       = 1;
            temperature[batch_idx] = 1;
        }

        std::memcpy(data.token_ids.data_ptr<int32_t>() + batch_idx * token_width,
                    complete_token_ids.data_ptr<int32_t>(),
                    seq_len * sizeof(int32_t));
        finished_mask[batch_idx] = stream->isSubGenerateDoneWithoutLock(0);
        ++batch_idx;
    }

    return data;
}

PPSampleResult makeSampleResult(const PPSamplingData&       data,
                                const SamplerOutput&        sampler_output,
                                const std::vector<int64_t>& output_vocab_ids) {
    PPSampleResult result;
    const auto     batch_size = data.logits_processor_configs.size();

    result.request_ids = data.request_ids.clone();
    result.new_token_ids =
        sampler_output.token_ids.narrow(1, sampler_output.token_ids.size(1) - 1, 1).to(torch::kCPU).contiguous();
    if (!output_vocab_ids.empty()) {
        auto* tokens = result.new_token_ids.data_ptr<int32_t>();
        for (int64_t index = 0; index < result.new_token_ids.numel(); ++index) {
            const auto compact_token = tokens[index];
            RTP_LLM_CHECK_WITH_INFO(compact_token >= 0 && static_cast<size_t>(compact_token) < output_vocab_ids.size(),
                                    "compact output token id %d is outside configured output vocabulary size %zu",
                                    compact_token,
                                    output_vocab_ids.size());
            const auto token_id = output_vocab_ids[compact_token];
            tokens[index]       = static_cast<int32_t>(token_id);
        }
    }
    result.sample_success   = sampler_output.success.to(torch::kCPU).contiguous();
    result.cum_log_probs    = sampler_output.cum_log_probs.defined() ?
                                  sampler_output.cum_log_probs.to(torch::kCPU).contiguous() :
                                  torch::empty({0}, torch::kFloat32);
    const auto* request_ids = result.request_ids.data_ptr<int64_t>();
    auto*       success     = result.sample_success.data_ptr<bool>();
    for (size_t index = 0; index < batch_size; ++index) {
        if (sampler_output.processor_errors[index].has_value()) {
            const auto& error = sampler_output.processor_errors[index].value();
            success[index]    = false;
            result.errors.push_back({request_ids[index], static_cast<int32_t>(error.code()), error.ToString()});
        } else if (!success[index]) {
            result.errors.push_back({request_ids[index],
                                     static_cast<int32_t>(ErrorCode::UNKNOWN_ERROR),
                                     "sampler generate token id failed"});
        }
    }
    return result;
}

}  // namespace

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

void PPExecutor::asyncSendSampleResult(const PPSampleResult& result, PPTickets& tickets) {
    sendObject(pp_serialization::serializeSampleResult(result), tickets);
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

absl::Status PPExecutor::processSampleResult(InflightBatch& batch) {
    (void)batch;
    return absl::UnimplementedError("PP sample result receive and stream update are not implemented");
}

PPExecutor::PPExecutor(const EngineInitParams&                params,
                       const std::shared_ptr<KVCacheManager>& cache_manager,
                       MlaOpsType                             mla_ops_type,
                       std::function<void()>                  profile_step_start,
                       std::function<void()>                  profile_step_finish):
    parallelism_config_(params.parallelism_config),
    pp_rank_(parallelism_config_.world_rank / parallelism_config_.tp_size),
    cache_manager_(cache_manager),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    slots_(parallelism_config_.pp_size + 1) {
    RTP_LLM_FAIL(
        "PPTransport backend is not implemented: per-rank plan receive and same-TP-lane tensor P2P are required");

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
        model_ = std::make_unique<PyWrappedModel>(model_init_params, params.py_model);
    } else {
        RTP_LLM_LOG_WARNING("PPExecutor py_model is None; stage forward is unavailable");
    }

    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : CacheConfig();
    batch_stream_processor_  = std::make_unique<NormalBatchStreamProcessor>(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, cache_config, false);
    LogitsProcessorFactory::init(params.model_config_, params.grammar_config, params.sp_config.tree_decode_config);

    output_vocab_ids_       = params.model_config_.output_vocab_ids;
    processor_eos_token_id_ = params.model_config_.special_tokens.eos_token_id;
    if (!output_vocab_ids_.empty()) {
        const auto& output_vocab_ids = params.model_config_.output_vocab_ids;
        const auto  eos_it           = std::lower_bound(
            output_vocab_ids.begin(), output_vocab_ids.end(), params.model_config_.special_tokens.eos_token_id);
        RTP_LLM_CHECK_WITH_INFO(eos_it != output_vocab_ids.end()
                                    && *eos_it == params.model_config_.special_tokens.eos_token_id,
                                "primary EOS token is absent from the configured output vocabulary");
        processor_eos_token_id_ = std::distance(output_vocab_ids.begin(), eos_it);
    }

    if (isLastStage() && isStageRoot()) {
        const auto initial_sampler_batch_size =
            static_cast<size_t>(std::max<int64_t>(1, params.runtime_config.max_generate_batch_size));
        sampler_ = std::make_unique<Sampler>(SamplerInitParams{initial_sampler_batch_size, false});
    }

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        const int   first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        const auto  moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        const bool  is_gated_activation = params.model_config_.isGatedActivation();
        const auto  moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

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
                                             params.metrics_reporter,
                                             params.eplb_config);
    }
}

PPExecutor::~PPExecutor() {
    for (auto& slot : slots_) {
        if (slot.has_value()) {
            waitAll(slot->plan_sends);
            waitAll(slot->activation_sends);
            waitAll(slot->sample_result_sends);
        }
    }
}

absl::StatusOr<PPExecutionPlan> PPExecutor::buildPlan(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    StreamGroups    stream_groups(streams);
    PPExecutionPlan plan;

    auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    plan.model_input          = std::move(model_input_status.value());
    plan.model_input.skip_run = streams.empty();

    if (!plan.model_input.skip_run) {
        plan.sampling = buildSamplingData(stream_groups);
    }

    return plan;
}

absl::StatusOr<SamplerInputs> PPExecutor::makeSamplerInputs(const PPSamplingData& data, const torch::Tensor& logits) {
    const auto batch_size = data.logits_processor_configs.size();
    RTP_LLM_CHECK(logits.dim() == 2 && logits.size(0) == data.request_ids.size(0));

    SamplerInputs inputs;
    inputs.logits               = logits;
    inputs.token_ids            = data.token_ids;
    inputs.input_lengths        = data.input_lengths;
    inputs.sequence_lengths     = data.sequence_lengths;
    inputs.num_beams_in         = torch::ones(data.request_ids.sizes(), torch::kLong);
    inputs.num_beams_out        = torch::ones(data.request_ids.sizes(), torch::kLong);
    inputs.top_k                = data.top_k.clone();
    inputs.top_p                = data.top_p;
    inputs.temperature          = data.temperature;
    inputs.repetition_penalty   = data.repetition_penalty;
    inputs.presence_penalty     = data.presence_penalty;
    inputs.frequency_penalty    = data.frequency_penalty;
    inputs.no_repeat_ngram_size = data.no_repeat_ngram_size;
    inputs.do_sample            = data.do_sample;
    inputs.finished_mask        = data.finished_mask;
    if (data.need_cum_log_probs) {
        inputs.cum_log_probs = torch::empty(data.request_ids.sizes(), torch::kFloat32);
    }

    inputs.batch_size     = batch_size;
    inputs.batch_size_out = batch_size;
    inputs.step           = data.token_ids.size(1) - 1;
    inputs.vocab_size     = logits.size(-1);
    inputs.generator.resize(batch_size);
    auto processor_states = std::make_shared<LogitsProcessorStates>();

    auto*       top_k         = inputs.top_k.data_ptr<int32_t>();
    auto*       cum_log_probs = inputs.cum_log_probs.defined() ? inputs.cum_log_probs.data_ptr<float>() : nullptr;
    const auto* request_ids   = data.request_ids.data_ptr<int64_t>();
    const auto* input_lengths = data.input_lengths.data_ptr<int32_t>();
    for (size_t index = 0; index < batch_size; ++index) {
        if (top_k[index] > 0) {
            top_k[index] = std::min(top_k[index], static_cast<int32_t>(logits.size(-1)));
        }

        auto state_it = sampling_states_.find(request_ids[index]);
        if (state_it == sampling_states_.end()) {
            const auto& processor_config = data.logits_processor_configs[index];
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
            generate_input->input_ids       = data.token_ids[index].narrow(0, 0, input_lengths[index]).clone();
            auto processors_result          = LogitsProcessorFactory::createLogitsProcessors(
                std::move(generate_input), 1, 1, processor_eos_token_id_);
            if (!processors_result.ok()) {
                return absl::InvalidArgumentError("failed to initialize sampling state for request_id="
                                                  + std::to_string(request_ids[index]) + ": "
                                                  + processors_result.status().ToString());
            }

            RequestSamplingState state;
            state.logits_processors = std::move(processors_result.value());
            if (data.random_seeds[index].has_value()) {
#if defined(USING_CUDA) || defined(USING_ROCM)
                state.generator = torch::make_generator<torch::CUDAGeneratorImpl>();
#else
                state.generator = torch::make_generator<torch::CPUGeneratorImpl>();
#endif
                state.generator.set_current_seed(data.random_seeds[index].value());
            }
            state_it = sampling_states_.emplace(request_ids[index], std::move(state)).first;
        }

        inputs.generator[index] = state_it->second.generator;
        if (cum_log_probs != nullptr) {
            cum_log_probs[index] = state_it->second.cum_log_prob;
        }
        for (const auto& processor : state_it->second.logits_processors) {
            processor_states->insert(processor, index, index + 1);
        }
    }
    inputs.logits_processor_states_ptr = std::move(processor_states);
    return std::move(inputs);
}

void PPExecutor::advanceSamplingStates(const PPSamplingData& data, PPSampleResult& result) {
    const auto batch_size = data.request_ids.size(0);

    const auto* request_ids      = data.request_ids.data_ptr<int64_t>();
    const auto* input_lengths    = data.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths = data.sequence_lengths.data_ptr<int32_t>();
    const auto* cum_log_probs    = data.need_cum_log_probs ? result.cum_log_probs.data_ptr<float>() : nullptr;
    auto*       success          = result.sample_success.data_ptr<bool>();
    for (int64_t index = 0; index < batch_size; ++index) {
        if (!success[index]) {
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
            success[index] = false;
            result.errors.push_back({request_ids[index], static_cast<int32_t>(error->code()), error->ToString()});
        } else if (cum_log_probs != nullptr) {
            state.cum_log_prob = cum_log_probs[index];
        }
    }
}

absl::Status PPExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t) {
    PPExecutionPlan plan;
    if (isFirstStage()) {
        auto plan_status = buildPlan(streams);
        RETURN_IF_STATUS_OR_ERROR(plan_status);
        plan = std::move(plan_status.value());
    } else {
        plan = receivePlan();
    }

    tpSyncModelInputs(plan.model_input, parallelism_config_);
    if (plan.model_input.skip_run) {
        return absl::OkStatus();
    }

    auto& slot = slots_[current_slot_];
    if (slot.has_value()) {
        waitAll(slot->plan_sends);
        waitAll(slot->activation_sends);
        waitAll(slot->sample_result_sends);
        slot.reset();
    }

    auto& inflight = slot.emplace();
    if (isFirstStage() && isStageRoot()) {
        inflight.streams = streams;
    }

    if (!isLastStage()) {
        asyncSendPlan(plan, !isStageRoot(), inflight.plan_sends);
    }

    PPTickets             tensor_receives;
    PPIntermediateTensors input_tensors;
    if (!isFirstStage()) {
        input_tensors = receiveTensors(tensor_receives);
    }

    waitAll(tensor_receives);

    if (profile_step_start_) {
        profile_step_start_();
    }

    RTP_LLM_CHECK_WITH_INFO(model_ != nullptr, "PP stage model is not initialized");

    GptModelInputs& local_model_input = plan.model_input;
    buffer_holder_.release();
    model_->releaseBuffers();
    if (cache_manager_ && local_model_input.kv_cache_update_mapping.defined()) {
        cache_manager_->blockBatchCopy(local_model_input.kv_cache_update_mapping);
    }

    PPIntermediateTensors output_tensors;
    auto                  model_output = model_->forwardPP(
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
        auto sampler_inputs_status = makeSamplerInputs(plan.sampling, model_output.logits);
        RETURN_IF_STATUS_OR_ERROR(sampler_inputs_status);
        auto sampler_inputs = std::move(sampler_inputs_status.value());
        auto sampler_output = sampler_->forward(sampler_inputs);
        auto sample_result  = makeSampleResult(plan.sampling, sampler_output, output_vocab_ids_);
        advanceSamplingStates(plan.sampling, sample_result);
        asyncSendSampleResult(sample_result, inflight.sample_result_sends);
    }

    if (profile_step_finish_) {
        profile_step_finish_();
    }

    current_slot_ = (current_slot_ + 1) % slots_.size();

    /** TODO: lzf, process of sample result could overlap with forward. */
    if (isFirstStage() && isStageRoot()) {
        RETURN_IF_STATUS_ERROR(processSampleResult(*slots_[current_slot_]));
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
