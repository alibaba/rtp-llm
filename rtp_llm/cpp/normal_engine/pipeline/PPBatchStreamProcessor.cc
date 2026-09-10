#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <ATen/Generator.h>
#if defined(USING_CUDA) || defined(USING_ROCM)
#include <ATen/cuda/CUDAGeneratorImpl.h>
#else
#include <ATen/CPUGeneratorImpl.h>
#endif

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {

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

PPBatchStreamProcessor::PPBatchStreamProcessor(const ModelConfig&                 model_config,
                                               const PDSepConfig&                 pd_sep_config,
                                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                                               const CacheConfig&                 cache_config,
                                               bool                               warm_up,
                                               bool                               mtp_enabled):
    NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
    mtp_enabled_(mtp_enabled),
    output_vocab_ids_(model_config.output_vocab_ids),
    processor_eos_token_id_(getProcessorEosTokenId(model_config)) {}

PPSamplingPlan PPBatchStreamProcessor::gatherSamplingPlan(const StreamGroups& stream_groups) const {
    RTP_LLM_CHECK(!stream_groups.empty());

    const auto   all_streams  = stream_groups.allStreams();
    const size_t stream_count = all_streams.size();
    const size_t token_width  = stream_groups.maxSeqLen() + 1;

    int64_t total_batch_size = 0;
    for (const auto& stream : all_streams) {
        const auto& config = *stream->generateConfig();
        RTP_LLM_CHECK_WITH_INFO(!stream->hasNumBeams(),
                                "PP sampling does not support beam search, "
                                "request_id=%ld",
                                stream->streamId());
        const auto grammar_count = config.json_schema.has_value() + config.regex.has_value() + config.ebnf.has_value()
                                   + config.structural_tag.has_value();
        RTP_LLM_CHECK_WITH_INFO(
            grammar_count <= 1, "only one grammar constraint may be set, request_id=%ld", stream->streamId());
        total_batch_size += stream->currentBatchSize();
    }

    static const auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

    PPSamplingPlan sampling_plan;
    sampling_plan.random_seeds.reserve(stream_count);
    sampling_plan.logits_processor_configs.reserve(stream_count);
    sampling_plan.num_return_sequences.reserve(stream_count);
    sampling_plan.request_ids = torch::empty({static_cast<int64_t>(stream_count)}, pinned_i64);
    sampling_plan.token_ids   = torch::empty({total_batch_size, static_cast<int64_t>(token_width)}, pinned_i32);
    const std::vector<int64_t> total_batch_shape{total_batch_size};
    sampling_plan.input_lengths        = torch::empty(total_batch_shape, pinned_i32);
    sampling_plan.sequence_lengths     = torch::empty(total_batch_shape, pinned_i32);
    sampling_plan.top_k                = torch::empty(total_batch_shape, pinned_i32);
    sampling_plan.top_p                = torch::empty(total_batch_shape, pinned_f32);
    sampling_plan.temperature          = torch::empty(total_batch_shape, pinned_f32);
    sampling_plan.repetition_penalty   = torch::empty(total_batch_shape, pinned_f32);
    sampling_plan.presence_penalty     = torch::empty(total_batch_shape, pinned_f32);
    sampling_plan.frequency_penalty    = torch::empty(total_batch_shape, pinned_f32);
    sampling_plan.no_repeat_ngram_size = torch::empty(total_batch_shape, pinned_i32);
    sampling_plan.do_sample            = torch::empty(total_batch_shape, pinned_bool);
    sampling_plan.finished_mask        = torch::empty(total_batch_shape, pinned_bool);
    if (mtp_enabled_) {
        sampling_plan.spec_do_sample  = torch::empty({static_cast<int64_t>(stream_count)}, pinned_bool);
        sampling_plan.force_sp_accept = torch::empty({static_cast<int64_t>(stream_count)}, torch::kBool);
    }

    auto* request_ids          = sampling_plan.request_ids.data_ptr<int64_t>();
    auto* input_lengths        = sampling_plan.input_lengths.data_ptr<int32_t>();
    auto* sequence_lengths     = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    auto* top_k                = sampling_plan.top_k.data_ptr<int32_t>();
    auto* top_p                = sampling_plan.top_p.data_ptr<float>();
    auto* temperature          = sampling_plan.temperature.data_ptr<float>();
    auto* repetition_penalty   = sampling_plan.repetition_penalty.data_ptr<float>();
    auto* presence_penalty     = sampling_plan.presence_penalty.data_ptr<float>();
    auto* frequency_penalty    = sampling_plan.frequency_penalty.data_ptr<float>();
    auto* no_repeat_ngram_size = sampling_plan.no_repeat_ngram_size.data_ptr<int32_t>();
    auto* do_sample            = sampling_plan.do_sample.data_ptr<bool>();
    auto* finished_mask        = sampling_plan.finished_mask.data_ptr<bool>();

    size_t stream_idx = 0;
    size_t batch_idx  = 0;
    for (const auto& stream : all_streams) {
        auto& config = *stream->generateConfig();
        sampling_plan.random_seeds.push_back(config.random_seed);
        sampling_plan.num_return_sequences.push_back(config.num_return_sequences);
        if (mtp_enabled_) {
            sampling_plan.spec_do_sample.data_ptr<bool>()[stream_idx]  = !config.top1();
            sampling_plan.force_sp_accept.data_ptr<bool>()[stream_idx] = stream->forceSpAccept();
        }

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
        processor_config.combo_token_size              = config.combo_token_size;
        processor_config.banned_combo_token_ids        = config.banned_combo_token_ids;
        processor_config.end_think_token_ids           = config.end_think_token_ids;
        processor_config.enable_cross_sequence_ban     = config.enable_cross_sequence_ban;
        processor_config.cross_seq_diverge_start_combo = config.cross_seq_diverge_start_combo;
        sampling_plan.logits_processor_configs.push_back(std::move(processor_config));

        const auto complete_token_ids    = stream->completeTokenIds();
        const auto complete_token_stride = complete_token_ids.size(1);
        const auto seq_len               = stream->seqLength();
        const auto stream_batch_size     = stream->currentBatchSize();
        request_ids[stream_idx]          = stream->streamId();

        for (int sequence_idx = 0; sequence_idx < stream_batch_size; ++sequence_idx) {
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

            std::memcpy(sampling_plan.token_ids.data_ptr<int32_t>() + batch_idx * token_width,
                        complete_token_ids.data_ptr<int32_t>() + sequence_idx * complete_token_stride,
                        seq_len * sizeof(int32_t));
            finished_mask[batch_idx] = stream->isSubGenerateDoneWithoutLock(sequence_idx);
            ++batch_idx;
        }
        ++stream_idx;
    }

    return sampling_plan;
}

PPOutputConfig PPBatchStreamProcessor::gatherOutputConfig(const StreamGroups& stream_groups) const {
    PPOutputConfig output_config;
    output_config.return_all_probs = stream_groups.needReturnAllProbs();
    const auto all_streams         = stream_groups.allStreams();
    output_config.prompt_logits_requests.reserve(all_streams.size());
    for (const auto& stream : all_streams) {
        const auto& config = *stream->generateConfig();
        output_config.return_logits |= stream->returnLogits();
        output_config.return_softmax_probs |= stream->calculateSoftmaxProbs();
        output_config.return_cum_log_probs |= stream->returnCumLogProbs();
        output_config.calculate_loss |= stream->calculateLoss();
        output_config.return_hidden_states |= config.return_hidden_states;
        output_config.return_all_hidden_states |= stream->needReturnHiddenStates();
        output_config.prompt_logits_requests.push_back(PPPromptLogitsRequest{config.return_prompt_logits,
                                                                             config.prompt_logits_top_k,
                                                                             config.prompt_logits_start,
                                                                             config.prompt_logits_end,
                                                                             config.return_target_logprob});
    }
    return output_config;
}

absl::StatusOr<SamplerInputs> PPBatchStreamProcessor::gatherSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                                       const PPOutputConfig& output_config,
                                                                       const torch::Tensor&  logits,
                                                                       SamplingStates&       sampling_states,
                                                                       bool                  score_batch,
                                                                       size_t                propose_step) const {
    RTP_LLM_CHECK(score_batch || propose_step == 0);
    const auto stream_count     = sampling_plan.request_ids.size(0);
    const auto sequence_count   = sampling_plan.token_ids.size(0);
    const auto total_batch_size = score_batch ? stream_count * static_cast<int64_t>(propose_step + 1) : sequence_count;
    RTP_LLM_CHECK(logits.dim() == 2 && logits.size(0) == total_batch_size);

    auto inputs       = allocateSamplerInputs(sampling_plan, output_config, total_batch_size, propose_step);
    inputs.vocab_size = logits.size(-1);
    RETURN_IF_STATUS_ERROR(fillSamplerInputs(inputs, sampling_plan, sampling_states, score_batch, propose_step));

    inputs.logits = score_batch || output_config.return_logits || output_config.return_softmax_probs ?
                        logits.clone() :
                        logits;
    if (output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        inputs.all_probs = torch::zeros({total_batch_size, logits.size(1)}, logits.options().dtype(torch::kFloat32));
        inputs.return_original_all_probs = output_config.return_all_probs == ReturnAllProbsMode::ORIGINAL;
    }
    return inputs;
}

SamplerInputs PPBatchStreamProcessor::allocateSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                           const PPOutputConfig& output_config,
                                                           size_t                total_batch_size,
                                                           size_t                propose_step) const {
    RTP_LLM_CHECK(sampling_plan.token_ids.dim() == 2 && sampling_plan.token_ids.size(1) > 0);

    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = total_batch_size;
    sampler_inputs.batch_size_out = total_batch_size;
    sampler_inputs.step           = sampling_plan.token_ids.size(1) - 1 + propose_step;

    const auto batch_size = static_cast<int64_t>(total_batch_size);
    sampler_inputs.token_ids =
        torch::empty({batch_size, static_cast<int64_t>(sampler_inputs.step + 1)}, pinned_i32);
    sampler_inputs.input_lengths        = torch::empty({batch_size}, pinned_i32);
    sampler_inputs.sequence_lengths     = torch::empty({batch_size}, pinned_i32);
    sampler_inputs.num_beams_in         = torch::empty({batch_size}, pinned_i64);
    sampler_inputs.num_beams_out        = torch::empty({batch_size}, pinned_i64);
    sampler_inputs.top_k                = torch::empty({batch_size}, pinned_i32);
    sampler_inputs.top_p                = torch::empty({batch_size}, pinned_f32);
    sampler_inputs.temperature          = torch::empty({batch_size}, pinned_f32);
    sampler_inputs.repetition_penalty   = torch::empty({batch_size}, pinned_f32);
    sampler_inputs.presence_penalty     = torch::empty({batch_size}, pinned_f32);
    sampler_inputs.frequency_penalty    = torch::empty({batch_size}, pinned_f32);
    sampler_inputs.no_repeat_ngram_size = torch::empty({batch_size}, pinned_i32);
    sampler_inputs.do_sample            = torch::empty({batch_size}, pinned_bool);
    sampler_inputs.finished_mask        = torch::empty({batch_size}, pinned_bool);
    if (output_config.return_cum_log_probs) {
        sampler_inputs.cum_log_probs = torch::empty({batch_size}, pinned_f32);
    }
    sampler_inputs.generator.resize(total_batch_size);
    return sampler_inputs;
}

absl::Status PPBatchStreamProcessor::fillSamplerInputs(SamplerInputs&        sampler_inputs,
                                                     const PPSamplingPlan& sampling_plan,
                                                     SamplingStates&       sampling_states,
                                                     bool                  score_batch,
                                                     size_t                propose_step) const {
    const auto stream_count   = sampling_plan.request_ids.size(0);
    const auto sequence_count = sampling_plan.token_ids.size(0);
    RTP_LLM_CHECK(sampling_plan.num_return_sequences.size() == static_cast<size_t>(stream_count));
    RTP_LLM_CHECK(score_batch || propose_step == 0);

    auto* token_ids            = sampler_inputs.token_ids.data_ptr<int32_t>();
    auto* input_lengths        = sampler_inputs.input_lengths.data_ptr<int32_t>();
    auto* sequence_lengths     = sampler_inputs.sequence_lengths.data_ptr<int32_t>();
    auto* num_beams_in         = sampler_inputs.num_beams_in.data_ptr<int64_t>();
    auto* num_beams_out        = sampler_inputs.num_beams_out.data_ptr<int64_t>();
    auto* top_k                = sampler_inputs.top_k.data_ptr<int32_t>();
    auto* top_p                = sampler_inputs.top_p.data_ptr<float>();
    auto* temperature          = sampler_inputs.temperature.data_ptr<float>();
    auto* repetition_penalty   = sampler_inputs.repetition_penalty.data_ptr<float>();
    auto* presence_penalty     = sampler_inputs.presence_penalty.data_ptr<float>();
    auto* frequency_penalty    = sampler_inputs.frequency_penalty.data_ptr<float>();
    auto* no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size.data_ptr<int32_t>();
    auto* do_sample            = sampler_inputs.do_sample.data_ptr<bool>();
    auto* finished_mask        = sampler_inputs.finished_mask.data_ptr<bool>();

    const auto* plan_token_ids            = sampling_plan.token_ids.data_ptr<int32_t>();
    const auto* plan_input_lengths        = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto* plan_sequence_lengths     = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    const auto* plan_top_k                = sampling_plan.top_k.data_ptr<int32_t>();
    const auto* plan_top_p                = sampling_plan.top_p.data_ptr<float>();
    const auto* plan_temperature          = sampling_plan.temperature.data_ptr<float>();
    const auto* plan_repetition_penalty   = sampling_plan.repetition_penalty.data_ptr<float>();
    const auto* plan_presence_penalty     = sampling_plan.presence_penalty.data_ptr<float>();
    const auto* plan_frequency_penalty    = sampling_plan.frequency_penalty.data_ptr<float>();
    const auto* plan_no_repeat_ngram_size = sampling_plan.no_repeat_ngram_size.data_ptr<int32_t>();
    const auto* plan_do_sample            = sampling_plan.do_sample.data_ptr<bool>();
    const auto* plan_finished_mask        = sampling_plan.finished_mask.data_ptr<bool>();
    const auto  plan_token_stride         = sampling_plan.token_ids.stride(0);
    const auto  token_stride              = sampler_inputs.token_ids.stride(0);

    auto processor_states = std::make_shared<LogitsProcessorStates>();
    auto* cum_log_probs =
        sampler_inputs.cum_log_probs.defined() ? sampler_inputs.cum_log_probs.data_ptr<float>() : nullptr;
    const auto* request_ids = sampling_plan.request_ids.data_ptr<int64_t>();

    int64_t sequence_offset = 0;
    int64_t sampling_offset = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const auto stream_sequence_count = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
        const auto stream_sampling_rows  = score_batch ? static_cast<int64_t>(propose_step + 1) : stream_sequence_count;
        RTP_LLM_CHECK(sequence_offset + stream_sequence_count <= sequence_count);
        RTP_LLM_CHECK(sampling_offset + stream_sampling_rows <= static_cast<int64_t>(sampler_inputs.batch_size));
        RTP_LLM_CHECK(!score_batch || stream_sequence_count == 1);

        auto state_it = sampling_states.find(request_ids[stream_idx]);
        if (state_it == sampling_states.end()) {
            auto state_status = createSamplingState(sampling_plan, stream_idx, sequence_offset);
            RETURN_IF_STATUS_OR_ERROR(state_status);
            state_it = sampling_states.emplace(request_ids[stream_idx], std::move(state_status.value())).first;
        }
        const auto& state               = state_it->second;
        const auto* state_cum_log_probs = cum_log_probs ? state.cum_log_probs.data_ptr<float>() : nullptr;

        for (int64_t row = 0; row < stream_sampling_rows; ++row) {
            const auto sequence_idx = score_batch ? 0 : row;
            const auto source_row   = sequence_offset + sequence_idx;
            const auto target_row   = sampling_offset + row;
            const auto seq_len      = plan_sequence_lengths[source_row];
            RTP_LLM_CHECK(seq_len >= 0 && seq_len <= sampling_plan.token_ids.size(1)
                          && seq_len <= static_cast<int64_t>(sampler_inputs.step));

            input_lengths[target_row]        = plan_input_lengths[source_row];
            sequence_lengths[target_row]     = seq_len + static_cast<int32_t>(propose_step);
            num_beams_in[target_row]         = 1;
            num_beams_out[target_row]        = 1;
            top_k[target_row]                = plan_top_k[source_row];
            if (top_k[target_row] > 0) {
                top_k[target_row] = std::min(top_k[target_row], static_cast<int32_t>(sampler_inputs.vocab_size));
            }
            top_p[target_row]                = plan_top_p[source_row];
            temperature[target_row]          = plan_temperature[source_row];
            repetition_penalty[target_row]   = plan_repetition_penalty[source_row];
            presence_penalty[target_row]     = plan_presence_penalty[source_row];
            frequency_penalty[target_row]    = plan_frequency_penalty[source_row];
            no_repeat_ngram_size[target_row] = plan_no_repeat_ngram_size[source_row];
            do_sample[target_row]            = plan_do_sample[source_row];
            finished_mask[target_row]        = plan_finished_mask[source_row];

            sampler_inputs.generator[target_row] = state.generator;
            if (cum_log_probs) {
                cum_log_probs[target_row] = state_cum_log_probs[sequence_idx];
            }
            std::memcpy(token_ids + target_row * token_stride,
                        plan_token_ids + source_row * plan_token_stride,
                        seq_len * sizeof(int32_t));
        }
        for (const auto& processor : state.logits_processors) {
            processor_states->insert(processor, sampling_offset, sampling_offset + stream_sampling_rows);
        }
        sequence_offset += stream_sequence_count;
        sampling_offset += stream_sampling_rows;
    }
    RTP_LLM_CHECK(sequence_offset == sequence_count);
    RTP_LLM_CHECK(sampling_offset == static_cast<int64_t>(sampler_inputs.batch_size));
    sampler_inputs.logits_processor_states_ptr = std::move(processor_states);
    return absl::OkStatus();
}

absl::StatusOr<SamplingState> PPBatchStreamProcessor::createSamplingState(const PPSamplingPlan& sampling_plan,
                                                                       int64_t               stream_idx,
                                                                       int64_t               sequence_offset) const {
    const auto  stream_sequence_count = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
    const auto  request_id            = sampling_plan.request_ids.data_ptr<int64_t>()[stream_idx];
    const auto* input_lengths         = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto& processor_config      = sampling_plan.logits_processor_configs[stream_idx];
    auto        config                = std::make_shared<GenerateConfig>();
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
    config->combo_token_size              = processor_config.combo_token_size;
    config->banned_combo_token_ids        = processor_config.banned_combo_token_ids;
    config->end_think_token_ids           = processor_config.end_think_token_ids;
    config->num_return_sequences          = sampling_plan.num_return_sequences[stream_idx];
    config->enable_cross_sequence_ban     = processor_config.enable_cross_sequence_ban;
    config->cross_seq_diverge_start_combo = processor_config.cross_seq_diverge_start_combo;

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = config;
    generate_input->input_ids =
        sampling_plan.token_ids[sequence_offset].narrow(0, 0, input_lengths[sequence_offset]).clone();
    auto processors_result = LogitsProcessorFactory::createLogitsProcessors(
        std::move(generate_input), stream_sequence_count, stream_sequence_count, processor_eos_token_id_);
    if (!processors_result.ok()) {
        return absl::InvalidArgumentError("failed to initialize sampling state for request_id="
                                          + std::to_string(request_id) + ": "
                                          + processors_result.status().ToString());
    }

    SamplingState state;
    state.logits_processors = std::move(processors_result.value());
    state.cum_log_probs     = torch::zeros({stream_sequence_count}, torch::kFloat32);
    if (sampling_plan.random_seeds[stream_idx].has_value()) {
#if defined(USING_CUDA) || defined(USING_ROCM)
        state.generator = torch::make_generator<torch::CUDAGeneratorImpl>();
#else
        state.generator = torch::make_generator<torch::CPUGeneratorImpl>();
#endif
        state.generator.set_current_seed(sampling_plan.random_seeds[stream_idx].value());
    }
    return state;
}

absl::StatusOr<PPExecutionResult> PPBatchStreamProcessor::makeExecutionResult(
    const PPExecutionPlan& plan, const GptModelOutputs& model_output, const SamplerOutput& sampler_output) const {
    const auto stream_count     = plan.sampling_plan.request_ids.size(0);
    const auto total_batch_size = plan.sampling_plan.token_ids.size(0);
    if (!sampler_output.token_ids.defined() || sampler_output.token_ids.dim() != 2
        || sampler_output.token_ids.size(0) != total_batch_size || sampler_output.token_ids.size(1) == 0
        || !sampler_output.success.defined() || sampler_output.success.dim() != 1
        || sampler_output.success.size(0) != total_batch_size) {
        return absl::InternalError("sampler returned invalid tensors for PP execution result");
    }

    PPExecutionResult result;
    result.request_ids = plan.sampling_plan.request_ids.to(torch::kCPU).contiguous();
    if (plan.output_config.return_logits) {
        result.logits = model_output.logits.to(torch::kCPU).contiguous();
    }

    const auto compact_token_ids =
        sampler_output.token_ids.narrow(1, sampler_output.token_ids.size(1) - 1, 1).contiguous();
    if (plan.output_config.return_softmax_probs) {
        auto probs = model_output.logits.to(torch::kFloat32).contiguous();
#if USING_CUDA
        cudaSoftmaxInplace(probs, at::cuda::getCurrentCUDAStream().stream());
#else
        probs = torch::softmax(probs, -1);
#endif
        result.softmax_probs = probs.gather(1, compact_token_ids.to(torch::kLong)).to(torch::kCPU).contiguous();
    }

    result.new_token_ids = compact_token_ids.to(torch::kCPU).contiguous();
    if (!output_vocab_ids_.empty()) {
        auto* tokens = result.new_token_ids.data_ptr<int32_t>();
        for (int64_t index = 0; index < result.new_token_ids.numel(); ++index) {
            const auto compact_token = tokens[index];
            RTP_LLM_CHECK_WITH_INFO(compact_token >= 0 && static_cast<size_t>(compact_token) < output_vocab_ids_.size(),
                                    "compact output token id %d is outside configured output vocabulary size %zu",
                                    compact_token,
                                    output_vocab_ids_.size());
            tokens[index] = static_cast<int32_t>(output_vocab_ids_[compact_token]);
        }
    }

    result.sample_success = sampler_output.success.to(torch::kCPU).contiguous();
    if (plan.output_config.return_cum_log_probs) {
        result.cum_log_probs = sampler_output.cum_log_probs.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        result.all_probs = sampler_output.all_probs.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_hidden_states) {
        result.hidden_states = model_output.hidden_states.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_all_hidden_states) {
        result.all_hidden_states = model_output.all_hidden_states.to(torch::kCPU).contiguous();
    }
    result.prompt_logits.resize(stream_count);

    /** Prompt loss and prompt logits are stream-level; return sequences share the first batch row. */
    if (plan.model_input.need_all_logits) {
        const int64_t decode_batch_size =
            plan.model_input.sequence_lengths.defined() ? plan.model_input.sequence_lengths.size(0) : 0;
        const auto lm_output_indexes = plan.model_input.lm_output_indexes.to(torch::kCPU, torch::kInt64).contiguous();
        RTP_LLM_CHECK(lm_output_indexes.numel() == total_batch_size);
        const auto*                indexes = lm_output_indexes.data_ptr<int64_t>();
        std::vector<torch::Tensor> losses;
        losses.reserve(stream_count);
        int64_t batch_idx = 0;
        for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            const int64_t stream_batch_size = std::max(plan.sampling_plan.num_return_sequences[stream_idx], 1);
            const int64_t start             = batch_idx == 0 ? 0 : indexes[batch_idx - 1] + 1;
            const int64_t end               = indexes[batch_idx] + 1;
            const int64_t token_size        = end - start;
            const int64_t loss_size         = token_size - 1;
            if (plan.output_config.calculate_loss && loss_size > 0) {
                auto labels = plan.model_input.combo_tokens.narrow(0, start + 1, loss_size)
                                  .to(model_output.all_logits.device(), torch::kLong);
                losses.push_back(torch::cross_entropy_loss(model_output.all_logits.narrow(0, start, loss_size),
                                                           labels,
                                                           torch::nullopt,
                                                           at::Reduction::None)
                                     .to(torch::kFloat32));
            }

            const auto& prompt_request = plan.output_config.prompt_logits_requests[stream_idx];
            if (batch_idx >= decode_batch_size && prompt_request.enabled) {
                auto request_logits = model_output.all_logits.narrow(0, start, token_size);
                auto request_tokens = plan.model_input.combo_tokens.narrow(0, start, token_size);
                auto output         = makePromptLogitsOutput(request_logits,
                                                     request_tokens,
                                                     prompt_request.top_k,
                                                     prompt_request.start,
                                                     prompt_request.end,
                                                     prompt_request.return_target_logprob);
                if (output.has_value()) {
                    result.prompt_logits[stream_idx] = std::move(output.value());
                }
            }
            batch_idx += stream_batch_size;
        }

        if (plan.output_config.calculate_loss && !losses.empty()) {
            result.loss = torch::cat(losses).to(torch::kCPU).contiguous();
        }
    }

    result.processor_errors = sampler_output.processor_errors;
    result.processor_errors.resize(total_batch_size);
    return result;
}

absl::Status PPBatchStreamProcessor::dispatchExecutionResult(const StreamGroups& stream_groups,
                                                             const PPExecutionResult& result) const {
    if (mtp_enabled_) {
        return dispatchMtpExecutionResult(stream_groups, result);
    }
    return dispatchNormalExecutionResult(stream_groups, result);
}

PPBatchStreamProcessor::ExecutionResultLayout
PPBatchStreamProcessor::validateExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                                const PPExecutionResult& result) const {
    const auto stream_count = static_cast<int64_t>(all_streams.size());
    RTP_LLM_CHECK_WITH_INFO(result.request_ids.defined() && result.request_ids.device().is_cpu()
                                && result.request_ids.scalar_type() == torch::kInt64
                                && result.request_ids.dim() == 1 && result.request_ids.size(0) == stream_count,
                            "PP execution result request count does not match the inflight stream count");
    RTP_LLM_CHECK_WITH_INFO(result.prompt_logits.size() == static_cast<size_t>(stream_count),
                            "PP prompt-logits result count does not match the inflight stream count");

    ExecutionResultLayout layout;
    const auto* request_ids = result.request_ids.data_ptr<int64_t>();
    int64_t stream_idx = 0;
    for (const auto& stream : all_streams) {
        const auto batch_size = static_cast<int64_t>(stream->currentBatchSize());
        const auto token_size = static_cast<int64_t>(stream->currentExecuteTokenSize());
        RTP_LLM_CHECK(batch_size > 0);
        layout.batch_size += batch_size;
        layout.token_size += token_size;
        layout.loss_size += std::max<int64_t>(token_size / batch_size - 1, 0);
        RTP_LLM_CHECK_WITH_INFO(request_ids[stream_idx] == stream->streamId(),
                                "PP execution result request order does not match the inflight stream order");
        ++stream_idx;
    }

    RTP_LLM_CHECK_WITH_INFO(
        result.new_token_ids.defined() && result.new_token_ids.device().is_cpu()
            && result.new_token_ids.scalar_type() == torch::kInt32 && result.new_token_ids.dim() == 2
            && result.new_token_ids.size(0) == layout.batch_size
            && result.sample_success.defined() && result.sample_success.device().is_cpu()
            && result.sample_success.scalar_type() == torch::kBool && result.sample_success.dim() == 1
            && result.sample_success.size(0) == layout.batch_size
            && result.processor_errors.size() == static_cast<size_t>(layout.batch_size),
        "PP execution result has invalid token, success or error fields");
    return layout;
}

void PPBatchStreamProcessor::validateRequestedOutputs(const PPOutputConfig& output_config,
                                                      const PPExecutionResult& result,
                                                      const ExecutionResultLayout& layout) const {
    const auto valid_batch_matrix = [&layout](bool requested, const torch::Tensor& tensor) {
        return !requested || (tensor.defined() && tensor.dim() == 2 && tensor.size(0) == layout.batch_size);
    };
    const bool valid_shapes =
        valid_batch_matrix(output_config.return_hidden_states, result.hidden_states)
        && valid_batch_matrix(output_config.return_logits, result.logits)
        && valid_batch_matrix(output_config.return_all_probs != ReturnAllProbsMode::NONE, result.all_probs)
        && (!output_config.return_softmax_probs
            || (result.softmax_probs.defined() && result.softmax_probs.dim() == 2
                && result.softmax_probs.size(0) == layout.batch_size && result.softmax_probs.size(1) == 1))
        && (!output_config.return_cum_log_probs
            || (result.cum_log_probs.defined() && result.cum_log_probs.dim() == 1
                && result.cum_log_probs.size(0) == layout.batch_size))
        && (!output_config.return_all_hidden_states
            || (result.all_hidden_states.defined() && result.all_hidden_states.dim() == 2
                && result.all_hidden_states.size(0) == layout.token_size))
        && (!output_config.calculate_loss || layout.loss_size == 0
            || (result.loss.defined() && result.loss.dim() == 1 && result.loss.numel() == layout.loss_size));
    RTP_LLM_CHECK_WITH_INFO(valid_shapes,
                            "PP execution result is missing a requested tensor or has invalid tensor shapes");
}

absl::Status PPBatchStreamProcessor::dispatchNormalExecutionResult(const StreamGroups& stream_groups,
                                                                   const PPExecutionResult& result) const {
    const auto all_streams = stream_groups.allStreams();
    const auto layout = validateExecutionResult(all_streams, result);
    RTP_LLM_CHECK_WITH_INFO(result.new_token_ids.size(1) == 1
                                && !result.accept_len.defined() && !result.propose_token_ids.defined(),
                            "ordinary PP execution result must not contain speculative fields");
    validateRequestedOutputs(gatherOutputConfig(stream_groups), result, layout);

    int64_t batch_idx = 0;
    int64_t stream_idx = 0;
    int64_t token_offset = 0;
    int64_t loss_offset = 0;
    for (const auto& stream : all_streams) {
        const auto stream_batch_size = static_cast<int64_t>(stream->currentBatchSize());
        const auto token_size = static_cast<int64_t>(stream->currentExecuteTokenSize());
        const auto loss_size = std::max<int64_t>(token_size / stream_batch_size - 1, 0);
        auto error_info =
            collectStreamSamplerError(result.processor_errors, result.sample_success, batch_idx, stream_batch_size);
        dispatchNormalSingleStream(
            stream, result, stream_idx, batch_idx, stream_batch_size, token_offset, loss_offset, std::move(error_info));
        stream->clearPPInflight();
        ++stream_idx;
        batch_idx += stream_batch_size;
        token_offset += token_size;
        loss_offset += loss_size;
    }
    return absl::OkStatus();
}

void PPBatchStreamProcessor::dispatchNormalSingleStream(const GenerateStreamPtr& stream,
                                                        const PPExecutionResult& result,
                                                        int64_t stream_idx,
                                                        int64_t batch_idx,
                                                        int64_t stream_batch_size,
                                                        int64_t token_offset,
                                                        int64_t loss_offset,
                                                        std::optional<ErrorInfo> error_info) const {
    const auto token_size = static_cast<int64_t>(stream->currentExecuteTokenSize());
    const auto loss_size = std::max<int64_t>(token_size / stream_batch_size - 1, 0);
    torch::Tensor hidden_states;
    if (stream->generateConfig()->return_hidden_states) {
        hidden_states = result.hidden_states.narrow(0, batch_idx, stream_batch_size).clone();
    }
    torch::Tensor logits;
    if (stream->returnLogits()) {
        logits = result.logits.narrow(0, batch_idx, stream_batch_size).clone();
    }
    torch::Tensor softmax_probs;
    if (stream->calculateSoftmaxProbs()) {
        softmax_probs = result.softmax_probs.narrow(0, batch_idx, stream_batch_size).clone();
    }
    torch::Tensor cum_log_probs;
    if (stream->returnCumLogProbs()) {
        cum_log_probs = result.cum_log_probs.narrow(0, batch_idx, stream_batch_size).clone();
    }
    torch::Tensor all_probs;
    if (stream->getReturnAllProbs() != ReturnAllProbsMode::NONE) {
        all_probs = result.all_probs.narrow(0, batch_idx, stream_batch_size).clone();
    }
    torch::Tensor loss;
    if (stream->calculateLoss() && loss_size > 0) {
        loss = result.loss.narrow(0, loss_offset, loss_size).clone();
    }
    torch::Tensor all_hidden_states;
    if (stream->needReturnHiddenStates()) {
        all_hidden_states = result.all_hidden_states.narrow(0, token_offset, token_size).clone();
    }
    auto prompt_logits = result.prompt_logits[stream_idx];
    stream->updateFromPP({result.new_token_ids.narrow(0, batch_idx, stream_batch_size),
                          1,
                          std::move(hidden_states),
                          std::move(logits),
                          std::move(softmax_probs),
                          std::move(cum_log_probs),
                          std::move(all_probs),
                          std::move(loss),
                          torch::Tensor(),
                          std::move(all_hidden_states),
                          true,
                          false,
                          std::move(prompt_logits),
                          std::move(error_info)});
}

torch::Tensor PPBatchStreamProcessor::gatherDraftNextPositionIds(const StreamGroups&   stream_groups,
                                                               const GptModelInputs& model_input) const {
    if (!mtp_enabled_ || model_input.is_target_verify || !model_input.combo_position_ids.defined()) {
        return {};
    }

    const auto streams         = stream_groups.allStreams();
    const auto position_factor = model_input_gatherer_config_.position_id_len_factor;
    auto position_ids = torch::empty({static_cast<int64_t>(streams.size() * position_factor)},
                                     torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    size_t row = 0;
    for (const auto& stream : streams) {
        stream->generateNextPositionId(position_ids.data_ptr<int32_t>() + row * position_factor);
        ++row;
    }
    return position_ids;
}

absl::StatusOr<GptModelInputs> PPBatchStreamProcessor::gatherTargetVerifyModelInput(
    const StreamGroups& stream_groups, size_t propose_step, TensorHolder& host_holder) const {
    auto model_input_status = gatherModelInput(stream_groups, host_holder);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto model_input = std::move(model_input_status.value());
    const auto batch_size = static_cast<int64_t>(stream_groups.size());
    const auto score_len = static_cast<int64_t>(propose_step + 1);
    const auto position_factor = model_input_gatherer_config_.position_id_len_factor;
    auto tokens = torch::empty({batch_size, score_len}, torch::kInt32).pin_memory();
    tokens.select(1, 0).copy_(model_input.combo_tokens.to(torch::kCPU));
    if (model_input.combo_position_ids.defined()) {
        model_input.combo_position_ids =
            torch::empty({batch_size * score_len * static_cast<int64_t>(position_factor)}, torch::kInt32).pin_memory();
    }

    int64_t row = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        const auto& proposals = stream->getSPOutputBuffer()->tokens;
        tokens[row].narrow(0, 1, propose_step).copy_(
            proposals.flatten().narrow(0, proposals.numel() - propose_step, propose_step));
        if (model_input.combo_position_ids.defined()) {
            auto* positions = model_input.combo_position_ids.data_ptr<int32_t>() + row * score_len * position_factor;
            stream->generateNextPositionId(positions);
            for (int64_t step = 1; step < score_len; ++step) {
                for (size_t dim = 0; dim < position_factor; ++dim) {
                    positions[step * position_factor + dim] = positions[dim] + step;
                }
            }
        }
        ++row;
    }
    host_holder.hold_host(tokens);
    model_input.combo_tokens = tokens.reshape({-1}).to(torch::kCUDA, /*non_blocking=*/true);
    model_input.prefix_lengths = model_input.sequence_lengths.clone();
    model_input.sequence_lengths = torch::empty({0}, model_input.sequence_lengths.options());
    model_input.input_lengths = torch::full({batch_size}, score_len, model_input.input_lengths.options());
    model_input.lm_output_indexes = torch::arange(batch_size * score_len, model_input.input_lengths.options());
    model_input.is_target_verify = true;
    return model_input;
}

void PPBatchStreamProcessor::validateMtpExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                                       const PPExecutionResult& result,
                                                       const ExecutionResultLayout& layout) const {
    const bool is_prefill = all_streams.empty() || all_streams.front()->isContextStream();
    RTP_LLM_CHECK_WITH_INFO(
        layout.batch_size == static_cast<int64_t>(all_streams.size())
            && result.accept_len.defined() && result.accept_len.device().is_cpu()
            && result.accept_len.scalar_type() == torch::kInt32 && result.accept_len.dim() == 1
            && result.accept_len.size(0) == layout.batch_size
            && result.propose_token_ids.defined() && result.propose_token_ids.device().is_cpu()
            && result.propose_token_ids.scalar_type() == torch::kInt32 && result.propose_token_ids.dim() == 2
            && result.propose_token_ids.size(0) == layout.batch_size && result.propose_token_ids.size(1) > 0
            && result.new_token_ids.size(1) == (is_prefill ? 1 : result.propose_token_ids.size(1) + 1),
        "PP MTP execution result requires single-sequence accepted-token and proposal fields");

    const auto* accept_lengths = result.accept_len.data_ptr<int32_t>();
    const auto* success = result.sample_success.data_ptr<bool>();
    int64_t row = 0;
    for (const auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->isContextStream() == is_prefill,
                                "PP MTP result batch must contain only prefill or only decode requests");
        // Failed rows are reported through StreamSpecUpdateInfo, without indexing
        // a possibly invalid accepted-token window.
        if (success[row] && !result.processor_errors[row].has_value()) {
            RTP_LLM_CHECK_WITH_INFO(accept_lengths[row] >= 1
                                        && accept_lengths[row] <= result.new_token_ids.size(1)
                                        && (!is_prefill || accept_lengths[row] == 1),
                                    "PP MTP execution result has an invalid accepted length");
            const auto sp_output_buffer = stream->getSPOutputBuffer();
            RTP_LLM_CHECK_WITH_INFO(sp_output_buffer
                                        && sp_output_buffer->propose_step
                                               == static_cast<size_t>(result.propose_token_ids.size(1)),
                                    "PP MTP execution result does not match the prepared proposal count");
        }
        ++row;
    }
}

absl::Status PPBatchStreamProcessor::dispatchMtpExecutionResult(const StreamGroups& stream_groups,
                                                                const PPExecutionResult& result) const {
    const auto all_streams = stream_groups.allStreams();
    const auto layout = validateExecutionResult(all_streams, result);
    validateMtpExecutionResult(all_streams, result, layout);

    int64_t row = 0;
    for (const auto& stream : all_streams) {
        auto error_info = collectStreamSamplerError(result.processor_errors, result.sample_success, row, 1);
        const int accepted_length = error_info.has_value() ? 0 : result.accept_len.data_ptr<int32_t>()[row];
        const auto draft_tokens = error_info.has_value() ? torch::Tensor() : result.propose_token_ids[row];
        auto new_tokens = result.new_token_ids.narrow(0, row, 1);
        if (!error_info.has_value()) {
            new_tokens = new_tokens.narrow(1, 0, accepted_length).contiguous();
        }

        stream->specUpdate({std::move(new_tokens),
                            accepted_length,
                            draft_tokens,
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor(),
                            true,
                            false,
                            std::move(error_info)},
                           false);
        stream->clearPPInflight();
        ++row;
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
