#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {

namespace {

bool isSupportedGrammarType(const std::string& type) {
    return type == "json" || type == "regex" || type == "ebnf" || type == "structural_tag";
}

}  // namespace

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

static RequestLogitsProcessorConfig makeLogitsProcessorConfig(const GenerateConfig& config) {
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
    return processor_config;
}

PPBatchStreamProcessor::PPBatchStreamProcessor(const ModelConfig&                 model_config,
                                               const PDSepConfig&                 pd_sep_config,
                                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                                               const CacheConfig&                 cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               bool                               warm_up):
    NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
    sp_enabled_(sp_config.type != SP_TYPE_NONE),
    is_dspark_(sp_config.type == SP_TYPE_DSPARK),
    propose_step_(sp_config.gen_num_per_cycle),
    dspark_mask_token_id_(static_cast<int32_t>(sp_config.sp_dspark_mask_token_id)),
    dspark_sample_from_anchor_(sp_config.sp_dspark_sample_from_anchor),
    output_vocab_ids_(model_config.output_vocab_ids),
    processor_eos_token_id_(getProcessorEosTokenId(model_config)) {}

absl::StatusOr<PPExecutionPlan>
PPBatchStreamProcessor::gatherExecutionPlan(const StreamGroups&                     stream_groups,
                                            const std::vector<RequestId>&           finished_request_ids,
                                            bool                                    shutdown,
                                            std::unordered_map<RequestId, int32_t>& token_history_lengths,
                                            TensorHolder&                           host_holder) const {
    const auto    streams          = stream_groups.allStreams();
    const int64_t stream_count     = streams.size();
    const int64_t total_batch_size = stream_groups.totalModelBatchSize();
    const bool    is_decode        = !stream_groups.decodeStreams().empty();
    const bool    need_verify      = sp_enabled_ && is_decode;
    const int64_t position_factor  = model_input_gatherer_config_.position_id_len_factor;

    static const auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

    PPExecutionPlan plan;
    auto&           model_input = plan.model_input;
    {
        auto model_input_status = gatherModelInput(stream_groups, host_holder);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input = std::move(model_input_status.value());
        /** PP rejects TorchSpec capture at construction; MTP features use their separate model outputs. */
        model_input.capture_hidden_states = false;
        model_input.skip_run              = stream_groups.empty();
        model_input.shutdown              = shutdown;

        if (need_verify) {
            fillTargetVerifyModelInput(streams, model_input);
        }
    }

    plan.finished_request_ids = finished_request_ids;
    plan.is_decode            = is_decode;

    if (model_input.skip_run) {
        return plan;
    }

    const bool need_sampling = !model_input.is_fake_stream && !model_input.skip_lm_head;
    auto&      sampling_plan = plan.sampling_plan;
    {
        sampling_plan.random_seeds.reserve(stream_count);
        sampling_plan.logits_processor_configs.reserve(stream_count);
        sampling_plan.num_return_sequences.reserve(stream_count);
        sampling_plan.request_ids = torch::empty({stream_count}, pinned_i64);
        /** The draft chain still consumes request metadata when target sampling is skipped. */
        if (need_sampling) {
            int64_t total_token_count = 0;
            for (const auto& stream : streams) {
                const auto history_it  = token_history_lengths.find(stream->streamId());
                const auto base_length = history_it == token_history_lengths.end() ? 0 : history_it->second;
                RTP_LLM_CHECK(base_length <= stream->seqLength());
                total_token_count +=
                    static_cast<int64_t>(stream->seqLength() - base_length) * stream->currentBatchSize();
            }
            sampling_plan.token_ids                            = torch::empty({total_token_count}, pinned_i32);
            sampling_plan.token_offsets                        = torch::empty({total_batch_size + 1}, pinned_i64);
            sampling_plan.token_offsets.data_ptr<int64_t>()[0] = 0;
        }

        const std::vector<int64_t> total_batch_shape{total_batch_size};
        sampling_plan.input_lengths        = torch::empty(total_batch_shape, pinned_i32);
        sampling_plan.sequence_lengths     = torch::empty(total_batch_shape, pinned_i32);
        sampling_plan.max_tokens           = torch::empty(total_batch_shape, pinned_i32);
        sampling_plan.top_k                = torch::empty(total_batch_shape, pinned_i32);
        sampling_plan.top_p                = torch::empty(total_batch_shape, pinned_f32);
        sampling_plan.temperature          = torch::empty(total_batch_shape, pinned_f32);
        sampling_plan.repetition_penalty   = torch::empty(total_batch_shape, pinned_f32);
        sampling_plan.presence_penalty     = torch::empty(total_batch_shape, pinned_f32);
        sampling_plan.frequency_penalty    = torch::empty(total_batch_shape, pinned_f32);
        sampling_plan.no_repeat_ngram_size = torch::empty(total_batch_shape, pinned_i32);
        sampling_plan.do_sample            = torch::empty(total_batch_shape, pinned_bool);
        sampling_plan.finished_mask        = torch::empty(total_batch_shape, pinned_bool);
        if (sp_enabled_) {
            sampling_plan.spec_do_sample  = torch::empty({stream_count}, pinned_bool);
            sampling_plan.force_sp_accept = torch::empty({stream_count}, torch::kBool);
        }
    }

    auto& output_config = plan.output_config;
    {
        output_config.requests.reserve(stream_count);
        if (!sp_enabled_) {
            output_config.return_all_probs = stream_groups.needReturnAllProbs();
        }
    }

    if (sp_enabled_ && !need_verify && model_input.combo_position_ids.defined()) {
        plan.draft_next_position_ids = torch::empty({stream_count * position_factor}, pinned_i32);
    }

    auto* request_ids          = sampling_plan.request_ids.data_ptr<RequestId>();
    auto* input_lengths        = sampling_plan.input_lengths.data_ptr<int32_t>();
    auto* sequence_lengths     = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    auto* max_tokens           = sampling_plan.max_tokens.data_ptr<int32_t>();
    auto* top_k                = sampling_plan.top_k.data_ptr<int32_t>();
    auto* top_p                = sampling_plan.top_p.data_ptr<float>();
    auto* temperature          = sampling_plan.temperature.data_ptr<float>();
    auto* repetition_penalty   = sampling_plan.repetition_penalty.data_ptr<float>();
    auto* presence_penalty     = sampling_plan.presence_penalty.data_ptr<float>();
    auto* frequency_penalty    = sampling_plan.frequency_penalty.data_ptr<float>();
    auto* no_repeat_ngram_size = sampling_plan.no_repeat_ngram_size.data_ptr<int32_t>();
    auto* do_sample            = sampling_plan.do_sample.data_ptr<bool>();
    auto* finished_mask        = sampling_plan.finished_mask.data_ptr<bool>();
    auto* token_ids            = need_sampling ? sampling_plan.token_ids.data_ptr<int32_t>() : nullptr;
    auto* token_offsets        = need_sampling ? sampling_plan.token_offsets.data_ptr<int64_t>() : nullptr;
    auto* spec_do_sample       = sp_enabled_ ? sampling_plan.spec_do_sample.data_ptr<bool>() : nullptr;
    auto* force_sp_accept      = sp_enabled_ ? sampling_plan.force_sp_accept.data_ptr<bool>() : nullptr;

    auto* draft_next_position_ids =
        plan.draft_next_position_ids.defined() ? plan.draft_next_position_ids.data_ptr<int32_t>() : nullptr;

    size_t stream_idx = 0;
    size_t batch_idx  = 0;
    /** Collect sampling, output and draft position fields in one traversal. */
    for (const auto& stream : streams) {
        const auto& config     = *stream->generateConfig();
        const auto  request_id = stream->streamId();
        const auto  seq_len    = stream->seqLength();
        {
            request_ids[stream_idx] = request_id;
            sampling_plan.random_seeds.push_back(config.random_seed);
            sampling_plan.num_return_sequences.push_back(config.num_return_sequences);
            if (sp_enabled_) {
                spec_do_sample[stream_idx]  = config.stochastic();
                force_sp_accept[stream_idx] = stream->forceSpAccept();
            }
            sampling_plan.logits_processor_configs.push_back(makeLogitsProcessorConfig(config));

            float sampling_temperature = config.do_sample ? config.temperature : 1.0f;
            /** Invalid temperatures distort target probabilities; normalize the shared base before verification. */
            if (is_dspark_ && need_verify && (!std::isfinite(sampling_temperature) || sampling_temperature < 0.0f)) {
                sampling_temperature = 1.0f;
            }

            const auto max_token_num     = static_cast<int32_t>(stream->maxTokenNum());
            const auto stream_batch_size = stream->currentBatchSize();
            const auto input_length      = stream->inputLength();
            const auto history_it        = token_history_lengths.find(request_id);
            const auto base_length       = history_it == token_history_lengths.end() ? 0 : history_it->second;
            const auto delta_length      = seq_len - base_length;
            for (int sequence_idx = 0; sequence_idx < stream_batch_size; ++sequence_idx) {
                input_lengths[batch_idx]        = input_length;
                sequence_lengths[batch_idx]     = seq_len;
                max_tokens[batch_idx]           = max_token_num;
                top_k[batch_idx]                = config.do_sample ? config.top_k : 1;
                top_p[batch_idx]                = config.do_sample ? config.top_p : 1.0f;
                temperature[batch_idx]          = sampling_temperature;
                repetition_penalty[batch_idx]   = config.repetition_penalty;
                presence_penalty[batch_idx]     = config.presence_penalty;
                frequency_penalty[batch_idx]    = config.frequency_penalty;
                no_repeat_ngram_size[batch_idx] = config.no_repeat_ngram_size.value_or(0);
                do_sample[batch_idx]            = config.do_sample;
                finished_mask[batch_idx]        = stream->isSubGenerateDoneWithoutLock(sequence_idx);

                if (need_sampling) {
                    token_offsets[batch_idx + 1] = token_offsets[batch_idx] + delta_length;
                    if (delta_length > 0) {
                        const auto complete_token_ids = stream->completeTokenIds();
                        std::memcpy(token_ids + token_offsets[batch_idx],
                                    complete_token_ids.data_ptr<int32_t>() + sequence_idx * complete_token_ids.size(1)
                                        + base_length,
                                    delta_length * sizeof(int32_t));
                    }
                }
                ++batch_idx;
            }
            if (need_sampling) {
                token_history_lengths[request_id] = seq_len;
            }
        }

        {
            auto& request = output_config.requests.emplace_back();
            /** SP verify rows include rejected tokens; these outputs require alignment with committed tokens. */
            if (!sp_enabled_) {
                request.return_logits            = stream->returnLogits();
                request.return_hidden_states     = config.return_hidden_states;
                request.return_all_hidden_states = stream->needReturnHiddenStates();
                request.return_cum_log_probs     = stream->returnCumLogProbs();
                request.return_all_probs         = stream->getReturnAllProbs() != ReturnAllProbsMode::NONE;
            }
            request.return_softmax_probs = stream->calculateSoftmaxProbs();
            request.calculate_loss       = stream->calculateLoss();
            request.prompt_logits        = {config.return_prompt_logits,
                                            config.prompt_logits_top_k,
                                            config.prompt_logits_start,
                                            config.prompt_logits_end,
                                            config.return_target_logprob};
            output_config.return_logits |= request.return_logits;
            output_config.return_softmax_probs |= request.return_softmax_probs;
            output_config.return_cum_log_probs |= request.return_cum_log_probs;
            output_config.calculate_loss |= request.calculate_loss;
            output_config.return_hidden_states |= request.return_hidden_states;
            output_config.return_all_hidden_states |= request.return_all_hidden_states;
        }

        if (plan.draft_next_position_ids.defined()) {
            stream->generateNextPositionId(draft_next_position_ids + stream_idx * position_factor);
        }
        ++stream_idx;
    }
    return plan;
}

void PPBatchStreamProcessor::fillTargetVerifyModelInput(const std::list<GenerateStreamPtr>& streams,
                                                        GptModelInputs&                     model_input) const {
    const int64_t     stream_count    = streams.size();
    const int64_t     position_factor = model_input_gatherer_config_.position_id_len_factor;
    static const auto pinned_i32      = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);

    /** Verification candidates extend model input; sampling history contains only committed tokens. */
    const auto score_len     = propose_step_ + 1;
    auto       verify_tokens = torch::empty({stream_count, score_len}, pinned_i32);
    verify_tokens.select(1, 0).copy_(model_input.combo_tokens);
    torch::Tensor verify_position_ids;
    if (model_input.combo_position_ids.defined()) {
        verify_position_ids = torch::empty({stream_count * score_len * position_factor}, pinned_i32);
    }

    int64_t stream_idx = 0;
    for (const auto& stream : streams) {
        const auto& proposals = stream->getSPOutputBuffer()->tokens;
        verify_tokens[stream_idx]
            .narrow(0, 1, propose_step_)
            .copy_(proposals.flatten().narrow(0, proposals.numel() - propose_step_, propose_step_));
        if (verify_position_ids.defined()) {
            const auto* base_positions =
                model_input.combo_position_ids.data_ptr<int32_t>() + stream_idx * position_factor;
            auto* positions = verify_position_ids.data_ptr<int32_t>() + stream_idx * score_len * position_factor;
            for (int64_t step = 0; step < score_len; ++step) {
                for (int64_t dim = 0; dim < position_factor; ++dim) {
                    positions[step * position_factor + dim] = base_positions[dim] + step;
                }
            }
        }
        ++stream_idx;
    }

    if (model_input.engram_token_windows.defined()) {
        model_input.engram_token_windows =
            mtp::makeEngramVerifyWindows(model_input.engram_token_windows, verify_tokens);
    }
    model_input.combo_tokens       = verify_tokens.reshape({-1});
    model_input.combo_position_ids = std::move(verify_position_ids);
    model_input.prefix_lengths     = model_input.sequence_lengths.clone();
    model_input.sequence_lengths   = torch::empty({0}, model_input.sequence_lengths.options());
    model_input.input_lengths      = torch::full({stream_count}, score_len, model_input.input_lengths.options());
    model_input.lm_output_indexes  = torch::arange(stream_count * score_len, model_input.input_lengths.options());
    model_input.is_target_verify   = true;
}

GrammarPrefetchContext PPBatchStreamProcessor::prepareLogitsProcessorsAsync(
    const PPSamplingPlan& sampling_plan, const SamplingStates& sampling_states) const {
    const auto stream_count = sampling_plan.request_ids.size(0);
    RTP_LLM_CHECK_WITH_INFO(sampling_plan.logits_processor_configs.size() == static_cast<size_t>(stream_count),
                            "PP grammar prefetch config count %zu does not match request count %ld",
                            sampling_plan.logits_processor_configs.size(),
                            stream_count);

    std::vector<GrammarKeyCpp> keys;
    keys.reserve(stream_count);
    const auto* request_ids = sampling_plan.request_ids.data_ptr<int64_t>();
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        if (sampling_states.find(request_ids[stream_idx]) != sampling_states.end()) {
            continue;
        }
        const auto& config = sampling_plan.logits_processor_configs[stream_idx];
        if (isSupportedGrammarType(config.grammar_type)) {
            keys.push_back({config.grammar_type, config.grammar_value});
        }
    }
    if (keys.empty()) {
        return {};
    }
    return LogitsProcessorFactory::prefetchGrammars(keys);
}

void PPBatchStreamProcessor::prepareSampling(const PPExecutionPlan&                              plan,
                                             const std::unordered_map<RequestId, torch::Tensor>& dspark_q,
                                             const torch::Tensor&                                d2t_map,
                                             SamplingStates&                                     sampling_states,
                                             PPExecutionResult&                                  result,
                                             SamplerOutput&                                      draft_sampler_output,
                                             speculative::SpeculativeSamplingParams&             verify_params,
                                             GrammarPrefetchContext*                             grammar_prefetch) {
    const auto& sampling_plan = plan.sampling_plan;
    const auto  stream_count  = sampling_plan.request_ids.size(0);
    const auto* request_ids   = sampling_plan.request_ids.data_ptr<RequestId>();
    const bool  is_verify     = sp_enabled_ && plan.model_input.is_target_verify;

    if (is_verify) {
        /** Exclude the committed anchor from proposal verification. */
        draft_sampler_output.token_ids =
            plan.model_input.combo_tokens.reshape({stream_count, propose_step_ + 1}).slice(1, 1).contiguous();
        draft_sampler_output.token_ids_are_point_mass = !is_dspark_;
        draft_sampler_output.all_probs                = torch::Tensor();

        verify_params.do_sample    = sampling_plan.spec_do_sample;
        verify_params.force_accept = sampling_plan.force_sp_accept;
        verify_params.generators.clear();
        verify_params.generators.reserve(stream_count);

        if (is_dspark_) {
            const int64_t target_vocab_size = model_input_gatherer_config_.vocab_size;
            if (!dspark_verify_q_buffer_.defined() || dspark_verify_q_buffer_.size(0) < stream_count) {
                dspark_verify_q_buffer_ =
                    torch::empty({stream_count, propose_step_, target_vocab_size},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }
            draft_sampler_output.all_probs = dspark_verify_q_buffer_.narrow(0, 0, stream_count);
            draft_sampler_output.all_probs.zero_();
        }
    }

    int64_t sequence_offset = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const auto stream_sequence_count = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
        const RequestId req_id                = request_ids[stream_idx];
        auto [state_it, new_request]          = sampling_states.try_emplace(req_id);
        auto& history                         = state_it->second.token_history;
        auto& state                           = state_it->second.sampling;
        if (new_request) {
            history.resize(stream_sequence_count);
        }
        /** Replaying these tokens into existing processors would apply each committed output twice. */
        const auto* offsets = sampling_plan.token_offsets.data_ptr<int64_t>();
        for (int32_t sequence_idx = 0; sequence_idx < stream_sequence_count; ++sequence_idx) {
            const auto row          = sequence_offset + sequence_idx;
            const auto delta_length = offsets[row + 1] - offsets[row];
            auto&      tokens       = history[sequence_idx];
            RTP_LLM_CHECK(static_cast<int64_t>(tokens.size()) + delta_length
                          == sampling_plan.sequence_lengths.data_ptr<int32_t>()[row]);
            if (delta_length > 0) {
                const auto* delta = sampling_plan.token_ids.data_ptr<int32_t>() + offsets[row];
                tokens.insert(tokens.end(), delta, delta + delta_length);
            }
        }
        if (new_request) {
            state.cum_log_probs = torch::zeros({stream_sequence_count}, torch::kFloat32);
            if (sampling_plan.random_seeds[stream_idx].has_value()) {
#if defined(USING_CUDA) || defined(USING_ROCM)
                state.generator = torch::make_generator<torch::CUDAGeneratorImpl>();
#else
                state.generator = torch::make_generator<torch::CPUGeneratorImpl>();
#endif
                state.generator.set_current_seed(sampling_plan.random_seeds[stream_idx].value());
            }

            auto error = initLogitsProcessors(
                state.logits_processors, sampling_plan, history, stream_idx, sequence_offset, grammar_prefetch);
            if (error.has_value()) {
                RTP_LLM_LOG_ERROR("PP tail processor initialization failed after stream initialization: "
                                  "request_id=%ld, error_code=%d, error=%s",
                                  req_id,
                                  static_cast<int>(error->code()),
                                  error->ToString().c_str());
                state.logits_processors.clear();
                result.request_errors[stream_idx] = std::move(error.value());
            }
        }
        sequence_offset += stream_sequence_count;

        if (!is_verify) {
            continue;
        }
        verify_params.generators.push_back(state.generator);

        if (!is_dspark_) {
            continue;
        }
        auto q_row = draft_sampler_output.all_probs[stream_idx];
        if (new_request) {
            /** PPScheduler's zero-padded handoff candidates require Q = delta(0). */
            q_row.select(1, 0).fill_(1.0f);
            continue;
        }

        const auto q_it = dspark_q.find(req_id);
        if (q_it == dspark_q.end()) {
            /** Keep the zero row as a placeholder; request_errors prevents its result from being committed. */
            result.request_errors[stream_idx] =
                ErrorInfo(ErrorCode::UNKNOWN_ERROR,
                          "missing PP DSpARK proposal probabilities, request_id=" + std::to_string(req_id));
            continue;
        }
        if (d2t_map.defined()) {
            q_row.index_copy_(1, d2t_map, q_it->second);
        } else {
            q_row.copy_(q_it->second);
        }
    }
}

std::optional<ErrorInfo>
PPBatchStreamProcessor::initLogitsProcessors(std::vector<BaseLogitsProcessorPtr>&     processors,
                                             const PPSamplingPlan&                    sampling_plan,
                                             const std::vector<std::vector<int32_t>>& token_history,
                                             int64_t                                  stream_idx,
                                             int64_t                                  sequence_offset,
                                             GrammarPrefetchContext*                  grammar_prefetch) const {
    const auto  stream_sequence_count = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
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
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported grammar type: " + processor_config.grammar_type);
    }
    config->combo_token_size              = processor_config.combo_token_size;
    config->banned_combo_token_ids        = processor_config.banned_combo_token_ids;
    config->end_think_token_ids           = processor_config.end_think_token_ids;
    config->num_return_sequences          = sampling_plan.num_return_sequences[stream_idx];
    config->enable_cross_sequence_ban     = processor_config.enable_cross_sequence_ban;
    config->cross_seq_diverge_start_combo = processor_config.cross_seq_diverge_start_combo;

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = config;
    generate_input->input_ids       = torch::empty({input_lengths[sequence_offset]}, torch::kInt32);
    std::memcpy(generate_input->input_ids.data_ptr<int32_t>(),
                token_history.front().data(),
                input_lengths[sequence_offset] * sizeof(int32_t));
    auto processors_result = LogitsProcessorFactory::createLogitsProcessors(std::move(generate_input),
                                                                            stream_sequence_count,
                                                                            stream_sequence_count,
                                                                            processor_eos_token_id_,
                                                                            grammar_prefetch);
    if (!processors_result.ok()) {
        return processors_result.status();
    }

    processors = std::move(processors_result.value());
    /**
     * First PD decode replays the committed anchor; initial prefill has no output history.
     * New samples are applied later by advanceSamplingStates().
     */
    const auto committed_count =
        sampling_plan.sequence_lengths.data_ptr<int32_t>()[sequence_offset] - input_lengths[sequence_offset];
    if (committed_count > 0) {
        const auto committed = torch::empty({stream_sequence_count, committed_count}, torch::kInt32);
        for (int32_t sequence_idx = 0; sequence_idx < stream_sequence_count; ++sequence_idx) {
            std::memcpy(committed.data_ptr<int32_t>() + sequence_idx * committed_count,
                        token_history[sequence_idx].data() + input_lengths[sequence_offset],
                        committed_count * sizeof(int32_t));
        }
        for (const auto& processor : processors) {
            const auto error = processor->updateStatus(committed, committed_count);
            if (error.has_value()) {
                return error.value();
            }
        }
    }
    return std::nullopt;
}

SamplerInputs PPBatchStreamProcessor::gatherSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                          const PPOutputConfig& output_config,
                                                          const torch::Tensor&  logits,
                                                          const SamplingStates& sampling_states,
                                                          bool                  score_batch,
                                                          size_t                propose_step,
                                                          const torch::Tensor&  verify_tokens) const {
    RTP_LLM_CHECK(score_batch || propose_step == 0);
    const auto stream_count     = sampling_plan.request_ids.size(0);
    const auto sequence_count   = sampling_plan.sequence_lengths.numel();
    const auto total_batch_size = score_batch ? stream_count * static_cast<int64_t>(propose_step + 1) : sequence_count;
    RTP_LLM_CHECK(logits.dim() == 2 && logits.size(0) == total_batch_size);

    auto inputs = allocateSamplerInputs(sampling_plan, output_config, total_batch_size, propose_step);
    if (score_batch) {
        /** Padded lm-head columns are not target tokens; exclude them from verification probabilities and processors. */
        const auto vocab_size = static_cast<int64_t>(model_input_gatherer_config_.vocab_size);
        RTP_LLM_CHECK_WITH_INFO(logits.size(1) >= vocab_size,
                                "PP target logits width %ld is smaller than vocab size %ld",
                                logits.size(1),
                                vocab_size);
        inputs.vocab_size = vocab_size;
        inputs.logits     = logits.narrow(1, 0, vocab_size).clone();
    } else {
        /** Mask padding while retaining output width; clone logits when callers need unmodified scores. */
        inputs.vocab_size = logits.size(1);
        inputs.logits = output_config.return_logits || output_config.return_softmax_probs ? logits.clone() : logits;
        const auto configured_vocab_size =
            output_vocab_ids_.empty() ? static_cast<int64_t>(model_input_gatherer_config_.vocab_size) :
                                       static_cast<int64_t>(output_vocab_ids_.size());
        const auto valid_vocab_size = std::min(logits.size(1), configured_vocab_size);
        if (logits.size(1) > valid_vocab_size) {
            inputs.logits.narrow(1, valid_vocab_size, logits.size(1) - valid_vocab_size)
                .fill_(-std::numeric_limits<float>::infinity());
        }
    }
    fillSamplerInputs(inputs, sampling_plan, sampling_states, score_batch, propose_step, verify_tokens);

    if (output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        inputs.all_probs = torch::zeros({total_batch_size, static_cast<int64_t>(inputs.vocab_size)},
                                        logits.options().dtype(torch::kFloat32));
        inputs.return_original_all_probs = output_config.return_all_probs == ReturnAllProbsMode::ORIGINAL;
    }
    return inputs;
}

SamplerInputs PPBatchStreamProcessor::allocateSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                           const PPOutputConfig& output_config,
                                                           size_t                total_batch_size,
                                                           size_t                propose_step) const {
    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = total_batch_size;
    sampler_inputs.batch_size_out = total_batch_size;
    sampler_inputs.step           = sampling_plan.sequence_lengths.max().item<int32_t>() + propose_step;

    const auto batch_size = static_cast<int64_t>(total_batch_size);
    sampler_inputs.token_ids = torch::zeros({batch_size, static_cast<int64_t>(sampler_inputs.step + 1)}, pinned_i32);
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

void PPBatchStreamProcessor::fillSamplerInputs(SamplerInputs&        sampler_inputs,
                                               const PPSamplingPlan& sampling_plan,
                                               const SamplingStates& sampling_states,
                                               bool                  score_batch,
                                               size_t                propose_step,
                                               const torch::Tensor&  verify_tokens) const {
    const auto stream_count   = sampling_plan.request_ids.size(0);
    const auto sequence_count = sampling_plan.sequence_lengths.numel();
    RTP_LLM_CHECK(sampling_plan.num_return_sequences.size() == static_cast<size_t>(stream_count));
    torch::Tensor verify_tokens_cpu;
    if (score_batch) {
        RTP_LLM_CHECK(verify_tokens.defined() && verify_tokens.scalar_type() == torch::kInt32
                      && verify_tokens.numel() == stream_count * static_cast<int64_t>(propose_step + 1));
        verify_tokens_cpu =
            verify_tokens.to(torch::kCPU).reshape({stream_count, static_cast<int64_t>(propose_step + 1)}).contiguous();
    }

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
    const auto  token_stride              = sampler_inputs.token_ids.stride(0);

    auto processor_states = std::make_shared<LogitsProcessorStates>();
    auto* cum_log_probs =
        sampler_inputs.cum_log_probs.defined() ? sampler_inputs.cum_log_probs.data_ptr<float>() : nullptr;
    const auto* request_ids = sampling_plan.request_ids.data_ptr<RequestId>();

    int64_t sequence_offset = 0;
    int64_t sampling_offset = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const auto stream_sequence_count = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
        const auto stream_sampling_rows  = score_batch ? static_cast<int64_t>(propose_step + 1) : stream_sequence_count;
        RTP_LLM_CHECK(sequence_offset + stream_sequence_count <= sequence_count);
        RTP_LLM_CHECK(sampling_offset + stream_sampling_rows <= static_cast<int64_t>(sampler_inputs.batch_size));
        RTP_LLM_CHECK(!score_batch || stream_sequence_count == 1);

        const auto& request_state       = sampling_states.at(request_ids[stream_idx]);
        const auto& state               = request_state.sampling;
        const auto* state_cum_log_probs = cum_log_probs ? state.cum_log_probs.data_ptr<float>() : nullptr;

        for (int64_t row = 0; row < stream_sampling_rows; ++row) {
            const auto sequence_idx = score_batch ? 0 : row;
            const auto source_row   = sequence_offset + sequence_idx;
            const auto target_row   = sampling_offset + row;
            const auto seq_len      = plan_sequence_lengths[source_row];
            RTP_LLM_CHECK(seq_len >= 0
                          && seq_len + (score_batch ? row : 0) <= static_cast<int64_t>(sampler_inputs.step));

            input_lengths[target_row]        = plan_input_lengths[source_row];
            sequence_lengths[target_row]     = seq_len + (score_batch ? static_cast<int32_t>(row) : 0);
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
                        request_state.token_history[sequence_idx].data(),
                        seq_len * sizeof(int32_t));
            if (score_batch && row > 0) {
                /** Verification row j sees committed history followed by c1..cj. */
                std::memcpy(token_ids + target_row * token_stride + seq_len,
                            verify_tokens_cpu.data_ptr<int32_t>() + stream_idx * (propose_step + 1) + 1,
                            row * sizeof(int32_t));
            }
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
}

/** Earlier request errors take precedence over sampler errors. */
static void collectRequestErrors(const PPSamplingPlan&                        sampling_plan,
                                 const std::vector<std::optional<ErrorInfo>>& processor_errors,
                                 const torch::Tensor&                         success_cpu,
                                 std::vector<ErrorInfo>&                      request_errors) {
    const auto stream_count    = sampling_plan.request_ids.size(0);
    int64_t    sampling_offset = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const auto stream_batch_size = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);
        auto&      error             = request_errors[stream_idx];
        if (error.ok()) {
            auto sampler_error =
                collectStreamSamplerError(processor_errors, success_cpu, sampling_offset, stream_batch_size);
            if (sampler_error.has_value()) {
                error = std::move(sampler_error.value());
            }
        }
        sampling_offset += stream_batch_size;
    }
}

void PPBatchStreamProcessor::fillExecutionResult(const PPExecutionPlan& plan,
                                                const GptModelOutputs& model_output,
                                                const SamplerOutput&   sampler_output,
                                                PPExecutionResult&     result) const {
    const auto  total_batch_size = plan.sampling_plan.sequence_lengths.numel();
    const auto& token_ids        = sampler_output.token_ids;
    RTP_LLM_CHECK_WITH_INFO(token_ids.defined() && token_ids.dim() == 2 && token_ids.size(0) == total_batch_size
                                && token_ids.size(1) > 0 && sampler_output.success.defined()
                                && sampler_output.success.numel() == total_batch_size,
                            "sampler returned invalid output for PP execution result");

    result.new_token_ids     = token_ids.narrow(1, token_ids.size(1) - 1, 1).to(torch::kCPU).contiguous();
    result.new_token_lengths = torch::ones({total_batch_size}, torch::kInt32);
    fillOptionalOutputs(plan, model_output, sampler_output, result);
    /** Output vocabulary pruning excludes SP, so only this path maps token ids. */
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
    collectRequestErrors(plan.sampling_plan,
                         sampler_output.processor_errors,
                         sampler_output.success.to(torch::kCPU).contiguous(),
                         result.request_errors);
}

void PPBatchStreamProcessor::fillExecutionResult(const PPExecutionPlan&                       plan,
                                                const GptModelOutputs&                       model_output,
                                                const SamplerOutput&                         target_sampler_output,
                                                const speculative::SpeculativeSamplerOutput& sp_output,
                                                PPExecutionResult&                           result) const {
    const auto total_batch_size = plan.sampling_plan.sequence_lengths.numel();
    auto       host_accepted    = mtp::getHostAcceptedTokens(sp_output);
    RTP_LLM_CHECK_WITH_INFO(host_accepted.token_ids.dim() == 2 && host_accepted.token_ids.size(0) == total_batch_size
                                && host_accepted.lengths.numel() == total_batch_size,
                            "invalid accepted tokens for PP verify result");
    const auto verify_token_count = host_accepted.token_ids.size(1);
    RTP_LLM_CHECK_WITH_INFO(target_sampler_output.success.defined()
                                && target_sampler_output.success.numel() == total_batch_size * verify_token_count,
                            "target sampler returned invalid success for PP verify result");

    result.new_token_ids     = host_accepted.token_ids.contiguous();
    result.new_token_lengths = std::move(host_accepted.lengths);
    fillOptionalOutputs(plan, model_output, target_sampler_output, result);
    /** Sampler failures outside the committed token prefix do not fail the request. */
    const auto verify_success =
        target_sampler_output.success.to(torch::kCPU).reshape({total_batch_size, verify_token_count});
    const auto accepted_rows =
        torch::arange(verify_token_count, torch::kInt32).unsqueeze(0) < result.new_token_lengths.unsqueeze(1);
    collectRequestErrors(plan.sampling_plan,
                         sp_output.processor_errors,
                         (verify_success | ~accepted_rows).all(1).contiguous(),
                         result.request_errors);
}

void PPBatchStreamProcessor::fillOptionalOutputs(const PPExecutionPlan& plan,
                                                 const GptModelOutputs& model_output,
                                                 const SamplerOutput&   sampler_output,
                                                 PPExecutionResult&     result) const {
    const auto& output_config = plan.output_config;
    const auto  stream_count  = plan.sampling_plan.request_ids.size(0);
    torch::Tensor logits;
    torch::Tensor softmax_probs;
    torch::Tensor hidden_states;
    torch::Tensor all_hidden_states;
    torch::Tensor cum_log_probs;
    torch::Tensor all_probs;
    if (output_config.return_logits) {
        /** FP32 contiguous logits can alias probs, whose CUDA softmax is in-place; the CPU copy preserves raw logits. */
        logits = model_output.logits.to(torch::kCPU).contiguous();
    }
    if (output_config.return_softmax_probs) {
        auto probs = model_output.logits.to(torch::kFloat32).contiguous();
#if USING_CUDA
        cudaSoftmaxInplace(probs, at::cuda::getCurrentCUDAStream().stream());
#else
        probs = torch::softmax(probs, -1);
#endif
        const auto& new_token_ids = result.new_token_ids;
        const auto  token_index   = new_token_ids.to(probs.device(), torch::kLong).reshape({probs.size(0), 1});
        softmax_probs = probs.gather(1, token_index).reshape(new_token_ids.sizes()).to(torch::kCPU).contiguous();
    }
    if (output_config.return_hidden_states) {
        hidden_states = model_output.hidden_states.to(torch::kCPU).contiguous();
    }
    if (output_config.return_all_hidden_states) {
        all_hidden_states = model_output.all_hidden_states.to(torch::kCPU).contiguous();
    }
    if (output_config.return_cum_log_probs) {
        cum_log_probs = sampler_output.cum_log_probs.to(torch::kCPU).contiguous();
    }
    if (output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        all_probs = sampler_output.all_probs.to(torch::kCPU).contiguous();
    }

    const bool    need_prompt_outputs = plan.model_input.need_all_logits && !plan.is_decode;
    torch::Tensor lm_output_indexes;
    if (output_config.return_all_hidden_states || need_prompt_outputs) {
        lm_output_indexes = plan.model_input.lm_output_indexes.to(torch::kCPU, torch::kInt64).contiguous();
    }

    /** Request views share batch CPU storage; serialization includes only each view's elements. */
    int64_t batch_idx = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const auto& request           = output_config.requests[stream_idx];
        auto&       output            = result.request_outputs[stream_idx];
        const auto  stream_batch_size = std::max(plan.sampling_plan.num_return_sequences[stream_idx], 1);
        if (request.return_logits) {
            output.logits = logits.narrow(0, batch_idx, stream_batch_size);
        }
        if (request.return_softmax_probs) {
            output.softmax_probs = softmax_probs.narrow(0, batch_idx, stream_batch_size);
        }
        if (request.return_cum_log_probs) {
            output.cum_log_probs = cum_log_probs.narrow(0, batch_idx, stream_batch_size);
        }
        if (request.return_all_probs) {
            output.all_probs = all_probs.narrow(0, batch_idx, stream_batch_size);
        }
        if (request.return_hidden_states) {
            output.hidden_states = hidden_states.narrow(0, batch_idx, stream_batch_size);
        }
        if (request.return_all_hidden_states || need_prompt_outputs) {
            const auto* indexes = lm_output_indexes.data_ptr<int64_t>();
            const auto  start   = batch_idx == 0 ? 0 : indexes[batch_idx - 1] + 1;
            if (request.return_all_hidden_states) {
                const auto end           = indexes[batch_idx + stream_batch_size - 1] + 1;
                output.all_hidden_states = all_hidden_states.narrow(0, start, end - start);
            }
            /** Return sequences share the prompt, so the first sequence suffices for loss and prompt logits. */
            if (need_prompt_outputs) {
                const auto token_size = indexes[batch_idx] + 1 - start;
                const auto loss_size  = token_size - 1;
                if (request.calculate_loss && loss_size > 0) {
                    auto labels = plan.model_input.combo_tokens.narrow(0, start + 1, loss_size)
                                      .to(model_output.all_logits.device(), torch::kLong);
                    output.loss = torch::cross_entropy_loss(model_output.all_logits.narrow(0, start, loss_size),
                                                            labels,
                                                            torch::nullopt,
                                                            at::Reduction::None)
                                      .to(torch::kCPU, torch::kFloat32)
                                      .contiguous();
                }
                const auto& prompt_request = request.prompt_logits;
                if (prompt_request.enabled) {
                    output.prompt_logits =
                        makePromptLogitsOutput(model_output.all_logits.narrow(0, start, token_size),
                                               plan.model_input.combo_tokens.narrow(0, start, token_size),
                                               prompt_request.top_k,
                                               prompt_request.start,
                                               prompt_request.end,
                                               prompt_request.return_target_logprob);
                }
            }
        }
        batch_idx += stream_batch_size;
    }
}

void PPBatchStreamProcessor::prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_prefill_input,
                                                                   const torch::Tensor& target_hidden_states,
                                                                   const torch::Tensor& sampled_token_ids,
                                                                   const torch::Tensor& next_position_ids,
                                                                   TensorHolder&        host_holder) const {
    mtp::prepareDraftPrefillAfterTargetPrefill(draft_prefill_input,
                                               target_hidden_states,
                                               sampled_token_ids,
                                               next_position_ids,
                                               model_input_gatherer_config_.position_id_len_factor,
                                               host_holder);
}

void PPBatchStreamProcessor::prepareDraftPrefillAfterVerify(GptModelInputs&       draft_prefill_input,
                                                            const torch::Tensor&  target_hidden_states,
                                                            const torch::Tensor&  accepted_token_ids,
                                                            const torch::Tensor&  accepted_lengths,
                                                            mtp::DraftInputLayout layout,
                                                            torch::Tensor&        hidden_states_d_t,
                                                            TensorHolder&         host_holder) const {
    mtp::prepareDraftPrefillAfterVerify(draft_prefill_input,
                                        target_hidden_states,
                                        accepted_token_ids,
                                        accepted_lengths,
                                        draft_prefill_input.input_lengths.numel(),
                                        propose_step_,
                                        layout,
                                        model_input_gatherer_config_.position_id_len_factor,
                                        hidden_states_d_t,
                                        host_holder);
}

void PPBatchStreamProcessor::prepareDSparkCommitInput(GptModelInputs&      commit_input,
                                                      const torch::Tensor& target_features,
                                                      bool                 cp_enabled) const {
    RTP_LLM_PROFILE_SCOPE("executor.pp.dspark_commit_prepare");

    RTP_LLM_CHECK_WITH_INFO(target_features.defined() && target_features.dim() == 2,
                            "PP DSpARK commit requires 2-D target features");
    /** CP checks feature rows against the padded local token count in handleInputs. */
    RTP_LLM_CHECK_WITH_INFO(cp_enabled || target_features.size(0) == commit_input.combo_tokens.numel(),
                            "PP DSpARK commit feature rows %ld do not match input rows %ld",
                            target_features.size(0),
                            commit_input.combo_tokens.numel());

    /** COMMIT keeps target geometry and rank-local features; Engram history belongs to target execution. */
    commit_input.last_hidden_states   = target_features;
    commit_input.engram_token_windows = torch::Tensor();
}

mtp::DSparkRoundState PPBatchStreamProcessor::buildDSparkRoundState(const GptModelInputs& target_input,
                                                                    const torch::Tensor&  draft_next_position_ids,
                                                                    bool                  is_decode,
                                                                    const torch::Tensor&  new_token_ids,
                                                                    const torch::Tensor&  new_token_lengths,
                                                                    TensorHolder&         host_holder) const {
    const auto            batch_size      = new_token_ids.size(0);
    const auto            position_factor = model_input_gatherer_config_.position_id_len_factor;
    auto                  prefix_lengths  = target_input.prefix_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
    mtp::DSparkRoundState round_state;
    if (is_decode) {
        auto last_indexes          = (new_token_lengths.to(torch::kLong) - 1).unsqueeze(1);
        round_state.anchors        = new_token_ids.gather(1, last_indexes).reshape({batch_size});
        round_state.committed_ends = prefix_lengths + new_token_lengths;
        if (target_input.combo_position_ids.defined()) {
            auto anchor_positions =
                target_input.combo_position_ids.reshape({batch_size, propose_step_ + 1, position_factor})
                    .select(1, 0)
                    .to(torch::kCPU);
            round_state.position_bases = anchor_positions + new_token_lengths.reshape({batch_size, 1});
        }
    } else {
        auto input_lengths         = target_input.input_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
        round_state.anchors        = new_token_ids.reshape({batch_size}).contiguous();
        round_state.committed_ends = prefix_lengths + input_lengths;
        if (draft_next_position_ids.defined()) {
            /** The plan captured the prompt's last position before target sampling appended the anchor. */
            round_state.position_bases = draft_next_position_ids.reshape({batch_size, position_factor}) + 1;
        }
    }
    host_holder.hold_host(round_state.anchors);
    round_state.anchors = round_state.anchors.to(torch::kCUDA, /*non_blocking=*/true);
    return round_state;
}

void PPBatchStreamProcessor::prepareDSparkProposeModelInput(const mtp::DSparkRoundState& round_state,
                                                            GptModelInputs&              draft_input,
                                                            TensorHolder&                host_holder) {
    draft_input.pd_separation = draft_input.pd_separation && !draft_input.skip_lm_head;
    draft_input.skip_lm_head  = false;
    mtp::prepareDSparkProposeInput(draft_input,
                                   round_state,
                                   propose_step_,
                                   dspark_mask_token_id_,
                                   dspark_sample_from_anchor_,
                                   dspark_propose_input_buffers_,
                                   host_holder);
    /** Only COMMIT publishes persistent draft KV state. */
    draft_input.request_id            = torch::Tensor();
    draft_input.request_pd_separation = torch::Tensor();
    draft_input.cache_keys            = torch::Tensor();
}

torch::Tensor PPBatchStreamProcessor::prepareDSparkDraftTemperature(const PPSamplingPlan& sampling_plan,
                                                                    torch::Device         device,
                                                                    TensorHolder&         host_holder) const {
    const auto      batch_size           = sampling_plan.temperature.numel();
    auto            draft_temperature    = sampling_plan.temperature.contiguous().clone();
    auto            draft_stochastic     = sampling_plan.spec_do_sample.contiguous();
    auto*           values               = draft_temperature.data_ptr<float>();
    const auto*     stochastic           = draft_stochastic.data_ptr<bool>();
    constexpr float kMinDraftTemperature = 1.0e-6f;
    for (int64_t row = 0; row < batch_size; ++row) {
        if (!std::isfinite(values[row]) || values[row] < 0.0f) {
            values[row] = 1.0f;
        }
        /** Greedy rejection requires an exact match; concentrate proposal Q on its argmax. */
        values[row] = stochastic[row] ? std::max(values[row], kMinDraftTemperature) : kMinDraftTemperature;
    }
    host_holder.hold_host(draft_temperature);
    return draft_temperature.to(device, /*non_blocking=*/true);
}

absl::Status PPBatchStreamProcessor::dispatchExecutionResult(const StreamGroups& stream_groups,
                                                             const PPExecutionResult& result) const {
    validateExecutionResult(stream_groups, result);
    if (stream_groups.allStreams().front()->generateConfig()->isPrefillOnly()) {
        return dispatchPrefillOnlyExecutionResult(stream_groups, result);
    }
    if (sp_enabled_) {
        return dispatchSpeculativeExecutionResult(stream_groups, result);
    }
    return dispatchNormalExecutionResult(stream_groups, result);
}

void PPBatchStreamProcessor::validateExecutionResult(const StreamGroups&      stream_groups,
                                                     const PPExecutionResult& result) const {
    RTP_LLM_CHECK_WITH_INFO(!stream_groups.empty(), "PP execution result requires a non-empty inflight batch");
    const auto stream_count = static_cast<int64_t>(stream_groups.size());
    RTP_LLM_CHECK_WITH_INFO(result.request_ids.defined() && result.request_ids.device().is_cpu()
                                && result.request_ids.scalar_type() == torch::kInt64
                                && result.request_ids.dim() == 1 && result.request_ids.size(0) == stream_count,
                            "PP execution result request count does not match the inflight stream count");
    RTP_LLM_CHECK_WITH_INFO(result.request_outputs.size() == static_cast<size_t>(stream_count),
                            "PP optional-output count does not match the inflight stream count");

    RTP_LLM_CHECK_WITH_INFO(result.request_errors.size() == static_cast<size_t>(stream_count),
                            "PP request error counts do not match the inflight stream count");
}

absl::Status PPBatchStreamProcessor::dispatchPrefillOnlyExecutionResult(const StreamGroups&      stream_groups,
                                                                       const PPExecutionResult& result) const {
    int64_t row = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        RTP_LLM_CHECK_WITH_INFO(stream->generateConfig()->isPrefillOnly() && stream->currentBatchSize() == 1
                                    && result.request_ids[row].item<int64_t>() == stream->streamId(),
                                "PP prefill-only result does not match the inflight request");
        StreamUpdateInfo update_info{torch::empty({stream->currentBatchSize(), 0}, torch::kInt32), 0};
        update_info.update_remote_generate = false;
        if (result.request_errors[row].hasError()) {
            update_info.error_info = result.request_errors[row];
        }
        stream->update(update_info, false);
        stream->clearPPInflight();
        ++row;
    }
    return absl::OkStatus();
}

absl::Status PPBatchStreamProcessor::dispatchSpeculativeExecutionResult(const StreamGroups& stream_groups,
                                                                        const PPExecutionResult& result) const {
    const auto all_streams = stream_groups.allStreams();
    const auto batch_size  = static_cast<int64_t>(all_streams.size());
    const bool is_prefill  = all_streams.empty() || all_streams.front()->isContextStream();
    RTP_LLM_CHECK_WITH_INFO(
        stream_groups.totalModelBatchSize() == all_streams.size() && result.new_token_lengths.device().is_cpu()
            && result.new_token_lengths.numel() == batch_size,
        "PP speculative execution result requires one new token count per request");
    RTP_LLM_CHECK_WITH_INFO(!is_dspark_ || !result.propose_all_probs.defined(),
                            "PP DSpARK proposal probabilities must remain on the last stage");
    RTP_LLM_CHECK_WITH_INFO(is_dspark_ || is_prefill || !result.propose_all_probs.defined(),
                            "ordinary PP decode results use token-only proposals");

    if (result.propose_all_probs.defined()) {
        RTP_LLM_CHECK_WITH_INFO(result.propose_token_ids.defined() && result.propose_token_ids.dim() == 2
                                    && result.propose_all_probs.device().is_cpu()
                                    && result.propose_all_probs.scalar_type() == torch::kFloat32
                                    && result.propose_all_probs.is_contiguous() && result.propose_all_probs.dim() == 3
                                    && result.propose_all_probs.size(0) == batch_size
                                    && result.propose_all_probs.size(1) == result.propose_token_ids.size(1),
                                "PP result probabilities must be contiguous CPU FP32 [B,K,V] and match tokens");
    }
    if (result.propose_hidden_states.defined()) {
        RTP_LLM_CHECK_WITH_INFO(is_prefill && !is_dspark_ && result.propose_hidden_states.device().is_cpu()
                                    && result.propose_hidden_states.dim() == 2
                                    && result.propose_hidden_states.size(0) == batch_size,
                                "PP MTP/EAGLE prefill handoff hidden must be CPU [B,H]");
    }

    const auto* request_ids       = result.request_ids.data_ptr<RequestId>();
    const auto* new_token_lengths = result.new_token_lengths.data_ptr<int32_t>();
    int64_t row = 0;
    for (const auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(request_ids[row] == stream->streamId(),
                                "PP execution result request order does not match the inflight stream order");
        RTP_LLM_CHECK_WITH_INFO(stream->isContextStream() == is_prefill,
                                "PP speculative result batch must contain only prefill or only decode requests");
        std::optional<ErrorInfo> error_info;
        if (result.request_errors[row].hasError()) {
            error_info = result.request_errors[row];
        }
        int           num_new_tokens = 0;
        torch::Tensor draft_tokens;
        torch::Tensor draft_all_probs;
        torch::Tensor draft_hidden_states;
        torch::Tensor new_tokens;
        if (!error_info.has_value()) {
            num_new_tokens = new_token_lengths[row];
            RTP_LLM_CHECK_WITH_INFO(num_new_tokens > 0,
                                    "PP speculative execution result requires a positive new token count");
            if (result.propose_token_ids.defined()) {
                draft_tokens = result.propose_token_ids[row];
            }
            if (result.propose_all_probs.defined()) {
                draft_all_probs = result.propose_all_probs[row];
            }
            if (result.propose_hidden_states.defined()) {
                draft_hidden_states = result.propose_hidden_states.narrow(0, row, 1);
            }
            new_tokens = result.new_token_ids.narrow(0, row, 1).narrow(1, 0, num_new_tokens).contiguous();
        }

        StreamSpecUpdateInfo update_info{std::move(new_tokens),
                                         num_new_tokens,
                                         draft_tokens,
                                         draft_hidden_states,
                                         draft_all_probs,
                                         torch::Tensor(),
                                         true,
                                         false,
                                         std::move(error_info)};
        if (!is_prefill && num_new_tokens > 0) {
            /** Report the verifier's acceptance before output clipping, as main does.
             * The last returned token is the target bonus/correction, not an accepted draft. */
            update_info.speculative_propose_step = stream->getSPOutputBuffer()->propose_step;
            update_info.accepted_draft_tokens    = num_new_tokens - 1;
        }
        stream->specUpdate(update_info, false);
        stream->clearPPInflight();
        ++row;
    }
    return absl::OkStatus();
}

absl::Status PPBatchStreamProcessor::dispatchNormalExecutionResult(const StreamGroups& stream_groups,
                                                                   const PPExecutionResult& result) const {
    const auto all_streams = stream_groups.allStreams();

    const auto* request_ids = result.request_ids.data_ptr<RequestId>();
    int64_t     batch_idx   = 0;
    int64_t     stream_idx  = 0;
    for (const auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(request_ids[stream_idx] == stream->streamId(),
                                "PP execution result request order does not match the inflight stream order");
        const auto stream_batch_size = static_cast<int64_t>(stream->currentBatchSize());
        dispatchNormalSingleStream(stream, result, stream_idx, batch_idx, stream_batch_size);
        stream->clearPPInflight();
        ++stream_idx;
        batch_idx += stream_batch_size;
    }
    return absl::OkStatus();
}

void PPBatchStreamProcessor::dispatchNormalSingleStream(const GenerateStreamPtr& stream,
                                                        const PPExecutionResult& result,
                                                        int64_t                  stream_idx,
                                                        int64_t                  batch_idx,
                                                        int64_t                  stream_batch_size) const {
    const auto& error = result.request_errors[stream_idx];
    if (error.hasError()) {
        StreamUpdateInfo update_info{};
        update_info.error_info = error;
        stream->update(update_info, false);
        return;
    }
    const auto& output = result.request_outputs[stream_idx];
    stream->update({result.new_token_ids.narrow(0, batch_idx, stream_batch_size),
                    1,
                    output.hidden_states,
                    output.logits,
                    output.softmax_probs,
                    output.cum_log_probs,
                    output.all_probs,
                    output.loss,
                    torch::Tensor(),
                    output.all_hidden_states,
                    true,
                    false,
                    output.prompt_logits,
                    std::nullopt},
                   false);
}

}  // namespace rtp_llm
