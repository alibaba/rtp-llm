#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

PPBatchStreamProcessor::PPBatchStreamProcessor(const ModelConfig&                 model_config,
                                               const PDSepConfig&                 pd_sep_config,
                                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                                               const CacheConfig&                 cache_config,
                                               bool                               warm_up):
    NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
    output_vocab_ids_(model_config.output_vocab_ids) {}

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
        RTP_LLM_CHECK_WITH_INFO(!config.return_prompt_logits,
                                "PP does not support return_prompt_logits, request_id=%ld",
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
        const auto& config = *stream->generateConfig();
        sampling_plan.random_seeds.push_back(config.random_seed);
        sampling_plan.num_return_sequences.push_back(config.num_return_sequences);

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
    for (const auto& stream : stream_groups.allStreams()) {
        output_config.return_logits |= stream->returnLogits();
        output_config.return_softmax_probs |= stream->calculateSoftmaxProbs();
        output_config.return_cum_log_probs |= stream->returnCumLogProbs();
        output_config.calculate_loss |= stream->calculateLoss();
        output_config.return_hidden_states |= stream->generateConfig()->return_hidden_states;
        output_config.return_all_hidden_states |= stream->needReturnHiddenStates();
    }
    return output_config;
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

    const auto compact_token_ids =
        sampler_output.token_ids.narrow(1, sampler_output.token_ids.size(1) - 1, 1).contiguous();
    if (plan.output_config.return_softmax_probs) {
        result.softmax_probs = torch::softmax(model_output.logits.to(torch::kFloat32), -1)
                                   .gather(1, compact_token_ids.to(torch::kLong))
                                   .to(torch::kCPU)
                                   .contiguous();
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
    if (plan.output_config.return_logits) {
        result.logits = model_output.logits.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_cum_log_probs) {
        result.cum_log_probs = sampler_output.cum_log_probs.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_all_probs != ReturnAllProbsMode::NONE) {
        result.all_probs = sampler_output.all_probs.to(torch::kCPU).contiguous();
    }
    /** need_all_logits makes hidden_states token-major; select one output row per batch row. */
    if (plan.output_config.return_hidden_states) {
        auto hidden_states = model_output.hidden_states;
        if (plan.model_input.need_all_logits) {
            auto indexes  = plan.model_input.lm_output_indexes.to(hidden_states.device(), torch::kLong);
            hidden_states = torch::index_select(hidden_states, 0, indexes);
        }
        result.hidden_states = hidden_states.to(torch::kCPU).contiguous();
    }
    if (plan.output_config.return_all_hidden_states) {
        result.all_hidden_states = model_output.all_hidden_states.to(torch::kCPU).contiguous();
    }
    /** Prompt loss is stream-level; all return sequences share the first batch row's prompt loss. */
    if (plan.output_config.calculate_loss) {
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
            const int64_t loss_size         = end - start - 1;
            if (loss_size > 0) {
                auto labels = plan.model_input.combo_tokens.narrow(0, start + 1, loss_size)
                                  .to(model_output.all_logits.device(), torch::kLong);
                losses.push_back(torch::cross_entropy_loss(model_output.all_logits.narrow(0, start, loss_size),
                                                           labels,
                                                           torch::nullopt,
                                                           at::Reduction::None)
                                     .to(torch::kFloat32));
            }
            batch_idx += stream_batch_size;
        }
        if (!losses.empty()) {
            result.loss = torch::cat(losses).to(torch::kCPU).contiguous();
        }
    }

    result.processor_errors = sampler_output.processor_errors;
    result.processor_errors.resize(total_batch_size);
    return result;
}

void PPBatchStreamProcessor::validateExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                                     const PPOutputConfig&               output_config,
                                                     const PPExecutionResult&            result) const {
    const auto stream_count = static_cast<int64_t>(all_streams.size());
    RTP_LLM_CHECK_WITH_INFO(result.request_ids.defined() && result.request_ids.dim() == 1
                                && result.request_ids.size(0) == stream_count,
                            "PP execution result request count does not match the inflight stream count");

    int64_t     total_batch_size = 0;
    int64_t     total_token_size = 0;
    int64_t     total_loss_size  = 0;
    const auto* request_ids      = result.request_ids.data_ptr<int64_t>();
    int64_t     stream_idx       = 0;
    for (const auto& stream : all_streams) {
        const auto stream_batch_size       = static_cast<int64_t>(stream->currentBatchSize());
        const auto token_size              = static_cast<int64_t>(stream->currentExecuteTokenSize());
        const auto token_size_per_sequence = token_size / stream_batch_size;
        total_batch_size += stream_batch_size;
        total_token_size += token_size;
        total_loss_size += std::max<int64_t>(token_size_per_sequence - 1, 0);
        RTP_LLM_CHECK_WITH_INFO(request_ids[stream_idx] == stream->streamId(),
                                "PP execution result request order does not match the inflight stream order");
        ++stream_idx;
    }

    const auto valid_batch_matrix = [total_batch_size](bool requested, const torch::Tensor& tensor) {
        return !requested || (tensor.defined() && tensor.dim() == 2 && tensor.size(0) == total_batch_size);
    };
    const bool valid_shapes =
        result.new_token_ids.defined() && result.new_token_ids.dim() == 2
        && result.new_token_ids.size(0) == total_batch_size && result.new_token_ids.size(1) == 1
        && result.sample_success.defined() && result.sample_success.dim() == 1
        && result.sample_success.size(0) == total_batch_size
        && valid_batch_matrix(output_config.return_hidden_states, result.hidden_states)
        && valid_batch_matrix(output_config.return_logits, result.logits)
        && valid_batch_matrix(output_config.return_all_probs != ReturnAllProbsMode::NONE, result.all_probs)
        && (!output_config.return_softmax_probs
            || (result.softmax_probs.defined() && result.softmax_probs.dim() == 2
                && result.softmax_probs.size(0) == total_batch_size && result.softmax_probs.size(1) == 1))
        && (!output_config.return_cum_log_probs
            || (result.cum_log_probs.defined() && result.cum_log_probs.dim() == 1
                && result.cum_log_probs.size(0) == total_batch_size))
        && (!output_config.return_all_hidden_states
            || (result.all_hidden_states.defined() && result.all_hidden_states.dim() == 2
                && result.all_hidden_states.size(0) == total_token_size))
        && (!output_config.calculate_loss || total_loss_size == 0
            || (result.loss.defined() && result.loss.dim() == 1 && result.loss.numel() == total_loss_size))
        && result.processor_errors.size() == static_cast<size_t>(total_batch_size);
    RTP_LLM_CHECK_WITH_INFO(valid_shapes,
                            "PP execution result is missing a requested tensor or has invalid tensor shapes");
}

absl::Status PPBatchStreamProcessor::dispatchExecutionResult(const StreamGroups&      stream_groups,
                                                             const PPExecutionResult& result) const {
    const auto all_streams   = stream_groups.allStreams();
    const auto output_config = gatherOutputConfig(stream_groups);
    validateExecutionResult(all_streams, output_config, result);

    int64_t batch_idx    = 0;
    int64_t token_offset = 0;
    int64_t loss_offset  = 0;
    for (const auto& stream : all_streams) {
        const auto stream_batch_size = static_cast<int64_t>(stream->currentBatchSize());
        const auto token_size        = static_cast<int64_t>(stream->currentExecuteTokenSize());
        const auto loss_size         = std::max<int64_t>(token_size / stream_batch_size - 1, 0);
        auto       error_info =
            collectStreamSamplerError(result.processor_errors, result.sample_success, batch_idx, stream_batch_size);
        dispatchSingleStream(
            stream, result, batch_idx, stream_batch_size, token_offset, loss_offset, std::move(error_info));
        stream->clearPPInflight();
        batch_idx += stream_batch_size;
        token_offset += token_size;
        loss_offset += loss_size;
    }

    return absl::OkStatus();
}

void PPBatchStreamProcessor::dispatchSingleStream(const GenerateStreamPtr& stream,
                                                  const PPExecutionResult& result,
                                                  int64_t                  batch_idx,
                                                  int64_t                  stream_batch_size,
                                                  int64_t                  token_offset,
                                                  int64_t                  loss_offset,
                                                  std::optional<ErrorInfo> error_info) const {
    const auto token_size = static_cast<int64_t>(stream->currentExecuteTokenSize());
    const auto loss_size  = std::max<int64_t>(token_size / stream_batch_size - 1, 0);

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
                          std::nullopt,
                          std::move(error_info)});
}

}  // namespace rtp_llm
