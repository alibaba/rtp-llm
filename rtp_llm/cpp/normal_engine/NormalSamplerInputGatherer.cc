#include <algorithm>
#include <cstring>
#include <limits>
#include "torch/all.h"
#include "rtp_llm/cpp/normal_engine/NormalSamplerInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

namespace rtp_llm {

bool NormalSamplerInputGatherer::anyStreamPenaltyEnabled(const std::list<GenerateStreamPtr>& streams) {
    return std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
        const auto& config = *stream->generateConfig();
        return config.repetition_penalty != 1.0f || config.presence_penalty != 0.0f || config.frequency_penalty != 0.0f
               || config.no_repeat_ngram_size.value_or(0) != 0;
    });
}

bool NormalSamplerInputGatherer::anyPenaltyRowActive(const SamplerInputs& inputs) {
    // An undefined array is the all-default state (see anyStreamPenaltyEnabled):
    // nothing to scan and nothing the kernels would act on.
    const auto any_float_not = [](const torch::Tensor& tensor, float default_value) {
        if (!tensor.defined()) {
            return false;
        }
        if (!tensor.device().is_cpu()) {
            return true;  // unexpected layout: keep the caller's safe side
        }
        const auto* data = tensor.data_ptr<float>();
        return std::any_of(
            data, data + tensor.numel(), [default_value](float value) { return value != default_value; });
    };
    const auto any_ngram = [](const torch::Tensor& tensor) {
        if (!tensor.defined()) {
            return false;
        }
        if (!tensor.device().is_cpu()) {
            return true;
        }
        const auto* data = tensor.data_ptr<int32_t>();
        return std::any_of(data, data + tensor.numel(), [](int32_t value) { return value != 0; });
    };
    return any_float_not(inputs.repetition_penalty, 1.0f) || any_float_not(inputs.presence_penalty, 0.0f)
           || any_float_not(inputs.frequency_penalty, 0.0f) || any_ngram(inputs.no_repeat_ngram_size);
}

absl::StatusOr<SamplerInputs> NormalSamplerInputGatherer::gather(const StreamGroups&    stream_groups,
                                                                 const GptModelInputs&  model_inputs,
                                                                 const GptModelOutputs& model_output) const {
    (void)model_inputs;
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_CHECK(!stream_groups.empty());
    auto               all_streams          = stream_groups.allStreams();
    auto               total_batch_size_in  = stream_groups.totalSamplerBatchSizeIn();
    auto               total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    ReturnAllProbsMode return_all_probs     = stream_groups.needReturnAllProbs();

    bool need_token_history = true;
#if USING_CUDA
    if (model_output.logits.is_cuda()) {
        // Readers of the transposed history: the repetition/presence/frequency
        // penalty kernel and beam search. An all-default-penalty single-beam
        // batch needs no host history at all (the sampler writes the last
        // column on device).
        need_token_history = anyStreamPenaltyEnabled(all_streams)
                             || std::any_of(all_streams.begin(), all_streams.end(), [](const auto& stream) {
                                    return stream->currentNumBeams() != 1 || stream->nextNumBeams() != 1;
                                });
    }
#endif
    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size_in, total_batch_size_out, 0, need_token_history);
    if (!need_token_history) {
        // Normal dispatch only consumes the last column without beams. The
        // sampler writes it on device; no host history needs to be uploaded.
        sampler_inputs.token_ids =
            torch::empty({static_cast<int64_t>(total_batch_size_in), static_cast<int64_t>(sampler_inputs.step + 1)},
                         model_output.logits.options().dtype(torch::kInt32));
    }
    fillSamplerCommonInputs(sampler_inputs, all_streams);

    setLogitsProcessorInputs(sampler_inputs, all_streams);

    size_t               total_decode_batch_size_in = 0;
    int                  batch_idx                  = 0;
    bool                 return_logits              = false;
    bool                 calculate_softmax_probs    = false;
    bool                 need_tiling                = false;
    for (auto& stream : all_streams) {
        auto complete_token_ids = stream->completeTokenIds();
        auto complete_seq_len   = complete_token_ids.size(1);
        auto seq_len            = stream->seqLength();
        auto current_batch_size = stream->currentBatchSize();
        auto sampler_batch_size =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();

        for (int i = 0; i < sampler_batch_size; ++i) {
            if (need_token_history) {
                int cur_batch = std::min(i, current_batch_size - 1);
                memcpy(sampler_inputs.token_ids.data_ptr<int32_t>() + ((batch_idx) * (sampler_inputs.step + 1)),
                       complete_token_ids.data_ptr<int32_t>() + cur_batch * complete_seq_len,
                       seq_len * sizeof(int));
            }
            reinterpret_cast<bool*>(sampler_inputs.finished_mask.data_ptr())[batch_idx] =
                stream->isSubGenerateDoneWithoutLock(i);
            batch_idx += 1;
        }
        need_tiling |= stream->needTilingForSampling();
        if (!stream->isContextStream()) {
            total_decode_batch_size_in += sampler_batch_size;
        }
        return_logits |= stream->returnLogits();
        calculate_softmax_probs |= stream->calculateSoftmaxProbs();
        sampler_inputs.return_logprobs |= stream->shouldComputeLogprobs();
        if (need_token_history) {
            RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                              stream->streamId(),
                              tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
        }
    }

    auto       vocab_size            = (size_t)model_output.logits.size(1);
    const auto output_vocab_size     = all_streams.front()->outputVocabSize();
    const auto configured_vocab_size = output_vocab_size > 0 ? output_vocab_size : all_streams.front()->vocabSize();
    const auto valid_vocab_size      = std::min(vocab_size, configured_vocab_size);
    sampler_inputs.vocab_size        = vocab_size;
    if (output_vocab_size > 0) {
        auto* top_k_values = sampler_inputs.top_k.data_ptr<int32_t>();
        for (size_t index = 0; index < sampler_inputs.batch_size; ++index) {
            if (top_k_values[index] > 0) {
                top_k_values[index] = std::min<int32_t>(top_k_values[index], static_cast<int32_t>(valid_vocab_size));
            }
        }
    }

    if (return_all_probs != ReturnAllProbsMode::NONE) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size_in, (int64_t)vocab_size},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        if (return_all_probs == ReturnAllProbsMode::ORIGINAL) {
            sampler_inputs.return_original_all_probs = true;
        }
    }

    // copy logits when needs tiling or returning logits
    torch::Tensor logits_tensor;
    if (need_tiling) {
        logits_tensor =
            torch::empty({(int64_t)total_batch_size_in, (int64_t)vocab_size}, model_output.logits.options());
        // copy decode batch logits
        if (total_decode_batch_size_in > 0) {
            logits_tensor.narrow(0, 0, total_decode_batch_size_in)
                .copy_(model_output.logits.narrow(0, 0, total_decode_batch_size_in));
        }
        // tile context batch logits
        size_t input_offset = total_decode_batch_size_in, logits_offset = total_decode_batch_size_in;
        for (auto& stream : stream_groups.contextStreams()) {
            auto sampler_batch_size =
                stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
            for (int i = 0; i < sampler_batch_size; ++i) {
                logits_tensor[input_offset].copy_(model_output.logits[logits_offset]);
                input_offset += 1;
            }
            logits_offset += 1;
        }
    } else if (return_logits || calculate_softmax_probs) {
        logits_tensor = model_output.logits.clone();
    } else {
        logits_tensor = model_output.logits;
    }
    if (vocab_size > valid_vocab_size) {
        logits_tensor.narrow(1, valid_vocab_size, vocab_size - valid_vocab_size)
            .fill_(-std::numeric_limits<float>::infinity());
    }
    sampler_inputs.logits = logits_tensor;

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      tensorDebugStringWithData<float>(sampler_inputs.logits.cpu(), 10).c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

SamplerInputs NormalSamplerInputGatherer::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              total_batch_size_in,
                                                                size_t              total_batch_size_out,
                                                                size_t              propose_step,
                                                                bool                materialize_host_token_ids) const {
    // TODO(xinfei.sxf) don't sample for chunk stream
    SamplerInputs sampler_inputs;
    sampler_inputs.step             = stream_groups.maxSeqLen() + propose_step;
    sampler_inputs.batch_size       = total_batch_size_in;
    sampler_inputs.batch_size_out   = total_batch_size_out;
    auto bs                         = (int64_t)total_batch_size_in;
    sampler_inputs.sequence_lengths = torch::empty({bs}, torch::kInt32);
    sampler_inputs.logits_processor_states_ptr.reset();
    sampler_inputs.input_lengths  = torch::empty({bs}, torch::kInt32);
    sampler_inputs.num_beams_in   = torch::empty({bs}, torch::kLong);
    sampler_inputs.num_beams_out  = torch::empty({bs}, torch::kLong);
    static const auto pinned_int  = torch::TensorOptions(torch::kInt).pinned_memory(true);
    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    sampler_inputs.top_k                = torch::empty({bs}, pinned_int);
    sampler_inputs.top_p                = torch::empty({bs}, pinned_f32);
    sampler_inputs.temperature          = torch::empty({bs}, pinned_f32);
    // Penalty inputs are materialized only when some stream can actually run the
    // repetition/presence/frequency or no-repeat-ngram kernel. An all-default
    // batch leaves all four undefined: Sampler::forward turns an undefined
    // tensor into std::nullopt, so processLogits never scans them for activity
    // and never launches the penalty kernels. That drops four pinned allocations,
    // their host fill and those scans from every step; fillSamplerCommonInputs
    // skips undefined arrays and every consumer reads them through
    // defined()/has_value().
    if (anyStreamPenaltyEnabled(stream_groups.allStreams())) {
        sampler_inputs.repetition_penalty   = torch::empty({bs}, pinned_f32);
        sampler_inputs.presence_penalty     = torch::empty({bs}, pinned_f32);
        sampler_inputs.frequency_penalty    = torch::empty({bs}, pinned_f32);
        sampler_inputs.no_repeat_ngram_size = torch::empty({bs}, pinned_i32);
    }
    sampler_inputs.do_sample            = torch::empty({bs}, pinned_bool);
    sampler_inputs.finished_mask        = torch::empty({bs}, torch::kBool);
    if (stream_groups.needReturnCumLogProbs()) {
        sampler_inputs.cum_log_probs = torch::empty({(int64_t)total_batch_size_in}, torch::kFloat32);
    }
    // Pin token_ids so Sampler::forward can non_blocking=true the H2D copy.
    // Without pinning, the .to(kCUDA) becomes a blocking pageable memcpy that
    // shows up as Memcpy Pageable→Device on the timeline (~33 MiB/rank/step
    // at bs=128 / step=65552).
    if (materialize_host_token_ids) {
        sampler_inputs.token_ids =
            torch::empty({(int64_t)total_batch_size_in, (int64_t)(sampler_inputs.step + 1)}, pinned_i32);
    }
    sampler_inputs.generator.resize(total_batch_size_in);
    return sampler_inputs;
}

void NormalSamplerInputGatherer::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    int*      input_lengths        = sampler_inputs.input_lengths.data_ptr<int32_t>();
    int*      sequence_lengths     = sampler_inputs.sequence_lengths.data_ptr<int32_t>();
    uint64_t* num_beams_in         = reinterpret_cast<uint64_t*>(sampler_inputs.num_beams_in.data_ptr<int64_t>());
    uint64_t* num_beams_out        = reinterpret_cast<uint64_t*>(sampler_inputs.num_beams_out.data_ptr<int64_t>());
    uint32_t* top_k                = reinterpret_cast<uint32_t*>(sampler_inputs.top_k.data_ptr<int32_t>());
    float*    top_p                = sampler_inputs.top_p.data_ptr<float>();
    float*    temperature          = sampler_inputs.temperature.data_ptr<float>();
    // Undefined penalty arrays mean "no stream can run a penalty kernel"
    // (see allocateSamplerInputs); they are skipped instead of written through a
    // null pointer. Each is guarded on its own so a caller that materialized only
    // some of them still gets exactly those filled.
    float* repetition_penalty =
        sampler_inputs.repetition_penalty.defined() ? sampler_inputs.repetition_penalty.data_ptr<float>() : nullptr;
    float* presence_penalty =
        sampler_inputs.presence_penalty.defined() ? sampler_inputs.presence_penalty.data_ptr<float>() : nullptr;
    float* frequency_penalty =
        sampler_inputs.frequency_penalty.defined() ? sampler_inputs.frequency_penalty.data_ptr<float>() : nullptr;
    int32_t*  no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size.defined() ?
                                         sampler_inputs.no_repeat_ngram_size.data_ptr<int32_t>() :
                                         nullptr;
    bool*     do_sample            = reinterpret_cast<bool*>(sampler_inputs.do_sample.data_ptr());

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        int sampler_batch_size;
        if (score_batch) {
            sampler_batch_size = stream->scoreLen();
        } else if (stream->needTilingForSampling()) {
            sampler_batch_size = stream->nextBatchSize();
        } else {
            sampler_batch_size = stream->currentBatchSize();
        }
        if (sampler_inputs.cum_log_probs.defined()) {
            const auto& cum_log_probs = stream->cumLogProbs();
            memcpy(sampler_inputs.cum_log_probs.data_ptr<float>() + batch_idx,
                   cum_log_probs.data_ptr<float>(),
                   cum_log_probs.numel() * sizeof(float));
        }
        for (int i = 0; i < sampler_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            sequence_lengths[batch_idx]   = stream->seqLength() + (score_batch ? i : propose_step);
            num_beams_in[batch_idx]       = stream->currentNumBeams();
            num_beams_out[batch_idx]      = stream->nextNumBeams();
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            if (repetition_penalty != nullptr) {
                repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            }
            if (presence_penalty != nullptr) {
                presence_penalty[batch_idx] = stream->generateConfig()->presence_penalty;
            }
            if (frequency_penalty != nullptr) {
                frequency_penalty[batch_idx] = stream->generateConfig()->frequency_penalty;
            }
            do_sample[batch_idx]          = stream->generateConfig()->do_sample;
            if (!do_sample[batch_idx]) {
                top_k[batch_idx]       = 1;
                top_p[batch_idx]       = 1;
                temperature[batch_idx] = 1;
            }
            if (no_repeat_ngram_size != nullptr) {
                no_repeat_ngram_size[batch_idx] = stream->generateConfig()->no_repeat_ngram_size.value_or(0);
            }
            sampler_inputs.generator[batch_idx] = stream->getGenerator();
            batch_idx += 1;
        }
    }
}

void NormalSamplerInputGatherer::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    size_t                   idx       = 0;
    std::for_each(all_streams.begin(), all_streams.end(), [&state_ptr, &idx](auto& stream) {
        const size_t batch_size = stream->currentBatchSize();
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor) {
                state_ptr->insert(processor, idx, idx + batch_size);
            }
        }
        idx += batch_size;
    });
    RTP_LLM_CHECK_WITH_INFO(idx == sampler_inputs.batch_size,
                            "logits processor rows mismatch: got %zu, expected %zu",
                            idx,
                            sampler_inputs.batch_size);
    sampler_inputs.logits_processor_states_ptr = state_ptr;
}

}  // namespace rtp_llm
