#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/logits_processor/LogitsProcessorStates.h"
#include <unordered_set>

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params): device_(params.device) {
    RTP_LLM_LOG_INFO("sampler max_batch_size: %ld", params.max_batch_size);
    const auto max_batch_size = params.max_batch_size;
    eos_ids_host_             = device_->allocateBuffer({DataType::TYPE_INT32, {max_batch_size}, AllocationType::HOST});
    std::fill_n(eos_ids_host_->data<int32_t>(), max_batch_size, params.eos_id);
    eos_ids_ = device_->allocateBuffer({DataType::TYPE_INT32, {max_batch_size}, AllocationType::DEVICE}, {"eos_id"});
    device_->copy({*eos_ids_, *eos_ids_host_});
};

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t from_batch_idx      = 0;
    size_t sample_to_batch_idx = 0;
    size_t from_seq_idx        = 0;  // accumulates batch_size * num_beams

    auto beam_sizes = inputs.num_beams->data<uint64_t>();

    const auto& input_tokens = *inputs.token_ids;
    auto        success      = CACHED_HOST_BUF(TYPE_BOOL, {inputs.batch_size});
    preprocessLogits(inputs);

    do {
        auto current_beam_size = beam_sizes[sample_to_batch_idx];
        while (sample_to_batch_idx + 1 < inputs.batch_size
               && beam_sizes[sample_to_batch_idx + 1] == current_beam_size) {
            sample_to_batch_idx++;
        }

        // now from_batch_idx to sample_to_batch_idx have the same beam size, sample once.
        const auto sample_batch_size = sample_to_batch_idx - from_batch_idx + 1;
        const auto sample_seq_num    = sample_batch_size;
        const auto sample_to_seq_idx = from_seq_idx + sample_seq_num;

        auto       sample_tokens      = input_tokens.view(from_seq_idx, sample_seq_num);
        auto       sample_logits      = inputs.logits->view(from_seq_idx, sample_seq_num);
        auto       input_lengths      = inputs.input_lengths->view(from_batch_idx, sample_batch_size);
        const auto decoder_batch_size = inputs.sequence_lengths->shape()[0];
        auto       sequence_lengths   = from_batch_idx < decoder_batch_size ?
                                            inputs.sequence_lengths->view(
                                        from_batch_idx, min(sample_batch_size, decoder_batch_size - from_batch_idx)) :
                                            Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {0}, nullptr);

        BufferPtr sample_cum_log_probs;
        if (inputs.cum_log_probs) {
            sample_cum_log_probs = device_->allocateBuffer({inputs.cum_log_probs->type(), {sample_seq_num}});
            device_->copy({*sample_cum_log_probs, inputs.cum_log_probs->view(from_seq_idx, sample_seq_num)});
        }

#define MAY_GET_BUFFER_VIEW(buffer_ptr)                                                                                \
    (buffer_ptr.get() ? buffer_ptr->view(from_batch_idx, sample_batch_size) : Buffer::emptyBuffer())

        if (current_beam_size == 1) {
            auto random_seeds         = MAY_GET_BUFFER_VIEW(inputs.random_seeds);
            auto repetition_penalty   = MAY_GET_BUFFER_VIEW(inputs.repetition_penalty);
            auto min_lengths          = MAY_GET_BUFFER_VIEW(inputs.min_lengths);
            auto no_repeat_ngram_size = MAY_GET_BUFFER_VIEW(inputs.no_repeat_ngram_size);
            auto all_probs = (inputs.all_probs.get() ? inputs.all_probs->view(from_batch_idx, sample_seq_num) :
                                                       Buffer::emptyBuffer());
            auto greedy_output =
                device_->sampleGreedy({sample_logits,
                                       input_lengths,
                                       sequence_lengths,
                                       sample_tokens,
                                       inputs.step,
                                       *inputs.top_k,
                                       *inputs.top_p,
                                       *inputs.temperature,
                                       inputs.random_seeds ? (OptionalBufferRef)random_seeds : nullopt,
                                       inputs.repetition_penalty ? (OptionalBufferRef)repetition_penalty : nullopt,
                                       inputs.min_lengths ? (OptionalBufferRef)min_lengths : nullopt,
                                       *eos_ids_,
                                       inputs.no_repeat_ngram_size ? (OptionalBufferRef)no_repeat_ngram_size : nullopt,
                                       inputs.cum_log_probs ? (OptionalBufferRef)*sample_cum_log_probs : nullopt,
                                       nullopt,  // output_log_probs
                                       inputs.all_probs ? (OptionalBufferRef)all_probs : nullopt});
            if (greedy_output.success) {
                device_->copy({success->view(from_seq_idx, sample_seq_num), *greedy_output.success});
            } else {
                std::fill(success->dataWithOffset<bool>(from_seq_idx),
                          success->dataWithOffset<bool>(from_seq_idx) + sample_seq_num,
                          true);
            }
        } else {
            size_t beam_batch_size = (size_t)(sample_batch_size / current_beam_size);
            RTP_LLM_LOG_DEBUG("current_beam_size is %d", current_beam_size);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            RTP_LLM_CHECK_WITH_INFO((sample_batch_size % current_beam_size == 0),
                                    "sample_batch_size[%d] must devide by current_beam_size[%d]",
                                    sample_batch_size,
                                    current_beam_size);
            auto beam_search_sequence_lengths =
                inputs.beam_search_sequence_lengths->view(from_batch_idx, sample_batch_size);
            auto beam_index                     = inputs.beam_index->view(from_batch_idx, sample_batch_size);
            auto org_sample_logits_shape        = sample_logits.shape();
            auto org_sample_tokens_shape        = sample_tokens.shape();
            auto org_input_lengths_shape        = input_lengths.shape();
            auto org_sequence_lengths_shape     = beam_search_sequence_lengths.shape();
            auto org_sample_cum_log_probs_shape = sample_cum_log_probs->shape();
            auto org_beam_index_shape           = beam_index.shape();
            sample_logits.updateShape({beam_batch_size, (size_t)current_beam_size, (size_t)inputs.logits->shape()[1]});
            sample_tokens.updateShape({beam_batch_size, (size_t)current_beam_size, (size_t)input_tokens.shape()[1]});
            beam_search_sequence_lengths.updateShape({beam_batch_size, (size_t)current_beam_size});
            sample_cum_log_probs->updateShape({beam_batch_size, (size_t)current_beam_size});
            input_lengths.updateShape({beam_batch_size, (size_t)current_beam_size});
            beam_index.updateShape({beam_batch_size, (size_t)current_beam_size});
            auto sample_logits_device = device_->clone({sample_logits, AllocationType::DEVICE});
            auto sample_tokens_device = device_->clone({sample_tokens, AllocationType::DEVICE});
            auto input_lengths_device = device_->clone({input_lengths, AllocationType::DEVICE});
            auto beam_search_sequence_lengths_device =
                device_->clone({beam_search_sequence_lengths, AllocationType::DEVICE});
            auto sample_cum_log_probs_device = device_->clone({*sample_cum_log_probs, AllocationType::DEVICE});
            auto beam_index_device           = device_->clone({beam_index, AllocationType::DEVICE});

            device_->sampleBeamSearch({*sample_logits_device,
                                       *sample_tokens_device,
                                       *input_lengths_device,
                                       *beam_search_sequence_lengths_device,
                                       *sample_cum_log_probs_device,
                                       *beam_index_device});
            device_->copy({sample_logits, *sample_logits_device});
            device_->copy({sample_tokens, *sample_tokens_device});
            device_->copy({input_lengths, *input_lengths_device});
            device_->copy({beam_search_sequence_lengths, *beam_search_sequence_lengths_device});
            device_->copy({*sample_cum_log_probs, *sample_cum_log_probs_device});
            device_->copy({beam_index, *beam_index_device});

            sample_logits.updateShape(org_sample_logits_shape);
            sample_tokens.updateShape(org_sample_tokens_shape);
            beam_search_sequence_lengths.updateShape(org_sequence_lengths_shape);
            sample_cum_log_probs->updateShape(org_sample_cum_log_probs_shape);
            input_lengths.updateShape(org_input_lengths_shape);
            beam_index.updateShape(org_beam_index_shape);
            std::fill(success->dataWithOffset<bool>(from_seq_idx),
                      success->dataWithOffset<bool>(from_seq_idx) + sample_seq_num,
                      true);
        }
        if (inputs.cum_log_probs) {
            device_->copy({inputs.cum_log_probs->view(from_seq_idx, sample_seq_num), *sample_cum_log_probs});
        }
        from_batch_idx      = sample_to_batch_idx + 1;
        sample_to_batch_idx = from_batch_idx;
        from_seq_idx        = sample_to_seq_idx;
    } while (from_batch_idx < inputs.batch_size);
    // TODO(xinfei.sxf) 优化copy token_ids
    return SamplerOutput({move(inputs.token_ids),
                          move(inputs.cum_log_probs),
                          move(inputs.all_probs),
                          move(inputs.beam_index),
                          move(success)});
}

void Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
}

}  // namespace rtp_llm
