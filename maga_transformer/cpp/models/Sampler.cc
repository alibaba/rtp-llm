#include "maga_transformer/cpp/models/Sampler.h"

#include <unordered_set>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params)
    : device_(params.device)
    , max_batch_size_(params.max_batch_size)
    {
        auto eos_ids_host = device_->allocateBuffer(
            {DataType::TYPE_INT32, {max_batch_size_}, AllocationType::HOST});
        std::fill_n(eos_ids_host->data<int32_t>(), max_batch_size_, params.eos_id);
        eos_ids_ = device_->allocateBuffer(
            {DataType::TYPE_INT32, {max_batch_size_}, AllocationType::DEVICE});
        device_->copy({*eos_ids_, *eos_ids_host});
    };

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    size_t from_batch_idx = 0;
    size_t sample_to_batch_idx = 0;
    size_t from_token_idx = 0; // accumulates batch_size * num_beams

    auto beam_sizes = inputs.num_beams->data<uint64_t>();
    auto current_beam_size = beam_sizes[0];

    const auto& input_tokens = *inputs.token_ids;

    do {
        while (sample_to_batch_idx + 1 < inputs.batch_size &&
               beam_sizes[sample_to_batch_idx + 1] == current_beam_size)
        {
            sample_to_batch_idx++;
        }

        // now from_batch_idx to sample_to_batch_idx have the same beam size, sample once.
        const auto sample_batch_size = sample_to_batch_idx - from_batch_idx + 1;
        const auto sample_token_batch_size = sample_batch_size * current_beam_size;
        const auto sample_to_token_idx = from_token_idx + sample_token_batch_size;

        auto sample_tokens = device_->allocateBuffer(
            {input_tokens.type(), {sample_batch_size * current_beam_size, inputs.step + 1}});
        auto sequence_lengths = device_->allocateBuffer(
                {inputs.sequence_lengths->type(), inputs.sequence_lengths->shape()});
        device_->copy({*sequence_lengths, *inputs.sequence_lengths});
        device_->copy({*sample_tokens, input_tokens, 0, from_token_idx, sample_token_batch_size});
        auto transposed_tokens = device_->transpose({*sample_tokens});
        auto sample_logits = inputs.logits->view(from_token_idx, sample_token_batch_size);
        auto sample_cum_log_probs = inputs.cum_log_probs->view(from_batch_idx, sample_batch_size);

#define MAY_GET_BUFFER_VIEW(buffer_ptr) \
        (buffer_ptr.get() ? buffer_ptr->view(from_batch_idx, sample_batch_size) : Buffer::emptyBuffer())

        if (current_beam_size == 1) {
            auto random_seeds = MAY_GET_BUFFER_VIEW(inputs.random_seeds);
            auto repetition_penalty = MAY_GET_BUFFER_VIEW(inputs.repetition_penalty);
            auto min_lengths = MAY_GET_BUFFER_VIEW(inputs.min_lengths);
            auto top_p_decay = MAY_GET_BUFFER_VIEW(inputs.top_p_decay);
            auto top_p_min = MAY_GET_BUFFER_VIEW(inputs.top_p_min);
            auto top_p_reset_ids = MAY_GET_BUFFER_VIEW(inputs.top_p_reset_ids);

            device_->sampleGreedy({
                sample_logits,
                *sequence_lengths,
                *transposed_tokens,
                inputs.step,
                *inputs.top_k,
                *inputs.top_p,
                *inputs.temperature,
                inputs.random_seeds ? (OptionalBufferRef)random_seeds : nullopt,
                inputs.repetition_penalty ? (OptionalBufferRef)repetition_penalty : nullopt,
                inputs.min_lengths ? (OptionalBufferRef)min_lengths : nullopt,
                inputs.top_p_decay ? (OptionalBufferRef)top_p_decay : nullopt,
                inputs.top_p_min ? (OptionalBufferRef)top_p_min : nullopt,
                inputs.top_p_reset_ids ? (OptionalBufferRef)top_p_reset_ids : nullopt,
                sample_cum_log_probs,
                nullopt, // output_log_probs
                inputs.index_log_prob.get() ? (OptionalBufferRef)*inputs.index_log_prob: nullopt
            });
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }

        auto output_tokens = device_->transpose({*transposed_tokens});
        device_->copy({input_tokens, *output_tokens, from_token_idx, 0, sample_token_batch_size});

        // current sampling done, update indices
        from_batch_idx = sample_to_batch_idx + 1;
        sample_to_batch_idx = from_batch_idx;
        from_token_idx = sample_to_token_idx;
    } while (from_batch_idx < inputs.batch_size);
    return SamplerOutput({move(inputs.token_ids), move(inputs.cum_log_probs), move(inputs.index_log_prob)});
}

} // namespace rtp_llm
