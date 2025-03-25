#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include <unordered_set>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params)
    : device_(params.device)
    {
        FT_LOG_INFO("sampler max_batch_size: %ld", params.max_batch_size);
        const auto max_batch_size = params.max_batch_size;
        eos_ids_host_ = device_->allocateBuffer(
            {DataType::TYPE_INT32, {max_batch_size}, AllocationType::HOST});
        std::fill_n(eos_ids_host_->data<int32_t>(), max_batch_size, params.eos_id);
        eos_ids_ = device_->allocateBuffer(
            {DataType::TYPE_INT32, {max_batch_size}, AllocationType::DEVICE}, {"eos_id"});
        device_->copy({*eos_ids_, *eos_ids_host_});
    };

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t from_batch_idx = 0;
    size_t sample_to_batch_idx = 0;
    size_t from_seq_idx = 0; // accumulates batch_size * num_beams

    auto beam_sizes = inputs.num_beams->data<uint64_t>();
    auto current_beam_size = beam_sizes[0];

    const auto& input_tokens = *inputs.token_ids;
    auto success = device_->allocateBuffer({DataType::TYPE_BOOL, {inputs.batch_size}, AllocationType::HOST});
    do {
        while (sample_to_batch_idx + 1 < inputs.batch_size &&
               beam_sizes[sample_to_batch_idx + 1] == current_beam_size)
        {
            sample_to_batch_idx++;
        }

        // now from_batch_idx to sample_to_batch_idx have the same beam size, sample once.
        const auto sample_batch_size = sample_to_batch_idx - from_batch_idx + 1;
        const auto sample_seq_num = sample_batch_size;
        const auto sample_to_seq_idx = from_seq_idx + sample_seq_num;

        auto sample_tokens = input_tokens.view(from_seq_idx, sample_seq_num);
        auto sample_logits = inputs.logits->view(from_seq_idx, sample_seq_num);
        auto input_lengths = inputs.input_lengths->view(from_batch_idx, sample_batch_size);
        const auto decoder_batch_size = inputs.sequence_lengths->shape()[0];
        auto sequence_lengths = from_batch_idx < decoder_batch_size
            ? inputs.sequence_lengths->view(from_batch_idx,
                                            min(sample_batch_size, decoder_batch_size - from_batch_idx))
            : Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {0}, nullptr);

        BufferPtr sample_cum_log_probs;
        if (inputs.cum_log_probs) {
            sample_cum_log_probs = device_->allocateBuffer(
                    {inputs.cum_log_probs->type(), {sample_seq_num}});
            device_->copy({*sample_cum_log_probs, inputs.cum_log_probs->view(from_seq_idx, sample_seq_num)});
        }

#define MAY_GET_BUFFER_VIEW(buffer_ptr) \
        (buffer_ptr.get() ? buffer_ptr->view(from_batch_idx, sample_batch_size) : Buffer::emptyBuffer())

        if (current_beam_size == 1) {
            auto random_seeds = MAY_GET_BUFFER_VIEW(inputs.random_seeds);
            auto repetition_penalty = MAY_GET_BUFFER_VIEW(inputs.repetition_penalty);
            auto min_lengths = MAY_GET_BUFFER_VIEW(inputs.min_lengths);
            auto no_repeat_ngram_size = MAY_GET_BUFFER_VIEW(inputs.no_repeat_ngram_size);
            auto all_probs = (inputs.all_probs.get() ? inputs.all_probs->view(from_batch_idx, sample_seq_num) : Buffer::emptyBuffer());
            auto greedy_output = device_->sampleGreedy({
                sample_logits,
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
                nullopt, // output_log_probs
                inputs.all_probs ? (OptionalBufferRef) all_probs: nullopt
            });
            if (greedy_output.success) {
                device_->copy({success->view(from_seq_idx, sample_seq_num), *greedy_output.success});
            } else {
                std::fill(success->dataWithOffset<bool>(from_seq_idx),
                          success->dataWithOffset<bool>(from_seq_idx) + sample_seq_num,
                          true);
            }
        } else {
            size_t beam_batch_size = (size_t)(sample_batch_size / current_beam_size);
            FT_LOG_DEBUG("current_beam_size is %d", current_beam_size);
            FT_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            FT_CHECK_WITH_INFO((sample_batch_size % current_beam_size == 0),
                "sample_batch_size[%d] must devide by current_beam_size[%d]");
            auto beam_search_sequence_lengths = inputs.beam_search_sequence_lengths->view(from_batch_idx, sample_batch_size);
            auto beam_index = inputs.beam_index->view(from_batch_idx, sample_batch_size);
            auto org_sample_logits_shape = sample_logits.shape();
            auto org_sample_tokens_shape = sample_tokens.shape();
            auto org_input_lengths_shape = input_lengths.shape();
            auto org_sequence_lengths_shape = beam_search_sequence_lengths.shape();
            auto org_sample_cum_log_probs_shape = sample_cum_log_probs->shape();
            auto org_beam_index_shape    = beam_index.shape();
            sample_logits.updateShape({beam_batch_size,
                                         (size_t)current_beam_size,
                                         (size_t)inputs.logits->shape()[1]});
            sample_tokens.updateShape({beam_batch_size,
                                         (size_t)current_beam_size,
                                         (size_t)input_tokens.shape()[1]});
            beam_search_sequence_lengths.updateShape({beam_batch_size,
                                          (size_t)current_beam_size});
            sample_cum_log_probs->updateShape({beam_batch_size,
                                                (size_t)current_beam_size});
            input_lengths.updateShape({beam_batch_size,
                                        (size_t)current_beam_size});
            beam_index.updateShape({beam_batch_size,
                                    (size_t)current_beam_size});
            auto sample_logits_device = device_->clone({sample_logits, AllocationType::DEVICE});
            auto sample_tokens_device = device_->clone({sample_tokens, AllocationType::DEVICE});
            auto input_lengths_device = device_->clone({input_lengths, AllocationType::DEVICE});
            auto beam_search_sequence_lengths_device = device_->clone({beam_search_sequence_lengths, AllocationType::DEVICE});
            auto sample_cum_log_probs_device = device_->clone({*sample_cum_log_probs, AllocationType::DEVICE});
            auto beam_index_device = device_->clone({beam_index, AllocationType::DEVICE});

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
        thinkLogicProcess(inputs, from_seq_idx, sample_seq_num);
        
        from_batch_idx = sample_to_batch_idx + 1;
        sample_to_batch_idx = from_batch_idx;
        from_seq_idx = sample_to_seq_idx;
    } while (from_batch_idx < inputs.batch_size);
    // TODO(xinfei.sxf) 优化copy token_ids
    return SamplerOutput({move(inputs.token_ids),
                          move(inputs.cum_log_probs),
                          move(inputs.all_probs),
                          move(inputs.beam_index),
                          move(success)});
}

void Sampler::thinkLogicProcess(const SamplerInputs& inputs, size_t from_seq_idx, size_t sample_seq_num) {
    int* input_lengths = inputs.input_lengths->data<int32_t>();
    int* sequence_lengths = inputs.sequence_lengths->data<int32_t>();
    int* max_thinking_tokens  = inputs.max_thinking_tokens->data<int32_t>();
    for (size_t idx = from_seq_idx; idx < from_seq_idx + sample_seq_num; idx++) {
        bool think_mode = inputs.think_modes;
        auto dfa_ptr = inputs.think_status_dfa_ptrs[idx];
        int num_new_tokens = 1;
        if (think_mode) {
            bool enforce = (sequence_lengths[idx] + num_new_tokens >= max_thinking_tokens[idx] + input_lengths[idx]);
            auto token_ids = inputs.token_ids->index(idx);
            auto logits = inputs.logits->index(idx);
            dfaForwardWithLogits(dfa_ptr, token_ids, logits, num_new_tokens, inputs.end_think_token_ids, inputs.vocab_size, enforce);
        }
    }
}

void Sampler::dfaForwardWithLogits(shared_ptr<StringContainDFA<size_t, int>> dfa_ptr, 
    ft::BufferPtr tokens_ids, ft::BufferPtr new_tokens_logits, int num_new_tokens, 
    std::vector<int> template_token_ids, size_t vocab_size, bool enforce) 
{
    const size_t step = tokens_ids->shape()[0];
    size_t original_status = dfa_ptr->status();
    for (size_t j = 0; j < num_new_tokens; ++j) {
        auto current_token_id = *(tokens_ids->dataWithOffset<int>(step - num_new_tokens + j));
        if (!dfa_ptr->isFinished()) {
            dfa_ptr->next(current_token_id);
        }
    }
    if (!dfa_ptr->isFinished() && enforce) {
        int offset = 0;
        for (size_t pos = original_status; pos < template_token_ids.size() && offset < num_new_tokens; pos++, offset++) {
            FT_LOG_INFO("sampler enforce transfer status");
            *(tokens_ids->dataWithOffset<int>(step - num_new_tokens + offset)) = template_token_ids[pos];
            memFill(new_tokens_logits, vocab_size, (size_t) template_token_ids[pos]);
            dfa_ptr->forceSetStatus(pos + 1);
        }
    }
}

void Sampler::memFill(ft::BufferPtr new_tokens_logits, size_t vocab_size, size_t index) {
    device_->bufMemset(*new_tokens_logits, 0);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor[index] = 1;
}

} // namespace rtp_llm
