#include "rtp_llm/cpp/models/Sampler.h"
#include "autil/Scope.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include <unordered_set>

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params): device_(params.device) {}

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#define MAY_GET_BUFFER_VIEW(buffer_ptr, offset, size)                                                                  \
    (buffer_ptr.get() ? buffer_ptr->view((offset), (size)) : Buffer::emptyBuffer())

#define SCOPED_UPDATE_BUFFER_SHAPE(buffer, ...)                                                                        \
    const auto        org_##buffer##_shape__ = buffer.shape();                                                         \
    autil::ScopeGuard guard_##buffer([&]() { buffer.updateShape(org_##buffer##_shape__); });                           \
    buffer.updateShape(__VA_ARGS__);

    preprocessLogits(inputs);

    uint64_t max_seq_len   = inputs.token_ids->shape()[1];
    auto     num_beams_in  = inputs.num_beams_in->data<uint64_t>();
    auto     num_beams_out = inputs.num_beams_out->data<uint64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    bool variable_num_beams = inputs.batch_size != inputs.batch_size_out;

    // allocate output buffers
    auto all_success = CACHED_HOST_BUF(TYPE_BOOL, {inputs.batch_size});
    auto all_beam_indices =
        has_num_beams ? device_->allocateBuffer({DataType::TYPE_INT32, {inputs.batch_size_out}, AllocationType::HOST}) :
                        nullptr;
    auto all_token_ids_out =
        variable_num_beams ? device_->allocateBuffer(
                                 {DataType::TYPE_INT32, {inputs.batch_size_out, max_seq_len}, AllocationType::HOST}) :
                             inputs.token_ids;
    auto all_cum_log_probs_out =
        variable_num_beams && inputs.cum_log_probs ?
            device_->allocateBuffer({DataType::TYPE_FP32, {inputs.batch_size_out}, AllocationType::HOST}) :
            inputs.cum_log_probs;

    size_t from_batch_idx_in = 0, to_batch_idx_in = 0;
    size_t from_batch_idx_out = 0;

    while (from_batch_idx_in < inputs.batch_size) {
        auto cur_num_beams_in  = num_beams_in[from_batch_idx_in];
        auto cur_num_beams_out = num_beams_out[from_batch_idx_in];
        ++to_batch_idx_in;
        while (to_batch_idx_in < inputs.batch_size && num_beams_in[to_batch_idx_in] == cur_num_beams_in
               && num_beams_out[to_batch_idx_in] == cur_num_beams_out) {
            ++to_batch_idx_in;
        }

        // now from_batch_idx to to_batch_idx have the same beam size, sample once.
        const auto batch_size_in    = to_batch_idx_in - from_batch_idx_in;
        const auto beam_batch_size  = batch_size_in / cur_num_beams_in;
        const auto batch_size_out   = beam_batch_size * cur_num_beams_out;
        const auto to_batch_idx_out = from_batch_idx_out + batch_size_out;

        auto success           = all_success->view(from_batch_idx_in, batch_size_in);
        auto logits            = inputs.logits->view(from_batch_idx_in, batch_size_in);
        auto token_ids_in      = inputs.token_ids->view(from_batch_idx_in, batch_size_in);
        auto token_ids_out     = all_token_ids_out->view(from_batch_idx_out, batch_size_out);
        auto input_lengths     = inputs.input_lengths->view(from_batch_idx_in, batch_size_in);
        auto sequence_lengths  = inputs.sequence_lengths->view(from_batch_idx_in, batch_size_in);
        auto cum_log_probs_in  = MAY_GET_BUFFER_VIEW(inputs.cum_log_probs, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_out = MAY_GET_BUFFER_VIEW(all_cum_log_probs_out, from_batch_idx_out, batch_size_out);

        if (cur_num_beams_in == 1 && cur_num_beams_out == 1) {
            const auto decoder_batch_size = inputs.sequence_lengths->shape()[0];
            auto       sequence_lengths_in =
                from_batch_idx_in < decoder_batch_size ?
                          inputs.sequence_lengths->view(from_batch_idx_in,
                                                  min(batch_size_in, decoder_batch_size - from_batch_idx_in)) :
                          Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {0}, nullptr);

            // TODO(zhangjianning.zjn): would be better to eliminate the copy
            if (inputs.cum_log_probs) {
                device_->copy({cum_log_probs_out, cum_log_probs_in});
            }

            auto top_k              = inputs.top_k->view(from_batch_idx_in, batch_size_in);
            auto top_p              = inputs.top_p->view(from_batch_idx_in, batch_size_in);
            auto temperature        = inputs.temperature->view(from_batch_idx_in, batch_size_in);
            auto repetition_penalty = MAY_GET_BUFFER_VIEW(inputs.repetition_penalty, from_batch_idx_in, batch_size_in);
            auto presence_penalty   = MAY_GET_BUFFER_VIEW(inputs.presence_penalty, from_batch_idx_in, batch_size_in);
            auto frequency_penalty  = MAY_GET_BUFFER_VIEW(inputs.frequency_penalty, from_batch_idx_in, batch_size_in);
            auto no_repeat_ngram_size =
                MAY_GET_BUFFER_VIEW(inputs.no_repeat_ngram_size, from_batch_idx_in, batch_size_in);
            auto all_probs = (inputs.all_probs.get() ? inputs.all_probs->view(from_batch_idx_in, batch_size_in) :
                                                       Buffer::emptyBuffer());
            auto do_sample = MAY_GET_BUFFER_VIEW(inputs.do_sample, from_batch_idx_in, batch_size_in);
            auto generator = std::vector<at::Generator>{inputs.generator.begin() + from_batch_idx_in,
                                                        inputs.generator.begin() + from_batch_idx_in + batch_size_in};
            auto logit_bias =
                std::vector<std::map<int, float>>{inputs.logit_bias.begin() + from_batch_idx_in,
                                                  inputs.logit_bias.begin() + from_batch_idx_in + batch_size_in};
            auto greedy_output =
                device_->sampleGreedy({logits,
                                       input_lengths,
                                       sequence_lengths,
                                       token_ids_in,
                                       inputs.step,
                                       top_k,
                                       top_p,
                                       temperature,
                                       inputs.repetition_penalty ? (OptionalBufferRef)repetition_penalty : nullopt,
                                       inputs.no_repeat_ngram_size ? (OptionalBufferRef)no_repeat_ngram_size : nullopt,
                                       inputs.cum_log_probs ? (OptionalBufferRef)cum_log_probs_out : nullopt,
                                       nullopt,  // output_log_probs
                                       inputs.all_probs ? (OptionalBufferRef)all_probs : nullopt,
                                       inputs.presence_penalty ? (OptionalBufferRef)presence_penalty : nullopt,
                                       inputs.frequency_penalty ? (OptionalBufferRef)frequency_penalty : nullopt,
                                       inputs.do_sample ? (OptionalBufferRef)do_sample : nullopt,
                                       generator,
                                       logit_bias});
            if (greedy_output.success) {
                device_->copy({success, *greedy_output.success});
                // TODO(zhangjianning.zjn): would be better to eliminate the copy
                if (variable_num_beams) {
                    device_->copy({token_ids_out, token_ids_in});
                }
            } else {
                std::fill(success.data<bool>(), success.data<bool>() + batch_size_in, true);
            }
        } else {
            RTP_LLM_LOG_DEBUG("current_num_beams_in is %d", cur_num_beams_in);
            RTP_LLM_LOG_DEBUG("current_num_beams_out is %d", cur_num_beams_out);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            RTP_LLM_CHECK_WITH_INFO((batch_size_in % cur_num_beams_in == 0),
                                    "sample_batch_size[%d] must devide by current_num_beams_in[%d]");

            const size_t vocab_size  = inputs.logits->shape()[1];
            const size_t max_seq_len = inputs.token_ids->shape()[1];

            auto beam_indices = all_beam_indices->view(from_batch_idx_out, batch_size_out);

            SCOPED_UPDATE_BUFFER_SHAPE(logits, {beam_batch_size, (size_t)cur_num_beams_in, vocab_size});
            SCOPED_UPDATE_BUFFER_SHAPE(token_ids_in, {beam_batch_size, (size_t)cur_num_beams_in, max_seq_len});
            SCOPED_UPDATE_BUFFER_SHAPE(token_ids_out, {beam_batch_size, (size_t)cur_num_beams_out, max_seq_len});
            SCOPED_UPDATE_BUFFER_SHAPE(input_lengths, {beam_batch_size, (size_t)cur_num_beams_in});
            SCOPED_UPDATE_BUFFER_SHAPE(sequence_lengths, {beam_batch_size, (size_t)cur_num_beams_in});
            SCOPED_UPDATE_BUFFER_SHAPE(cum_log_probs_in, {beam_batch_size, (size_t)cur_num_beams_in});
            SCOPED_UPDATE_BUFFER_SHAPE(cum_log_probs_out, {beam_batch_size, (size_t)cur_num_beams_out});

            auto logits_device           = device_->clone({logits, AllocationType::DEVICE});
            auto token_ids_in_device     = device_->clone({token_ids_in, AllocationType::DEVICE});
            auto input_lengths_device    = device_->clone({input_lengths, AllocationType::DEVICE});
            auto sequence_lengths_device = device_->clone({sequence_lengths, AllocationType::DEVICE});
            auto cum_log_probs_in_device = device_->clone({cum_log_probs_in, AllocationType::DEVICE});

            auto output = device_->sampleBeamSearch({*logits_device,
                                                     token_ids_in_device,
                                                     input_lengths_device,
                                                     sequence_lengths_device,
                                                     cum_log_probs_in_device,
                                                     cur_num_beams_out});

            device_->copy({token_ids_out, *output.token_ids});
            device_->copy({cum_log_probs_out, *output.cum_log_probs});
            device_->copy({beam_indices, *output.beam_indices});

            std::fill(success.data<bool>(), success.data<bool>() + batch_size_in, true);
        }

        // prepare for next sampling
        from_batch_idx_in  = to_batch_idx_in;
        from_batch_idx_out = to_batch_idx_out;
    }
    // TODO(xinfei.sxf) 优化copy token_ids
    return SamplerOutput({std::move(all_token_ids_out),
                          std::move(all_cum_log_probs_out),
                          std::move(inputs.all_probs),
                          std::move(all_beam_indices),
                          std::move(all_success)});
}

void Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
}

}  // namespace rtp_llm
