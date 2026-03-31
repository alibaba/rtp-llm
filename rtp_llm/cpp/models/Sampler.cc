#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include <unordered_set>

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params) {}

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    preprocessLogits(inputs);

    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    bool variable_num_beams = inputs.batch_size != inputs.batch_size_out;

    // allocate output tensors
    auto all_success = torch::empty({(int64_t)inputs.batch_size}, torch::kBool);
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
    // When !variable_num_beams, share data with inputs — sampleGreedy writes in-place
    auto all_token_ids_out     = variable_num_beams ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len}, torch::kInt32) :
                                     inputs.token_ids;
    auto all_cum_log_probs_out = variable_num_beams && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
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

        auto success           = all_success.narrow(0, from_batch_idx_in, batch_size_in);
        auto logits            = inputs.logits.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_in      = inputs.token_ids.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_out     = all_token_ids_out.narrow(0, from_batch_idx_out, batch_size_out);
        auto input_lengths     = inputs.input_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto sequence_lengths  = inputs.sequence_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_in  = mayNarrow(inputs.cum_log_probs, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_out = mayNarrow(all_cum_log_probs_out, from_batch_idx_out, batch_size_out);

        if (cur_num_beams_in == 1 && cur_num_beams_out == 1) {
            const auto decoder_batch_size = (int64_t)inputs.sequence_lengths.size(0);
            auto       sequence_lengths_in =
                (int64_t)from_batch_idx_in < decoder_batch_size ?
                          inputs.sequence_lengths.narrow(
                        0,
                        from_batch_idx_in,
                        min((int64_t)batch_size_in, decoder_batch_size - (int64_t)from_batch_idx_in)) :
                          torch::empty({0}, torch::kInt32);

            // TODO(zhangjianning.zjn): would be better to eliminate the copy
            if (cum_log_probs_out.defined() && cum_log_probs_in.defined()) {
                cum_log_probs_out.copy_(cum_log_probs_in);
            }

            auto top_k                = inputs.top_k.narrow(0, from_batch_idx_in, batch_size_in);
            auto top_p                = inputs.top_p.narrow(0, from_batch_idx_in, batch_size_in);
            auto temperature          = inputs.temperature.narrow(0, from_batch_idx_in, batch_size_in);
            auto repetition_penalty   = mayOptNarrow(inputs.repetition_penalty, from_batch_idx_in, batch_size_in);
            auto presence_penalty     = mayOptNarrow(inputs.presence_penalty, from_batch_idx_in, batch_size_in);
            auto frequency_penalty    = mayOptNarrow(inputs.frequency_penalty, from_batch_idx_in, batch_size_in);
            auto no_repeat_ngram_size = mayOptNarrow(inputs.no_repeat_ngram_size, from_batch_idx_in, batch_size_in);
            auto all_probs            = mayOptNarrow(inputs.all_probs, from_batch_idx_in, batch_size_in);
            auto do_sample            = mayOptNarrow(inputs.do_sample, from_batch_idx_in, batch_size_in);
            auto generator            = std::vector<at::Generator>{inputs.generator.begin() + from_batch_idx_in,
                                                                   inputs.generator.begin() + from_batch_idx_in + batch_size_in};

            auto greedy_output = execSampleGreedy(
                {logits,
                 input_lengths,
                 sequence_lengths_in,
                 token_ids_in,
                 inputs.step,
                 top_k,
                 top_p,
                 temperature,
                 repetition_penalty,
                 no_repeat_ngram_size,
                 cum_log_probs_out.defined() ? std::optional<torch::Tensor>(cum_log_probs_out) : std::nullopt,
                 std::nullopt,  // output_log_probs
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator});
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
                // TODO(zhangjianning.zjn): would be better to eliminate the copy
                if (variable_num_beams) {
                    memcpy(token_ids_out.data_ptr(), token_ids_in.data_ptr(), token_ids_in.nbytes());
                }
            } else {
                success.fill_(true);
            }
        } else {
            RTP_LLM_LOG_DEBUG("current_num_beams_in is %d", cur_num_beams_in);
            RTP_LLM_LOG_DEBUG("current_num_beams_out is %d", cur_num_beams_out);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            RTP_LLM_CHECK_WITH_INFO((batch_size_in % cur_num_beams_in == 0),
                                    "sample_batch_size[%d] must devide by current_num_beams_in[%d]");

            const size_t vocab_size      = inputs.logits.size(1);
            const size_t max_seq_len_val = inputs.token_ids.size(1);

            auto beam_indices = all_beam_indices.narrow(0, from_batch_idx_out, batch_size_out);

            // Reshape for beam search: [batch, beams, ...]
            auto logits_reshaped =
                logits.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)vocab_size});
            auto token_ids_in_reshaped =
                token_ids_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)max_seq_len_val});
            auto input_lengths_reshaped = input_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto sequence_lengths_reshaped =
                sequence_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto cum_log_probs_in_reshaped =
                cum_log_probs_in.defined() ?
                    cum_log_probs_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in}) :
                    torch::zeros({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});

            auto logits_t           = logits_reshaped.to(torch::kCUDA);
            auto token_ids_in_t     = token_ids_in_reshaped.to(torch::kCUDA);
            auto input_lengths_t    = input_lengths_reshaped.to(torch::kCUDA);
            auto sequence_lengths_t = sequence_lengths_reshaped.to(torch::kCUDA);
            auto cum_log_probs_in_t = cum_log_probs_in_reshaped.to(torch::kCUDA);

            auto output = execSampleBeamSearch({logits_t,
                                                token_ids_in_t,
                                                input_lengths_t,
                                                sequence_lengths_t,
                                                cum_log_probs_in_t,
                                                (size_t)cur_num_beams_out});

            auto token_ids_out_reshaped =
                token_ids_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out, (int64_t)max_seq_len_val});
            auto cum_log_probs_out_reshaped =
                cum_log_probs_out.defined() ?
                    cum_log_probs_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}) :
                    torch::Tensor();

            token_ids_out_reshaped.copy_(output.token_ids);
            if (cum_log_probs_out_reshaped.defined()) {
                cum_log_probs_out_reshaped.copy_(output.cum_log_probs);
            }
            beam_indices.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}).copy_(output.beam_indices);

            success.fill_(true);
        }

        // prepare for next sampling
        from_batch_idx_in  = to_batch_idx_in;
        from_batch_idx_out = to_batch_idx_out;
    }

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
