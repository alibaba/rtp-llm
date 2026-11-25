#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <algorithm>
#include <exception>
#include <unordered_set>
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params):
    fixed_max_batch_size_(params.max_batch_size > 0 && params.fixed_max_batch_size),
    copy_stream_(cuda_graph::graphGetStreamFromPool(false)) {
    if (params.max_batch_size > 0) {
        allocateGreedySamplingBuffers(params.max_batch_size);
    }
}

void Sampler::allocateGreedySamplingBuffers(size_t max_batch_size) {
    waitGreedySamplingBufferEvents();
    max_batch_size_ = max_batch_size;
    auto pinned_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    for (auto& slot : greedy_sampling_buffer_slots_) {
        auto& buffers                = slot.buffers;
        buffers.seed_host            = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.offset_host          = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.output_ids_ptrs_host = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.max_batch_size       = max_batch_size_;
        slot.ready_event.reset();
    }
}

void Sampler::ensureGreedySamplingBuffers(size_t batch_size) {
    if (batch_size <= max_batch_size_) {
        return;
    }
    // Fixed users fail fast on impossible batch sizes. Dynamic users wait for all
    // pending slot events before rebuilding every slot together.
    RTP_LLM_CHECK_WITH_INFO(!fixed_max_batch_size_,
                            "sampler batch size [%lu] exceeds initialized max batch size [%lu]",
                            batch_size,
                            max_batch_size_);
    RTP_LLM_LOG_INFO("grow greedy sampling buffers from batch size [%lu] to [%lu]", max_batch_size_, batch_size);
    allocateGreedySamplingBuffers(batch_size);
}

void Sampler::waitGreedySamplingBufferEvents() {
    for (auto& slot : greedy_sampling_buffer_slots_) {
        if (slot.ready_event) {
            slot.ready_event->synchronize();
            slot.ready_event.reset();
        }
    }
}

GreedySamplingBuffers& Sampler::nextGreedySamplingBuffers(size_t batch_size) {
    ensureGreedySamplingBuffers(batch_size);
    auto& slot = greedy_sampling_buffer_slots_[greedy_sampling_buffer_index_];
    if (slot.ready_event) {
        slot.ready_event->synchronize();
        slot.ready_event.reset();
    }
    current_greedy_sampling_slot_ = &slot;
    greedy_sampling_buffer_index_ = (greedy_sampling_buffer_index_ + 1) % greedy_sampling_buffer_slots_.size();
    return slot.buffers;
}

void Sampler::markGreedySamplingBufferReady() {
    if (current_greedy_sampling_slot_ != nullptr) {
        auto* slot = current_greedy_sampling_slot_;
        try {
            slot->ready_event             = runtimeCreateEvent();
            current_greedy_sampling_slot_ = nullptr;
        } catch (...) {
            current_greedy_sampling_slot_ = nullptr;
            slot->ready_event.reset();
            runtimeSyncAndCheck();
            throw;
        }
    }
}
SamplerOutput Sampler::prepareDeferredVerification(const SamplerInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("sampler.prepare_deferred_verification");
    const auto rows = static_cast<int64_t>(inputs.batch_size);
    TORCH_CHECK(inputs.rejection_rows > 0 && rows > 0 && rows % inputs.rejection_rows == 0,
                "fixed-length verification requires one complete block of target rows per request");
    const int64_t batch = rows / inputs.rejection_rows;
    TORCH_CHECK(inputs.rejection_generators.empty() || inputs.rejection_generators.size() == batch,
                "expected one rejection generator per request");

    // do_sample=false and explicit top_k=1 both mean greedy, including in mixed batches.
    auto        mask        = torch::empty({batch}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    const auto* sample      = inputs.do_sample.defined() ? inputs.do_sample.data_ptr<bool>() : nullptr;
    const auto* top_k       = inputs.top_k.data_ptr<int32_t>();
    auto*       output_mask = mask.data_ptr<bool>();
    for (int64_t b = 0; b < batch; ++b) {
        const auto row = b * inputs.rejection_rows;
        output_mask[b] = (!sample || sample[row]) && top_k[row] != 1;
    }

    SamplerOutput output;
    output.verify_do_sample = mask.to(inputs.logits.device(), /*non_blocking=*/true);
    output.success          = torch::ones({rows}, inputs.logits.options().dtype(torch::kBool));
    output.token_ids        = inputs.token_ids;
    output.cum_log_probs    = inputs.cum_log_probs;

    GreedyParams params{};
    params.logits           = inputs.logits;
    params.input_lengths    = inputs.input_lengths;
    params.sequence_lengths = inputs.sequence_lengths;
    params.token_ids        = inputs.token_ids;
    params.step             = inputs.step;
    params.top_k            = inputs.top_k;
    params.top_p            = inputs.top_p;
    params.temperature      = inputs.temperature;
    if (inputs.do_sample.defined()) {
        params.do_sample = inputs.do_sample;
    }
    if (inputs.repetition_penalty.defined()) {
        params.repetition_penalty = inputs.repetition_penalty;
    }
    if (inputs.presence_penalty.defined()) {
        params.presence_penalty = inputs.presence_penalty;
    }
    if (inputs.frequency_penalty.defined()) {
        params.frequency_penalty = inputs.frequency_penalty;
    }
    if (inputs.no_repeat_ngram_size.defined()) {
        params.no_repeat_ngram_size = inputs.no_repeat_ngram_size;
    }
    params.defer_token_sampling          = true;
    params.deferred_rejection_generators = inputs.rejection_generators;
    params.deferred_rejection_rows       = inputs.rejection_generators.empty() ? 0 : inputs.rejection_rows;
    params.buffer_holder                 = &deferred_buffers_;

    auto filtered = execSampleGreedy(params);
    TORCH_CHECK(filtered.deferred_token_sampling, "target preparation must return deferred verification inputs");
    output.deferred_token_sampling = true;
    output.verify_logits           = filtered.deferred_logits.defined() ? filtered.deferred_logits : inputs.logits;
    output.verify_candidate_ids    = std::move(filtered.deferred_token_ids);
    output.verify_dense_fallback   = std::move(filtered.deferred_dense_fallback);
    output.verify_fallback_rows    = std::move(filtered.deferred_fallback_rows);
    output.prepared_rejection_rng  = std::move(filtered.prepared_rejection_rng);
    output.verify_log_normalizer   = std::move(filtered.deferred_log_normalizer);
    deferred_buffers_.release();
    return output;
}

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    RTP_LLM_CHECK_WITH_INFO(!forward_in_progress_.exchange(true),
                            "Sampler::forward is single-threaded and must not be called concurrently or reentrantly");
    struct SamplerForwardGuard {
        std::atomic<bool>& forward_in_progress;
        ~SamplerForwardGuard() {
            forward_in_progress.store(false);
        }
    } sampler_forward_guard{forward_in_progress_};

    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    auto processor_errors = preprocessLogits(inputs);
    if (inputs.defer_token_sampling) {
        auto output                     = prepareDeferredVerification(inputs);
        output.processor_errors         = std::move(processor_errors);
        output.raw_logprobs_logits      = inputs.raw_logprobs_logits;
        output.raw_logprobs_row_indices = inputs.raw_logprobs_row_indices;
        return output;
    }

    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    const bool requires_independent_output = !std::equal(num_beams_in, num_beams_in + inputs.batch_size, num_beams_out);

    // allocate output tensors
    // Keep success on CUDA to avoid a blocking D2H copy: the GPU sampling kernel writes success
    // directly, and callers that need CPU access should call .cpu() explicitly.
    auto all_success =
        torch::empty({(int64_t)inputs.batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
#if USING_ROCM || (defined(USE_PPU) && USE_PPU)
    // ROCm: hipMemcpyAsync from pageable memory is truly async, and
    // Tensor::record_stream() rejects at::hip streams.
    // PPU can reject the asynchronous token ID copy on the dedicated stream.
    auto inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA);
#else
    torch::Tensor inputs_token_ids_cuda;
    {
        auto main_stream     = cuda_graph::graphGetCurrentStream();
        auto copy_done_event = cuda_graph::makeGraphEvent();
        {
            cuda_graph::GraphStreamGuard guard(copy_stream_);
            inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA, /*non_blocking=*/true);
            copy_done_event.record(copy_stream_);
        }
        copy_done_event.block(main_stream);
        inputs_token_ids_cuda.record_stream(main_stream);
    }
#endif

    auto all_token_ids_out     = requires_independent_output ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) :
                                     inputs_token_ids_cuda;
    auto all_cum_log_probs_out = requires_independent_output && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
                                     inputs.cum_log_probs;

    size_t from_batch_idx_in = 0, to_batch_idx_in = 0;
    size_t from_batch_idx_out      = 0;
    auto&  greedy_sampling_buffers = nextGreedySamplingBuffers(inputs.batch_size);
    struct GreedySamplingBufferGuard {
        Sampler* sampler = nullptr;
        ~GreedySamplingBufferGuard() {
            if (sampler == nullptr) {
                return;
            }
            try {
                sampler->markGreedySamplingBufferReady();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event");
            }
        }
    } greedy_sampling_buffer_guard{this};

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
        auto token_ids_in      = inputs_token_ids_cuda.narrow(0, from_batch_idx_in, batch_size_in);
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

            GreedySamplingBuffers greedy_sampling_buffer_slice{
                greedy_sampling_buffers.seed_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.offset_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.output_ids_ptrs_host.narrow(0, from_batch_idx_in, batch_size_in),
                batch_size_in};
            GreedySamplingBuffers* greedy_sampling_buffer_ptr = &greedy_sampling_buffer_slice;

            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
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
                 inputs.return_original_all_probs,
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator,
                 greedy_sampling_buffer_ptr});
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
            } else {
                success.fill_(true);
            }
            // execSampleGreedy updates token_ids_in in place. Mixed beam transitions use a
            // separate output tensor even when the aggregate input/output row counts match.
            if (requires_independent_output) {
                token_ids_out.copy_(token_ids_in);
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

    auto output                     = SamplerOutput({std::move(all_token_ids_out),
                                                     std::move(all_cum_log_probs_out),
                                                     std::move(inputs.all_probs),
                                                     std::move(all_beam_indices),
                                                     std::move(all_success),
                                                     std::move(processor_errors)});
    output.raw_logprobs_logits      = inputs.raw_logprobs_logits;
    output.raw_logprobs_row_indices = inputs.raw_logprobs_row_indices;
    return output;
}

std::vector<std::optional<ErrorInfo>> Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        return inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
    return {};
}

}  // namespace rtp_llm
