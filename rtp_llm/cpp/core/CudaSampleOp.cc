#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/CommonDefines.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include <cstddef>
#include <random>
#include <memory>
#endif

using namespace std;

namespace rtp_llm {

#if USING_CUDA

using SamplerT = float;

namespace {

void processLogits(const GreedyParams&  params,
                   const torch::Tensor& device_tokens,
                   const torch::Tensor& transposed_tokens) {
    const auto vocab_size_padded  = params.logits.size(1);
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    const auto batch_size         = params.logits.size(0);
    const auto step               = params.step;
    auto       cur_stream         = at::cuda::getCurrentCUDAStream().stream();

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
    }

    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            // Build sequence_lengths: clone input_lengths, then overwrite first decoder_batch_size entries
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA));
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
            // NOTE: here step is max_len - 1
        }
    }

    if (decoder_batch_size && params.no_repeat_ngram_size.has_value()) {
        const auto& no_repeat_ngram_size = params.no_repeat_ngram_size.value();
        if (any_of(no_repeat_ngram_size.data_ptr<int32_t>(),
                   no_repeat_ngram_size.data_ptr<int32_t>() + decoder_batch_size,
                   [](auto s) { return s != 0; })) {
            auto no_repeat_ngram_size_gpu = no_repeat_ngram_size.to(torch::kCUDA);
            // Build array of pointers to each batch's token ids
            auto output_ids_ptrs =
                torch::empty({(int64_t)decoder_batch_size}, torch::TensorOptions().dtype(torch::kInt64)).pin_memory();
            for (int64_t i = 0; i < (int64_t)decoder_batch_size; i++) {
                output_ids_ptrs.data_ptr<int64_t>()[i] = (int64_t)(device_tokens.data_ptr<int32_t>() + i * (step + 1));
            }
            auto output_ids_ptrs_gpu  = output_ids_ptrs.to(torch::kCUDA);
            auto sequence_lengths_gpu = params.sequence_lengths.to(torch::kCUDA);

            tensorrt_llm::kernels::invokeBanRepeatNgram(params.logits.data_ptr<float>(),
                                                        (int32_t const**)(output_ids_ptrs_gpu.data_ptr()),
                                                        nullptr,  // finished_buf
                                                        nullptr,  // parent_ids_buf
                                                        nullptr,  // batch_slot
                                                        sequence_lengths_gpu.data_ptr<int32_t>(),
                                                        decoder_batch_size,
                                                        1,  // beam_width
                                                        step + 1,
                                                        no_repeat_ngram_size_gpu.data_ptr<int32_t>(),
                                                        vocab_size_padded,
                                                        step + 1,
                                                        cur_stream);
        }
    }
}

}  // anonymous namespace

static GreedyOutput flashinferSampleGreedy(const GreedyParams& params, const torch::Tensor& transposed_tokens) {
    const auto batch_size = params.logits.size(0);
    auto       cur_stream = at::cuda::getCurrentCUDAStream().stream();

    // [batch_size, vocab_size] — compute softmax probabilities.
    // Copy result back to logits to preserve the in-place behavior of the original kernel,
    // since callers may reuse the logits tensor across iterations.
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);
    auto success = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // [1, batch_size] — last row of transposed_tokens
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0));

    torch::TensorOptions options          = torch::TensorOptions(probs_t.scalar_type()).device(torch::kCUDA);
    constexpr bool       deterministic    = true;
    constexpr int        max_top_k_rounds = 32;
    auto                 uniform_samples  = torch::rand({max_top_k_rounds, (int)batch_size}, options);
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
        if (params.generator[i].defined()) {
            uniform_samples.index({torch::indexing::Slice(), i}) =
                torch::rand({max_top_k_rounds}, params.generator[i], nullopt, options);
        }
    }

    torch::Tensor success_t = success;
    torch::Tensor top_k_t   = params.top_k;
    torch::Tensor top_p_t   = params.top_p;
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    // top_k/top_p are CPU tensors with int32/float32 dtype
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    auto top_p_ptr = params.top_p.data_ptr<float>();

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        success = torch::Tensor();  // mark as undefined — all succeeded
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t <= 0; })) {
        top_p_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_p_t, 1.0, deterministic, (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    } else if (std::all_of(top_p_ptr, top_p_ptr + batch_size, [&](auto t) { return std::abs(t - 1.0f) < 1e-7; })) {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_k_t, 0, deterministic, (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_top_p_sampling_from_probs(probs_t,
                                        uniform_samples,
                                        samples_t,
                                        success_t,
                                        top_k_t,
                                        1.0,
                                        top_p_t,
                                        1.0,
                                        deterministic,
                                        (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
            top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, (int64_t)cur_stream);
            top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    }

    if (params.cum_log_probs.has_value()) {

        // [batch_size]
        auto cum_log_probs_t = params.cum_log_probs.value();
        // [batch_size]
        auto token_probs_t     = output_all_probs_t.gather(1, samples_t.transpose(1, 0).to(torch::kLong)).squeeze(1);
        auto token_probs_t_log = token_probs_t.log();
        cum_log_probs_t.add_(token_probs_t_log.to(cum_log_probs_t.device()));
    }

    // Copy results back: transpose and copy to token_ids
    auto transposed_t = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(transposed_t);
    check_cuda_error();
    return {success};
}

GreedyOutput sampleGreedy(const GreedyParams& params) {
    // [batch_size, step + 1] — clone to GPU
    auto device_tokens = params.token_ids.to(torch::kCUDA);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    const auto batch_size        = params.logits.size(0);
    bool       has_not_do_sample = params.do_sample.has_value()
                             && std::any_of(params.do_sample.value().data_ptr<bool>(),
                                            params.do_sample.value().data_ptr<bool>() + batch_size,
                                            [&](auto t) { return !t; });
    bool need_do_sample = (!params.do_sample.has_value())
                          || std::any_of(params.do_sample.value().data_ptr<bool>(),
                                         params.do_sample.value().data_ptr<bool>() + batch_size,
                                         [&](auto t) { return t; });
    if (need_do_sample) {
        torch::Tensor selected_logits;
        torch::Tensor mask_tensor;
        if (has_not_do_sample) {
            auto do_sample_gpu = params.do_sample.value().to(torch::kCUDA);
            mask_tensor        = do_sample_gpu.reshape({(int64_t)batch_size, 1}).logical_not();
            selected_logits    = params.logits.masked_select(mask_tensor);
        }
        processLogits(params, device_tokens, transposed_tokens);
        if (has_not_do_sample) {
            params.logits.masked_scatter_(mask_tensor, selected_logits);
        }
    }

    // fast path for topk = 1
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);

        return GreedyOutput{};
    }

    return flashinferSampleGreedy(params, transposed_tokens);
}

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    chain_speculative_sampling(params.draft_probs_d,
                               params.draft_token_ids_d,
                               params.uniform_samples_d,
                               params.target_probs_d,
                               params.output_token_ids_d,
                               params.output_accepted_token_num_d,
                               params.output_emitted_token_num_d,
                               false,
                               int64_t(stream));
}

#else  // !USING_CUDA — ROCm platform

}  // namespace rtp_llm — temporarily close for includes

// Forward-declare penalty kernels to avoid transitive amd_bfloat16.h breakage
// (sampling_penalty_kernels.h → cuda_shims.h → amd_bfloat16.h is broken in .cc on ROCm 6.4.3 + clang 19)
namespace rtp_llm {
template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        int          batch_size,
                                        int          vocab_size,
                                        int          vocab_size_padd,
                                        hipStream_t  stream);
template<typename T>
void invokeBatchApplyRepetitionPenalty(T*           logits,
                                       int*         penalty_ws,
                                       const float* repetition_penalty,
                                       const float* presence_penalty,
                                       const float* frequency_penalty,
                                       const int*   output_ids,
                                       int          batch_size,
                                       int          local_batch_size,
                                       int          vocab_size,
                                       const int*   input_lengths,
                                       int          max_input_length,
                                       int          step,
                                       hipStream_t  stream);
}  // namespace rtp_llm

#include <ATen/hip/HIPContext.h>
#include "rtp_llm/cpp/kernels/rocm/sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {  // reopen

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size         = params.logits.size(0);
    const auto vocab_size_padded  = params.logits.size(1);
    const auto step               = params.step;
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    auto       cur_stream         = at::hip::getCurrentHIPStream().stream();

    // [batch_size, step + 1] — clone to GPU
    auto device_tokens = params.token_ids.to(torch::kCUDA);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    // 1. Apply temperature penalty
    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
    }

    // 2. Apply repetition/presence/frequency penalty
    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA));
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
        }
    }

    // 3. Fast path for topk = 1
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);

        return GreedyOutput{};
    }

    // 4. Compute softmax probabilities
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // 5. Prepare sampling parameters
    constexpr bool deterministic = true;
    auto           seed_h        = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    auto           offset_h      = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
        auto [sd, ofst] = get_seed_and_offset(
            batch_size * 32, params.generator[i].defined() ? std::make_optional(params.generator[i]) : std::nullopt);
        seed_h.data_ptr<int64_t>()[i]   = static_cast<int64_t>(sd);
        offset_h.data_ptr<int64_t>()[i] = static_cast<int64_t>(ofst);
    }

    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).flatten();
    auto top_k_t   = params.top_k;
    auto top_p_t   = params.top_p;
    auto top_p_ptr = params.top_p.data_ptr<float>();

    bool          need_output_all_probs = params.output_all_probs.has_value();
    torch::Tensor output_all_probs_t;
    if (need_output_all_probs) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    // 6. Sample
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_output_all_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
        }
    } else {
        // Use pure PyTorch sampling instead of FlashInfer ROCm kernels.
        // FlashInfer's ROCm sampling kernels (top_p_sampling_from_probs,
        // top_k_sampling_from_probs, top_k_top_p_sampling_from_probs) crash
        // with GPU memory access faults in multi-GPU (TP>1) configurations.
        // torch::multinomial is well-tested and handles all cases correctly.
        //
        // Apply top_k filtering if needed
        auto filtered_probs = probs_t;
        bool has_top_k      = !std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
        if (has_top_k) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                int k = top_k_ptr[b] <= 0 ? vocab_size_padded : top_k_ptr[b];
                if ((int64_t)k < vocab_size_padded) {
                    auto row                    = filtered_probs[b];
                    auto [topk_vals, topk_inds] = row.topk(k);
                    auto min_val                = topk_vals[-1];
                    row.masked_fill_(row < min_val, 0.0f);
                }
            }
        }
        // Apply top_p (nucleus) filtering if needed
        bool has_top_p =
            !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });
        if (has_top_p) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                float p = top_p_ptr[b];
                if (std::abs(p - 1.0f) >= 1e-7) {
                    auto row                            = filtered_probs[b];
                    auto [sorted_probs, sorted_indices] = row.sort(/*descending=*/true);
                    auto cumsum                         = sorted_probs.cumsum(0);
                    auto mask                           = cumsum - sorted_probs > p;
                    sorted_probs.masked_fill_(mask, 0.0f);
                    row.scatter_(0, sorted_indices, sorted_probs);
                }
            }
        }
        // Re-normalize and sample
        auto row_sums  = filtered_probs.sum(-1, /*keepdim=*/true);
        filtered_probs = filtered_probs / row_sums.clamp_min(1e-10);
        auto selected  = torch::multinomial(filtered_probs, 1, /*replacement=*/false).squeeze(-1);
        samples_t.copy_(selected);
        if (need_output_all_probs) {
            output_all_probs_t.copy_(filtered_probs);
        }
    }

    // 7. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        cum_log_probs_t.add_(probs_t.log());
    }

    // 8. Copy results back
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);
    return GreedyOutput{};
}

}  // namespace rtp_llm

// Forward-declare in global namespace (matches rtp_llm/cpp/rocm/speculative_sampling/sampling.cu)
void chain_speculative_sampling(at::Tensor draft_probs,
                                at::Tensor draft_token_ids,
                                at::Tensor uniform_samples,
                                at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num,
                                bool       deterministic,
                                int64_t    hip_stream);

namespace rtp_llm {

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    ::chain_speculative_sampling(params.draft_probs_d,
                                 params.draft_token_ids_d,
                                 params.uniform_samples_d,
                                 params.target_probs_d,
                                 params.output_token_ids_d,
                                 params.output_accepted_token_num_d,
                                 params.output_emitted_token_num_d,
                                 false,
                                 int64_t(stream));
}

#endif  // USING_CUDA

}  // namespace rtp_llm
