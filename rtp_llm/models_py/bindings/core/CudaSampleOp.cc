#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"

#include <limits>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/common/kernels/vocab_prune/mapping.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include <cstddef>
#include <random>
#include <memory>
#endif

using namespace std;

namespace rtp_llm {

namespace {

#if USING_CUDA || USING_ROCM

struct RejectionSamplingLaunchConfig {
    int batch_size;
    int num_speculative_tokens;
    int target_vocab_size;
    int target_token_stride;
};

void checkRejectionSamplingTensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t dim) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be on CUDA/HIP device", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == dtype, "%s dtype mismatch", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == dim,
                            "%s must be %ld-D, got %ld-D",
                            name,
                            static_cast<long>(dim),
                            static_cast<long>(tensor.dim()));
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
}

void checkSameDevice(const torch::Tensor& tensor, const char* name, const c10::Device& device) {
    RTP_LLM_CHECK_WITH_INFO(tensor.device() == device, "%s must be on the same device as draft_probs_d", name);
}

RejectionSamplingLaunchConfig validateRejectionSamplingParams(const RejectionSamplingParams& params) {
    checkRejectionSamplingTensor(params.draft_probs_d, "draft_probs_d", torch::kFloat32, 3);
    const auto device = params.draft_probs_d.device();

    checkRejectionSamplingTensor(params.draft_token_ids_d, "draft_token_ids_d", torch::kInt32, 2);
    checkRejectionSamplingTensor(params.uniform_samples_d, "uniform_samples_d", torch::kFloat32, 2);
    checkRejectionSamplingTensor(params.target_probs_d, "target_probs_d", torch::kFloat32, 3);
    checkRejectionSamplingTensor(params.target_token_ids_d, "target_token_ids_d", torch::kInt32, 2);
    checkRejectionSamplingTensor(params.output_token_ids_d, "output_token_ids_d", torch::kInt32, 2);
    checkRejectionSamplingTensor(params.output_accepted_token_num_d, "output_accepted_token_num_d", torch::kInt32, 1);
    checkRejectionSamplingTensor(params.do_sample_d, "do_sample_d", torch::kBool, 1);

    checkSameDevice(params.draft_token_ids_d, "draft_token_ids_d", device);
    checkSameDevice(params.uniform_samples_d, "uniform_samples_d", device);
    checkSameDevice(params.target_probs_d, "target_probs_d", device);
    checkSameDevice(params.target_token_ids_d, "target_token_ids_d", device);
    checkSameDevice(params.output_token_ids_d, "output_token_ids_d", device);
    checkSameDevice(params.output_accepted_token_num_d, "output_accepted_token_num_d", device);
    checkSameDevice(params.do_sample_d, "do_sample_d", device);

    const int64_t batch_size             = params.draft_probs_d.size(0);
    const int64_t num_speculative_tokens = params.draft_probs_d.size(1);
    const int64_t target_vocab_size      = params.draft_probs_d.size(2);
    const int64_t target_token_stride    = params.target_token_ids_d.size(1);

    RTP_LLM_CHECK_WITH_INFO(target_vocab_size > 0, "target_vocab_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(target_token_stride > 0, "target_token_ids_d stride dimension must be positive");
    RTP_LLM_CHECK_WITH_INFO(batch_size <= std::numeric_limits<int>::max(), "batch_size too large");
    RTP_LLM_CHECK_WITH_INFO(num_speculative_tokens <= std::numeric_limits<int>::max(),
                            "num_speculative_tokens too large");
    RTP_LLM_CHECK_WITH_INFO(target_vocab_size <= std::numeric_limits<int>::max(), "target_vocab_size too large");
    RTP_LLM_CHECK_WITH_INFO(target_token_stride <= std::numeric_limits<int>::max(), "target_token_stride too large");

    const int64_t target_token_rows = batch_size * (num_speculative_tokens + 1);

    RTP_LLM_CHECK_WITH_INFO(params.draft_token_ids_d.size(0) == batch_size, "draft_token_ids_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.draft_token_ids_d.size(1) == num_speculative_tokens,
                            "draft_token_ids_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.uniform_samples_d.size(0) == batch_size, "uniform_samples_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.uniform_samples_d.size(1) == num_speculative_tokens + 1,
                            "uniform_samples_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(0) == batch_size, "target_probs_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(1) == num_speculative_tokens + 1,
                            "target_probs_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(2) == target_vocab_size, "target_probs_d shape[2] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_token_ids_d.size(0) == target_token_rows,
                            "target_token_ids_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_token_ids_d.size(0) == batch_size, "output_token_ids_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_token_ids_d.size(1) == num_speculative_tokens + 1,
                            "output_token_ids_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_accepted_token_num_d.size(0) == batch_size,
                            "output_accepted_token_num_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.do_sample_d.size(0) == batch_size, "do_sample_d shape[0] mismatch");

    return {static_cast<int>(batch_size),
            static_cast<int>(num_speculative_tokens),
            static_cast<int>(target_vocab_size),
            static_cast<int>(target_token_stride)};
}

#endif  // USING_CUDA || USING_ROCM

}  // anonymous namespace

#if USING_CUDA

using SamplerT = float;

namespace {

torch::Tensor getSamplingHostBufferSlice(const torch::Tensor& buffer, int64_t batch_size, const char* buffer_name) {
    RTP_LLM_CHECK_WITH_INFO(buffer.defined(), "%s is not initialized", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.device().is_cpu(), "%s must be a CPU tensor", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.scalar_type() == torch::kInt64, "%s must be an int64 tensor", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.numel() >= batch_size,
                            "%s capacity [%ld] is smaller than batch size [%ld]",
                            buffer_name,
                            buffer.numel(),
                            batch_size);
    return buffer.narrow(0, 0, batch_size);
}

std::pair<torch::Tensor, torch::Tensor> makeSamplingSeedOffsetTensors(const std::vector<at::Generator>& generators,
                                                                      int64_t                           batch_size,
                                                                      int                               increment,
                                                                      GreedySamplingBuffers* sampling_buffers) {
    RTP_LLM_CHECK_WITH_INFO(generators.size() >= static_cast<size_t>(batch_size),
                            "sampling generators size [%lu] is smaller than batch size [%ld]",
                            generators.size(),
                            batch_size);
    const bool    use_persistent_buffers = sampling_buffers != nullptr;
    auto          options                = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    torch::Tensor seed_h;
    torch::Tensor offset_h;
    if (use_persistent_buffers) {
        seed_h   = getSamplingHostBufferSlice(sampling_buffers->seed_host, batch_size, "sampling seed buffer");
        offset_h = getSamplingHostBufferSlice(sampling_buffers->offset_host, batch_size, "sampling offset buffer");
    } else {
        seed_h   = torch::empty({batch_size}, options);
        offset_h = torch::empty({batch_size}, options);
    }
    auto seed_ptr = seed_h.data_ptr<int64_t>();
    auto off_ptr  = offset_h.data_ptr<int64_t>();

    for (int64_t i = 0; i < batch_size; ++i) {
        // Undefined generator entries intentionally use PyTorch's default CUDA generator.
        auto generator      = (i < static_cast<int64_t>(generators.size()) && generators[i].defined()) ?
                                  std::make_optional(generators[i]) :
                                  std::nullopt;
        auto [seed, offset] = get_seed_and_offset(increment, generator);
        seed_ptr[i]         = static_cast<int64_t>(seed);
        off_ptr[i]          = static_cast<int64_t>(offset);
    }

    auto seed_d   = seed_h.to(torch::kCUDA, /*non_blocking=*/use_persistent_buffers).contiguous();
    auto offset_d = offset_h.to(torch::kCUDA, /*non_blocking=*/use_persistent_buffers).contiguous();
    return {seed_d, offset_d};
}

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
        auto temperature_gpu = params.temperature.to(torch::kCUDA, true);
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
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA, true);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA, true), true);
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA, true);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA, true);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA, true);
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
            auto no_repeat_ngram_size_gpu = no_repeat_ngram_size.to(torch::kCUDA, true);
            // Build array of pointers to each batch's token ids
            const bool    use_persistent_buffers = params.sampling_buffers != nullptr;
            torch::Tensor output_ids_ptrs;
            if (use_persistent_buffers) {
                output_ids_ptrs = getSamplingHostBufferSlice(params.sampling_buffers->output_ids_ptrs_host,
                                                             decoder_batch_size,
                                                             "sampling output ids ptrs buffer");
            } else {
                output_ids_ptrs = torch::empty({(int64_t)decoder_batch_size},
                                               torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
            }
            for (int64_t i = 0; i < (int64_t)decoder_batch_size; i++) {
                output_ids_ptrs.data_ptr<int64_t>()[i] = (int64_t)(device_tokens.data_ptr<int32_t>() + i * (step + 1));
            }
            auto output_ids_ptrs_gpu  = output_ids_ptrs.to(torch::kCUDA, use_persistent_buffers);
            auto sequence_lengths_gpu = params.sequence_lengths.to(torch::kCUDA, true);

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
    params.logits.copy_(probs_t, true);
    auto success = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // [1, batch_size] — last row of transposed_tokens
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0));

    constexpr bool deterministic       = true;
    constexpr int  max_sampling_rounds = 32;
    auto [seed_t, offset_t] =
        makeSamplingSeedOffsetTensors(params.generator, batch_size, max_sampling_rounds, params.sampling_buffers);

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
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    auto top_p_ptr = params.top_p.data_ptr<float>();

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    const bool need_renorm_probs  = output_all_probs_t.defined() && !params.return_original_all_probs;
    const bool all_top_k_one      = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; });
    const bool all_top_k_no_limit = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
    const bool all_top_p_one =
        std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });

    if (all_top_k_one) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);
        success = torch::Tensor();  // mark as undefined — all succeeded
        if (need_renorm_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else if (all_top_k_no_limit) {
        top_p_sampling_from_probs(probs_t,
                                  samples_t.squeeze(0),
                                  success_t,
                                  std::nullopt,
                                  top_p_t,
                                  1.0,
                                  deterministic,
                                  seed_t,
                                  0,
                                  offset_t,
                                  0,
                                  (int64_t)cur_stream);
        if (need_renorm_probs) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    } else {
        // top_k<=0 means "no limit" in RTP config. The combined FlashInfer
        // kernel takes a top_k array, so normalize mixed batches after the
        // pure top-p route has been selected.
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [](auto t) { return t <= 0 ? 1 << 30 : t; });
        if (all_top_p_one) {
            top_k_sampling_from_probs(probs_t,
                                      samples_t.squeeze(0),
                                      success_t,
                                      std::nullopt,
                                      top_k_t,
                                      0,
                                      deterministic,
                                      seed_t,
                                      0,
                                      offset_t,
                                      0,
                                      (int64_t)cur_stream);
            if (need_renorm_probs) {
                top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
            }
        } else {
            top_k_top_p_sampling_from_probs(probs_t,
                                            samples_t.squeeze(0),
                                            success_t,
                                            std::nullopt,
                                            top_k_t,
                                            0,
                                            top_p_t,
                                            1.0,
                                            deterministic,
                                            seed_t,
                                            0,
                                            offset_t,
                                            0,
                                            (int64_t)cur_stream);
            if (need_renorm_probs) {
                torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
                top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, (int64_t)cur_stream);
                top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
            }
        }
    }

    if (params.return_original_all_probs && output_all_probs_t.defined()) {
        top_k_renorm_probs(probs_t, output_all_probs_t, std::nullopt, 1 << 30, (int64_t)cur_stream);
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
    params.token_ids.copy_(transposed_t, true);
    check_cuda_error();
    return {success};
}

GreedyOutput sampleGreedy(const GreedyParams& params) {
    // [batch_size, step + 1] — clone to GPU
    auto device_tokens = params.token_ids.to(torch::kCUDA, true);
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
            auto do_sample_gpu = params.do_sample.value().to(torch::kCUDA, true);
            mask_tensor        = do_sample_gpu.reshape({(int64_t)batch_size, 1}).logical_not();
            selected_logits    = params.logits.masked_select(mask_tensor);
        }
        processLogits(params, device_tokens, transposed_tokens);
        if (has_not_do_sample) {
            params.logits.masked_scatter_(mask_tensor, selected_logits);
        }
    }

    // fast path for topk = 1
    // TODO: Support output_all_probs and cum_log_probs in this fast path.
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value() && !params.cum_log_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens, true);

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
                               true,
                               int64_t(stream));
}

void rejectionSampling(const RejectionSamplingParams& params) {
    auto config = validateRejectionSamplingParams(params);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    check_cuda_value(invokeRejectionSampling(params.draft_probs_d.data_ptr<float>(),
                                             params.draft_token_ids_d.data_ptr<int32_t>(),
                                             params.uniform_samples_d.data_ptr<float>(),
                                             params.target_probs_d.data_ptr<float>(),
                                             params.target_token_ids_d.data_ptr<int32_t>(),
                                             config.target_token_stride,
                                             params.output_token_ids_d.data_ptr<int32_t>(),
                                             params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                             params.do_sample_d.data_ptr<bool>(),
                                             config.batch_size,
                                             config.num_speculative_tokens,
                                             config.target_vocab_size,
                                             stream));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        auto* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();
        if (params.tokens.dtype() == torch::kInt32) {
            auto* tokens_ptr = params.tokens.data_ptr<int32_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = static_cast<int32_t>(d2t_map_ptr[token_id]);
                    }
                }
            }
        } else if (params.tokens.dtype() == torch::kInt64) {
            auto* tokens_ptr = params.tokens.data_ptr<int64_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = d2t_map_ptr[token_id];
                    }
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else if (params.tokens.is_cuda() && params.d2t_map.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (params.tokens.dtype() == torch::kInt32) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int32_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else if (params.tokens.dtype() == torch::kInt64) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int64_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "tokens and d2t_map must be on the same device");
    }
}

#elif USING_ROCM

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

template<typename IdType>
void invokeMappingDraft2Target(IdType*     tokens,
                               int         batch_size,
                               int         token_offset,
                               int         token_stride,
                               int64_t*    d2t_map,
                               int         d2t_map_size,
                               hipStream_t stream);
}  // namespace rtp_llm

#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/kernels/sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {  // reopen

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size         = params.logits.size(0);
    const auto vocab_size_padded  = params.logits.size(1);
    const auto step               = params.step;
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    auto       cur_stream         = at::hip::getCurrentHIPStream().stream();

    // [batch_size, step + 1] — clone to GPU
    // On ROCm, hipMemcpyAsync from pageable memory is truly async (unlike CUDA where it
    // falls back to sync). Use blocking transfer to avoid memory access faults.
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
    // TODO: Support output_all_probs and cum_log_probs in this fast path.
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value() && !params.cum_log_probs.has_value()) {
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
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).flatten();
    auto top_p_ptr = params.top_p.data_ptr<float>();

    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });
    const bool need_renorm_probs  = output_all_probs_t.defined() && !params.return_original_all_probs;
    const bool all_top_k_one      = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; });
    const bool all_top_k_no_limit = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
    const bool all_top_p_one =
        std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });

    // The ROCm sampling kernels consume these arrays on the GPU. Passing the pinned host
    // tensors directly happens to work on some single-GPU setups, but can fault when the
    // active device is not the one which mapped the host allocation. Stage every array on
    // the logits device before issuing batched sampling launches.
    if (!all_top_k_one && !all_top_k_no_limit) {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [](auto t) { return t <= 0 ? 1 << 30 : t; });
    }
    auto top_k_t = params.top_k.to(params.logits.device(), /*non_blocking=*/true).contiguous();
    auto top_p_t = params.top_p.to(params.logits.device(), /*non_blocking=*/true).contiguous();

    // 6. Sample
    if (all_top_k_one) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_renorm_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
        }
    } else {
        const bool    use_persistent_buffers = params.sampling_buffers != nullptr;
        auto          host_options           = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
        torch::Tensor seed_h;
        torch::Tensor offset_h;
        if (use_persistent_buffers) {
            RTP_LLM_CHECK_WITH_INFO(params.sampling_buffers->seed_host.numel() >= batch_size,
                                    "sampling seed buffer capacity is smaller than batch size");
            RTP_LLM_CHECK_WITH_INFO(params.sampling_buffers->offset_host.numel() >= batch_size,
                                    "sampling offset buffer capacity is smaller than batch size");
            seed_h   = params.sampling_buffers->seed_host.narrow(0, 0, batch_size);
            offset_h = params.sampling_buffers->offset_host.narrow(0, 0, batch_size);
        } else {
            seed_h   = torch::empty({(int64_t)batch_size}, host_options);
            offset_h = torch::empty({(int64_t)batch_size}, host_options);
        }
        constexpr int max_sampling_rounds = 32;
        for (int64_t i = 0; i < (int64_t)batch_size; ++i) {
            auto generator      = i < static_cast<int64_t>(params.generator.size()) && params.generator[i].defined() ?
                                      std::make_optional(params.generator[i]) :
                                      std::nullopt;
            auto [seed, offset] = get_seed_and_offset(max_sampling_rounds, generator);
            seed_h.data_ptr<int64_t>()[i]   = static_cast<int64_t>(seed);
            offset_h.data_ptr<int64_t>()[i] = static_cast<int64_t>(offset);
        }
        auto seed_t   = seed_h.to(params.logits.device(), /*non_blocking=*/true).contiguous();
        auto offset_t = offset_h.to(params.logits.device(), /*non_blocking=*/true).contiguous();

        if (all_top_k_no_limit) {
            top_p_sampling_from_probs(probs_t,
                                      samples_t,
                                      std::nullopt,
                                      top_p_t,
                                      1.0,
                                      deterministic,
                                      seed_t,
                                      offset_t,
                                      reinterpret_cast<uintptr_t>(cur_stream));
            if (need_renorm_probs) {
                top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(cur_stream));
            }
        } else if (all_top_p_one) {
            top_k_sampling_from_probs(probs_t,
                                      samples_t,
                                      std::nullopt,
                                      top_k_t,
                                      0,
                                      deterministic,
                                      seed_t,
                                      offset_t,
                                      reinterpret_cast<uintptr_t>(cur_stream));
            if (need_renorm_probs) {
                top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
            }
        } else {
            // Match the CUDA kernel's joint top-k/top-p semantics. Construct the
            // corresponding normalized intersection only when the caller needs it.
            top_k_top_p_sampling_from_probs(probs_t,
                                            samples_t,
                                            std::nullopt,
                                            top_k_t,
                                            0,
                                            top_p_t,
                                            1.0,
                                            deterministic,
                                            seed_t,
                                            offset_t,
                                            reinterpret_cast<uintptr_t>(cur_stream));
            if (need_renorm_probs) {
                auto top_k_probs_t = torch::zeros_like(output_all_probs_t);
                top_k_renorm_probs(probs_t, top_k_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
                top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(cur_stream));
                output_all_probs_t.masked_fill_(top_k_probs_t.eq(0), 0.0);
                output_all_probs_t.div_(output_all_probs_t.sum(-1, /*keepdim=*/true).clamp_min_(1e-10));
            }
        }
    }

    if (params.return_original_all_probs && output_all_probs_t.defined()) {
        top_k_renorm_probs(probs_t, output_all_probs_t, std::nullopt, 1 << 30, reinterpret_cast<uintptr_t>(cur_stream));
    }

    // 7. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        auto token_probs_t   = output_all_probs_t.gather(1, samples_t.to(torch::kLong).unsqueeze(1)).squeeze(1);
        cum_log_probs_t.add_(token_probs_t.log().to(cum_log_probs_t.device()));
    }

    // 8. Copy results back
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);
    return GreedyOutput{};
}

}  // namespace rtp_llm

// Forward-declare in global namespace (matches rtp_llm/models_py/bindings/rocm/speculative_sampling/sampling.cu)
void chain_speculative_sampling(at::Tensor draft_probs,
                                at::Tensor draft_token_ids,
                                at::Tensor uniform_samples,
                                at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num,
                                bool       deterministic,
                                int64_t    hip_stream);

template<typename DType, typename IdType>
hipError_t invokeRejectionSampling(DType*      draft_probs,
                                   IdType*     draft_token_ids,
                                   DType*      uniform_samples,
                                   DType*      target_probs,
                                   IdType*     target_token_ids,
                                   int         target_token_stride,
                                   IdType*     output_token_ids,
                                   IdType*     output_accepted_token_num,
                                   bool*       do_sample,
                                   int         batch_size,
                                   int         num_speculative_tokens,
                                   int         target_vocab_size,
                                   hipStream_t stream);

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
                                 true,
                                 int64_t(stream));
}

void rejectionSampling(const RejectionSamplingParams& params) {
    auto config = validateRejectionSamplingParams(params);
    auto stream = at::hip::getCurrentHIPStream().stream();

    hipError_t err = ::invokeRejectionSampling(params.draft_probs_d.data_ptr<float>(),
                                               params.draft_token_ids_d.data_ptr<int32_t>(),
                                               params.uniform_samples_d.data_ptr<float>(),
                                               params.target_probs_d.data_ptr<float>(),
                                               params.target_token_ids_d.data_ptr<int32_t>(),
                                               config.target_token_stride,
                                               params.output_token_ids_d.data_ptr<int32_t>(),
                                               params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                               params.do_sample_d.data_ptr<bool>(),
                                               config.batch_size,
                                               config.num_speculative_tokens,
                                               config.target_vocab_size,
                                               stream);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess, "invokeRejectionSampling failed: %s", hipGetErrorString(err));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        auto* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();
        if (params.tokens.dtype() == torch::kInt32) {
            auto* tokens_ptr = params.tokens.data_ptr<int32_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = static_cast<int32_t>(d2t_map_ptr[token_id]);
                    }
                }
            }
        } else if (params.tokens.dtype() == torch::kInt64) {
            auto* tokens_ptr = params.tokens.data_ptr<int64_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = d2t_map_ptr[token_id];
                    }
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else if (params.tokens.is_cuda() && params.d2t_map.is_cuda()) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        if (params.tokens.dtype() == torch::kInt32) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int32_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else if (params.tokens.dtype() == torch::kInt64) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int64_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "tokens and d2t_map must be on the same device");
    }
}

#else  // Generic PyTorch fallback

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size        = params.logits.size(0);
    const auto vocab_size_padded = params.logits.size(1);
    const auto device            = params.logits.device();

    auto          device_tokens     = params.token_ids.to(device);
    auto          transposed_tokens = device_tokens.transpose(0, 1).contiguous();
    torch::Tensor original_logits;
    torch::Tensor do_sample_device;
    if (params.do_sample.has_value()) {
        RTP_LLM_CHECK_WITH_INFO(params.do_sample.value().numel() == batch_size,
                                "do_sample length must match batch size");
        do_sample_device = params.do_sample.value().to(device).reshape({batch_size, 1});
        if (!params.do_sample.value().all().item<bool>()) {
            original_logits = params.logits.clone();
        }
    }

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [](auto temperature) { return temperature != 1.0f; })) {
        auto temperature = params.temperature.to(device).reshape({batch_size, 1});
        params.logits.div_(temperature);
    }

    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        const bool  has_penalty        = std::any_of(repetition_penalty.data_ptr<float>(),
                                             repetition_penalty.data_ptr<float>() + batch_size,
                                             [](auto penalty) { return penalty != 1.0f; })
                                 || std::any_of(presence_penalty.data_ptr<float>(),
                                                presence_penalty.data_ptr<float>() + batch_size,
                                                [](auto penalty) { return penalty != 0.0f; })
                                 || std::any_of(frequency_penalty.data_ptr<float>(),
                                                frequency_penalty.data_ptr<float>() + batch_size,
                                                [](auto penalty) { return penalty != 0.0f; });
        if (has_penalty) {
            auto       effective_lengths  = params.input_lengths.clone();
            const auto decoder_batch_size = params.sequence_lengths.size(0);
            if (decoder_batch_size > 0) {
                effective_lengths.narrow(0, 0, decoder_batch_size).copy_(params.sequence_lengths);
            }

            auto history = device_tokens.slice(1, 0, params.step + 1).to(torch::kLong);
            auto positions =
                torch::arange(history.size(1), torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
            auto lengths = effective_lengths.to(device).to(torch::kLong).unsqueeze(1);
            auto valid   = positions.lt(lengths).logical_and(history.ge(0)).logical_and(history.lt(vocab_size_padded));
            auto safe_history = history.clamp(0, vocab_size_padded - 1);
            auto counts       = torch::zeros({batch_size, vocab_size_padded},
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));
            counts.scatter_add_(1, safe_history, valid.to(torch::kInt32));

            auto seen                 = counts.gt(0);
            auto repetition_penalty_d = repetition_penalty.to(device).reshape({batch_size, 1});
            auto repeated_logits      = torch::where(
                params.logits.lt(0), params.logits * repetition_penalty_d, params.logits / repetition_penalty_d);
            params.logits.copy_(torch::where(seen, repeated_logits, params.logits));
            params.logits.sub_(presence_penalty.to(device).reshape({batch_size, 1}) * seen.to(params.logits.dtype())
                               + frequency_penalty.to(device).reshape({batch_size, 1})
                                     * counts.to(params.logits.dtype()));
        }
    }

    const auto decoder_batch_size = params.sequence_lengths.size(0);
    if (decoder_batch_size > 0 && params.no_repeat_ngram_size.has_value()) {
        RTP_LLM_CHECK_WITH_INFO(params.no_repeat_ngram_size.value().numel() >= decoder_batch_size,
                                "no_repeat_ngram_size length is smaller than decoder batch size");
        RTP_LLM_CHECK_WITH_INFO(params.sequence_lengths.device().is_cpu()
                                    && params.no_repeat_ngram_size.value().device().is_cpu()
                                    && device_tokens.device().is_cpu(),
                                "generic no-repeat n-gram fallback requires CPU tensors");
        auto       sequence_lengths_ptr = params.sequence_lengths.data_ptr<int32_t>();
        auto       ngram_sizes_ptr      = params.no_repeat_ngram_size.value().data_ptr<int32_t>();
        auto       tokens_ptr           = device_tokens.data_ptr<int32_t>();
        const auto token_stride         = device_tokens.size(1);

        for (int64_t batch = 0; batch < decoder_batch_size; ++batch) {
            const int64_t ngram_size = ngram_sizes_ptr[batch];
            const int64_t length     = sequence_lengths_ptr[batch] + 1;
            RTP_LLM_CHECK_WITH_INFO(length >= 0 && length < token_stride,
                                    "sequence length is outside the token history");
            if (ngram_size <= 0 || length < ngram_size) {
                continue;
            }
            const auto* row = tokens_ptr + batch * token_stride;
            for (int64_t start = 0; start <= length - ngram_size; ++start) {
                bool matches_suffix = true;
                for (int64_t offset = 0; offset < ngram_size - 1; ++offset) {
                    if (row[start + offset] != row[length - ngram_size + 1 + offset]) {
                        matches_suffix = false;
                        break;
                    }
                }
                if (matches_suffix) {
                    const auto banned_token = row[start + ngram_size - 1];
                    if (banned_token >= 0 && banned_token < vocab_size_padded) {
                        auto logits_row = params.logits.select(0, batch);
                        logits_row.index_put_({banned_token}, -std::numeric_limits<float>::infinity());
                    }
                }
            }
        }
    }

    if (original_logits.defined()) {
        params.logits.copy_(torch::where(do_sample_device, params.logits, original_logits));
    }

    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    // TODO: Support output_all_probs and cum_log_probs in this top-k=1 fast path.
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto top_k) { return top_k == 1; })
        && !params.output_all_probs.has_value() && !params.cum_log_probs.has_value()) {
        auto samples = transposed_tokens.select(0, transposed_tokens.size(0) - 1);
        samples.copy_(torch::argmax(params.logits, -1, /*keepdim=*/false));
        params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
        return GreedyOutput{};
    }

    auto probs = torch::softmax(params.logits, -1);
    params.logits.copy_(probs);
    auto filtered_probs = probs.clone();
    auto top_p_ptr      = params.top_p.data_ptr<float>();
    std::transform(
        top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [](auto top_p) { return std::abs(top_p) < 1e-7 ? 1.0f : top_p; });

    int64_t max_bounded_top_k = 0;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t top_k = top_k_ptr[batch];
        if (top_k > 0 && top_k < vocab_size_padded) {
            max_bounded_top_k = std::max(max_bounded_top_k, top_k);
        }
    }
    if (max_bounded_top_k > 0) {
        auto top_k_device = params.top_k.to(device).to(torch::kLong);
        auto bounded_rows = top_k_device.gt(0).logical_and(top_k_device.lt(vocab_size_padded));
        auto row_indices  = torch::nonzero(bounded_rows).squeeze(1);
        auto active_probs = filtered_probs.index_select(0, row_indices);
        auto active_top_k = top_k_device.index_select(0, row_indices);

        // PyTorch topk accepts one k for the whole tensor. Compute the largest requested
        // k once, then gather each row's actual kth value and filter the active rows in one
        // batched operation. Using a threshold preserves the old fallback's tie behavior.
        auto topk_values   = std::get<0>(active_probs.topk(max_bounded_top_k, /*dim=*/-1));
        auto kth_positions = active_top_k.sub(1).unsqueeze(1);
        auto kth_values    = topk_values.gather(1, kth_positions);
        active_probs.masked_fill_(active_probs.lt(kth_values), 0.0f);
        filtered_probs.index_copy_(0, row_indices, active_probs);
    }

    const bool has_top_p =
        !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto top_p) { return std::abs(top_p - 1.0f) < 1e-7; });
    if (has_top_p) {
        auto top_p_device = params.top_p.to(device);
        auto active_rows  = top_p_device.sub(1.0f).abs().ge(1e-7);
        auto row_indices  = torch::nonzero(active_rows).squeeze(1);
        auto active_probs = filtered_probs.index_select(0, row_indices);
        auto active_top_p = top_p_device.index_select(0, row_indices).unsqueeze(1);

        auto [sorted_probs, sorted_indices] = active_probs.sort(/*dim=*/-1, /*descending=*/true);
        auto cumulative_probs               = sorted_probs.cumsum(-1);
        auto outside_nucleus                = cumulative_probs - sorted_probs > active_top_p;
        sorted_probs.masked_fill_(outside_nucleus, 0.0f);
        active_probs.zero_().scatter_(1, sorted_indices, sorted_probs);
        filtered_probs.index_copy_(0, row_indices, active_probs);
    }

    auto row_sums = filtered_probs.sum(-1, /*keepdim=*/true);
    filtered_probs.div_(row_sums.clamp_min(1e-10));
    auto selected = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kLong).device(device));
    // TODO: Use batched multinomial sampling when no per-request generator is specified.
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        c10::optional<at::Generator> generator = c10::nullopt;
        if (batch < static_cast<int64_t>(params.generator.size()) && params.generator[batch].defined()) {
            generator = params.generator[batch];
        }
        auto sample = at::multinomial(filtered_probs.select(0, batch),
                                      /*num_samples=*/1,
                                      /*replacement=*/false,
                                      generator);
        selected.select(0, batch).copy_(sample.select(0, 0));
    }
    transposed_tokens.select(0, transposed_tokens.size(0) - 1).copy_(selected);

    torch::Tensor probability_output;
    if (params.output_all_probs.has_value()) {
        probability_output = params.output_all_probs.value();
    } else if (params.cum_log_probs.has_value()) {
        probability_output = torch::empty_like(probs);
    }
    if (probability_output.defined()) {
        probability_output.copy_(params.return_original_all_probs ? probs : filtered_probs);
    }
    if (params.cum_log_probs.has_value()) {
        auto selected_probs = probability_output.gather(1, selected.to(torch::kLong).unsqueeze(1)).squeeze(1);
        params.cum_log_probs.value().add_(selected_probs.log());
    }

    params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
    return GreedyOutput{};
}

void chainSpeculativeSampling(const SpeculativeSamplingParams&) {
    RTP_LLM_CHECK_WITH_INFO(false, "chain speculative sampling is not supported on this device");
}

void rejectionSampling(const RejectionSamplingParams&) {
    RTP_LLM_CHECK_WITH_INFO(false, "rejection sampling is not supported on this device");
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");
    RTP_LLM_CHECK_WITH_INFO(params.tokens.device() == params.d2t_map.device(),
                            "mappingDraft2Target: tokens and d2t_map must be on the same device");

    auto token_rows    = params.tokens.reshape({-1, params.token_stride});
    auto active_tokens = token_rows.narrow(0, 0, params.batch_size).slice(1, params.token_offset, params.token_stride);
    auto valid_tokens  = active_tokens.ge(0).logical_and(active_tokens.lt(params.d2t_map.numel()));
    auto safe_tokens   = active_tokens.to(torch::kLong).clamp(0, params.d2t_map.numel() - 1);
    auto mapped_tokens = params.d2t_map.index_select(0, safe_tokens.flatten()).reshape_as(active_tokens);
    active_tokens.copy_(torch::where(valid_tokens, mapped_tokens.to(active_tokens.dtype()), active_tokens));
}

#endif  // USING_CUDA / USING_ROCM / generic fallback

}  // namespace rtp_llm
