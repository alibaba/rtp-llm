#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <algorithm>

#include <limits>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
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
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
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

#elif USING_XPU  // XPU platform — pure PyTorch sampling

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size        = params.logits.size(0);
    const auto vocab_size_padded = params.logits.size(1);
    const auto step              = params.step;
    auto       device            = getTorchDevice();  // returns torch::kXPU

    // Verify sampling parameters are on CPU (constructed by SamplerInputGatherer)
    RTP_LLM_CHECK(params.temperature.is_cpu());
    RTP_LLM_CHECK(params.top_k.is_cpu());
    RTP_LLM_CHECK(params.top_p.is_cpu());

    // [batch_size, step + 1] -> GPU
    auto device_tokens     = params.token_ids.to(device);
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    // 0. Unsupported feature guard
    if (params.no_repeat_ngram_size.has_value()) {
        const auto& nrn = params.no_repeat_ngram_size.value();
        if (std::any_of(nrn.data_ptr<int32_t>(),
                        nrn.data_ptr<int32_t>() + batch_size,
                        [](int32_t s) { return s != 0; })) {
            RTP_LLM_CHECK_WITH_INFO(false,
                "no_repeat_ngram_size is not yet supported on XPU. "
                "Set no_repeat_ngram_size=0 when running on Intel GPU.");
        }
    }

    // 0.5 do_sample: match CUDA semantics — save raw logits for
    // do_sample=false rows so they can be restored after penalties,
    // giving those rows argmax on unmodified logits.
    torch::Tensor saved_logits;
    if (params.do_sample.has_value()) {
        auto top_k_ptr = params.top_k.data_ptr<int32_t>();
        bool any_greedy = false;
        for (int64_t b = 0; b < batch_size; b++) {
            if (!params.do_sample.value().data_ptr<bool>()[b]) {
                top_k_ptr[b] = 1;
                any_greedy = true;
            }
        }
        if (any_greedy) {
            saved_logits = params.logits.clone();
        }
    }

    // 1. Temperature
    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [](float t) { return t != 1.0f; })) {
        for (int64_t b = 0; b < batch_size; b++) {
            float t = params.temperature.data_ptr<float>()[b];
            if (t != 1.0f && t > 0.0f) {
                params.logits[b].div_(t);
            }
        }
    }

    // 1.5 Force greedy on per-row temperature==0 in mixed batches.
    // The all-zero temperature fast path below is skipped when only some rows
    // are 0, so mark those rows top_k=1 to make the per-row top_k filter pick
    // the argmax token deterministically (instead of softmax/multinomial).
    {
        auto top_k_force = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
        for (int64_t b = 0; b < batch_size; b++) {
            if (params.temperature.data_ptr<float>()[b] == 0.0f) {
                top_k_force[b] = 1;
            }
        }
    }

    // 2. Repetition / presence / frequency penalty
    // Uses vectorized PyTorch ops to avoid slow element-wise CPU access on device tensors.
    if (params.repetition_penalty.has_value()) {
        TORCH_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value(),
            "XPU sampling: repetition_penalty is set but presence_penalty and/or "
            "frequency_penalty are missing. All three must be provided together.");
        const auto& rep_pen  = params.repetition_penalty.value();
        const auto& pres_pen = params.presence_penalty.value();
        const auto& freq_pen = params.frequency_penalty.value();
        // These tensors, along with sequence_lengths/input_lengths, are dereferenced on
        // the host via data_ptr below; ensure they all live on CPU before reading them.
        RTP_LLM_CHECK(rep_pen.is_cpu() && pres_pen.is_cpu() && freq_pen.is_cpu());
        RTP_LLM_CHECK(params.sequence_lengths.is_cpu() && params.input_lengths.is_cpu());
        for (int64_t b = 0; b < batch_size; b++) {
            float rp = rep_pen.data_ptr<float>()[b];
            float pp = pres_pen.data_ptr<float>()[b];
            float fp = freq_pen.data_ptr<float>()[b];
            if (rp == 1.0f && pp == 0.0f && fp == 0.0f) continue;
            auto row = params.logits[b];
            // Use per-row actual length: decode rows use sequence_lengths,
            // context rows use input_lengths, to avoid reading padding tokens.
            const auto decoder_batch_size = params.sequence_lengths.size(0);
            int actual_len;
            if (b < decoder_batch_size) {
                actual_len = params.sequence_lengths.data_ptr<int32_t>()[b];
            } else {
                actual_len = params.input_lengths.data_ptr<int32_t>()[b];
            }
            actual_len = std::min(actual_len, (int)step);
            if (actual_len <= 0) continue;
            auto past_tokens = transposed_tokens.slice(0, 0, actual_len).select(1, b);

            // Build frequency histogram on-device: freq_count[token_id] = count
            // This replaces the O(step * unique_tokens) CPU-side nested loop.
            auto freq_count = torch::zeros({vocab_size_padded},
                                           past_tokens.options().dtype(torch::kFloat));
            freq_count.scatter_add_(0, past_tokens.to(torch::kLong),
                                    torch::ones({past_tokens.size(0)},
                                                torch::TensorOptions().dtype(torch::kFloat)
                                                    .device(past_tokens.device())));
            auto appeared = freq_count > 0;  // mask of tokens that appeared

            // Apply repetition penalty: score = score/rp if score>0, score*rp if score<0
            if (rp != 1.0f) {
                auto pos_mask = (row > 0) & appeared;
                auto neg_mask = (row < 0) & appeared;
                // Divide positive scores by rp, multiply negative scores by rp
                auto adjusted = torch::where(pos_mask, row / rp,
                                torch::where(neg_mask, row * rp, row));
                row.copy_(adjusted);
            }
            // Apply presence penalty: subtract pp for every appeared token
            if (pp != 0.0f) {
                row.sub_(pp * appeared.to(torch::kFloat));
            }
            // Apply frequency penalty: subtract fp*count for every token
            if (fp != 0.0f) {
                row.sub_(fp * freq_count);
            }
        }
    }

    // 2.5 Restore raw logits for do_sample=false rows (after penalties).
    if (saved_logits.defined() && params.do_sample.has_value()) {
        for (int64_t b = 0; b < batch_size; b++) {
            if (!params.do_sample.value().data_ptr<bool>()[b]) {
                params.logits[b].copy_(saved_logits[b]);
            }
        }
        saved_logits.reset();
    }

    // 2.6 Temperature==0 fast path (greedy argmax)
    // When temperature is 0, the intent is greedy decoding.
    // Skip the expensive softmax→multinomial path and use argmax directly.
    if (std::all_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [](float t) { return t == 0.0f; })
        && !params.output_all_probs.has_value()) {
        auto samples_t = transposed_tokens.slice(0, step, step + 1).squeeze(0);
        auto selected  = torch::argmax(params.logits, -1, false);
        samples_t.copy_(selected);
        if (params.cum_log_probs.has_value()) {
            auto probs = torch::softmax(params.logits, -1);
            auto token_prob = probs.gather(1, selected.unsqueeze(-1).to(torch::kLong)).squeeze(1);
            auto cum_log_probs_t = params.cum_log_probs.value();
            cum_log_probs_t.add_(token_prob.log().to(cum_log_probs_t.device()));
        }
        params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
        return GreedyOutput{};
    }

    // 3. Top-k=1 fast path (greedy argmax)
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [](uint32_t t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        auto samples_t      = transposed_tokens.slice(0, step, step + 1).squeeze(0);
        auto selected       = torch::argmax(params.logits, -1, false);
        samples_t.copy_(selected);
        if (params.cum_log_probs.has_value()) {
            auto probs       = torch::softmax(params.logits, -1);
            auto token_prob  = probs.gather(1, selected.unsqueeze(-1).to(torch::kLong)).squeeze(1);
            auto cum_log_probs_t = params.cum_log_probs.value();
            cum_log_probs_t.add_(token_prob.log().to(cum_log_probs_t.device()));
        }
        params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
        return GreedyOutput{};
    }

    // 4. Softmax -> probabilities
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // 5. Apply top_k filtering
    // Clone so in-place top_k/top_p filtering below does not mutate probs_t,
    // which must remain the original (pre-filter) softmax source for
    // return_original_all_probs and cum_log_probs.
    auto filtered_probs = probs_t.clone();
    bool has_top_k = !std::all_of(top_k_ptr, top_k_ptr + batch_size, [](uint32_t t) { return t <= 0; });
    if (has_top_k) {
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t k = top_k_ptr[b] <= 0 ? vocab_size_padded : (int64_t)top_k_ptr[b];
            // Clamp k into [1, vocab_size_padded] so an out-of-range top_k can
            // never trigger an out-of-bounds topk() call.
            if (k < 1) k = 1;
            if (k > vocab_size_padded) k = vocab_size_padded;
            if (k < vocab_size_padded) {
                auto row                    = filtered_probs[b];
                auto [topk_vals, topk_inds] = row.topk(k);
                // Zero the row and scatter back only the top-K values.
                // This guarantees exactly K candidates survive even when
                // multiple tokens share the same probability as the K-th.
                row.zero_();
                row.scatter_(0, topk_inds, topk_vals);
            }
        }
    }

    // 6. Apply top_p filtering
    // Operate on a private copy: params.top_p is conceptually an input and may be
    // reused by the caller, but the std::transform below rewrites it in place.
    auto top_p_host = params.top_p.clone();
    auto top_p_ptr  = top_p_host.data_ptr<float>();
    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [](float t) { return std::abs(t) < 1e-7f ? 1.0f : t; });
    bool has_top_p = !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](float t) { return std::abs(t - 1.0f) < 1e-7f; });
    if (has_top_p) {
        for (int64_t b = 0; b < batch_size; b++) {
            float p = top_p_ptr[b];
            if (std::abs(p - 1.0f) >= 1e-7f) {
                auto row                            = filtered_probs[b];
                auto [sorted_probs, sorted_indices] = row.sort(/*dim=*/0, /*descending=*/true);
                auto cumsum                         = sorted_probs.cumsum(0);
                auto mask                           = cumsum - sorted_probs > p;
                sorted_probs.masked_fill_(mask, 0.0f);
                row.scatter_(0, sorted_indices, sorted_probs);
            }
        }
    }

    // 7. Re-normalize and sample
    auto row_sums  = filtered_probs.sum(-1, true);
    // Guard against invalid probability distributions (all-zero / NaN / Inf rows).
    // Fall back to argmax on the original logits for any degenerate row.
    auto row_valid = (row_sums.squeeze(-1) > 0) & row_sums.squeeze(-1).isfinite();
    filtered_probs = filtered_probs / row_sums.clamp_min(1e-10f);
    // Fix degenerate rows BEFORE multinomial to prevent crash on XPU.
    // Replace invalid rows with a uniform distribution so multinomial won't throw.
    // Done purely on-device (torch::where) to avoid any per-row D2H .item() syncs.
    float uniform_val = 1.0f / static_cast<float>(filtered_probs.size(1));
    filtered_probs    = torch::where(row_valid.unsqueeze(-1),
                                  filtered_probs,
                                  torch::full_like(filtered_probs, uniform_val));
    auto selected  = torch::multinomial(filtered_probs, 1, false).squeeze(-1);
    // For any degenerate row, override the (uniform) draw with argmax on the
    // original logits. Pure-device select, no host sync.
    auto fallback = torch::argmax(params.logits, -1, false);
    selected      = torch::where(row_valid, selected, fallback);

    // Use per-request generators when available (respects request-level random seeds).
    // Skip the D2H transfer entirely when no generators are defined — the common
    // case in greedy/top-k sampling where no per-request seed is set.
    bool has_any_generator = std::any_of(
        params.generator.begin(), params.generator.end(),
        [](const c10::optional<at::Generator>& g) { return g.has_value() && g->defined(); });
    if (has_any_generator) {
        // Copy row_valid to host ONCE so the loop incurs no per-row D2H syncs,
        // and skip degenerate rows so the argmax fallback is preserved.
        auto  row_valid_cpu  = row_valid.to(torch::kCPU);
        auto* row_valid_host = row_valid_cpu.data_ptr<bool>();
        for (int64_t b = 0; b < batch_size; b++) {
            if (params.generator[b].defined() && row_valid_host[b]) {
                // selected[b] = ... does NOT write back in-place in C++ libtorch;
                // use select(0,b).copy_() to update the underlying storage.
                auto sampled = torch::multinomial(
                    filtered_probs[b].unsqueeze(0), 1, false, params.generator[b]).squeeze();
                selected.select(0, b).copy_(sampled);
            }
        }
    }

    auto samples_t = transposed_tokens.slice(0, step, step + 1).squeeze(0);
    samples_t.copy_(selected);

    bool need_output_all_probs = params.output_all_probs.has_value();
    if (need_output_all_probs) {
        if (params.return_original_all_probs) {
            // Write pre-filter softmax probabilities (before top_k/top_p filtering)
            params.output_all_probs.value().copy_(probs_t);
        } else {
            params.output_all_probs.value().copy_(filtered_probs);
        }
    }

    // 8. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        // Use the same probability source as output_all_probs for consistency
        auto& prob_source = (params.return_original_all_probs) ? probs_t : filtered_probs;
        auto token_prob = prob_source.gather(1, selected.unsqueeze(-1).to(torch::kLong)).squeeze(1);
        auto log_prob   = token_prob.log();
        // Degenerate rows (row_valid=false) emitted the argmax fallback token;
        // their prob source may be 0/NaN, so log_prob is -inf/NaN. Zero those
        // updates so a single invalid row cannot corrupt cum_log_probs (which
        // would cascade into beam ranking / stop criteria).
        log_prob = torch::where(row_valid, log_prob, torch::zeros_like(log_prob));
        auto cum_log_probs_t = params.cum_log_probs.value();
        cum_log_probs_t.add_(log_prob.to(cum_log_probs_t.device()));
    }

    // 9. Copy back
    params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
    // Signal degenerate rows (invalid prob distribution: all-zero / NaN / Inf)
    // to the caller via success=false instead of silently trusting the argmax
    // fallback. This matches the CUDA/flashinfer contract (the dispatcher turns
    // a false row into a reported error). The uniform/argmax fallback above only
    // exists to keep multinomial from crashing on XPU; rows remain marked failed.
    return GreedyOutput{row_valid};
}

// XPU: Speculative (draft-model) sampling is not supported.
// This requires chain_speculative_sampling kernel which performs rejection sampling
// between draft and target model probabilities. A pure PyTorch implementation would
// need: (1) compute acceptance probability min(1, target_prob/draft_prob),
// (2) accept/reject each drafted token, (3) resample rejected positions.
// TODO(xpu): Implement PyTorch fallback when speculative decoding is needed on XPU.
void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    RTP_LLM_CHECK_WITH_INFO(false,
        "Speculative sampling is not supported on XPU. "
        "Disable speculative decoding (draft model) when running on Intel GPU.");
}

// XPU: Rejection sampling (speculative-decode accept/reject) is not supported.
// Same rationale as chainSpeculativeSampling above — requires the rejection
// sampling kernel or a PyTorch fallback. Throw a clear error instead of
// silently emitting wrong tokens.
void rejectionSampling(const RejectionSamplingParams& params) {
    RTP_LLM_CHECK_WITH_INFO(false,
        "Rejection sampling is not supported on XPU. "
        "Disable speculative decoding (draft model) when running on Intel GPU.");
}

#else  // ROCm platform (fallback)

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

    bool          need_renorm_probs = params.output_all_probs.has_value() && !params.return_original_all_probs;
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
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
        if (need_renorm_probs) {
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
                    auto [sorted_probs, sorted_indices] = row.sort(/*dim=*/0, /*descending=*/true);
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
        if (need_renorm_probs) {
            output_all_probs_t.copy_(filtered_probs);
        }
    }

    if (params.return_original_all_probs && output_all_probs_t.defined()) {
        top_k_renorm_probs(probs_t, output_all_probs_t, std::nullopt, 1 << 30, reinterpret_cast<uintptr_t>(cur_stream));
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

#endif  // USING_CUDA / USING_XPU / USING_ROCM

}  // namespace rtp_llm
