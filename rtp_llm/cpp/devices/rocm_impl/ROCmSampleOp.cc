#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/sampling_topk_kernels.h"
#include "rtp_llm/cpp/kernels/sampling_topp_kernels.h"
#include "rtp_llm/cpp/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/rocm/sampling/sampling.h"

using namespace std;

namespace rtp_llm {

using SamplerT = float;

// batch sampling explained:
// topk = [4, 0, 4]. topp = [0.0, 0.5, 0.5]
// then topk_decode handles [4, x, 4 + 0.5]
//      topp_decode handles [x, 0.5, x]
// where "x" are skipped.
// topk should has higher proirity than topp.

GreedyOutput ROCmDevice::sampleGreedy(const GreedyParams& params) {
    const auto& logits            = params.logits;
    const auto  batch_size        = logits.shape()[0];
    const auto  vocab_size_padded = logits.shape()[1];
    const auto  step              = params.step;
    RUNTIME_ASSERT_OP_ARG(batch_size == params.token_ids.shape()[0],
                          "logits.shape[0] should equal to token_ids.shape[0], but %d vs %d",
                          batch_size,
                          params.token_ids.shape()[0]);
    RUNTIME_ASSERT_OP_ARG((step == params.token_ids.shape()[1] - 1),
                          "step should equal to token_ids.shape[1] - 1, but %d vs %d",
                          step,
                          params.token_ids.shape()[1] - 1);
    auto device_tokens     = clone({params.token_ids});
    auto transposed_tokens = transpose({*device_tokens});

    auto& top_k       = params.top_k;
    auto& top_p       = params.top_p;
    auto& temperature = params.temperature;
    ROCM_CHECK_VALUE(top_k.size() == batch_size, "top_k.size() != batch_size");
    ROCM_CHECK_VALUE(top_p.size() == batch_size, "top_p.size() != batch_size");
    ROCM_CHECK_VALUE(temperature.size() == batch_size, "temperature.size() != batch_size");

    if (std::any_of(
            temperature.data<float>(), temperature.data<float>() + batch_size, [&](auto t) { return t != 1.0f; })) {
        BufferPtr temperature_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
        copy({*temperature_buf, temperature});
        invokeBatchApplyTemperaturePenalty(logits.data<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_buf->data<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           stream_);
    }

    const auto decoder_batch_size = params.sequence_lengths.shape()[0];
    if (decoder_batch_size) {
        auto sequence_lengths = clone({params.sequence_lengths});
        auto input_lengths    = clone({params.input_lengths});

        if (step > 1 && params.repetition_penalty && decoder_batch_size) {
            RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
            auto& repetition_penalty = params.repetition_penalty->get();
            auto& presence_penalty   = params.presence_penalty->get();
            auto& frequency_penalty  = params.frequency_penalty->get();
            if (std::any_of(repetition_penalty.data<float>(),
                            repetition_penalty.data<float>() + batch_size,
                            [&](auto t) { return t != 1.0f; })
                || std::any_of(presence_penalty.data<float>(),
                               presence_penalty.data<float>() + batch_size,
                               [&](auto t) { return t != 0.0f; })
                || std::any_of(frequency_penalty.data<float>(),
                               frequency_penalty.data<float>() + batch_size,
                               [&](auto t) { return t != 0.0f; })) {
                copy({sequence_lengths->view(0, decoder_batch_size), params.sequence_lengths});
                auto penalty_ws = allocateBuffer({DataType::TYPE_INT32, {batch_size, vocab_size_padded}});
                bufMemset(*penalty_ws, 0);
                auto repetition_penalty_gpu = clone({repetition_penalty, AllocationType::DEVICE});
                auto presence_penalty_gpu   = clone({presence_penalty, AllocationType::DEVICE});
                auto frequency_penalty_gpu  = clone({frequency_penalty, AllocationType::DEVICE});
                invokeBatchApplyRepetitionPenalty(logits.data<float>(),
                                                  penalty_ws->data<int32_t>(),
                                                  repetition_penalty_gpu->data<float>(),
                                                  presence_penalty_gpu->data<float>(),
                                                  frequency_penalty_gpu->data<float>(),
                                                  transposed_tokens->data<int32_t>(),
                                                  batch_size,
                                                  batch_size,  // local_batch_size
                                                  vocab_size_padded,
                                                  sequence_lengths->data<int32_t>(),
                                                  step + 1,  // max_input_length
                                                  step + 1,  // step
                                                  stream_);
                // NOTE: here step is max_len - 1
            }
        }
    }

    // fast path for topk = 1
    if (std::all_of(top_k.data<uint32_t>(), top_k.data<uint32_t>() + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        BufferPtr     logits_ref      = params.logits.slice(0, params.logits.shape()[0]);
        Buffer        samples         = transposed_tokens->view(transposed_tokens->shape()[0] - 1, 1);
        torch::Tensor samples_t       = Buffer2torchTensor(samples, false);
        torch::Tensor probs_t         = Buffer2torchTensor(*logits_ref, false);
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transpose({*transposed_tokens});
        copy({params.token_ids, *output_tokens});

        return GreedyOutput{};
    }

    bool deterministic = true;
    auto seed_h   = allocateBuffer({DataType::TYPE_INT64, {batch_size}, AllocationType::HOST});
    auto offset_h = allocateBuffer({DataType::TYPE_INT64, {batch_size}, AllocationType::HOST});
    for (int i = 0; i < batch_size; i++) {
        auto [sd, ofst] = get_seed_and_offset(batch_size * 32,
                                              params.generator[i].defined() ?
                                              std::make_optional(params.generator[i]) :
                                              std::nullopt);
        seed_h->data<int64_t>()[i]   = static_cast<int64_t>(sd);
        offset_h->data<int64_t>()[i] = static_cast<int64_t>(ofst);
    }
    auto seed = Buffer2torchTensor(seed_h, false);
    auto offset = Buffer2torchTensor(offset_h, false);

    auto logits_ref = params.logits.slice(0, params.logits.shape()[0]);
    auto probs      = softmax({logits_ref, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_INVALID, std::nullopt});
    auto samples    = transposed_tokens->view(transposed_tokens->shape()[0] - 1, 1);

    bool          need_output_all_probs = params.output_all_probs.has_value();
    torch::Tensor probs_t               = Buffer2torchTensor(probs, false);
    torch::Tensor samples_t             = Buffer2torchTensor(samples, false).flatten();
    torch::Tensor top_k_t               = Buffer2torchTensor(top_k, false);
    torch::Tensor top_p_t               = Buffer2torchTensor(top_p, false);
    torch::Tensor output_all_probs_t;
    if (need_output_all_probs) {
        output_all_probs_t = Buffer2torchTensor(params.output_all_probs.value().get(), false);
    }
    std::transform(top_p.data<float>(), top_p.data<float>() + batch_size, top_p.data<float>(), [&](auto t) {
        return std::abs(t) < 1e-7 ? 1.0 : t;
    });
    if (std::all_of(top_k.data<uint32_t>(), top_k.data<uint32_t>() + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_output_all_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(stream_));
        }
    } else if (std::all_of(
                   top_k.data<uint32_t>(), top_k.data<uint32_t>() + batch_size, [&](auto t) { return t <= 0; })) {
        top_p_sampling_from_probs(probs_t,
                                  samples_t,
                                  std::nullopt,
                                  top_p_t,
                                  1.0,
                                  deterministic,
                                  seed,
                                  offset,
                                  reinterpret_cast<uintptr_t>(stream_));
        if (need_output_all_probs) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(stream_));
        }
    } else if (std::all_of(top_p.data<float>(), top_p.data<float>() + batch_size, [&](auto t) {
                   return std::abs(t - 1.0f) < 1e-7;
               })) {
        std::transform(top_k.data<uint32_t>(),
                       top_k.data<uint32_t>() + batch_size,
                       top_k.data<uint32_t>(),
                       [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_sampling_from_probs(probs_t,
                                  samples_t,
                                  std::nullopt,
                                  top_k_t,
                                  0,
                                  deterministic,
                                  seed,
                                  offset,
                                  reinterpret_cast<uintptr_t>(stream_));
        if (need_output_all_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(stream_));
        }
    } else {
        std::transform(top_k.data<uint32_t>(),
                       top_k.data<uint32_t>() + batch_size,
                       top_k.data<uint32_t>(),
                       [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_top_p_sampling_from_probs(probs_t,
                                        samples_t,
                                        std::nullopt,
                                        top_k_t,
                                        0,
                                        top_p_t,
                                        1.0,
                                        deterministic,
                                        seed,
                                        offset,
                                        reinterpret_cast<uintptr_t>(stream_));
        if (need_output_all_probs) {
            torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
            top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, reinterpret_cast<uintptr_t>(stream_));
            top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(stream_));
        }
    }
    if (params.cum_log_probs.has_value()) {
        Buffer2torchTensor(*params.cum_log_probs).add_(probs_t.log());
    }
    auto output_tokens = transpose({*transposed_tokens});
    copy({params.token_ids, *output_tokens});
    check_cuda_error();
    return GreedyOutput{};
}

}  // namespace rtp_llm
