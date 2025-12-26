#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp//core/BufferHelper.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include <cstddef>
#include <random>
#include <memory>

using namespace std;

namespace rtp_llm {

using SamplerT = float;

void CudaDevice::processLogits(const GreedyParams& params,
                               const BufferPtr&    device_tokens,
                               const BufferPtr&    transposed_tokens) {
    auto&      logits             = params.logits;
    const auto vocab_size_padded  = params.logits.shape()[1];
    const auto decoder_batch_size = params.sequence_lengths.shape()[0];
    const auto batch_size         = logits.shape()[0];
    const auto step               = params.step;

    if (std::any_of(params.temperature.data<float>(), params.temperature.data<float>() + batch_size, [&](auto t) {
            return t != 1.0f;
        })) {
        BufferPtr temperature_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
        copy({*temperature_buf, params.temperature});
        invokeBatchApplyTemperaturePenalty(logits.data<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_buf->data<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           stream_);
    }

    if (params.repetition_penalty) {
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
            || std::any_of(frequency_penalty.data<float>(), frequency_penalty.data<float>() + batch_size, [&](auto t) {
                   return t != 0.0f;
               })) {
            auto sequence_lengths = clone({params.input_lengths});
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

    if (decoder_batch_size && params.no_repeat_ngram_size) {
        const auto& no_repeat_ngram_size = params.no_repeat_ngram_size.value().get();
        if (any_of(no_repeat_ngram_size.data<int32_t>(),
                   no_repeat_ngram_size.data<int32_t>() + decoder_batch_size,
                   [](auto s) { return s != 0; })) {
            auto no_repeat_ngram_size_buf = clone({no_repeat_ngram_size});
            auto output_ids_ptrs = allocateBuffer({DataType::TYPE_UINT64, {decoder_batch_size}, AllocationType::HOST});
            for (int i = 0; i < decoder_batch_size; i++) {
                output_ids_ptrs->data<uint64_t>()[i] = (uint64_t)(device_tokens->data<int32_t>() + i * (step + 1));
            }
            auto output_ids_ptrs_device = clone({*output_ids_ptrs, AllocationType::DEVICE});
            auto sequence_lengths       = clone({params.sequence_lengths});

            tensorrt_llm::kernels::invokeBanRepeatNgram(logits.data<float>(),
                                                        (int32_t const**)(output_ids_ptrs_device->data()),
                                                        nullptr,  // finished_buf
                                                        nullptr,  // parent_ids_buf
                                                        nullptr,  // batch_slot
                                                        sequence_lengths->data<int32_t>(),
                                                        decoder_batch_size,
                                                        1,  // beam_width
                                                        step + 1,
                                                        no_repeat_ngram_size_buf->data<int32_t>(),
                                                        vocab_size_padded,
                                                        step + 1,
                                                        stream_);
        }
    }
}

GreedyOutput CudaDevice::flashinferSampleGreedy(const GreedyParams& params, const BufferPtr& transposed_tokens) {
    const auto batch_size = params.logits.shape()[0];
    auto&      top_k      = params.top_k;
    auto&      top_p      = params.top_p;

    // [batch_size, vocab_size]
    auto      logits_ref = params.logits.slice(0, params.logits.shape()[0]);
    // [batch_size, vocab_size]
    auto      probs   = softmax({logits_ref, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_INVALID, std::nullopt});
    BufferPtr success = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});
    // [1, batch_size]
    auto      samples = transposed_tokens->view(transposed_tokens->shape()[0] - 1, 1);

    torch::TensorOptions options =
        torch::TensorOptions(dataTypeToTorchType(probs->type())).device(torch::Device(torch::kCUDA));
    constexpr bool deterministic = true;
    constexpr int max_top_k_rounds = 32;
    auto uniform_samples = torch::rand({max_top_k_rounds, (int)batch_size}, options);
    for (int i = 0; i < batch_size; i++) {
        if (params.generator[i].defined()) {
            uniform_samples.index({torch::indexing::Slice(), i}) = torch::rand({max_top_k_rounds}, params.generator[i], nullopt, options);
        }
    }

    torch::Tensor probs_t               = Buffer2torchTensor(probs, false);
    torch::Tensor samples_t             = Buffer2torchTensor(samples, false);
    torch::Tensor success_t             = Buffer2torchTensor(success, false);
    torch::Tensor top_k_t               = Buffer2torchTensor(top_k, false);
    torch::Tensor top_p_t               = Buffer2torchTensor(top_p, false);
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = Buffer2torchTensor(*params.output_all_probs, false);
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    std::transform(top_p.data<float>(), top_p.data<float>() + batch_size, top_p.data<float>(), [&](auto t) {
        return std::abs(t) < 1e-7 ? 1.0 : t;
    });
    
    if (std::all_of(top_k.data<uint32_t>(), top_k.data<uint32_t>() + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        success.reset();
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)stream_);
        }
    } else if (std::all_of(
                   top_k.data<uint32_t>(), top_k.data<uint32_t>() + batch_size, [&](auto t) { return t <= 0; })) {
        top_p_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_p_t, 1.0, deterministic, (int64_t)stream_);
        if (output_all_probs_t.defined()) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, (int64_t)stream_);
        }
    } else if (std::all_of(top_p.data<float>(), top_p.data<float>() + batch_size, [&](auto t) {
                   return std::abs(t - 1.0f) < 1e-7;
               })) {
        std::transform(top_k.data<uint32_t>(),
                       top_k.data<uint32_t>() + batch_size,
                       top_k.data<uint32_t>(),
                       [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_k_t, 0, deterministic, (int64_t)stream_);
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)stream_);
        }
    } else {
        std::transform(top_k.data<uint32_t>(),
                       top_k.data<uint32_t>() + batch_size,
                       top_k.data<uint32_t>(),
                       [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_top_p_sampling_from_probs(probs_t,
                                        uniform_samples,
                                        samples_t,
                                        success_t,
                                        top_k_t,
                                        1.0,
                                        top_p_t,
                                        1.0,
                                        deterministic,
                                        (int64_t)stream_);
        if (output_all_probs_t.defined()) {
            torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
            top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, (int64_t)stream_);
            top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, (int64_t)stream_);
        }
    }

    if (params.cum_log_probs.has_value()) {

        // [batch_size]
        auto cum_log_probs_t = Buffer2torchTensor(*params.cum_log_probs, false);
        // [batch_size]
        auto token_probs_t = output_all_probs_t.gather(1, samples_t.transpose(1, 0).to(torch::kLong)).squeeze(1);
        auto token_probs_t_log = token_probs_t.log();
        cum_log_probs_t.add_(token_probs_t_log.to(cum_log_probs_t.device()));
    }

    auto output_tokens = transpose({*transposed_tokens});
    copy({params.token_ids, *output_tokens});
    check_cuda_error();
    return {success};
}

GreedyOutput CudaDevice::sampleGreedy(const GreedyParams& params) {
    // [batch_size, step + 1]
    auto device_tokens     = clone({params.token_ids});
    // [step + 1, batch_size]
    auto transposed_tokens = transpose({*device_tokens});

    const auto batch_size        = params.logits.shape()[0];
    bool       has_not_do_sample = params.do_sample
                             && std::any_of(params.do_sample.value().get().data<bool>(),
                                            params.do_sample.value().get().data<bool>() + batch_size,
                                            [&](auto t) { return !t; });
    bool need_do_sample = (!params.do_sample.has_value())
                          || std::any_of(params.do_sample.value().get().data<bool>(),
                                         params.do_sample.value().get().data<bool>() + batch_size,
                                         [&](auto t) { return t; });
    if (need_do_sample) {
        torch::Tensor selected_logits;
        torch::Tensor mask_tensor;
        if (has_not_do_sample) {
            BufferPtr do_sample_mask = clone({params.do_sample.value().get()});
            mask_tensor = Buffer2torchTensor(do_sample_mask->reshape({batch_size, 1}), false).logical_not();
            torch::Tensor logits_tensor = Buffer2torchTensor(params.logits, false);
            selected_logits             = logits_tensor.masked_select(mask_tensor);
        }
        processLogits(params, device_tokens, transposed_tokens);
        if (has_not_do_sample) {
            torch::Tensor logits_tensor = Buffer2torchTensor(params.logits, false);
            logits_tensor.masked_scatter_(mask_tensor, selected_logits);
        }
    }

    // fast path for topk = 1
    auto& top_k = params.top_k;
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

    return flashinferSampleGreedy(params, transposed_tokens);
}

}  // namespace rtp_llm
