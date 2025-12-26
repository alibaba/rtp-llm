#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/sampling_topk_kernels.h"
#include "rtp_llm/cpp/kernels/sampling_topp_kernels.h"
#include "rtp_llm/cpp/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace rtp_llm {

using SamplerT = float;

// batch sampling explained:
// topk = [4, 0, 4]. topp = [0.0, 0.5, 0.5]
// then topk_decode handles [4, x, 4 + 0.5]
//      topp_decode handles [x, 0.5, x]
// where "x" are skipped.
// topk should has higher proirity than topp.

inline void _saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName) {
    auto          tensor_cpu = tensor.contiguous().cpu();
    auto          pickled    = torch::pickle_save(tensor_cpu);
    std::ofstream fout(fileName, std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.close();
}
GreedyOutput ROCmDevice::sampleGreedy(const GreedyParams& params) {

    static int cnt = 0;
    const char* env_dir = std::getenv("XBJ_DUMP");
    if (env_dir != nullptr) {
        ++cnt;
        std::ostringstream oss;
        oss << env_dir << "/logits_" << cnt << ".pt";
        _saveTorchDataTofile(Buffer2torchTensor(params.logits, false), oss.str());
    }

    
    const auto& logits     = params.logits;
    const auto  batch_size = logits.shape()[0];
    RUNTIME_ASSERT_OP_ARG(batch_size < init_params_.max_batch_size,
                          "batch_size exceeded device limit %d: %d",
                          init_params_.max_batch_size,
                          batch_size);
    const auto vocab_size_padded = logits.shape()[1];
    const auto step              = params.step;
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

    // 1. prepare buffers
    auto& top_k       = params.top_k;
    auto& top_p       = params.top_p;
    auto& temperature = params.temperature;
    auto& random_seed = params.random_seed;
    ROCM_CHECK_VALUE(top_k.size() == batch_size, "top_k.size() != batch_size");
    ROCM_CHECK_VALUE(top_p.size() == batch_size, "top_p.size() != batch_size");
    ROCM_CHECK_VALUE(temperature.size() == batch_size, "temperature.size() != batch_size");

    auto default_top_k = top_k.data<uint32_t>()[0];
    auto default_top_p = top_p.data<float>()[0];
    auto max_top_k     = *max_element(top_k.data<uint32_t>(), top_k.dataWithOffset<uint32_t>(top_k.size()));
    if (max_top_k == 0) {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    auto max_top_p = *max_element(top_p.data<SamplerT>(), top_p.dataWithOffset<SamplerT>(top_p.size()));
    // RTP_LLM_LOG_WARNING("max_top_k: %d, max_top_p: %f", max_top_k, max_top_p);

    size_t topk_ws_size;
    size_t topp_ws_size;
    size_t cub_temp_storage_size;  // useless variable

    // these two kernel calls are only for querying workspace size,
    // all the args are ignored.
    invokeTopKSampling<SamplerT>(nullptr,
                                 topk_ws_size,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 max_top_k,
                                 max_top_p,
                                 vocab_size_padded,
                                 nullptr,
                                 nullptr,
                                 stream_,
                                 batch_size,
                                 nullptr);

    invokeTopPSampling<SamplerT>(nullptr,  // workspace
                                 topp_ws_size,
                                 cub_temp_storage_size,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 batch_size,
                                 vocab_size_padded,
                                 nullptr,
                                 max_top_p,
                                 nullptr,
                                 stream_,
                                 &device_prop_,
                                 nullptr);

    // RTP_LLM_LOG_WARNING("topk_ws_size: %d, topp_ws_size: %d", topk_ws_size, topp_ws_size);

    // see BaseSamplingLayer<T>::allocateBuffer ------------------
    auto skip_top_k_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});
    auto skip_top_p_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});

    auto topp_id_vals_buf      = allocateBuffer({DataType::TYPE_INT32, {batch_size * vocab_size_padded}});
    auto topp_offset_buf       = allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}});
    auto begin_topp_offset_buf = allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}});

    // TopKSamplingLayer<T>::allocateBuffer
    auto top_k_workspace   = allocateBuffer({topk_ws_size});
    auto top_p_workspace   = allocateBuffer({topp_ws_size});
    auto runtime_top_k_buf = allocateBuffer({DataType::TYPE_UINT32, {batch_size}});
    copy({*runtime_top_k_buf, top_k});
    auto runtime_top_p_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
    copy({*runtime_top_p_buf, top_p});

    auto cum_log_probs = params.cum_log_probs.has_value() ? params.cum_log_probs.value().get().data<float>() : nullptr;
    auto output_log_probs =
        params.output_log_probs.has_value() ? params.output_log_probs.value().get().data<float>() : nullptr;
    auto output_all_probs =
        params.output_all_probs.has_value() ? params.output_all_probs.value().get().data<float>() : nullptr;

    // 3. prepare common inputs

    // 3.1. setup random seeds
    if (random_seed) {
        auto& seeds = random_seed.value().get();
        if (seeds.size() == 1) {
            invokeCurandInitialize(
                (curandState_t*)curandstate_buf_->data(), batch_size, seeds.data<uint64_t>()[0], stream_);
        } else {
            auto random_seeds_buf = allocateBuffer({DataType::TYPE_UINT64, {batch_size}});
            RUNTIME_ASSERT_OP_ARG((seeds.size() == batch_size),
                                  "random_seed.size() should equal to batch_size, but %d vs %d",
                                  seeds.size(),
                                  batch_size);
            copy({*random_seeds_buf, seeds});
            invokeCurandBatchInitialize((curandState_t*)curandstate_buf_->data(),
                                        batch_size,
                                        (unsigned long long*)random_seeds_buf->data(),
                                        stream_);
        }
    }

    // 3.2. compute logits penalty
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

    if (params.min_lengths && params.eos_ids) {
        auto min_lengths_buf = clone({params.min_lengths.value().get()});
        // move this to NormalExecutor
        auto sequence_lengths = clone({params.sequence_lengths});
        auto input_lengths    = clone({params.input_lengths});
        invokeMinLengthPenaltyNew(logits.data<float>(),
                                  min_lengths_buf->data<int32_t>(),
                                  params.eos_ids.value().get().data<int32_t>(),
                                  sequence_lengths->data<int32_t>(),
                                  input_lengths->data<int32_t>(),
                                  decoder_batch_size,
                                  batch_size,
                                  vocab_size_padded,
                                  stream_);
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

    // 4. run sampling
    // 4.1 run top_k
    invokeSetupTopKRuntimeArgs(batch_size,
                               default_top_k,
                               runtime_top_k_buf->data<uint>(),
                               batch_size,
                               default_top_p,
                               runtime_top_p_buf->data<float>(),
                               batch_size,
                               skip_top_k_decode_buf->data<bool>(),
                               stream_);

    invokeBatchTopKSampling(top_k_workspace->data(),
                            topk_ws_size,
                            logits.data<float>(),
                            transposed_tokens->dataWithOffset<int32_t>(step * batch_size),
                            nullptr,  // sequence_length
                            nullptr,  // finished
                            cum_log_probs,
                            output_log_probs,
                            (curandState_t*)curandstate_buf_->data(),
                            max_top_k,  // useless because runtime_top_k_buf_ is never nullptr. Keep for legacy.
                            (int32_t*)runtime_top_k_buf->data<uint32_t>(),
                            1.0f,  // useless because runtime_top_p_buf_ is never nullptr. Keep for legacy.
                            runtime_top_p_buf->data<float>(),
                            vocab_size_padded,
                            nullptr,  // end_id
                            output_all_probs,
                            stream_,
                            batch_size,
                            skip_top_k_decode_buf->data<bool>());

    // 4.2. run top_p
    // NOTE: running top_k could write values to runtime bufs, so need to copy again.
    copy({*runtime_top_k_buf, top_k});
    copy({*runtime_top_p_buf, top_p});

    invokeSetupTopPRuntimeArgs(batch_size,
                               default_top_k,
                               runtime_top_k_buf->data<uint>(),
                               batch_size,
                               default_top_p,
                               runtime_top_p_buf->data<float>(),
                               batch_size,
                               skip_top_p_decode_buf->data<bool>(),
                               nullptr,  // initial_top_p_buf,
                               nullptr,  // top_p_decay_buf,
                               nullptr,
                               nullptr,  // top_p_min_buf,
                               nullptr,
                               nullptr,  // top_p_reset_ids_buf,
                               nullptr,
                               stream_);

    invokeTopPInitialize(topp_id_vals_buf->data<int32_t>(),
                         topp_offset_buf->data<int32_t>(),
                         begin_topp_offset_buf->data<int32_t>(),
                         batch_size,
                         vocab_size_padded,
                         stream_);

    invokeAddBiasSoftMax(logits.data<SamplerT>(),
                         (SamplerT*)nullptr,  // bias
                         nullptr,             // end_id
                         nullptr,             // finished
                         batch_size,
                         vocab_size_padded,
                         vocab_size_padded,
                         stream_);

    invokeBatchTopPSampling(top_p_workspace->data(),
                            topp_ws_size,
                            cub_temp_storage_size,
                            transposed_tokens->dataWithOffset<int32_t>(step * batch_size),
                            nullptr,  // sequence_length
                            nullptr,  // finished
                            cum_log_probs,
                            output_log_probs,
                            logits.data<float>(),
                            topp_id_vals_buf->data<int32_t>(),
                            topp_offset_buf->data<int32_t>(),
                            begin_topp_offset_buf->data<int32_t>(),
                            (curandState_t*)curandstate_buf_->data(),
                            batch_size,
                            vocab_size_padded,
                            nullptr,  // end_id
                            max_top_p,
                            runtime_top_p_buf->data<float>(),
                            output_all_probs,
                            stream_,
                            &device_prop_,
                            skip_top_p_decode_buf->data<bool>());

    auto output_tokens = transpose({*transposed_tokens});
    copy({params.token_ids, *output_tokens});
    check_cuda_error();
    return GreedyOutput{};
}

}  // namespace rtp_llm
