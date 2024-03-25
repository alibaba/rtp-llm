#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"
#include "src/fastertransformer/cuda/memory_utils.h"

using namespace std;

namespace fastertransformer {

using SamplerT = float;

void CudaDevice::sampleGreedy(const GreedyParams& params) {
    const auto& logits = params.logits;
    const auto batch_size = logits.shape()[0];
    const auto vocab_size_padded = logits.shape()[1];
    const auto step = params.token_ids.shape()[0] - 1;

    // 1. confirm buffer sizes
    auto& top_k = params.top_k;
    auto& random_seed = params.random_seed;

    auto default_top_k = 1;
    auto runtime_top_k_size = batch_size;
    auto max_top_k = *max_element(top_k.data<int32_t>(), top_k.dataWithOffset<int32_t>(top_k.size()));
    FT_LOG_INFO("max_top_k: %d", max_top_k);
    size_t topk_ws_size;

    // this is to query workspace size
    invokeTopKSampling<SamplerT>(nullptr,
                                topk_ws_size,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                max_top_k,
                                1.0f,
                                vocab_size_padded,
                                nullptr,
                                stream_,
                                batch_size,
                                nullptr);
    FT_LOG_INFO("topk_ws_size: %d", topk_ws_size);

    auto default_top_p = 0.0f;
    auto runtime_top_p_size = 0;
    auto repetition_penalty_type = RepetitionPenaltyType::None;

    // 2. allocate buffers

    // see BaseSamplingLayer<T>::allocateBuffer ------------------
    auto curandstate_buf = allocateBuffer({batch_size * sizeof(curandState_t)});
    auto random_seeds_buf = allocateBuffer({DataType::TYPE_UINT64, {batch_size}});
    auto temperature_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
    auto skip_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});
    // penalty_logits_
    // repetition_penalty_buf_
    // min_lengths_buf_
    // runtime_logits_buf_
    // temperature_        = new float[batch_size];
    // repetition_penalty_ = new float[batch_size];
    // min_lengths_        = new int[batch_size];
    // skip_decode_        = new bool[batch_size];
    // ------------------------------------------------------------

    // TopKSamplingLayer<T>::allocateBuffer
    auto top_k_workspace = allocateBuffer({topk_ws_size});
    auto runtime_top_k_buf = allocateBuffer({DataType::TYPE_UINT32, {batch_size}});
    auto runtime_top_p_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});

    // TODO: integrate TopPSamplingLayer

    // 3. prepare kernel inputs
    copy({*runtime_top_k_buf, top_k});

    if (random_seed) {
        auto& seeds = random_seed.value().get();
        if (seeds.size() == 1) {
            invokeCurandInitialize(
                (curandState_t *)curandstate_buf->data(), batch_size,
                seeds.data<int64_t>()[0], stream_);
        } else {
            copy({*random_seeds_buf, seeds});
            invokeCurandBatchInitialize(
                (curandState_t *)curandstate_buf->data(), batch_size,
                (unsigned long long *)random_seeds_buf->data(), stream_);
        }
    }
    invokeSetupTopKRuntimeArgs(batch_size,
                                default_top_k,
                                (uint *)runtime_top_k_buf->data(),
                                runtime_top_k_size,
                                default_top_p,
                                (float *)runtime_top_p_buf->data(),
                                runtime_top_p_size,
                                (bool *)skip_decode_buf->data(),
                                stream_);

    // 4. kernel call
    auto cum_log_probs = params.cum_log_probs.has_value() ?
                         params.cum_log_probs.value().get().data<float>() : nullptr;
    auto output_log_probs = params.output_log_probs.has_value() ?
                            params.output_log_probs.value().get().data<float>() : nullptr;

    invokeBatchTopKSampling(
        top_k_workspace->data(),
        topk_ws_size,
        logits.data<float>(),
        params.token_ids.dataWithOffset<int32_t>(step * batch_size),
        params.input_lenghts.data<int32_t>(),
        nullptr, // finished
        cum_log_probs,
        output_log_probs,
        nullptr, // output_index_logits
        nullptr, // token_id_for_index_prob,
        (curandState_t *)curandstate_buf->data(),
        max_top_k,  // useless because runtime_top_k_buf_ is never nullptr. Keep for legacy.
        runtime_top_k_buf->data<int32_t>(),
        1.0f,  // useless because runtime_top_p_buf_ is never nullptr. Keep for legacy.
        runtime_top_p_buf->data<float>(),
        vocab_size_padded,
        nullptr, // end_id
        stream_,
        batch_size,
        skip_decode_buf->data<bool>());
}

} // namespace fastertransformer
