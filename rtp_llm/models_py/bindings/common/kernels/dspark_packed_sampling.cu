#include "rtp_llm/models_py/bindings/common/kernels/dspark_packed_sampling.h"
#if USING_CUDA
#include <c10/cuda/CUDAException.h>
#define SAMPLING_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK
#elif USING_ROCM
#include <c10/hip/HIPException.h>
#define SAMPLING_LAUNCH_CHECK C10_HIP_KERNEL_LAUNCH_CHECK
#endif

namespace rtp_llm {
namespace {
struct Params {
    float*   f[6];
    int32_t* i[5];
    int64_t* seed;
    int64_t* offset;
    bool*    sample;
    float*   cum;
    int64_t* token_ptrs;
};
template<typename T>
T* optionalPtr(const std::optional<torch::Tensor>& t) {
    return t ? t->data_ptr<T>() : nullptr;
}
Params pointers(const GreedyParams& p) {
    return {{p.top_p.data_ptr<float>(),
             p.temperature.data_ptr<float>(),
             optionalPtr<float>(p.repetition_penalty),
             optionalPtr<float>(p.presence_penalty),
             optionalPtr<float>(p.frequency_penalty),
             p.inverse_temperature.data_ptr<float>()},
            {p.top_k.data_ptr<int32_t>(),
             optionalPtr<int32_t>(p.no_repeat_ngram_size),
             p.input_lengths.data_ptr<int32_t>(),
             p.sequence_lengths.data_ptr<int32_t>(),
             p.sequence_last_indexes.data_ptr<int32_t>()},
            p.seed.defined() ? p.seed.data_ptr<int64_t>() : nullptr,
            p.offset.defined() ? p.offset.data_ptr<int64_t>() : nullptr,
            optionalPtr<bool>(p.do_sample),
            optionalPtr<float>(p.cum_log_probs),
            p.output_ids_ptrs.defined() ? p.output_ids_ptrs.data_ptr<int64_t>() : nullptr};
}

template<bool History>
__global__ void gather_sampling_rows(
    Params src, Params dst, const int64_t* map, int packed_rows, const int32_t* history, int32_t* tokens, int width) {
    const int p = History ? blockIdx.x : blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= packed_rows)
        return;
    const int d = map[p];
    if (!History || threadIdx.x == 0) {
        for (int f = 0; f < 6; ++f)
            if (src.f[f])
                dst.f[f][p] = src.f[f][d];
        for (int f = 0; f < 5; ++f)
            if (src.i[f])
                dst.i[f][p] = src.i[f][d];
        dst.sample[p] = src.sample[d];
        if (src.cum)
            dst.cum[p] = src.cum[d];
        if (src.seed) {
            dst.seed[p]   = src.seed[d];
            dst.offset[p] = src.offset[d];
        }
        if (dst.token_ptrs) {
            dst.token_ptrs[p] = reinterpret_cast<int64_t>(tokens + static_cast<int64_t>(p) * width);
        }
    }
    if (History) {
        for (int c = threadIdx.x; c < width; c += blockDim.x) {
            tokens[static_cast<int64_t>(p) * width + c] = history[static_cast<int64_t>(d) * width + c];
        }
    }
}

struct Result {
    int32_t* tokens;
    float*   probs;
    int32_t* indices;
    float*   sampling;
    float*   cum;
    bool*    success;
    int64_t  token_stride, prob_stride, sampling_stride;
};
template<typename T>
T* ptr(const torch::Tensor& t) {
    return t.defined() ? t.data_ptr<T>() : nullptr;
}
Result pointers(const SamplingResult& p) {
    return {p.tokens.data_ptr<int32_t>() + p.tokens.size(1) - 1,
            ptr<float>(p.probs),
            ptr<int32_t>(p.indices),
            ptr<float>(p.sampling_probs),
            ptr<float>(p.cum_log_probs),
            ptr<bool>(p.success),
            p.tokens.stride(0),
            p.probs.defined() ? p.probs.stride(0) : 0,
            p.sampling_probs.defined() ? p.sampling_probs.stride(0) : 0};
}
template<bool Vectorized>
__global__ void
scatter_sampling_rows(Result src, Result dst, const int32_t* inverse, int prob_width, int sampling_width) {
    const int d = blockIdx.y;
    const int p = inverse[d];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        dst.tokens[d * dst.token_stride] = p < 0 ? 0 : src.tokens[p * src.token_stride];
        if (dst.cum)
            dst.cum[d] = p < 0 ? 0.0f : src.cum[p];
        if (dst.success)
            dst.success[d] = p >= 0 && (!src.success || src.success[p]);
    }
    if constexpr (Vectorized) {
        const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if (c < prob_width) {
            *reinterpret_cast<float4*>(dst.probs + d * dst.prob_stride + c) =
                p < 0 ? make_float4(0, 0, 0, 0) : *reinterpret_cast<const float4*>(src.probs + p * src.prob_stride + c);
            if (dst.indices) {
                *reinterpret_cast<int4*>(dst.indices + d * dst.prob_stride + c) =
                    p < 0 ? make_int4(-1, -1, -1, -1) :
                            *reinterpret_cast<const int4*>(src.indices + p * src.prob_stride + c);
            }
        }
        if (c < sampling_width) {
            *reinterpret_cast<float4*>(dst.sampling + d * dst.sampling_stride + c) =
                p < 0 ? make_float4(0, 0, 0, 0) :
                        *reinterpret_cast<const float4*>(src.sampling + p * src.sampling_stride + c);
        }
    } else {
        for (int c = blockIdx.x * 1024 + threadIdx.x; c < (blockIdx.x + 1) * 1024; c += blockDim.x) {
            if (c < prob_width) {
                dst.probs[d * dst.prob_stride + c] = p < 0 ? 0.0f : src.probs[p * src.prob_stride + c];
                if (dst.indices)
                    dst.indices[d * dst.prob_stride + c] = p < 0 ? -1 : src.indices[p * src.prob_stride + c];
            }
            if (c < sampling_width) {
                dst.sampling[d * dst.sampling_stride + c] = p < 0 ? 0.0f : src.sampling[p * src.sampling_stride + c];
            }
        }
    }
}
}  // namespace

void invokeSamplingGather(const GreedyParams&  dense,
                          const GreedyParams&  packed,
                          const torch::Tensor& map,
                          const torch::Tensor& history,
                          const torch::Tensor& tokens,
                          SamplingStream       stream) {
    const int   q       = packed.top_k.numel();
    const auto* row_map = map.data_ptr<int64_t>();
    if (history.defined()) {
        gather_sampling_rows<true><<<q, 128, 0, stream>>>(pointers(dense),
                                                          pointers(packed),
                                                          row_map,
                                                          q,
                                                          history.data_ptr<int32_t>(),
                                                          tokens.data_ptr<int32_t>(),
                                                          tokens.size(1));
    } else {
        gather_sampling_rows<false><<<(q + 127) / 128, 128, 0, stream>>>(
            pointers(dense), pointers(packed), row_map, q, nullptr, tokens.data_ptr<int32_t>(), tokens.size(1));
    }
    SAMPLING_LAUNCH_CHECK();
}

void invokeSamplingScatter(const SamplingResult& packed,
                           const SamplingResult& dense,
                           const torch::Tensor&  inverse,
                           SamplingStream        stream) {
    const int  w = dense.probs.defined() ? dense.probs.size(1) : 0;
    const int  s = dense.sampling_probs.defined() ? dense.sampling_probs.size(1) : 0;
    const dim3 grid(std::max(1, (std::max(w, s) + 1023) / 1024), inverse.numel());
    const auto aligned = [](const torch::Tensor& t) {
        return !t.defined()
               || (t.size(1) % 4 == 0 && t.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
    };
    if (aligned(packed.probs) && aligned(dense.probs) && aligned(packed.indices) && aligned(dense.indices)
        && aligned(packed.sampling_probs) && aligned(dense.sampling_probs)) {
        scatter_sampling_rows<true>
            <<<grid, 256, 0, stream>>>(pointers(packed), pointers(dense), inverse.data_ptr<int32_t>(), w, s);
    } else {
        scatter_sampling_rows<false>
            <<<grid, 256, 0, stream>>>(pointers(packed), pointers(dense), inverse.data_ptr<int32_t>(), w, s);
    }
    SAMPLING_LAUNCH_CHECK();
}
}  // namespace rtp_llm
