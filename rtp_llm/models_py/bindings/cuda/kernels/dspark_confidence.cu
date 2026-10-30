#include "rtp_llm/models_py/bindings/cuda/kernels/dspark_confidence.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rtp_llm {
namespace {

constexpr int kStatsColumns = 6;

struct LengthView {
    const void* data     = nullptr;
    int64_t     stride   = 0;
    bool        is_int64 = false;

    __device__ int64_t get(int64_t row) const {
        return is_int64 ? static_cast<const int64_t*>(data)[row * stride] :
                          static_cast<const int32_t*>(data)[row * stride];
    }
};

struct StatsParams {
    const float* rates;
    float*       output;
    LengthView   accepted, valid;
    const bool*  eligible;
    int64_t      eligible_stride, batch, gamma;
    bool         reduce_x, vectorized;
};

struct StatsValue {
    float values[kStatsColumns];

    __device__ void add(const StatsValue& other) {
#pragma unroll
        for (int i = 0; i < kStatsColumns; ++i) {
            values[i] = __fadd_rn(values[i], other.values[i]);
        }
    }
};

__device__ StatsValue confidenceObservation(const StatsParams& p, int64_t row, int64_t position) {
    // Match int64 Tensor subtraction, including its wraparound at INT64_MIN.
    const auto accepted  = static_cast<int64_t>(static_cast<uint64_t>(p.accepted.get(row)) - 1);
    const bool attempted = position <= accepted && (!p.valid.data || position < p.valid.get(row))
                           && (!p.eligible || p.eligible[row * p.eligible_stride]);
    const float rate     = p.rates[row * p.gamma + position];
    const bool  valid    = isfinite(rate) && rate >= 0.0f && rate <= 1.0f;
    const bool  observed = attempted && valid;
    // Select before arithmetic so a censored NaN never reaches the reduction.
    const float prediction = observed ? rate : 0.0f;
    const float label      = observed && position < accepted ? 1.0f : 0.0f;
    const float error      = __fsub_rn(prediction, label);
    return {{observed ? 1.0f : 0.0f,
             label,
             prediction,
             fabsf(error),
             __fmul_rn(error, error),
             attempted && !valid ? 1.0f : 0.0f}};
}

__global__ void dsparkConfidenceStatsKernel(StatsParams p) {
    const int64_t position = blockIdx.x;
    const int64_t lane     = threadIdx.x;
    const int64_t stride   = blockDim.x;
    // ATen's FP32 sum uses four independent accumulators, then combines them
    // left-to-right. A conventional warp sum changes the results even at B=32.
    StatsValue partial[4]{};
    if (p.vectorized) {
        for (int64_t row = lane * 4; row + 3 < p.batch; row += stride * 4) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                partial[i].add(confidenceObservation(p, row + i, position));
            }
        }
        const int64_t tail = p.batch - p.batch % 4 + lane;
        if (tail < p.batch) {
            partial[0].add(confidenceObservation(p, tail, position));
        }
    } else {
        for (int64_t start = lane; start < p.batch; start += stride * 4) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t row = start + i * stride;
                if (row < p.batch) {
                    partial[i].add(confidenceObservation(p, row, position));
                }
            }
        }
    }
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        partial[0].add(partial[i]);
    }

    __shared__ StatsValue shared[512];
    StatsValue            value     = partial[0];
    int                   remaining = blockDim.x;
    // A non-fastest-dimension sum uses ATen's block-y tree. Gamma=1 instead
    // uses its block-x tree and warp shuffle, after vectorized input loading.
    const int shared_stop = p.reduce_x ? 32 : 1;
    shared[lane]          = value;
    for (int offset = remaining / 2; offset >= shared_stop; offset >>= 1) {
        __syncthreads();
        if (lane < offset) {
            value.add(shared[lane + offset]);
            shared[lane] = value;
        }
    }
    if (p.reduce_x) {
        __syncthreads();
        remaining = min(remaining, 32);
        for (int offset = remaining / 2; offset > 0; offset >>= 1) {
            StatsValue other;
#pragma unroll
            for (int i = 0; i < kStatsColumns; ++i) {
                other.values[i] = __shfl_down_sync(__activemask(), value.values[i], offset);
            }
            value.add(other);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kStatsColumns; ++i) {
            p.output[position * kStatsColumns + i] = value.values[i];
        }
    }
}

int floorPowerOfTwo(int64_t value, int limit) {
    int result = 1;
    while (result < limit && result * 2 <= value) {
        result *= 2;
    }
    return result;
}

LengthView lengthView(const torch::Tensor& value) {
    return value.defined() ? LengthView{value.data_ptr(), value.stride(0), value.scalar_type() == torch::kInt64} :
                             LengthView{};
}

struct ConfidenceParams {
    LengthView   anchors, tokens;
    const float *weights, *hidden;
    float*       output;
    int64_t      count, gamma, rank, vocab;
    int64_t      token_row_stride, token_col_stride;
    int64_t      embedding_row_stride, embedding_col_stride;
    int64_t      weight_stride, hidden_row_stride, hidden_col_stride;
    float        inverse_temperature;
};

template<typename Embedding>
__global__ void dsparkConfidenceRatesKernel(const Embedding* embeddings, ConfidenceParams p) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < p.count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = idx / p.gamma, step = idx % p.gamma;
        const int64_t token =
            step == 0 ? p.anchors.get(row) : p.tokens.get(row * p.token_row_stride + (step - 1) * p.token_col_stride);
        CUDA_KERNEL_ASSERT(token >= 0 && token < p.vocab);
        float dot = 0.0f;
        for (int64_t k = 0; k < p.rank; ++k) {
            // Conversion is exact for both FP16 and BF16, as in to(FP32).mm.
            const float embedding =
                static_cast<float>(embeddings[token * p.embedding_row_stride + k * p.embedding_col_stride]);
            dot = __fmaf_rn(embedding, p.weights[k * p.weight_stride], dot);
        }
        const float logit = __fadd_rn(p.hidden[row * p.hidden_row_stride + step * p.hidden_col_stride], dot);
        // ATen divides by a scalar using an FP32 reciprocal multiplication.
        const float scaled = __fmul_rn(logit, p.inverse_temperature);
        // fminf/fmaxf would turn NaN into a finite boundary unlike Tensor.clamp.
        const float bounded = scaled < -60.0f ? -60.0f : (scaled > 60.0f ? 60.0f : scaled);
        p.output[idx]       = 1.0f / (1.0f + expf(-bounded));
    }
}

}  // namespace

torch::Tensor tryDSparkConfidenceStats(const torch::Tensor& confidence_rates,
                                       const torch::Tensor& accept_len,
                                       const torch::Tensor& valid_lengths,
                                       const torch::Tensor& eligible_rows) {
    if (!confidence_rates.is_cuda() || !confidence_rates.is_contiguous()) {
        return {};
    }
    const int64_t batch = confidence_rates.size(0), gamma = confidence_rates.size(1);
    if (gamma > std::numeric_limits<int32_t>::max() || batch > std::numeric_limits<int32_t>::max()
        || confidence_rates.numel() > std::numeric_limits<int32_t>::max() / static_cast<int64_t>(sizeof(float))) {
        return {};
    }
    const c10::cuda::CUDAGuard guard(confidence_rates.device());
    if (batch == 0 || gamma == 0) {
        return torch::zeros({gamma, kStatsColumns}, confidence_rates.options());
    }

    // Match ATen Reduce.cuh's FP32 reduction configuration on freshly allocated
    // contiguous [B, gamma] intermediates. Only reductions within one CTA are
    // fused; the existing implementation retains its multi-CTA behavior.
    const bool reduce_x    = gamma == 1;
    const bool vectorized  = reduce_x && batch >= 128;
    const int  output_vec  = reduce_x ? 1 : (gamma % 4 == 0 ? 4 : (gamma % 2 == 0 ? 2 : 1));
    const int  max_threads = 512 / output_vec;
    const int  dim0        = floorPowerOfTwo(reduce_x ? batch / (vectorized ? 4 : 1) : gamma / output_vec, max_threads);
    const int  dim1        = floorPowerOfTwo(reduce_x ? gamma : batch, max_threads);
    int        width       = std::min(dim0, 32);
    const int  height      = std::min(dim1, max_threads / width);
    width                  = std::min(dim0, max_threads / height);
    const int  input_width = reduce_x ? width : 1;
    const int  values_per_thread = (batch + input_width - 1) / input_width;
    const bool reduce_y          = values_per_thread >= std::min(height * 16, 256);
    const int  threads           = input_width * (reduce_y ? height : 1);
    if (reduce_y && (batch + threads - 1) / threads >= 256) {
        return {};
    }

    auto        output = torch::empty({gamma, kStatsColumns}, confidence_rates.options());
    StatsParams params{confidence_rates.data_ptr<float>(),
                       output.data_ptr<float>(),
                       lengthView(accept_len),
                       lengthView(valid_lengths),
                       eligible_rows.defined() ? eligible_rows.data_ptr<bool>() : nullptr,
                       eligible_rows.defined() ? eligible_rows.stride(0) : 0,
                       batch,
                       gamma,
                       reduce_x,
                       vectorized};
    dsparkConfidenceStatsKernel<<<static_cast<int>(gamma), threads, 0, at::cuda::getCurrentCUDAStream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void invokeDSparkConfidenceRates(const torch::Tensor& anchors,
                                 const torch::Tensor& token_ids,
                                 const torch::Tensor& markov_w1,
                                 const torch::Tensor& confidence_w_markov,
                                 const torch::Tensor& hidden_logits,
                                 torch::Tensor&       confidence_rates,
                                 float                temperature,
                                 cudaStream_t         stream) {
    const int64_t batch = confidence_rates.size(0), gamma = confidence_rates.size(1);
    if (batch == 0 || gamma == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(confidence_rates.device());
    TORCH_CHECK(anchors.scalar_type() == torch::kInt32 || anchors.scalar_type() == torch::kInt64,
                "DSpARK confidence requires int32/int64 anchors");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt32 || token_ids.scalar_type() == torch::kInt64,
                "DSpARK confidence requires int32/int64 sampled tokens");
    TORCH_CHECK(confidence_rates.is_contiguous(), "DSpARK confidence requires contiguous output");
    ConfidenceParams params{lengthView(anchors),
                            {token_ids.data_ptr(), 1, token_ids.scalar_type() == torch::kInt64},
                            confidence_w_markov.data_ptr<float>(),
                            hidden_logits.data_ptr<float>(),
                            confidence_rates.data_ptr<float>(),
                            batch * gamma,
                            gamma,
                            markov_w1.size(1),
                            markov_w1.size(0),
                            token_ids.stride(0),
                            token_ids.stride(1),
                            markov_w1.stride(0),
                            markov_w1.stride(1),
                            confidence_w_markov.stride(0),
                            hidden_logits.stride(0),
                            hidden_logits.stride(1),
                            1.0f / temperature};
    const int        grid = static_cast<int>(std::min<int64_t>((params.count + 127) / 128, 65535));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kBFloat16, torch::kHalf, markov_w1.scalar_type(), "dsparkConfidenceRates", [&] {
            dsparkConfidenceRatesKernel<<<grid, 128, 0, stream>>>(markov_w1.data_ptr<scalar_t>(), params);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
