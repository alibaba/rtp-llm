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

struct DraftCommitParams {
    const int32_t* sampled;
    const void*    candidates;
    const float*   probabilities;
    const int64_t* map;
    int32_t*       tokens_out;
    float*         probabilities_out;
    int32_t*       candidates_out;
    const void*    embeddings;
    void*          next_embedding;
    int64_t        sampled_stride, candidate_stride, probability_stride, map_stride, map_size;
    int64_t        token_stride, output_probability_stride, output_candidate_stride;
    int64_t        width, output_width, embedding_rows, embedding_width, embedding_row_stride, embedding_col_stride;
};

template<typename Bits, typename Candidate>
__global__ void dsparkDraftStepCommitKernel(DraftCommitParams p) {
    const int64_t      row = blockIdx.x;
    const auto*        candidates = static_cast<const Candidate*>(p.candidates);
    __shared__ int32_t next_token;
    if (threadIdx.x == 0) {
        int64_t token = p.sampled[row * p.sampled_stride];
        if (p.candidates) {
            CUDA_KERNEL_ASSERT(token >= 0 && token < p.width);
            token = candidates[row * p.candidate_stride + token];
        }
        if (p.map) {
            CUDA_KERNEL_ASSERT(token >= 0 && token < p.map_size);
            token = p.map[token * p.map_stride];
        }
        next_token                         = static_cast<int32_t>(token);
        p.tokens_out[row * p.token_stride] = next_token;
        if (p.next_embedding) {
            CUDA_KERNEL_ASSERT(next_token >= 0 && next_token < p.embedding_rows);
        }
    }

    // A mapped compact distribution retains its original draft vocabulary
    // coordinates, exactly like zero_().scatter_() in the existing path.
    const bool scatter_probabilities = p.probabilities_out && p.candidates && !p.candidates_out;
    if (scatter_probabilities) {
        for (int64_t column = threadIdx.x; column < p.output_width; column += blockDim.x) {
            p.probabilities_out[row * p.output_probability_stride + column] = 0.0f;
        }
    }
    __syncthreads();
    for (int64_t column = threadIdx.x; column < p.width; column += blockDim.x) {
        if (p.probabilities_out) {
            int64_t output_column = column;
            if (scatter_probabilities) {
                output_column = candidates[row * p.candidate_stride + column];
                CUDA_KERNEL_ASSERT(output_column >= 0 && output_column < p.output_width);
            }
            p.probabilities_out[row * p.output_probability_stride + output_column] =
                p.probabilities[row * p.probability_stride + column];
        }
        if (p.candidates_out) {
            p.candidates_out[row * p.output_candidate_stride + column] =
                static_cast<int32_t>(candidates[row * p.candidate_stride + column]);
        }
    }
    if (p.next_embedding) {
        const auto* weights = static_cast<const Bits*>(p.embeddings);
        auto*       output  = static_cast<Bits*>(p.next_embedding);
        for (int64_t column = threadIdx.x; column < p.embedding_width; column += blockDim.x) {
            output[row * p.embedding_width + column] =
                weights[static_cast<int64_t>(next_token) * p.embedding_row_stride + column * p.embedding_col_stride];
        }
    }
}

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

bool tryDSparkDraftStepCommit(const torch::Tensor& sampled_tokens,
                              const torch::Tensor& candidate_ids,
                              const torch::Tensor& probabilities,
                              const torch::Tensor& d2t_map,
                              const torch::Tensor& markov_w1,
                              const torch::Tensor& output_tokens,
                              const torch::Tensor& output_probabilities,
                              const torch::Tensor& output_candidate_ids,
                              const torch::Tensor& next_embedding) {
    if (!sampled_tokens.defined() || !sampled_tokens.is_cuda() || sampled_tokens.dim() != 1
        || sampled_tokens.scalar_type() != torch::kInt32 || !output_tokens.defined() || !markov_w1.defined()
        || markov_w1.dim() != 2
        || (markov_w1.scalar_type() != torch::kFloat32 && markov_w1.scalar_type() != torch::kFloat16
            && markov_w1.scalar_type() != torch::kBFloat16)) {
        return false;
    }
    const auto    device           = sampled_tokens.device();
    const int64_t batch            = sampled_tokens.size(0);
    const auto    vector_on_device = [&](const torch::Tensor& tensor, c10::ScalarType dtype) {
        return tensor.is_cuda() && tensor.device() == device && tensor.dim() == 1 && tensor.size(0) == batch
               && tensor.scalar_type() == dtype;
    };
    const auto rows_on_device = [&](const torch::Tensor& tensor, c10::ScalarType dtype, int64_t width) {
        return tensor.is_cuda() && tensor.device() == device && tensor.dim() == 2 && tensor.size(0) == batch
               && tensor.size(1) == width && tensor.stride(1) == 1 && tensor.scalar_type() == dtype
               && (batch <= 1 || tensor.stride(0) >= width);
    };
    if (!vector_on_device(output_tokens, torch::kInt32) || !markov_w1.is_cuda() || markov_w1.device() != device
        || (batch > 1 && output_tokens.stride(0) <= 0) || batch > std::numeric_limits<int32_t>::max()
        || (probabilities.defined() && probabilities.dim() != 2)
        || (candidate_ids.defined() && candidate_ids.dim() != 2)
        || (output_probabilities.defined() && output_probabilities.dim() != 2)) {
        return false;
    }
    const int64_t width = probabilities.defined() ? probabilities.size(1) :
                          candidate_ids.defined() ? candidate_ids.size(1) :
                                                    0;
    if ((probabilities.defined() && !rows_on_device(probabilities, torch::kFloat32, width))
        || (candidate_ids.defined() && !rows_on_device(candidate_ids, torch::kInt64, width)
            && !rows_on_device(candidate_ids, torch::kInt32, width))
        || (d2t_map.defined()
            && (!d2t_map.is_cuda() || d2t_map.device() != device || d2t_map.dim() != 1
                || d2t_map.scalar_type() != torch::kInt64))
        || (output_probabilities.defined()
            && (!probabilities.defined()
                || !rows_on_device(output_probabilities, torch::kFloat32, output_probabilities.size(1))
                || (!candidate_ids.defined() && output_probabilities.size(1) != width)))
        || (output_candidate_ids.defined()
            && (!candidate_ids.defined() || !rows_on_device(output_candidate_ids, torch::kInt32, width)
                || (output_probabilities.defined() && output_probabilities.size(1) != width)))
        || (next_embedding.defined()
            && (!rows_on_device(next_embedding, markov_w1.scalar_type(), markov_w1.size(1))
                || !next_embedding.is_contiguous()))) {
        return false;
    }
    if (batch == 0) {
        return true;
    }
    const c10::cuda::CUDAGuard guard(device);
    DraftCommitParams          params{
        sampled_tokens.data_ptr<int32_t>(),
        candidate_ids.defined() ? candidate_ids.data_ptr() : nullptr,
        probabilities.defined() ? probabilities.data_ptr<float>() : nullptr,
        d2t_map.defined() ? d2t_map.data_ptr<int64_t>() : nullptr,
        output_tokens.data_ptr<int32_t>(),
        output_probabilities.defined() ? output_probabilities.data_ptr<float>() : nullptr,
        output_candidate_ids.defined() ? output_candidate_ids.data_ptr<int32_t>() : nullptr,
        markov_w1.data_ptr(),
        next_embedding.defined() ? next_embedding.data_ptr() : nullptr,
        sampled_tokens.stride(0),
        candidate_ids.defined() ? candidate_ids.stride(0) : 0,
        probabilities.defined() ? probabilities.stride(0) : 0,
        d2t_map.defined() ? d2t_map.stride(0) : 0,
        d2t_map.defined() ? d2t_map.numel() : 0,
        output_tokens.stride(0),
        output_probabilities.defined() ? output_probabilities.stride(0) : 0,
        output_candidate_ids.defined() ? output_candidate_ids.stride(0) : 0,
        candidate_ids.defined() || output_probabilities.defined() || output_candidate_ids.defined() ? width : 0,
        output_probabilities.defined() ? output_probabilities.size(1) : 0,
        markov_w1.size(0),
        markov_w1.size(1),
        markov_w1.stride(0),
        markov_w1.stride(1)};
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto launch = [&](auto candidate_type) {
        using Candidate = decltype(candidate_type);
        if (markov_w1.element_size() == 4) {
            dsparkDraftStepCommitKernel<uint32_t, Candidate><<<static_cast<int>(batch), 256, 0, stream>>>(params);
        } else {
            dsparkDraftStepCommitKernel<uint16_t, Candidate><<<static_cast<int>(batch), 256, 0, stream>>>(params);
        }
    };
    if (candidate_ids.defined() && candidate_ids.scalar_type() == torch::kInt32) {
        launch(int32_t{});
    } else {
        launch(int64_t{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

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
