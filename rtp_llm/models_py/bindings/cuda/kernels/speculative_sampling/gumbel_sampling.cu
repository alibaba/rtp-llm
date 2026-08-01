#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/gumbel_sampling.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace rtp_llm {
namespace {

constexpr uint32_t kPhiloxKeyA   = 0x9E3779B9U;
constexpr uint32_t kPhiloxKeyB   = 0xBB67AE85U;
constexpr uint32_t kPhiloxRoundA = 0xD2511F53U;
constexpr uint32_t kPhiloxRoundB = 0xCD9E8D57U;
constexpr float    kTritonRandScale = 4.6566127342e-10F;
constexpr int      kThreads         = 256;
constexpr int      kVocabBlockSize  = 1024;

struct Philox4x32 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

__device__ __forceinline__ Philox4x32 philox(uint64_t seed, uint64_t offset) {
    uint32_t c0 = static_cast<uint32_t>(offset);
    uint32_t c1 = static_cast<uint32_t>(offset >> 32);
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const uint32_t old_c0 = c0;
        const uint32_t old_c2 = c2;
        c0 = __umulhi(kPhiloxRoundB, old_c2) ^ c1 ^ k0;
        c2 = __umulhi(kPhiloxRoundA, old_c0) ^ c3 ^ k1;
        c1 = kPhiloxRoundB * old_c2;
        c3 = kPhiloxRoundA * old_c0;
        k0 += kPhiloxKeyA;
        k1 += kPhiloxKeyB;
    }
    return {c0, c1, c2, c3};
}

__device__ __forceinline__ float flippedGumbel(uint64_t request_seed, int64_t position, int64_t vocab_id) {
    // vLLM: gumbel_seed = tl.randint(seed, pos); u = tl.rand(gumbel_seed, vocab_id).
    const uint32_t position_seed = philox(request_seed, static_cast<uint64_t>(position)).x;
    const uint32_t bits = philox(static_cast<uint64_t>(position_seed), static_cast<uint64_t>(vocab_id)).x;
    const int32_t  signed_bits = static_cast<int32_t>(bits);
    const uint32_t magnitude = signed_bits < 0 ? ~bits : bits;  // -x-1 with wrap semantics.
    const float    u = fmaxf(static_cast<float>(magnitude) * kTritonRandScale, kTritonRandScale);
    return -logf(-log1pf(-u));
}

__device__ __forceinline__ double fp64Gumbel(uint64_t request_seed, int64_t position, int64_t vocab_id) {
    // vLLM USE_FP64: lo, hi, _, _ = tl.randint4x(gumbel_seed, vocab_id),
    // concatenate the two unsigned words and map the full 64-bit integer to
    // [0,1). Zero is clamped to float64 tiny before the ordinary transform.
    const uint32_t position_seed = philox(request_seed, static_cast<uint64_t>(position)).x;
    const auto     words = philox(static_cast<uint64_t>(position_seed), static_cast<uint64_t>(vocab_id));
    const uint64_t bits  = (static_cast<uint64_t>(words.y) << 32) | static_cast<uint64_t>(words.x);
    const double u = fmax(static_cast<double>(bits) * 5.421010862427522170037e-20,
                          std::numeric_limits<double>::min());
    return -log(-log(u));
}

template<bool INPUT_IS_PROBS, bool APPLY_TEMPERATURE, bool USE_FP64>
__global__ void gumbelBlockArgmaxKernel(const float* __restrict__ values,
                                        const int64_t* __restrict__ seeds,
                                        const int64_t* __restrict__ positions,
                                        const float* __restrict__ temperatures,
                                        int64_t* __restrict__ local_indices,
                                        std::conditional_t<USE_FP64, double, float>* __restrict__ local_values,
                                        int64_t rows,
                                        int64_t vocab_size,
                                        int64_t vocab_blocks) {
    // Match vLLM's two-level reduction geometry: independent 1024-token
    // vocabulary tiles expose enough parallelism for a 129k vocabulary, then
    // one small per-row reduction below chooses among tile winners. A single
    // block per row is mathematically equivalent but is prohibitively serial.
    const int64_t row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const int64_t vocab_block = blockIdx.x;
    const int64_t block_start = vocab_block * kVocabBlockSize;
    const float temperature = temperatures[row];
    const bool  stochastic  = temperature != 0.0F;
    const auto* row_values  = values + row * vocab_size;

    using AccT = std::conditional_t<USE_FP64, double, float>;
    AccT    best_value = -std::numeric_limits<AccT>::infinity();
    int64_t best_index = std::numeric_limits<int64_t>::max();
    for (int64_t local_col = threadIdx.x; local_col < kVocabBlockSize; local_col += blockDim.x) {
        const int64_t col = block_start + local_col;
        if (col >= vocab_size) {
            continue;
        }
        AccT score = static_cast<AccT>(row_values[col]);
        if constexpr (INPUT_IS_PROBS) {
            score = score > static_cast<AccT>(0) ? log(score) : -std::numeric_limits<AccT>::infinity();
        } else if constexpr (APPLY_TEMPERATURE) {
            if (stochastic && temperature != 1.0F) {
                // Match vLLM: the source logit is fp32 and temperature is
                // applied before the optional fp64 promotion used by the
                // noise/reduction path.
                score = static_cast<AccT>(row_values[col] / temperature);
            }
        }
        if (stochastic) {
            if constexpr (USE_FP64) {
                score += fp64Gumbel(static_cast<uint64_t>(seeds[row]), positions[row], col);
            } else {
                score += flippedGumbel(static_cast<uint64_t>(seeds[row]), positions[row], col);
            }
        }
        if (score > best_value || (score == best_value && col < best_index)) {
            best_value = score;
            best_index = col;
        }
    }

    __shared__ AccT    values_shared[kThreads];
    __shared__ int64_t indices_shared[kThreads];
    values_shared[threadIdx.x]  = best_value;
    indices_shared[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const AccT    other_value = values_shared[threadIdx.x + stride];
            const int64_t other_index = indices_shared[threadIdx.x + stride];
            if (other_value > values_shared[threadIdx.x]
                || (other_value == values_shared[threadIdx.x] && other_index < indices_shared[threadIdx.x])) {
                values_shared[threadIdx.x]  = other_value;
                indices_shared[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int64_t out = row * vocab_blocks + vocab_block;
        local_indices[out] = indices_shared[0];
        local_values[out]  = values_shared[0];
    }
}

template<typename AccT>
__global__ void finalizeGumbelArgmaxKernel(const AccT* __restrict__ local_values,
                                           const int64_t* __restrict__ local_indices,
                                           int64_t* __restrict__ output,
                                           int64_t vocab_blocks) {
    const int64_t row = blockIdx.x;
    AccT    best_value = -std::numeric_limits<AccT>::infinity();
    int64_t best_index = std::numeric_limits<int64_t>::max();
    for (int64_t block = threadIdx.x; block < vocab_blocks; block += blockDim.x) {
        const int64_t offset = row * vocab_blocks + block;
        const AccT    value  = local_values[offset];
        const int64_t index  = local_indices[offset];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    __shared__ AccT    values_shared[kThreads];
    __shared__ int64_t indices_shared[kThreads];
    values_shared[threadIdx.x]  = best_value;
    indices_shared[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const AccT    other_value = values_shared[threadIdx.x + stride];
            const int64_t other_index = indices_shared[threadIdx.x + stride];
            if (other_value > values_shared[threadIdx.x]
                || (other_value == values_shared[threadIdx.x] && other_index < indices_shared[threadIdx.x])) {
                values_shared[threadIdx.x]  = other_value;
                indices_shared[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = indices_shared[0];
    }
}

template<bool INPUT_IS_PROBS, bool APPLY_TEMPERATURE, bool USE_FP64>
torch::Tensor launchGumbelSample(const torch::Tensor& values,
                                 const torch::Tensor& seeds,
                                 const torch::Tensor& positions,
                                 const torch::Tensor& temperatures,
                                 cudaStream_t         stream) {
    const int64_t rows         = values.size(0);
    const int64_t vocab_size   = values.size(1);
    const int64_t vocab_blocks = (vocab_size + kVocabBlockSize - 1) / kVocabBlockSize;
    auto output = torch::empty({rows}, values.options().dtype(torch::kInt64));
    auto local_indices = torch::empty({rows, vocab_blocks}, values.options().dtype(torch::kInt64));
    using AccT = std::conditional_t<USE_FP64, double, float>;
    auto local_values = torch::empty(
        {rows, vocab_blocks}, values.options().dtype(USE_FP64 ? torch::kFloat64 : torch::kFloat32));
    const dim3 grid(static_cast<unsigned int>(vocab_blocks), static_cast<unsigned int>(rows));
    gumbelBlockArgmaxKernel<INPUT_IS_PROBS, APPLY_TEMPERATURE, USE_FP64><<<grid, kThreads, 0, stream>>>(
        values.data_ptr<float>(),
        seeds.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(),
        temperatures.data_ptr<float>(),
        local_indices.data_ptr<int64_t>(),
        local_values.data_ptr<AccT>(),
        rows,
        vocab_size,
        vocab_blocks);
    finalizeGumbelArgmaxKernel<AccT><<<rows, kThreads, 0, stream>>>(local_values.data_ptr<AccT>(),
                                                                   local_indices.data_ptr<int64_t>(),
                                                                   output.data_ptr<int64_t>(),
                                                                   vocab_blocks);
    return output;
}

template<typename DraftT, typename TargetT>
__global__ void greedyTokenVerifyKernel(const DraftT* __restrict__ draft_tokens,
                                         const TargetT* __restrict__ target_tokens,
                                         int32_t* __restrict__ output_tokens,
                                         int32_t* __restrict__ output_lengths,
                                         int64_t width) {
    const int64_t row = blockIdx.x;
    const auto* draft = draft_tokens + row * (width - 1);
    const auto* target = target_tokens + row * width;
    auto* output = output_tokens + row * width;
    int64_t accepted = 0;
    while (accepted + 1 < width
           && static_cast<int64_t>(draft[accepted]) == static_cast<int64_t>(target[accepted])) {
        output[accepted] = static_cast<int32_t>(draft[accepted]);
        ++accepted;
    }
    output[accepted] = static_cast<int32_t>(target[accepted]);
    for (int64_t col = accepted + 1; col < width; ++col) {
        output[col] = 0;
    }
    output_lengths[row] = static_cast<int32_t>(accepted + 1);
}

template<typename DraftT>
void launchGreedyTokenVerify(const torch::Tensor& draft_tokens,
                             const torch::Tensor& target_tokens,
                             torch::Tensor&       output_tokens,
                             torch::Tensor&       output_lengths,
                             cudaStream_t         stream) {
    const auto rows  = draft_tokens.size(0);
    const auto width = target_tokens.size(1);
    if (target_tokens.scalar_type() == torch::kInt64) {
        greedyTokenVerifyKernel<DraftT, int64_t><<<rows, 1, 0, stream>>>(draft_tokens.data_ptr<DraftT>(),
                                                                         target_tokens.data_ptr<int64_t>(),
                                                                         output_tokens.data_ptr<int32_t>(),
                                                                         output_lengths.data_ptr<int32_t>(),
                                                                         width);
    } else {
        greedyTokenVerifyKernel<DraftT, int32_t><<<rows, 1, 0, stream>>>(draft_tokens.data_ptr<DraftT>(),
                                                                         target_tokens.data_ptr<int32_t>(),
                                                                         output_tokens.data_ptr<int32_t>(),
                                                                         output_lengths.data_ptr<int32_t>(),
                                                                         width);
    }
}

}  // namespace

torch::Tensor gumbelSample(const torch::Tensor& values,
                           const torch::Tensor& seeds,
                           const torch::Tensor& positions,
                           const torch::Tensor& temperatures,
                           bool                 input_is_probs,
                           bool                 apply_temperature,
                           bool                 use_fp64) {
    TORCH_CHECK(values.is_cuda() && seeds.is_cuda() && positions.is_cuda() && temperatures.is_cuda(),
                "gumbelSample expects CUDA tensors");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32 && values.dim() == 2 && values.is_contiguous(),
                "gumbelSample values must be contiguous float32 [rows, vocab]");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64 && positions.scalar_type() == torch::kInt64,
                "gumbelSample seeds/positions must be int64");
    TORCH_CHECK(temperatures.scalar_type() == torch::kFloat32,
                "gumbelSample temperatures must be float32");
    TORCH_CHECK(seeds.is_contiguous() && positions.is_contiguous() && temperatures.is_contiguous(),
                "gumbelSample metadata tensors must be contiguous");
    TORCH_CHECK(values.device() == seeds.device() && values.device() == positions.device()
                    && values.device() == temperatures.device(),
                "gumbelSample inputs must be on the same CUDA device");
    const int64_t rows = values.size(0);
    TORCH_CHECK(rows > 0, "gumbelSample requires a non-empty batch");
    TORCH_CHECK(seeds.numel() == rows && positions.numel() == rows && temperatures.numel() == rows,
                "gumbelSample metadata rows do not match values");
    TORCH_CHECK(values.size(1) > 0, "gumbelSample requires a non-empty vocab");

    c10::cuda::CUDAGuard guard(values.device());
    auto stream = at::cuda::getCurrentCUDAStream(values.device().index()).stream();
    if (input_is_probs && use_fp64) {
        if (apply_temperature) {
            auto output = launchGumbelSample<true, true, true>(values, seeds, positions, temperatures, stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return output;
        }
        auto output = launchGumbelSample<true, false, true>(values, seeds, positions, temperatures, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    } else if (input_is_probs) {
        if (apply_temperature) {
            auto output = launchGumbelSample<true, true, false>(values, seeds, positions, temperatures, stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return output;
        }
        auto output = launchGumbelSample<true, false, false>(values, seeds, positions, temperatures, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    } else if (apply_temperature && use_fp64) {
        auto output = launchGumbelSample<false, true, true>(values, seeds, positions, temperatures, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    } else if (apply_temperature) {
        auto output = launchGumbelSample<false, true, false>(values, seeds, positions, temperatures, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    } else if (use_fp64) {
        auto output = launchGumbelSample<false, false, true>(values, seeds, positions, temperatures, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    auto output = launchGumbelSample<false, false, false>(values, seeds, positions, temperatures, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> greedyTokenVerify(const torch::Tensor& draft_tokens,
                                                           const torch::Tensor& target_tokens) {
    TORCH_CHECK(draft_tokens.is_cuda() && target_tokens.is_cuda(),
                "greedyTokenVerify expects CUDA tensors");
    TORCH_CHECK(draft_tokens.dim() == 2 && target_tokens.dim() == 2,
                "greedyTokenVerify expects draft [B,k] and target [B,k+1]");
    TORCH_CHECK(draft_tokens.size(0) == target_tokens.size(0)
                    && target_tokens.size(1) == draft_tokens.size(1) + 1,
                "greedyTokenVerify shape mismatch");
    TORCH_CHECK(draft_tokens.is_contiguous() && target_tokens.is_contiguous(),
                "greedyTokenVerify inputs must be contiguous");
    TORCH_CHECK((draft_tokens.scalar_type() == torch::kInt32 || draft_tokens.scalar_type() == torch::kInt64)
                    && (target_tokens.scalar_type() == torch::kInt32 || target_tokens.scalar_type() == torch::kInt64),
                "greedyTokenVerify token tensors must be int32 or int64");
    TORCH_CHECK(draft_tokens.device() == target_tokens.device(),
                "greedyTokenVerify inputs must be on the same CUDA device");
    c10::cuda::CUDAGuard guard(draft_tokens.device());
    auto int_options = draft_tokens.options().dtype(torch::kInt32);
    auto output_tokens = torch::empty(target_tokens.sizes(), int_options);
    auto output_lengths = torch::empty({draft_tokens.size(0)}, int_options);
    auto stream = at::cuda::getCurrentCUDAStream(draft_tokens.device().index()).stream();
    if (draft_tokens.scalar_type() == torch::kInt64) {
        launchGreedyTokenVerify<int64_t>(draft_tokens, target_tokens, output_tokens, output_lengths, stream);
    } else {
        launchGreedyTokenVerify<int32_t>(draft_tokens, target_tokens, output_tokens, output_lengths, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_tokens, output_lengths};
}

}  // namespace rtp_llm
