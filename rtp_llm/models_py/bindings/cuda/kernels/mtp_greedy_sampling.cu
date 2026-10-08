#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_greedy_sampling.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace rtp_llm {
namespace {

constexpr int kThreads = 256;
constexpr int kTile    = 4096;

template<typename T>
struct alignas(sizeof(T) * 4) Pack {
    T values[4];
};

template<typename T>
__global__ void mtpGreedyOneHotKernel(const int64_t* selected,
                                     const int64_t* d2t_map,
                                     int64_t map_size,
                                     int64_t map_stride,
                                     int32_t* tokens,
                                     T* probabilities,
                                     int64_t vocab_size) {
    const int64_t row   = blockIdx.x;
    const int64_t token = selected[row];
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        tokens[row] = static_cast<int32_t>(token < map_size ? d2t_map[token * map_stride] : token);
    }
    for (int64_t begin = static_cast<int64_t>(blockIdx.y) * kTile; begin < vocab_size;
         begin += static_cast<int64_t>(gridDim.y) * kTile) {
        const int64_t end = min(vocab_size, begin + kTile);
        if (vocab_size % 4 == 0) {
            auto* output = reinterpret_cast<Pack<T>*>(probabilities + row * vocab_size);
            for (int64_t col = begin + threadIdx.x * 4; col < end; col += blockDim.x * 4) {
                Pack<T> values;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    values.values[i] = static_cast<T>(col + i == token ? 1 : 0);
                }
                output[col / 4] = values;
            }
        } else {
            for (int64_t col = begin + threadIdx.x; col < end; col += blockDim.x) {
                probabilities[row * vocab_size + col] = static_cast<T>(col == token ? 1 : 0);
            }
        }
    }
}

}  // namespace

void invokeMtpGreedySampling(const torch::Tensor& logits,
                             const torch::Tensor& d2t_map,
                             torch::Tensor&       token_ids,
                             torch::Tensor&       probabilities,
                             cudaStream_t         stream) {
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2, "greedy sampling expects CUDA [batch, vocab] logits");
    c10::cuda::CUDAGuard device_guard(logits.device());
    const bool has_map = d2t_map.defined() && d2t_map.numel() > 0;
    if (has_map) {
        TORCH_CHECK(d2t_map.device() == logits.device() && d2t_map.scalar_type() == torch::kInt64
                        && d2t_map.dim() == 1,
                    "greedy draft-to-target mapping must be int64 [vocab] on the logits device");
    }
    // Reuse ATen's dtype/stride-aware reduction and its NaN/tie semantics.
    const auto selected = torch::argmax(logits, -1, true);
    token_ids           = torch::empty(selected.sizes(), selected.options().dtype(torch::kInt32));
    probabilities       = torch::empty(logits.sizes(), logits.options());
    if (logits.size(0) == 0) {
        return;
    }
    const int64_t vocab_size = logits.size(1);
    const dim3 grid(logits.size(0), std::min<int64_t>((vocab_size + kTile - 1) / kTile, 65535));
    AT_DISPATCH_ALL_TYPES_AND2(torch::kFloat16, torch::kBFloat16, logits.scalar_type(), "mtp_greedy_one_hot", [&] {
        mtpGreedyOneHotKernel<scalar_t><<<grid, kThreads, 0, stream>>>(selected.data_ptr<int64_t>(),
                                                                    has_map ? d2t_map.data_ptr<int64_t>() : nullptr,
                                                                    has_map ? d2t_map.numel() : 0,
                                                                    has_map ? d2t_map.stride(0) : 0,
                                                                    token_ids.data_ptr<int32_t>(),
                                                                    probabilities.data_ptr<scalar_t>(),
                                                                    vocab_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

}  // namespace rtp_llm
