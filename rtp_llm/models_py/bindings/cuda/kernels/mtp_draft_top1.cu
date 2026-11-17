#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_draft_top1.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace rtp_llm {
namespace {
constexpr int kThreads = 256;
constexpr int kChunk   = 4096;
using Candidate        = unsigned long long;

__device__ Candidate blockMax(Candidate value) {
    __shared__ Candidate warps[kThreads / 32];
    for (int delta = 16; delta > 0; delta /= 2) {
        auto other = __shfl_down_sync(0xffffffff, value, delta);
        value      = value > other ? value : other;
    }
    if (threadIdx.x % 32 == 0) {
        warps[threadIdx.x / 32] = value;
    }
    __syncthreads();
    value = threadIdx.x < kThreads / 32 ? warps[threadIdx.x] : 0;
    if (threadIdx.x < 32) {
        for (int delta = 16; delta > 0; delta /= 2) {
            auto other = __shfl_down_sync(0xffffffff, value, delta);
            value      = value > other ? value : other;
        }
    }
    return value;
}

__device__ Candidate encode(float value, unsigned int token) {
    auto bits = __float_as_uint(value);
    // PyTorch argmax chooses the first NaN, or the first maximum. Normalize
    // signed zero and NaN payloads. Integer ordering also preserves subnormals.
    if ((bits & 0x7fffffff) == 0) {
        bits = 0;
    }
    unsigned int score = (bits & 0x7fffffff) > 0x7f800000 ? 0xffffffff :
                         (bits & 0x80000000)              ? ~bits :
                                                            (bits ^ 0x80000000);
    return (static_cast<Candidate>(score) << 32) | static_cast<unsigned int>(~token);
}

template<typename T>
__global__ void localTop1(const T* logits, Candidate* partials, int cols, int parts, int offset, int vocab) {
    const int row  = blockIdx.x;
    const int part = blockIdx.y;
    Candidate best = 0;  // Smaller than every valid value, including -inf.
    for (int col = part * kChunk + threadIdx.x; col < min(cols, (part + 1) * kChunk); col += kThreads) {
        if (offset + col < vocab) {
            const auto value = encode(static_cast<float>(logits[static_cast<int64_t>(row) * cols + col]), offset + col);
            best             = best > value ? best : value;
        }
    }
    best = blockMax(best);
    if (threadIdx.x == 0) {
        partials[static_cast<int64_t>(row) * parts + part] = best;
    }
}

__global__ void reduceLocal(const Candidate* partials, Candidate* candidates, int parts) {
    Candidate best = 0;
    for (int part = threadIdx.x; part < parts; part += kThreads) {
        auto value = partials[static_cast<int64_t>(blockIdx.x) * parts + part];
        best       = best > value ? best : value;
    }
    best = blockMax(best);
    if (threadIdx.x == 0) {
        candidates[blockIdx.x] = best;
    }
}

__global__ void globalTop1(const Candidate* candidates, int64_t* tokens, int rows, int tp_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    Candidate best = 0;
    for (int rank = 0; rank < tp_size; ++rank) {
        auto value = candidates[static_cast<int64_t>(rank) * rows + row];
        best       = best > value ? best : value;
    }
    tokens[row] = static_cast<unsigned int>(~static_cast<unsigned int>(best));
}
}  // namespace

torch::Tensor mtpDraftLocalTop1(const torch::Tensor& logits, int tp_size, int tp_rank, int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2,
                "MTP top1 requires contiguous CUDA logits [rows, local_vocab]");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kBFloat16,
                "MTP top1 requires FP32 or BF16 logits");
    TORCH_CHECK(tp_size > 0 && tp_rank >= 0 && tp_rank < tp_size && logits.size(1) > 0
                    && logits.size(1) <= INT_MAX / tp_size && logits.size(0) <= INT_MAX && vocab_size > 0
                    && vocab_size <= logits.size(1) * tp_size,
                "invalid MTP top1 vocabulary or TP layout");
    const c10::cuda::CUDAGuard guard(logits.device());
    const int                  rows       = logits.size(0);
    const int                  cols       = logits.size(1);
    const int                  parts      = (cols + kChunk - 1) / kChunk;
    auto                       candidates = torch::empty({tp_size, rows}, logits.options().dtype(torch::kInt64));
    if (rows == 0) {
        return candidates;
    }
    auto partials    = torch::empty({rows, parts}, candidates.options());
    auto stream      = c10::cuda::getCurrentCUDAStream();
    auto partial_ptr = reinterpret_cast<Candidate*>(partials.data_ptr<int64_t>());
    auto output_ptr  = reinterpret_cast<Candidate*>(candidates.data_ptr<int64_t>()) + tp_rank * rows;
    if (logits.scalar_type() == torch::kBFloat16) {
        localTop1<<<dim3(rows, parts), kThreads, 0, stream>>>(
            logits.data_ptr<c10::BFloat16>(), partial_ptr, cols, parts, tp_rank * cols, vocab_size);
    } else {
        localTop1<<<dim3(rows, parts), kThreads, 0, stream>>>(
            logits.data_ptr<float>(), partial_ptr, cols, parts, tp_rank * cols, vocab_size);
    }
    reduceLocal<<<rows, kThreads, 0, stream>>>(partial_ptr, output_ptr, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return candidates;
}

torch::Tensor mtpDraftGlobalTop1(const torch::Tensor& candidates) {
    TORCH_CHECK(candidates.is_cuda() && candidates.is_contiguous() && candidates.dim() == 2
                    && candidates.scalar_type() == torch::kInt64 && candidates.size(0) > 0
                    && candidates.size(0) <= INT_MAX && candidates.size(1) <= INT_MAX,
                "MTP top1 requires CUDA int64 candidates [tp, rows]");
    const c10::cuda::CUDAGuard guard(candidates.device());
    const int                  rows   = candidates.size(1);
    auto                       tokens = torch::empty({rows, 1}, candidates.options());
    if (rows > 0) {
        globalTop1<<<(rows + kThreads - 1) / kThreads, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const Candidate*>(candidates.data_ptr<int64_t>()),
            tokens.data_ptr<int64_t>(),
            rows,
            candidates.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return tokens;
}
}  // namespace rtp_llm
