#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_greedy_sampling.h"

#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <climits>
#include <cmath>

namespace rtp_llm {
namespace {

constexpr int kThreads     = 256;
constexpr int kReduceChunk = 8192;
constexpr int kOutputChunk = 4096;

struct ArgMaxPair {
    float value;
    int   index;
};
static_assert(sizeof(ArgMaxPair) == sizeof(int64_t), "partial storage must hold one argmax pair");

// Match ATen argmax: a NaN wins over non-NaNs; NaN/score ties select
// the first logical vocabulary index, regardless of AllGather rank layout.
struct FirstArgMax {
    __device__ ArgMaxPair operator()(const ArgMaxPair& a, const ArgMaxPair& b) const {
        const bool a_nan = isnan(a.value);
        const bool b_nan = isnan(b.value);
        if (a_nan != b_nan) {
            return a_nan ? a : b;
        }
        if (!a_nan && a.value != b.value) {
            return a.value > b.value ? a : b;
        }
        return a.index < b.index ? a : b;
    }
};

__global__ void mtpGreedyRankMajorPartialsKernel(
    const float* logits, ArgMaxPair* partials, int batch_size, int local_vocab, int vocab_size, int num_partials) {
    const int  row   = blockIdx.x;
    const int  chunk = blockIdx.y;
    const int  end   = min(vocab_size, (chunk + 1) * kReduceChunk);
    ArgMaxPair best{-CUDART_INF_F, INT_MAX};
    for (int index = chunk * kReduceChunk + threadIdx.x; index < end; index += blockDim.x) {
        const int     rank        = index / local_vocab;
        const int     local_index = index - rank * local_vocab;
        const int64_t offset      = (static_cast<int64_t>(rank) * batch_size + row) * local_vocab + local_index;
        best                      = FirstArgMax{}(best, {logits[offset], index});
    }
    using Reduce = cub::BlockReduce<ArgMaxPair, kThreads>;
    __shared__ typename Reduce::TempStorage storage;
    best = Reduce(storage).Reduce(best, FirstArgMax{});
    if (threadIdx.x == 0) {
        partials[static_cast<int64_t>(row) * num_partials + chunk] = best;
    }
}

// Each output tile reduces the small partial array independently. Repeating
// this reduction avoids a third launch and lets the dense one-hot stores use
// enough CTAs even for small batches. The extra partial reads are small next
// to the required [B,V] output, and no [B,V] logits transpose is materialized.
__global__ void mtpGreedyRankMajorFinishKernel(const ArgMaxPair* partials,
                                               const int64_t*    d2t_map,
                                               int               map_size,
                                               int32_t*          tokens,
                                               float*            probabilities,
                                               const int32_t*    sequence_lengths,
                                               int32_t*          next_sequence_lengths,
                                               const int32_t*    position_ids,
                                               int32_t*          next_position_ids,
                                               int               position_factor,
                                               int               vocab_size,
                                               int               num_partials) {
    const int  row  = blockIdx.x;
    const int  tile = blockIdx.y;
    ArgMaxPair best{-CUDART_INF_F, INT_MAX};
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        best = FirstArgMax{}(best, partials[static_cast<int64_t>(row) * num_partials + i]);
    }
    using Reduce = cub::BlockReduce<ArgMaxPair, kThreads>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ int                          selected;
    best = Reduce(storage).Reduce(best, FirstArgMax{});
    if (threadIdx.x == 0) {
        selected = best.index;
        if (tile == 0) {
            const int64_t token = d2t_map != nullptr && best.index < map_size ? d2t_map[best.index] : best.index;
            tokens[row]         = static_cast<int32_t>(token);
            if (sequence_lengths != nullptr) {
                next_sequence_lengths[row] = static_cast<int32_t>(static_cast<uint32_t>(sequence_lengths[row]) + 1U);
            }
        }
    }
    __syncthreads();
    const int end = min(vocab_size, (tile + 1) * kOutputChunk);
    for (int index = tile * kOutputChunk + threadIdx.x; index < end; index += blockDim.x) {
        probabilities[static_cast<int64_t>(row) * vocab_size + index] = index == selected ? 1.0f : 0.0f;
    }
    if (tile == 0 && position_ids != nullptr) {
        for (int i = threadIdx.x; i < position_factor; i += blockDim.x) {
            const int64_t index      = static_cast<int64_t>(row) * position_factor + i;
            next_position_ids[index] = static_cast<int32_t>(static_cast<uint32_t>(position_ids[index]) + 1U);
        }
    }
}

}  // namespace

bool tryInvokeMtpGreedyRankMajor(const torch::Tensor& logits,
                                 const torch::Tensor& d2t_map,
                                 int64_t              vocab_size,
                                 const torch::Tensor& sequence_lengths,
                                 const torch::Tensor& position_ids,
                                 torch::Tensor&       token_ids,
                                 torch::Tensor&       probabilities,
                                 torch::Tensor&       next_sequence_lengths,
                                 torch::Tensor&       next_position_ids,
                                 cudaStream_t         stream) {
    if (!logits.defined() || !logits.is_cuda() || logits.scalar_type() != torch::kFloat32 || !logits.is_contiguous()
        || logits.dim() != 3) {
        return false;
    }
    const int64_t tp_size = logits.size(0), batch_size = logits.size(1), local_vocab = logits.size(2);
    if (batch_size <= 0 || batch_size > INT_MAX || tp_size <= 0 || tp_size > INT_MAX || local_vocab <= 0
        || local_vocab > INT_MAX || vocab_size <= 0 || vocab_size > tp_size * local_vocab
        || vocab_size > static_cast<int64_t>(kOutputChunk) * 65535) {
        return false;
    }
    const auto device = logits.device();
    if (d2t_map.defined() && d2t_map.numel() > 0
        && (d2t_map.device() != device || d2t_map.scalar_type() != torch::kInt64 || !d2t_map.is_contiguous()
            || d2t_map.dim() != 1 || d2t_map.numel() > INT_MAX)) {
        return false;
    }
    const auto valid_metadata = [device](const torch::Tensor& tensor) {
        return tensor.device() == device && tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous();
    };
    if (sequence_lengths.defined() && (!valid_metadata(sequence_lengths) || sequence_lengths.numel() != batch_size)) {
        return false;
    }
    if (position_ids.defined()
        && (!valid_metadata(position_ids) || position_ids.numel() == 0 || position_ids.numel() % batch_size != 0
            || position_ids.numel() / batch_size > INT_MAX)) {
        return false;
    }

    const auto num_partials = (vocab_size + kReduceChunk - 1) / kReduceChunk;
    auto       partials     = torch::empty({batch_size, num_partials}, logits.options().dtype(torch::kInt64));
    token_ids               = torch::empty({batch_size, 1}, logits.options().dtype(torch::kInt32));
    probabilities           = torch::empty({batch_size, vocab_size}, logits.options());
    next_sequence_lengths   = sequence_lengths.defined() ? torch::empty_like(sequence_lengths) : torch::Tensor();
    next_position_ids       = position_ids.defined() ? torch::empty_like(position_ids) : torch::Tensor();
    mtpGreedyRankMajorPartialsKernel<<<dim3(batch_size, num_partials), kThreads, 0, stream>>>(
        logits.data_ptr<float>(),
        reinterpret_cast<ArgMaxPair*>(partials.data_ptr<int64_t>()),
        static_cast<int>(batch_size),
        static_cast<int>(local_vocab),
        static_cast<int>(vocab_size),
        static_cast<int>(num_partials));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const bool has_map = d2t_map.defined() && d2t_map.numel() > 0;
    mtpGreedyRankMajorFinishKernel<<<dim3(batch_size, (vocab_size + kOutputChunk - 1) / kOutputChunk),
                                     kThreads,
                                     0,
                                     stream>>>(
        reinterpret_cast<const ArgMaxPair*>(partials.data_ptr<int64_t>()),
        has_map ? d2t_map.data_ptr<int64_t>() : nullptr,
        has_map ? static_cast<int>(d2t_map.numel()) : 0,
        token_ids.data_ptr<int32_t>(),
        probabilities.data_ptr<float>(),
        sequence_lengths.defined() ? sequence_lengths.data_ptr<int32_t>() : nullptr,
        sequence_lengths.defined() ? next_sequence_lengths.data_ptr<int32_t>() : nullptr,
        position_ids.defined() ? position_ids.data_ptr<int32_t>() : nullptr,
        position_ids.defined() ? next_position_ids.data_ptr<int32_t>() : nullptr,
        position_ids.defined() ? static_cast<int>(position_ids.numel() / batch_size) : 0,
        static_cast<int>(vocab_size),
        static_cast<int>(num_partials));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

}  // namespace rtp_llm
