#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace rtp_llm {

namespace {

__global__ void mtpTargetVerifyPrepareKernel(const int32_t* __restrict__ sequence_lengths,
                                             const int32_t* __restrict__ position_bases,
                                             int32_t* __restrict__ input_lengths,
                                             int32_t* __restrict__ prefix_lengths,
                                             int32_t* __restrict__ sequence_lengths_plus_1,
                                             int32_t* __restrict__ lm_output_indexes,
                                             int32_t* __restrict__ position_ids,
                                             int32_t position_factor,
                                             int32_t tokens_per_batch,
                                             int32_t batch_size) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size * tokens_per_batch) {
        return;
    }
    if (idx < batch_size) {
        input_lengths[idx]           = tokens_per_batch;
        prefix_lengths[idx]          = sequence_lengths[idx];
        sequence_lengths_plus_1[idx] = sequence_lengths[idx] + 1;
    }
    lm_output_indexes[idx] = idx;
    const int32_t request  = idx / tokens_per_batch;
    const int32_t step     = idx % tokens_per_batch;
    for (int32_t axis = 0; axis < position_factor; ++axis) {
        position_ids[idx * position_factor + axis] = position_bases[request * position_factor + axis] + step;
    }
}

__global__ void mtpSpecDecodeMetadataPrepareKernel(int32_t* __restrict__ input_lengths,
                                                   int32_t* __restrict__ lm_output_indexes,
                                                   int32_t tokens_per_batch,
                                                   int32_t batch_size) {
    const int32_t total_tokens = batch_size * tokens_per_batch;
    const int32_t idx          = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < batch_size) {
        input_lengths[idx] = tokens_per_batch;
    }
    if (idx < total_tokens) {
        lm_output_indexes[idx] = idx;
    }
}

__global__ void mtpSpecDecodeTokensMetadataPrepareKernel(const int32_t* __restrict__ token0,
                                                         const int32_t* __restrict__ token1,
                                                         const int32_t* __restrict__ token2,
                                                         const int32_t* __restrict__ token3,
                                                         const int32_t* __restrict__ token4,
                                                         const int32_t* __restrict__ token5,
                                                         const int32_t* __restrict__ token6,
                                                         const int32_t* __restrict__ token7,
                                                         int32_t* __restrict__ spec_tokens,
                                                         int32_t* __restrict__ input_lengths,
                                                         int32_t* __restrict__ lm_output_indexes,
                                                         int32_t tokens_per_batch,
                                                         int32_t batch_size) {
    const int32_t total_tokens = batch_size * tokens_per_batch;
    const int32_t idx          = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_tokens) {
        return;
    }

    const int32_t  batch_idx = idx / tokens_per_batch;
    const int32_t  token_idx = idx - batch_idx * tokens_per_batch;
    const int32_t* src       = nullptr;
    switch (token_idx) {
        case 0:
            src = token0;
            break;
        case 1:
            src = token1;
            break;
        case 2:
            src = token2;
            break;
        case 3:
            src = token3;
            break;
        case 4:
            src = token4;
            break;
        case 5:
            src = token5;
            break;
        case 6:
            src = token6;
            break;
        case 7:
            src = token7;
            break;
    }

    spec_tokens[idx]       = src[batch_idx];
    if (lm_output_indexes) {
        lm_output_indexes[idx] = idx;
    }
    if (input_lengths && token_idx == 0) {
        input_lengths[batch_idx] = tokens_per_batch;
    }
}

void checkCudaI32Vector(const torch::Tensor& tensor, const char* name, int64_t batch_size) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(
        tensor.numel() >= batch_size, "%s numel %ld is smaller than batch_size %ld", name, tensor.numel(), batch_size);
}

__device__ int64_t swapBlockIndex(int64_t index, int64_t source, int64_t destination) {
    if (index == source) {
        return destination;
    }
    return index == destination ? source : index;
}

__global__ void mtpAdvanceCacheTableKernel(torch::PackedTensorAccessor64<int32_t, 3> current,
                                           torch::PackedTensorAccessor64<int32_t, 3> next,
                                           const int32_t*                            previous_seq_lengths,
                                           const int32_t*                            accept_lengths,
                                           uint64_t                                  linear_groups,
                                           int64_t                                   tokens_per_block) {
    const int64_t group   = blockIdx.x / current.size(1);
    const int64_t request = blockIdx.x % current.size(1);
    const int64_t accept  = accept_lengths[request];
    const int64_t before  = static_cast<int64_t>(previous_seq_lengths[request]) - 1;
    const int64_t after   = before + accept;
    const bool    linear  = ((linear_groups >> group) & 1) != 0 && accept > 1;

    const auto old_row = current[group][request];
    auto       new_row = next[group][request];
    for (int64_t destination = threadIdx.x; destination < current.size(2); destination += blockDim.x) {
        int64_t source = destination;
        if (linear) {
            // Host commit applies the boundary swap and then the final-token
            // swap in place. Walk that permutation backwards for an out-of-place gather.
            source = swapBlockIndex(source, before / tokens_per_block + accept - 1, (after - 1) / tokens_per_block);
            if ((before + 1) % tokens_per_block > (after - 1) % tokens_per_block) {
                const int64_t boundary = before / tokens_per_block + tokens_per_block - before % tokens_per_block - 1;
                source                 = swapBlockIndex(source, boundary, after / tokens_per_block - 1);
            }
        }
        new_row[destination] = old_row[source];
    }
}

}  // namespace

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  const torch::Tensor& position_bases,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
                                  torch::Tensor&       position_ids,
                                  int32_t              tokens_per_batch,
                                  cudaStream_t         stream) {
    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(sequence_lengths, "sequence_lengths", batch_size);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(prefix_lengths, "prefix_lengths", batch_size);
    checkCudaI32Vector(sequence_lengths_plus_1, "sequence_lengths_plus_1", batch_size);
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch > 0, "tokens_per_batch must be positive");
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", batch_size * tokens_per_batch);
    int32_t position_factor = 0;
    if (position_bases.defined()) {
        checkCudaI32Vector(position_bases, "position_bases", batch_size);
        RTP_LLM_CHECK_WITH_INFO(position_bases.numel() % batch_size == 0,
                                "target position bases must contain complete per-request tuples");
        position_factor = position_bases.numel() / batch_size;
        checkCudaI32Vector(position_ids, "position_ids", position_bases.numel() * tokens_per_batch);
    } else {
        RTP_LLM_CHECK_WITH_INFO(!position_ids.defined(), "position output requires position bases");
    }

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size * tokens_per_batch + block_size - 1) / block_size);
    mtpTargetVerifyPrepareKernel<<<grid_size, block_size, 0, stream>>>(
        sequence_lengths.data_ptr<int32_t>(),
        position_bases.defined() ? position_bases.data_ptr<int32_t>() : nullptr,
        input_lengths.data_ptr<int32_t>(),
        prefix_lengths.data_ptr<int32_t>(),
        sequence_lengths_plus_1.data_ptr<int32_t>(),
        lm_output_indexes.data_ptr<int32_t>(),
        position_ids.defined() ? position_ids.data_ptr<int32_t>() : nullptr,
        position_factor,
        tokens_per_batch,
        static_cast<int32_t>(batch_size));
}

void invokeMtpSpecDecodeMetadataPrepare(torch::Tensor& input_lengths,
                                        torch::Tensor& lm_output_indexes,
                                        int32_t        tokens_per_batch,
                                        cudaStream_t   stream) {
    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    const int64_t total_tokens = batch_size * tokens_per_batch;
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", total_tokens);

    constexpr int block_size = 256;
    const int64_t work_items = std::max<int64_t>(batch_size, total_tokens);
    const int     grid_size  = static_cast<int>((work_items + block_size - 1) / block_size);
    mtpSpecDecodeMetadataPrepareKernel<<<grid_size, block_size, 0, stream>>>(input_lengths.data_ptr<int32_t>(),
                                                                             lm_output_indexes.data_ptr<int32_t>(),
                                                                             tokens_per_batch,
                                                                             static_cast<int32_t>(batch_size));
}

void invokeMtpSpecDecodeTokensMetadataPrepare(const std::vector<torch::Tensor>& token_columns,
                                              torch::Tensor&                    spec_tokens,
                                              torch::Tensor&                    input_lengths,
                                              torch::Tensor&                    lm_output_indexes,
                                              int32_t                           tokens_per_batch,
                                              cudaStream_t                      stream) {
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch > 0, "tokens_per_batch must be positive");
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch <= 8, "tokens_per_batch %d exceeds fused kernel max 8", tokens_per_batch);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int32_t>(token_columns.size()) == tokens_per_batch,
                            "token_columns size %ld must equal tokens_per_batch %d",
                            token_columns.size(),
                            tokens_per_batch);

    const int64_t batch_size = token_columns.front().numel();
    if (batch_size <= 0) {
        return;
    }
    const int64_t total_tokens = batch_size * tokens_per_batch;
    checkCudaI32Vector(spec_tokens, "spec_tokens", total_tokens);
    if (input_lengths.defined()) {
        checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    }
    if (lm_output_indexes.defined()) {
        checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", total_tokens);
    }
    for (size_t i = 0; i < token_columns.size(); ++i) {
        checkCudaI32Vector(token_columns[i], "token_columns", batch_size);
    }

    const int32_t* ptrs[8] = {};
    for (size_t i = 0; i < token_columns.size(); ++i) {
        ptrs[i] = token_columns[i].data_ptr<int32_t>();
    }

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((total_tokens + block_size - 1) / block_size);
    mtpSpecDecodeTokensMetadataPrepareKernel<<<grid_size, block_size, 0, stream>>>(
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        spec_tokens.data_ptr<int32_t>(),
        input_lengths.defined() ? input_lengths.data_ptr<int32_t>() : nullptr,
        lm_output_indexes.defined() ? lm_output_indexes.data_ptr<int32_t>() : nullptr,
        tokens_per_batch,
        static_cast<int32_t>(batch_size));
}

// Fused kernel: next_seq_len[i] = prev_seq_len[i] + accept_len[i]
//               hidden_idx[i]  = (int64_t)(accept_len[i] - 1)
__global__ void mtpDispatchStatePrepareKernel(const int32_t* __restrict__ accept_len,
                                              const int32_t* __restrict__ prev_seq_len,
                                              int32_t* __restrict__ next_seq_len,
                                              int64_t* __restrict__ hidden_idx,
                                              int32_t batch_size) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size) {
        return;
    }
    const int32_t al   = accept_len[idx];
    next_seq_len[idx]  = prev_seq_len[idx] + al;
    hidden_idx[idx]    = static_cast<int64_t>(al - 1);
}

void invokeMtpDispatchStatePrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& prev_seq_len,
                                   torch::Tensor&       next_seq_len,
                                   torch::Tensor&       hidden_idx,
                                   int64_t              batch_size,
                                   cudaStream_t         stream) {
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(accept_len, "accept_len", batch_size);
    checkCudaI32Vector(prev_seq_len, "prev_seq_len", batch_size);
    checkCudaI32Vector(next_seq_len, "next_seq_len", batch_size);
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.defined() && hidden_idx.is_cuda(), "hidden_idx must be CUDA");
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.scalar_type() == torch::kInt64, "hidden_idx must be int64");
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.is_contiguous(), "hidden_idx must be contiguous");
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.numel() >= batch_size,
                            "hidden_idx numel %ld < batch_size %ld",
                            hidden_idx.numel(),
                            batch_size);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpDispatchStatePrepareKernel<<<grid_size, block_size, 0, stream>>>(accept_len.data_ptr<int32_t>(),
                                                                        prev_seq_len.data_ptr<int32_t>(),
                                                                        next_seq_len.data_ptr<int32_t>(),
                                                                        hidden_idx.data_ptr<int64_t>(),
                                                                        static_cast<int32_t>(batch_size));
}

void invokeMtpAdvanceCacheTable(const torch::Tensor& current,
                                const torch::Tensor& previous_seq_lengths,
                                const torch::Tensor& accept_lengths,
                                torch::Tensor&       next,
                                uint64_t             linear_groups,
                                int64_t              tokens_per_block,
                                cudaStream_t         stream) {
    constexpr int threads = 128;
    const int64_t rows    = current.size(0) * current.size(1);
    mtpAdvanceCacheTableKernel<<<rows, threads, 0, stream>>>(current.packed_accessor64<int32_t, 3>(),
                                                             next.packed_accessor64<int32_t, 3>(),
                                                             previous_seq_lengths.data_ptr<int32_t>(),
                                                             accept_lengths.data_ptr<int32_t>(),
                                                             linear_groups,
                                                             tokens_per_block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
