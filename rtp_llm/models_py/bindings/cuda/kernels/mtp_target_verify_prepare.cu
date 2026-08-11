#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <climits>

namespace rtp_llm {

namespace {

__global__ void mtpTargetVerifyPrepareKernel(const int32_t* __restrict__ sequence_lengths,
                                             int32_t* __restrict__ input_lengths,
                                             int32_t* __restrict__ prefix_lengths,
                                             int32_t* __restrict__ sequence_lengths_plus_1,
                                             int32_t* __restrict__ lm_output_indexes,
                                             int32_t tokens_per_batch,
                                             int32_t batch_size) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size) {
        return;
    }
    input_lengths[idx]           = tokens_per_batch;
    prefix_lengths[idx]          = sequence_lengths[idx];
    sequence_lengths_plus_1[idx] = sequence_lengths[idx] + 1;
    lm_output_indexes[idx]       = idx * tokens_per_batch;
}

__global__ void mtpMsaTargetVerifyAddressingPrepareKernel(const int32_t* __restrict__ request_block_table,
                                                          const int32_t* __restrict__ prefix_lengths,
                                                          const int32_t* __restrict__ input_lengths,
                                                          int32_t* __restrict__ physical_block_table,
                                                          int32_t* __restrict__ positions,
                                                          int32_t* __restrict__ sequence_lengths,
                                                          bool* __restrict__ valid_token_mask,
                                                          int64_t request_block_stride,
                                                          int32_t tokens_per_batch,
                                                          int32_t max_blocks,
                                                          int32_t total_tokens) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t work_items = static_cast<int64_t>(total_tokens) * max_blocks;
    if (linear_idx >= work_items) {
        return;
    }

    const int32_t token_row   = static_cast<int32_t>(linear_idx / max_blocks);
    const int32_t block_idx   = static_cast<int32_t>(linear_idx - static_cast<int64_t>(token_row) * max_blocks);
    const int32_t request_idx = token_row / tokens_per_batch;
    const int32_t token_idx   = token_row - request_idx * tokens_per_batch;
    physical_block_table[linear_idx] =
        request_block_table[static_cast<int64_t>(request_idx) * request_block_stride + block_idx];

    if (block_idx == 0) {
        const int32_t position      = prefix_lengths[request_idx] + token_idx;
        const bool    valid         = input_lengths[request_idx] > 0;
        positions[token_row]        = position;
        sequence_lengths[token_row] = valid ? position + 1 : 0;
        valid_token_mask[token_row] = valid;
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
    lm_output_indexes[idx] = idx;
    if (token_idx == 0) {
        input_lengths[batch_idx] = tokens_per_batch;
    }
}

__global__ void mtpPrefillShiftAppendKernel(const int32_t* __restrict__ combo_tokens_in,
                                            const int32_t* __restrict__ input_lengths,
                                            const int32_t* __restrict__ batch_offsets,
                                            const int32_t* __restrict__ new_all_token_ids,
                                            int32_t* __restrict__ combo_tokens_out,
                                            int32_t token_stride,
                                            int32_t batch_size,
                                            int32_t total_tokens) {
    const int32_t global_idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= total_tokens) {
        return;
    }
    // Binary search for the batch this token belongs to. batch_offsets[b] holds
    // the exclusive end offset for batch b (i.e. cumulative input_lengths up to b+1).
    int32_t lo = 0;
    int32_t hi = batch_size - 1;
    while (lo < hi) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (batch_offsets[mid] <= global_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int32_t batch_idx         = lo;
    const int32_t batch_start       = (batch_idx == 0) ? 0 : batch_offsets[batch_idx - 1];
    const int32_t position_in_batch = global_idx - batch_start;
    const int32_t input_length      = input_lengths[batch_idx];

    if (position_in_batch == input_length - 1) {
        // Last position: write the new accepted token (last column of new_all_token_ids).
        combo_tokens_out[global_idx] = new_all_token_ids[batch_idx * token_stride + token_stride - 1];
    } else if (position_in_batch < input_length - 1) {
        // Shift left by 1: out[i] = in[i+1] within the batch.
        combo_tokens_out[global_idx] = combo_tokens_in[global_idx + 1];
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

}  // namespace

std::vector<torch::Tensor> mtpMsaTargetVerifyAddressingPrepare(const torch::Tensor& request_block_table,
                                                               const torch::Tensor& prefix_lengths,
                                                               const torch::Tensor& input_lengths,
                                                               int64_t              tokens_per_batch) {
    RTP_LLM_CHECK_WITH_INFO(request_block_table.defined() && request_block_table.is_cuda(),
                            "request_block_table must be CUDA");
    RTP_LLM_CHECK_WITH_INFO(request_block_table.scalar_type() == torch::kInt32, "request_block_table must be int32");
    RTP_LLM_CHECK_WITH_INFO(request_block_table.dim() == 2, "request_block_table must be rank 2");
    RTP_LLM_CHECK_WITH_INFO(request_block_table.stride(1) == 1,
                            "request_block_table inner dimension must be contiguous");
    RTP_LLM_CHECK_WITH_INFO(
        tokens_per_batch > 0 && tokens_per_batch <= 8, "tokens_per_batch must be in [1, 8], got %ld", tokens_per_batch);

    const int64_t batch_size = request_block_table.size(0);
    const int64_t max_blocks = request_block_table.size(1);
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0, "request_block_table batch must be positive");
    RTP_LLM_CHECK_WITH_INFO(max_blocks > 0, "request_block_table width must be positive");
    checkCudaI32Vector(prefix_lengths, "prefix_lengths", batch_size);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    RTP_LLM_CHECK_WITH_INFO(prefix_lengths.numel() == batch_size,
                            "prefix_lengths numel %ld must equal batch_size %ld",
                            prefix_lengths.numel(),
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(input_lengths.numel() == batch_size,
                            "input_lengths numel %ld must equal batch_size %ld",
                            input_lengths.numel(),
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(prefix_lengths.get_device() == request_block_table.get_device()
                                && input_lengths.get_device() == request_block_table.get_device(),
                            "all addressing tensors must be on the same CUDA device");
    RTP_LLM_CHECK_WITH_INFO(batch_size <= INT32_MAX && max_blocks <= INT32_MAX
                                && batch_size * tokens_per_batch <= INT32_MAX,
                            "MTP MSA addressing shape exceeds int32 launch bounds");

    const int64_t total_tokens = batch_size * tokens_per_batch;
    const int64_t work_items   = total_tokens * max_blocks;
    constexpr int block_size   = 256;
    RTP_LLM_CHECK_WITH_INFO((work_items + block_size - 1) / block_size <= INT32_MAX,
                            "MTP MSA addressing launch grid exceeds int32 bounds");
    const c10::cuda::CUDAGuard device_guard(request_block_table.device());
    auto physical_block_table = torch::empty({total_tokens, max_blocks}, request_block_table.options());
    auto positions            = torch::empty({total_tokens}, prefix_lengths.options());
    auto sequence_lengths     = torch::empty({total_tokens}, prefix_lengths.options());
    auto valid_token_mask     = torch::empty({total_tokens}, prefix_lengths.options().dtype(torch::kBool));

    const int          grid_size = static_cast<int>((work_items + block_size - 1) / block_size);
    const cudaStream_t stream    = at::cuda::getCurrentCUDAStream(request_block_table.get_device()).stream();
    mtpMsaTargetVerifyAddressingPrepareKernel<<<grid_size, block_size, 0, stream>>>(
        request_block_table.data_ptr<int32_t>(),
        prefix_lengths.data_ptr<int32_t>(),
        input_lengths.data_ptr<int32_t>(),
        physical_block_table.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        sequence_lengths.data_ptr<int32_t>(),
        valid_token_mask.data_ptr<bool>(),
        request_block_table.stride(0),
        static_cast<int32_t>(tokens_per_batch),
        static_cast<int32_t>(max_blocks),
        static_cast<int32_t>(total_tokens));
    const auto launch_error = cudaGetLastError();
    TORCH_CHECK(launch_error == cudaSuccess,
                "MTP MSA target-verify addressing kernel launch failed: ",
                cudaGetErrorString(launch_error));
    return {physical_block_table, positions, sequence_lengths, valid_token_mask};
}

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
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
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", batch_size);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpTargetVerifyPrepareKernel<<<grid_size, block_size, 0, stream>>>(sequence_lengths.data_ptr<int32_t>(),
                                                                       input_lengths.data_ptr<int32_t>(),
                                                                       prefix_lengths.data_ptr<int32_t>(),
                                                                       sequence_lengths_plus_1.data_ptr<int32_t>(),
                                                                       lm_output_indexes.data_ptr<int32_t>(),
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

    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    const int64_t total_tokens = batch_size * tokens_per_batch;
    checkCudaI32Vector(spec_tokens, "spec_tokens", total_tokens);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", total_tokens);
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
        input_lengths.data_ptr<int32_t>(),
        lm_output_indexes.data_ptr<int32_t>(),
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
    const int32_t al  = accept_len[idx];
    next_seq_len[idx] = prev_seq_len[idx] + al;
    hidden_idx[idx]   = static_cast<int64_t>(al - 1);
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
    RTP_LLM_CHECK_WITH_INFO(
        hidden_idx.numel() >= batch_size, "hidden_idx numel %ld < batch_size %ld", hidden_idx.numel(), batch_size);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpDispatchStatePrepareKernel<<<grid_size, block_size, 0, stream>>>(accept_len.data_ptr<int32_t>(),
                                                                        prev_seq_len.data_ptr<int32_t>(),
                                                                        next_seq_len.data_ptr<int32_t>(),
                                                                        hidden_idx.data_ptr<int64_t>(),
                                                                        static_cast<int32_t>(batch_size));
}

// REBASE CONFLICT CONTEXT(518707c73): source branch added this fused
// shift/append launcher to eliminate sync-heavy CPU token manipulation. Keep it
// with the new base dispatch-state prepare launcher above.
void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream) {
    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    const int64_t total_tokens = combo_tokens_in.numel();
    if (total_tokens <= 0) {
        return;
    }
    checkCudaI32Vector(combo_tokens_in, "combo_tokens_in", total_tokens);
    checkCudaI32Vector(combo_tokens_out, "combo_tokens_out", total_tokens);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(batch_offsets, "batch_offsets", batch_size);
    RTP_LLM_CHECK_WITH_INFO(new_all_token_ids.defined() && new_all_token_ids.is_cuda(),
                            "new_all_token_ids must be CUDA");
    RTP_LLM_CHECK_WITH_INFO(new_all_token_ids.scalar_type() == torch::kInt32,
                            "new_all_token_ids must be int32 (got %s)",
                            c10::toString(new_all_token_ids.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(new_all_token_ids.is_contiguous(), "new_all_token_ids must be contiguous");
    RTP_LLM_CHECK_WITH_INFO(new_all_token_ids.numel() >= batch_size * token_stride,
                            "new_all_token_ids numel %ld < batch_size %ld * token_stride %d",
                            new_all_token_ids.numel(),
                            batch_size,
                            token_stride);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((total_tokens + block_size - 1) / block_size);
    mtpPrefillShiftAppendKernel<<<grid_size, block_size, 0, stream>>>(combo_tokens_in.data_ptr<int32_t>(),
                                                                      input_lengths.data_ptr<int32_t>(),
                                                                      batch_offsets.data_ptr<int32_t>(),
                                                                      new_all_token_ids.data_ptr<int32_t>(),
                                                                      combo_tokens_out.data_ptr<int32_t>(),
                                                                      token_stride,
                                                                      static_cast<int32_t>(batch_size),
                                                                      static_cast<int32_t>(total_tokens));
}

}  // namespace rtp_llm
