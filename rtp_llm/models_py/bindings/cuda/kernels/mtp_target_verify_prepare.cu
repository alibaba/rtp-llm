#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

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

void checkCudaI32Vector(const torch::Tensor& tensor, const char* name, int64_t batch_size) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(
        tensor.numel() >= batch_size, "%s numel %ld is smaller than batch_size %ld", name, tensor.numel(), batch_size);
}

}  // namespace

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

namespace {

constexpr int32_t kLinearPatchWidth = static_cast<int32_t>(MTP_LINEAR_BLOCK_PATCH_WIDTH);

__device__ __forceinline__ int32_t findPatchPosition(const int32_t* positions, int32_t count, int32_t target) {
    for (int32_t i = 0; i < count; ++i) {
        if (positions[i] == target) {
            return i;
        }
    }
    return -1;
}

__device__ __forceinline__ void addPatchPosition(int32_t* positions, int32_t& count, int32_t position) {
    if (position < 0 || findPatchPosition(positions, count, position) >= 0) {
        return;
    }
    if (count < kLinearPatchWidth) {
        positions[count++] = position;
    }
}

__device__ __forceinline__ void
swapPatchValues(int32_t* values, const int32_t* positions, int32_t count, int32_t src, int32_t dst) {
    if (src == dst) {
        return;
    }
    const int32_t src_slot = findPatchPosition(positions, count, src);
    const int32_t dst_slot = findPatchPosition(positions, count, dst);
    if (src_slot < 0 || dst_slot < 0) {
        return;
    }
    const int32_t tmp = values[src_slot];
    values[src_slot]  = values[dst_slot];
    values[dst_slot]  = tmp;
}

__global__ void mtpLinearKvCacheBlockPatchBuildKernel(const int32_t* __restrict__ block_ids,
                                                      const int32_t* __restrict__ group_types,
                                                      const int32_t* __restrict__ valid_block_counts,
                                                      const int32_t* __restrict__ prev_seq_len,
                                                      const int32_t* __restrict__ accept_len,
                                                      int32_t* __restrict__ positions,
                                                      int32_t* __restrict__ source_slots,
                                                      int32_t* __restrict__ before_values,
                                                      int32_t* __restrict__ after_values,
                                                      int32_t* __restrict__ patch_valid,
                                                      int32_t seq_size_per_block,
                                                      int32_t group_num,
                                                      int32_t batch_size,
                                                      int32_t row_width) {
    const int32_t batch_id = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (batch_id >= batch_size) {
        return;
    }

    int32_t patch_positions[kLinearPatchWidth] = {-1, -1, -1, -1};
    int32_t patch_count                        = 0;
    bool    has_cached_swap                    = false;
    int32_t cached_src                         = 0;
    int32_t cached_dst                         = 0;
    int32_t final_src                          = 0;
    int32_t final_dst                          = 0;

    const int32_t accepted       = accept_len[batch_id];
    const int32_t cur_cached_len = prev_seq_len[batch_id] - 1;
    if (accepted > 1 && cur_cached_len >= 0) {
        const int32_t nxt_cached_len = cur_cached_len + accepted;
        has_cached_swap =
            (cur_cached_len + 1) % seq_size_per_block > (nxt_cached_len + seq_size_per_block - 1) % seq_size_per_block;
        const int32_t base_block_idx = cur_cached_len / seq_size_per_block;
        cached_src                   = base_block_idx + seq_size_per_block - cur_cached_len % seq_size_per_block - 1;
        cached_dst                   = nxt_cached_len / seq_size_per_block - 1;
        final_src                    = base_block_idx + accepted - 1;
        final_dst                    = (nxt_cached_len - 1) / seq_size_per_block;

        if (has_cached_swap && cached_src != cached_dst) {
            addPatchPosition(patch_positions, patch_count, cached_src);
            addPatchPosition(patch_positions, patch_count, cached_dst);
        }
        if (final_src != final_dst) {
            addPatchPosition(patch_positions, patch_count, final_src);
            addPatchPosition(patch_positions, patch_count, final_dst);
        }
    }

    const int32_t position_offset                       = batch_id * kLinearPatchWidth;
    int32_t       patch_source_slots[kLinearPatchWidth] = {-1, -1, -1, -1};
    for (int32_t slot = 0; slot < patch_count; ++slot) {
        patch_source_slots[slot] = slot;
    }
    if (has_cached_swap) {
        swapPatchValues(patch_source_slots, patch_positions, patch_count, cached_src, cached_dst);
    }
    swapPatchValues(patch_source_slots, patch_positions, patch_count, final_src, final_dst);
    for (int32_t slot = 0; slot < kLinearPatchWidth; ++slot) {
        positions[position_offset + slot]    = patch_positions[slot];
        source_slots[position_offset + slot] = patch_source_slots[slot];
    }

    for (int32_t group_id = 0; group_id < group_num; ++group_id) {
        const int32_t value_offset                   = (batch_id * group_num + group_id) * kLinearPatchWidth;
        patch_valid[batch_id * group_num + group_id] = 0;
        for (int32_t slot = 0; slot < kLinearPatchWidth; ++slot) {
            before_values[value_offset + slot] = -1;
            after_values[value_offset + slot]  = -1;
        }
        if (patch_count == 0 || group_types[group_id] != static_cast<int32_t>(CacheGroupType::LINEAR)) {
            continue;
        }

        const int32_t valid_block_count = valid_block_counts[group_id * batch_size + batch_id];
        bool          indices_valid     = valid_block_count > 0;
        for (int32_t slot = 0; slot < patch_count; ++slot) {
            indices_valid &= patch_positions[slot] >= 0 && patch_positions[slot] < valid_block_count
                             && patch_positions[slot] < row_width;
        }
        if (!indices_valid) {
            continue;
        }

        const int32_t* row                       = block_ids + (group_id * batch_size + batch_id) * row_width;
        int32_t        values[kLinearPatchWidth] = {-1, -1, -1, -1};
        for (int32_t slot = 0; slot < patch_count; ++slot) {
            values[slot] = row[patch_positions[slot]];
        }

        for (int32_t slot = 0; slot < patch_count; ++slot) {
            before_values[value_offset + slot] = values[slot];
        }
        for (int32_t slot = 0; slot < patch_count; ++slot) {
            after_values[value_offset + slot] = values[patch_source_slots[slot]];
        }
        patch_valid[batch_id * group_num + group_id] = 1;
    }
}

__global__ void mtpLinearKvCacheBlockPatchApplyKernel(int32_t* __restrict__ block_ids,
                                                      const int32_t* __restrict__ group_types,
                                                      const int32_t* __restrict__ valid_block_counts,
                                                      const int32_t* __restrict__ positions,
                                                      const int32_t* __restrict__ source_slots,
                                                      const int32_t* __restrict__ before_values,
                                                      const int32_t* __restrict__ after_values,
                                                      const int32_t* __restrict__ patch_valid,
                                                      const int32_t* __restrict__ pending_patches,
                                                      int32_t group_num,
                                                      int32_t batch_size,
                                                      int32_t row_width) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= group_num * batch_size) {
        return;
    }

    const int32_t group_id = idx / batch_size;
    const int32_t batch_id = idx - group_id * batch_size;
    if (pending_patches[batch_id] == 0 || group_types[group_id] != static_cast<int32_t>(CacheGroupType::LINEAR)
        || patch_valid[batch_id * group_num + group_id] == 0) {
        return;
    }

    const int32_t* patch_positions    = positions + batch_id * kLinearPatchWidth;
    const int32_t* patch_source_slots = source_slots + batch_id * kLinearPatchWidth;
    int32_t        patch_count        = 0;
    while (patch_count < kLinearPatchWidth && patch_positions[patch_count] >= 0) {
        ++patch_count;
    }
    if (patch_count == 0) {
        return;
    }

    const int32_t valid_block_count = valid_block_counts[group_id * batch_size + batch_id];
    for (int32_t slot = 0; slot < patch_count; ++slot) {
        if (patch_positions[slot] >= valid_block_count || patch_positions[slot] >= row_width) {
            return;
        }
    }

    int32_t*       row                        = block_ids + idx * row_width;
    const int32_t  value_offset               = (batch_id * group_num + group_id) * kLinearPatchWidth;
    const int32_t* before                     = before_values + value_offset;
    const int32_t* after                      = after_values + value_offset;
    int32_t        current[kLinearPatchWidth] = {-1, -1, -1, -1};
    bool           matches_before             = true;
    bool           matches_after              = true;
    for (int32_t slot = 0; slot < patch_count; ++slot) {
        current[slot] = row[patch_positions[slot]];
        matches_before &= current[slot] == before[slot];
        matches_after &= current[slot] == after[slot];
    }
    if (matches_after) {
        return;
    }
    if (matches_before) {
        for (int32_t slot = 0; slot < patch_count; ++slot) {
            row[patch_positions[slot]] = after[slot];
        }
        return;
    }

    // The allocator may append or backfill between rounds. Preserve those new
    // IDs by applying the saved permutation to the fresh tuple rather than
    // blindly restoring after_values from the previous round.
    int32_t permuted[kLinearPatchWidth]    = {-1, -1, -1, -1};
    bool    source_used[kLinearPatchWidth] = {false, false, false, false};
    for (int32_t dst_slot = 0; dst_slot < patch_count; ++dst_slot) {
        const int32_t src_slot = patch_source_slots[dst_slot];
        if (src_slot < 0 || src_slot >= patch_count || source_used[src_slot]) {
            return;
        }
        source_used[src_slot] = true;
        permuted[dst_slot]    = current[src_slot];
    }
    for (int32_t slot = 0; slot < patch_count; ++slot) {
        row[patch_positions[slot]] = permuted[slot];
    }
}

void checkCudaI32Matrix(const torch::Tensor& tensor, const char* name, int64_t rows, int64_t columns) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined() && tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == columns,
                            "%s must have shape [%ld, %ld]",
                            name,
                            rows,
                            columns);
}

void checkCudaI32PatchValues(const torch::Tensor& tensor, const char* name, int64_t batch_size, int64_t group_num) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined() && tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == 3 && tensor.size(0) == batch_size && tensor.size(1) == group_num
                                && tensor.size(2) == kLinearPatchWidth,
                            "%s must have shape [%ld, %ld, %d]",
                            name,
                            batch_size,
                            group_num,
                            kLinearPatchWidth);
}

void checkLinearBlockTable(const torch::Tensor& block_ids,
                           const torch::Tensor& group_types,
                           const torch::Tensor& valid_block_counts,
                           int64_t&             group_num,
                           int64_t&             batch_size,
                           int64_t&             row_width) {
    RTP_LLM_CHECK_WITH_INFO(block_ids.defined() && block_ids.is_cuda(), "block_ids must be CUDA");
    RTP_LLM_CHECK_WITH_INFO(block_ids.scalar_type() == torch::kInt32, "block_ids must be int32");
    RTP_LLM_CHECK_WITH_INFO(block_ids.is_contiguous(), "block_ids must be contiguous");
    RTP_LLM_CHECK_WITH_INFO(block_ids.dim() == 3, "block_ids must have shape [group, batch, row_width]");
    group_num  = block_ids.size(0);
    batch_size = block_ids.size(1);
    row_width  = block_ids.size(2);
    RTP_LLM_CHECK_WITH_INFO(group_num <= INT32_MAX && batch_size <= INT32_MAX && row_width <= INT32_MAX,
                            "linear block patch tensor dimensions exceed int32");
    checkCudaI32Vector(group_types, "group_types", group_num);
    checkCudaI32Matrix(valid_block_counts, "valid_block_counts", group_num, batch_size);
}

}  // namespace

void invokeMtpLinearKvCacheBlockPatchBuild(const torch::Tensor& block_ids,
                                           const torch::Tensor& group_types,
                                           const torch::Tensor& valid_block_counts,
                                           const torch::Tensor& prev_seq_len,
                                           const torch::Tensor& accept_len,
                                           torch::Tensor&       positions,
                                           torch::Tensor&       source_slots,
                                           torch::Tensor&       before_values,
                                           torch::Tensor&       after_values,
                                           torch::Tensor&       patch_valid,
                                           int32_t              seq_size_per_block,
                                           cudaStream_t         stream) {
    int64_t group_num  = 0;
    int64_t batch_size = 0;
    int64_t row_width  = 0;
    checkLinearBlockTable(block_ids, group_types, valid_block_counts, group_num, batch_size, row_width);
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be positive");
    checkCudaI32Vector(prev_seq_len, "prev_seq_len", batch_size);
    checkCudaI32Vector(accept_len, "accept_len", batch_size);
    checkCudaI32Matrix(positions, "positions", batch_size, kLinearPatchWidth);
    checkCudaI32Matrix(source_slots, "source_slots", batch_size, kLinearPatchWidth);
    checkCudaI32PatchValues(before_values, "before_values", batch_size, group_num);
    checkCudaI32PatchValues(after_values, "after_values", batch_size, group_num);
    checkCudaI32Matrix(patch_valid, "patch_valid", batch_size, group_num);
    if (group_num == 0 || batch_size == 0 || row_width == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpLinearKvCacheBlockPatchBuildKernel<<<grid_size, block_size, 0, stream>>>(block_ids.data_ptr<int32_t>(),
                                                                                group_types.data_ptr<int32_t>(),
                                                                                valid_block_counts.data_ptr<int32_t>(),
                                                                                prev_seq_len.data_ptr<int32_t>(),
                                                                                accept_len.data_ptr<int32_t>(),
                                                                                positions.data_ptr<int32_t>(),
                                                                                source_slots.data_ptr<int32_t>(),
                                                                                before_values.data_ptr<int32_t>(),
                                                                                after_values.data_ptr<int32_t>(),
                                                                                patch_valid.data_ptr<int32_t>(),
                                                                                seq_size_per_block,
                                                                                static_cast<int32_t>(group_num),
                                                                                static_cast<int32_t>(batch_size),
                                                                                static_cast<int32_t>(row_width));
}

void invokeMtpLinearKvCacheBlockPatchApply(torch::Tensor&       block_ids,
                                           const torch::Tensor& group_types,
                                           const torch::Tensor& valid_block_counts,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& source_slots,
                                           const torch::Tensor& before_values,
                                           const torch::Tensor& after_values,
                                           const torch::Tensor& patch_valid,
                                           const torch::Tensor& pending_patches,
                                           cudaStream_t         stream) {
    int64_t group_num  = 0;
    int64_t batch_size = 0;
    int64_t row_width  = 0;
    checkLinearBlockTable(block_ids, group_types, valid_block_counts, group_num, batch_size, row_width);
    checkCudaI32Matrix(positions, "positions", batch_size, kLinearPatchWidth);
    checkCudaI32Matrix(source_slots, "source_slots", batch_size, kLinearPatchWidth);
    checkCudaI32PatchValues(before_values, "before_values", batch_size, group_num);
    checkCudaI32PatchValues(after_values, "after_values", batch_size, group_num);
    checkCudaI32Matrix(patch_valid, "patch_valid", batch_size, group_num);
    checkCudaI32Vector(pending_patches, "pending_patches", batch_size);
    if (group_num == 0 || batch_size == 0 || row_width == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int64_t work_items = group_num * batch_size;
    const int     grid_size  = static_cast<int>((work_items + block_size - 1) / block_size);
    mtpLinearKvCacheBlockPatchApplyKernel<<<grid_size, block_size, 0, stream>>>(block_ids.data_ptr<int32_t>(),
                                                                                group_types.data_ptr<int32_t>(),
                                                                                valid_block_counts.data_ptr<int32_t>(),
                                                                                positions.data_ptr<int32_t>(),
                                                                                source_slots.data_ptr<int32_t>(),
                                                                                before_values.data_ptr<int32_t>(),
                                                                                after_values.data_ptr<int32_t>(),
                                                                                patch_valid.data_ptr<int32_t>(),
                                                                                pending_patches.data_ptr<int32_t>(),
                                                                                static_cast<int32_t>(group_num),
                                                                                static_cast<int32_t>(batch_size),
                                                                                static_cast<int32_t>(row_width));
}

}  // namespace rtp_llm
