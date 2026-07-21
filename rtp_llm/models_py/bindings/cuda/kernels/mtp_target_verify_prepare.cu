#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>
#include <limits>
#include <math_constants.h>

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

template<typename scalar_t>
__global__ void mtpRowLogSoftmaxStatsKernel(const scalar_t* __restrict__ logits,
                                            float* __restrict__ row_max_output,
                                            float* __restrict__ row_shifted_logsumexp,
                                            int64_t row_stride,
                                            int64_t real_vocab_size) {
    const int64_t row        = static_cast<int64_t>(blockIdx.x);
    const auto*   row_logits = logits + row * row_stride;

    float thread_max = -CUDART_INF_F;
    for (int64_t col = threadIdx.x; col < real_vocab_size; col += blockDim.x) {
        thread_max = fmaxf(thread_max, static_cast<float>(row_logits[col]));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float row_max = reduction[0];
    if (row_max == CUDART_INF_F || row_max == -CUDART_INF_F) {
        if (threadIdx.x == 0) {
            row_max_output[row]        = row_max;
            row_shifted_logsumexp[row] = 0.0f;
        }
        return;
    }

    float thread_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < real_vocab_size; col += blockDim.x) {
        thread_sum += expf(static_cast<float>(row_logits[col]) - row_max);
    }
    reduction[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_max_output[row]        = row_max;
        row_shifted_logsumexp[row] = logf(reduction[0]);
    }
}

template<typename scalar_t>
void launchMtpRowLogSoftmaxStats(const torch::Tensor& logits,
                                 torch::Tensor&       row_max,
                                 torch::Tensor&       row_shifted_logsumexp,
                                 int64_t              real_vocab_size,
                                 cudaStream_t         stream) {
    constexpr int block_size = 256;
    mtpRowLogSoftmaxStatsKernel<scalar_t>
        <<<static_cast<unsigned int>(logits.size(0)), block_size, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                                                               row_max.data_ptr<float>(),
                                                                               row_shifted_logsumexp.data_ptr<float>(),
                                                                               logits.stride(0),
                                                                               real_vocab_size);
}

void invokeMtpRowLogSoftmaxStats(const torch::Tensor& logits,
                                 torch::Tensor&       row_max,
                                 torch::Tensor&       row_shifted_logsumexp,
                                 int64_t              real_vocab_size,
                                 cudaStream_t         stream) {
    RTP_LLM_CHECK_WITH_INFO(logits.defined() && logits.is_cuda(), "MTP logits must be a CUDA tensor");
    RTP_LLM_CHECK_WITH_INFO(logits.dim() == 2, "MTP logits must be 2-D, got dim=%ld", logits.dim());
    RTP_LLM_CHECK_WITH_INFO(logits.stride(1) == 1, "MTP logits innermost stride must be 1, got %ld", logits.stride(1));
    RTP_LLM_CHECK_WITH_INFO(real_vocab_size > 0 && real_vocab_size <= logits.size(1),
                            "real_vocab_size %ld must be in [1, logits width %ld]",
                            real_vocab_size,
                            logits.size(1));
    auto check_output = [&logits](const torch::Tensor& output, const char* name) {
        RTP_LLM_CHECK_WITH_INFO(output.defined() && output.is_cuda(), "MTP %s must be a CUDA tensor", name);
        RTP_LLM_CHECK_WITH_INFO(output.scalar_type() == torch::kFloat32, "MTP %s must be float32", name);
        RTP_LLM_CHECK_WITH_INFO(output.is_contiguous() && output.numel() == logits.size(0),
                                "MTP %s must be contiguous with one value per logits row",
                                name);
    };
    check_output(row_max, "row_max");
    check_output(row_shifted_logsumexp, "row_shifted_logsumexp");
    if (logits.size(0) == 0) {
        return;
    }

    switch (logits.scalar_type()) {
        case torch::kFloat32:
            launchMtpRowLogSoftmaxStats<float>(logits, row_max, row_shifted_logsumexp, real_vocab_size, stream);
            break;
        case torch::kFloat16:
            launchMtpRowLogSoftmaxStats<at::Half>(logits, row_max, row_shifted_logsumexp, real_vocab_size, stream);
            break;
        case torch::kBFloat16:
            launchMtpRowLogSoftmaxStats<at::BFloat16>(logits, row_max, row_shifted_logsumexp, real_vocab_size, stream);
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "unsupported MTP logits dtype %s", c10::toString(logits.scalar_type()));
    }
}

namespace {

constexpr int kMtpSelectedLogprobsBlockSize = 256;
constexpr int kMtpSelectedLogprobsMaxTopK   = 20;

struct MtpTopCandidate {
    float   value;
    int32_t token_id;
};

__device__ __forceinline__ bool mtpTopCandidateBetter(const MtpTopCandidate& lhs, const MtpTopCandidate& rhs) {
    const bool lhs_nan = isnan(lhs.value);
    const bool rhs_nan = isnan(rhs.value);
    if (lhs_nan != rhs_nan) {
        // Keep NaNs visible and deterministic instead of silently dropping
        // them from Top-K. The row normalizer is also NaN in this case.
        return lhs_nan;
    }
    if (lhs.value > rhs.value) {
        return true;
    }
    if (lhs.value < rhs.value) {
        return false;
    }
    if (lhs.token_id >= 0 && rhs.token_id < 0) {
        return true;
    }
    if (lhs.token_id < 0 && rhs.token_id >= 0) {
        return false;
    }
    return lhs.token_id < rhs.token_id;
}

template<int LOCAL_TOP_K>
__device__ __forceinline__ void
insertMtpLocalTopK(float value, int32_t token_id, float (&values)[LOCAL_TOP_K], int32_t (&token_ids)[LOCAL_TOP_K]) {
    const MtpTopCandidate candidate{value, token_id};
    const MtpTopCandidate tail{values[LOCAL_TOP_K - 1], token_ids[LOCAL_TOP_K - 1]};
    if (!mtpTopCandidateBetter(candidate, tail)) {
        return;
    }

    int  insertion_pos = LOCAL_TOP_K - 1;
    bool keep_moving   = true;
#pragma unroll
    for (int pos = LOCAL_TOP_K - 1; pos > 0; --pos) {
        const MtpTopCandidate previous{values[pos - 1], token_ids[pos - 1]};
        const bool            move_previous = keep_moving && mtpTopCandidateBetter(candidate, previous);
        if (move_previous) {
            values[pos]    = previous.value;
            token_ids[pos] = previous.token_id;
            insertion_pos  = pos - 1;
        } else {
            keep_moving = false;
        }
    }
    values[insertion_pos]    = value;
    token_ids[insertion_pos] = token_id;
}

template<typename scalar_t, int LOCAL_TOP_K>
__global__ void mtpSelectedRowLogProbsKernel(const scalar_t* __restrict__ logits,
                                             const int64_t* __restrict__ source_row_indices,
                                             const int32_t* __restrict__ emitted_token_ids_i32,
                                             const int64_t* __restrict__ emitted_token_ids_i64,
                                             float* __restrict__ token_logprobs,
                                             int32_t* __restrict__ top_logprob_token_ids,
                                             float* __restrict__ top_logprobs,
                                             int64_t logits_rows,
                                             int64_t row_stride,
                                             int64_t vocab_size,
                                             int32_t top_k) {
    const int64_t selected_row = static_cast<int64_t>(blockIdx.x);
    const int64_t source_row   = source_row_indices[selected_row];
    const int     tid          = static_cast<int>(threadIdx.x);

    if (source_row < 0 || source_row >= logits_rows) {
        if (tid == 0) {
            token_logprobs[selected_row] = -CUDART_INF_F;
        }
        for (int rank = tid; rank < top_k; rank += blockDim.x) {
            const int64_t output_offset          = selected_row * top_k + rank;
            top_logprob_token_ids[output_offset] = -1;
            top_logprobs[output_offset]          = -CUDART_INF_F;
        }
        return;
    }

    const scalar_t* row_logits = logits + source_row * row_stride;

    float   thread_max = -CUDART_INF_F;
    int     has_nan    = 0;
    float   local_top_values[LOCAL_TOP_K > 0 ? LOCAL_TOP_K : 1];
    int32_t local_top_token_ids[LOCAL_TOP_K > 0 ? LOCAL_TOP_K : 1];
    if constexpr (LOCAL_TOP_K > 0) {
#pragma unroll
        for (int rank = 0; rank < LOCAL_TOP_K; ++rank) {
            local_top_values[rank]    = -CUDART_INF_F;
            local_top_token_ids[rank] = -1;
        }
    }

    for (int64_t token_id = tid; token_id < vocab_size; token_id += blockDim.x) {
        const float value = static_cast<float>(row_logits[token_id]);
        has_nan |= isnan(value);
        thread_max = fmaxf(thread_max, value);
        if constexpr (LOCAL_TOP_K > 0) {
            insertMtpLocalTopK<LOCAL_TOP_K>(
                value, static_cast<int32_t>(token_id), local_top_values, local_top_token_ids);
        }
    }

    __shared__ float   reduction_values[kMtpSelectedLogprobsBlockSize];
    __shared__ int32_t reduction_token_ids[kMtpSelectedLogprobsBlockSize];
    __shared__ int32_t reduction_owners[kMtpSelectedLogprobsBlockSize];
    __shared__ int32_t nan_flags[kMtpSelectedLogprobsBlockSize];
    __shared__ int32_t winning_owner;

    reduction_values[tid] = thread_max;
    nan_flags[tid]        = has_nan;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction_values[tid] = fmaxf(reduction_values[tid], reduction_values[tid + stride]);
            nan_flags[tid] |= nan_flags[tid + stride];
        }
        __syncthreads();
    }

    const float row_max     = reduction_values[0];
    const bool  row_has_nan = nan_flags[0] != 0;
    float       row_shifted_logsumexp;
    if (row_has_nan) {
        row_shifted_logsumexp = CUDART_NAN_F;
    } else if (row_max == CUDART_INF_F || row_max == -CUDART_INF_F) {
        row_shifted_logsumexp = 0.0f;
    } else {
        float thread_sum = 0.0f;
        for (int64_t token_id = tid; token_id < vocab_size; token_id += blockDim.x) {
            thread_sum += expf(static_cast<float>(row_logits[token_id]) - row_max);
        }
        reduction_values[tid] = thread_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction_values[tid] += reduction_values[tid + stride];
            }
            __syncthreads();
        }
        row_shifted_logsumexp = logf(reduction_values[0]);
    }

    if (tid == 0) {
        int64_t emitted_token_id =
            emitted_token_ids_i64 != nullptr ? emitted_token_ids_i64[source_row] : emitted_token_ids_i32[source_row];
        emitted_token_id = emitted_token_id < 0 ? 0 : emitted_token_id;
        emitted_token_id = emitted_token_id >= vocab_size ? vocab_size - 1 : emitted_token_id;
        token_logprobs[selected_row] =
            (static_cast<float>(row_logits[emitted_token_id]) - row_max) - row_shifted_logsumexp;
    }

    if constexpr (LOCAL_TOP_K > 0) {
        int local_cursor = 0;
        for (int rank = 0; rank < top_k; ++rank) {
            reduction_values[tid]    = local_top_values[local_cursor];
            reduction_token_ids[tid] = local_top_token_ids[local_cursor];
            reduction_owners[tid]    = tid;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    const MtpTopCandidate current{reduction_values[tid], reduction_token_ids[tid]};
                    const MtpTopCandidate other{reduction_values[tid + stride], reduction_token_ids[tid + stride]};
                    if (mtpTopCandidateBetter(other, current)) {
                        reduction_values[tid]    = other.value;
                        reduction_token_ids[tid] = other.token_id;
                        reduction_owners[tid]    = reduction_owners[tid + stride];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int64_t output_offset          = selected_row * top_k + rank;
                top_logprob_token_ids[output_offset] = reduction_token_ids[0];
                top_logprobs[output_offset]          = (reduction_values[0] - row_max) - row_shifted_logsumexp;
                winning_owner                        = reduction_owners[0];
            }
            __syncthreads();
            if (tid == winning_owner) {
                ++local_cursor;
            }
            __syncthreads();
        }
    }
}

template<typename scalar_t, int LOCAL_TOP_K>
void launchMtpSelectedRowLogProbs(const torch::Tensor& logits,
                                  const torch::Tensor& source_row_indices,
                                  const torch::Tensor& emitted_token_ids,
                                  torch::Tensor&       token_logprobs,
                                  torch::Tensor&       top_logprob_token_ids,
                                  torch::Tensor&       top_logprobs,
                                  int32_t              top_k,
                                  cudaStream_t         stream) {
    const int32_t* emitted_token_ids_i32 =
        emitted_token_ids.scalar_type() == torch::kInt32 ? emitted_token_ids.data_ptr<int32_t>() : nullptr;
    const int64_t* emitted_token_ids_i64 =
        emitted_token_ids.scalar_type() == torch::kInt64 ? emitted_token_ids.data_ptr<int64_t>() : nullptr;
    int32_t* top_ids_ptr   = top_k > 0 ? top_logprob_token_ids.data_ptr<int32_t>() : nullptr;
    float*   top_probs_ptr = top_k > 0 ? top_logprobs.data_ptr<float>() : nullptr;
    mtpSelectedRowLogProbsKernel<scalar_t, LOCAL_TOP_K>
        <<<static_cast<unsigned int>(source_row_indices.numel()), kMtpSelectedLogprobsBlockSize, 0, stream>>>(
            logits.data_ptr<scalar_t>(),
            source_row_indices.data_ptr<int64_t>(),
            emitted_token_ids_i32,
            emitted_token_ids_i64,
            token_logprobs.data_ptr<float>(),
            top_ids_ptr,
            top_probs_ptr,
            logits.size(0),
            logits.stride(0),
            logits.size(1),
            top_k);
}

template<typename scalar_t>
void dispatchMtpSelectedRowLogProbsTopK(const torch::Tensor& logits,
                                        const torch::Tensor& source_row_indices,
                                        const torch::Tensor& emitted_token_ids,
                                        torch::Tensor&       token_logprobs,
                                        torch::Tensor&       top_logprob_token_ids,
                                        torch::Tensor&       top_logprobs,
                                        int32_t              top_k,
                                        cudaStream_t         stream) {
#define LAUNCH_MTP_SELECTED_LOGPROBS(LOCAL_TOP_K)                                                                      \
    launchMtpSelectedRowLogProbs<scalar_t, LOCAL_TOP_K>(logits,                                                        \
                                                        source_row_indices,                                            \
                                                        emitted_token_ids,                                             \
                                                        token_logprobs,                                                \
                                                        top_logprob_token_ids,                                         \
                                                        top_logprobs,                                                  \
                                                        top_k,                                                         \
                                                        stream)
    if (top_k == 0) {
        LAUNCH_MTP_SELECTED_LOGPROBS(0);
    } else if (top_k == 1) {
        LAUNCH_MTP_SELECTED_LOGPROBS(1);
    } else if (top_k == 2) {
        LAUNCH_MTP_SELECTED_LOGPROBS(2);
    } else if (top_k <= 4) {
        LAUNCH_MTP_SELECTED_LOGPROBS(4);
    } else if (top_k <= 8) {
        LAUNCH_MTP_SELECTED_LOGPROBS(8);
    } else if (top_k <= 16) {
        LAUNCH_MTP_SELECTED_LOGPROBS(16);
    } else {
        LAUNCH_MTP_SELECTED_LOGPROBS(20);
    }
#undef LAUNCH_MTP_SELECTED_LOGPROBS
}

void checkMtpSelectedLogprobsCudaTensor(const torch::Tensor& tensor, const torch::Device& device, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.device() == device, "%s must be on the logits device", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
}

}  // namespace

void invokeMtpSelectedRowLogProbs(const torch::Tensor& logits,
                                  const torch::Tensor& source_row_indices,
                                  const torch::Tensor& emitted_token_ids,
                                  torch::Tensor&       token_logprobs,
                                  torch::Tensor&       top_logprob_token_ids,
                                  torch::Tensor&       top_logprobs,
                                  int64_t              top_k,
                                  cudaStream_t         stream) {
    RTP_LLM_CHECK_WITH_INFO(logits.defined() && logits.is_cuda(), "MTP logits must be a CUDA tensor");
    RTP_LLM_CHECK_WITH_INFO(logits.dim() == 2, "MTP logits must be 2-D, got dim=%ld", logits.dim());
    RTP_LLM_CHECK_WITH_INFO(logits.stride(1) == 1, "MTP logits innermost stride must be 1, got %ld", logits.stride(1));
    RTP_LLM_CHECK_WITH_INFO(logits.stride(0) >= logits.size(1),
                            "MTP logits row stride %ld must cover width %ld",
                            logits.stride(0),
                            logits.size(1));
    RTP_LLM_CHECK_WITH_INFO(logits.size(1) > 0, "MTP logits vocabulary must be non-empty");
    RTP_LLM_CHECK_WITH_INFO(logits.size(1) <= std::numeric_limits<int32_t>::max(),
                            "MTP logits vocabulary %ld exceeds int32 token IDs",
                            logits.size(1));
    RTP_LLM_CHECK_WITH_INFO(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kFloat16
                                || logits.scalar_type() == torch::kBFloat16,
                            "unsupported MTP logits dtype %s",
                            c10::toString(logits.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(top_k >= 0 && top_k <= kMtpSelectedLogprobsMaxTopK,
                            "MTP selected-row top_k %ld must be in [0, %d]",
                            top_k,
                            kMtpSelectedLogprobsMaxTopK);
    RTP_LLM_CHECK_WITH_INFO(
        top_k <= logits.size(1), "MTP selected-row top_k %ld exceeds vocabulary %ld", top_k, logits.size(1));

    const auto device = logits.device();
    checkMtpSelectedLogprobsCudaTensor(source_row_indices, device, "source_row_indices");
    RTP_LLM_CHECK_WITH_INFO(source_row_indices.scalar_type() == torch::kInt64, "source_row_indices must be int64");
    RTP_LLM_CHECK_WITH_INFO(
        source_row_indices.dim() == 1, "source_row_indices must be 1-D, got dim=%ld", source_row_indices.dim());
    const int64_t selected_rows = source_row_indices.numel();
    RTP_LLM_CHECK_WITH_INFO(selected_rows <= std::numeric_limits<unsigned int>::max(),
                            "selected MTP rows %ld exceed CUDA grid capacity",
                            selected_rows);

    checkMtpSelectedLogprobsCudaTensor(emitted_token_ids, device, "emitted_token_ids");
    RTP_LLM_CHECK_WITH_INFO(emitted_token_ids.scalar_type() == torch::kInt32
                                || emitted_token_ids.scalar_type() == torch::kInt64,
                            "emitted_token_ids must be int32 or int64");
    RTP_LLM_CHECK_WITH_INFO(emitted_token_ids.numel() >= logits.size(0),
                            "emitted_token_ids numel %ld must cover all %ld dense logits rows",
                            emitted_token_ids.numel(),
                            logits.size(0));

    checkMtpSelectedLogprobsCudaTensor(token_logprobs, device, "token_logprobs");
    RTP_LLM_CHECK_WITH_INFO(token_logprobs.scalar_type() == torch::kFloat32, "token_logprobs must be float32");
    RTP_LLM_CHECK_WITH_INFO(token_logprobs.dim() == 1 && token_logprobs.size(0) == selected_rows,
                            "token_logprobs must have shape [%ld]",
                            selected_rows);

    checkMtpSelectedLogprobsCudaTensor(top_logprob_token_ids, device, "top_logprob_token_ids");
    RTP_LLM_CHECK_WITH_INFO(top_logprob_token_ids.scalar_type() == torch::kInt32,
                            "top_logprob_token_ids must be int32");
    RTP_LLM_CHECK_WITH_INFO(top_logprob_token_ids.dim() == 2 && top_logprob_token_ids.size(0) == selected_rows
                                && top_logprob_token_ids.size(1) == top_k,
                            "top_logprob_token_ids must have shape [%ld, %ld]",
                            selected_rows,
                            top_k);

    checkMtpSelectedLogprobsCudaTensor(top_logprobs, device, "top_logprobs");
    RTP_LLM_CHECK_WITH_INFO(top_logprobs.scalar_type() == torch::kFloat32, "top_logprobs must be float32");
    RTP_LLM_CHECK_WITH_INFO(top_logprobs.dim() == 2 && top_logprobs.size(0) == selected_rows
                                && top_logprobs.size(1) == top_k,
                            "top_logprobs must have shape [%ld, %ld]",
                            selected_rows,
                            top_k);

    if (selected_rows == 0) {
        return;
    }

    switch (logits.scalar_type()) {
        case torch::kFloat32:
            dispatchMtpSelectedRowLogProbsTopK<float>(logits,
                                                      source_row_indices,
                                                      emitted_token_ids,
                                                      token_logprobs,
                                                      top_logprob_token_ids,
                                                      top_logprobs,
                                                      static_cast<int32_t>(top_k),
                                                      stream);
            break;
        case torch::kFloat16:
            dispatchMtpSelectedRowLogProbsTopK<at::Half>(logits,
                                                         source_row_indices,
                                                         emitted_token_ids,
                                                         token_logprobs,
                                                         top_logprob_token_ids,
                                                         top_logprobs,
                                                         static_cast<int32_t>(top_k),
                                                         stream);
            break;
        case torch::kBFloat16:
            dispatchMtpSelectedRowLogProbsTopK<at::BFloat16>(logits,
                                                             source_row_indices,
                                                             emitted_token_ids,
                                                             token_logprobs,
                                                             top_logprob_token_ids,
                                                             top_logprobs,
                                                             static_cast<int32_t>(top_k),
                                                             stream);
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "unsupported MTP logits dtype %s", c10::toString(logits.scalar_type()));
    }
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
