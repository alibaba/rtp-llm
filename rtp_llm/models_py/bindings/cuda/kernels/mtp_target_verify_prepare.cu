#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>
#include <climits>
#include <cuda_bf16.h>
#include <limits>
#include <math_constants.h>

namespace rtp_llm {

namespace {
__global__ void deepinferGreedyFinalizeRtp(const int32_t* draft,
                                           const int32_t* target,
                                           int32_t*       out,
                                           int32_t*       lengths,
                                           int64_t*       rows,
                                           int64_t*       total,
                                           int            batch,
                                           int            steps,
                                           int            token_stride) {
    __shared__ int64_t partial[256];
    int64_t            sum = 0;
    for (int b = threadIdx.x; b < batch; b += blockDim.x) {
        int last = steps;
        for (int j = 0; j < steps; j++) {
            int token = target[(b * (steps + 1) + j) * token_stride + token_stride - 1];
            if (token != draft[b * steps + j]) {
                last = j;
                break;
            }
        }
        for (int j = 0; j <= steps; j++)
            out[b * (steps + 1) + j] = j <= last ? target[(b * (steps + 1) + j) * token_stride + token_stride - 1] : 0;
        lengths[b] = last + 1;
        rows[b]    = last + 1;
        sum += last + 1;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset)
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        total[0] = partial[0];
}
__global__ void mtpAcceptMetricsKernel(const int32_t* lengths, int64_t* rows, int64_t* total, int32_t count) {
    __shared__ int64_t sums[256];
    int64_t            sum = 0;
    for (int32_t i = threadIdx.x; i < count; i += blockDim.x) {
        int64_t value = lengths[i];
        rows[i]       = value;
        sum += value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        total[0] = sums[0];
}

__global__ void mtpCleanRejectedTokensKernel(int32_t* tokens, int64_t count) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && tokens[i] == -1)
        tokens[i] = 0;
}

__global__ void mtpPublishStateKernel(const int32_t*       lengths,
                                      const int32_t*       tokens,
                                      const int32_t*       previous,
                                      const __nv_bfloat16* hidden,
                                      int32_t*             next,
                                      int32_t*             selected,
                                      __nv_bfloat16*       output_hidden,
                                      int32_t              stride,
                                      int32_t              width) {
    const int32_t row      = blockIdx.y;
    const int32_t col      = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t accepted = lengths[row];
    if (col == 0) {
        selected[row] = tokens[row * stride + accepted - 1];
        if (next != nullptr)
            next[row] = previous[row] + accepted;
    }
    if (col < width) {
        output_hidden[row * width + col] = hidden[(row * stride + accepted - 1) * width + col];
    }
}

struct MtpPreviousRows {
    const int32_t* rows[128];
};
__global__ void mtpPublishStateRowsKernel(const int32_t*       lengths,
                                          const int32_t*       tokens,
                                          MtpPreviousRows      previous,
                                          const __nv_bfloat16* hidden,
                                          int32_t*             next,
                                          int32_t*             selected,
                                          __nv_bfloat16*       output_hidden,
                                          int32_t              stride,
                                          int32_t              width,
                                          int32_t              offset) {
    const int32_t row      = blockIdx.y + offset;
    const int32_t col      = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t accepted = lengths[row];
    if (col == 0) {
        selected[row] = tokens[row * stride + accepted - 1];
        next[row]     = *previous.rows[blockIdx.y] + accepted;
    }
    if (col < width) {
        output_hidden[row * width + col] = hidden[(row * stride + accepted - 1) * width + col];
    }
}

struct MtpMaxToken {
    float score;
    int   index;
};
__device__ __forceinline__ MtpMaxToken mtpPick(MtpMaxToken a, MtpMaxToken b) {
    if (isnan(a.score))
        return isnan(b.score) && b.index < a.index ? b : a;
    if (isnan(b.score))
        return b;
    return b.score > a.score || (b.score == a.score && b.index < a.index) ? b : a;
}
__device__ __forceinline__ MtpMaxToken mtpReduce(MtpMaxToken value) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        MtpMaxToken other{__shfl_xor_sync(0xffffffff, value.score, off), __shfl_xor_sync(0xffffffff, value.index, off)};
        value = mtpPick(value, other);
    }
    __shared__ MtpMaxToken shared[8];
    int                    lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? shared[lane] : MtpMaxToken{-INFINITY, INT_MAX};
#pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            MtpMaxToken other{__shfl_xor_sync(0xffffffff, value.score, off),
                              __shfl_xor_sync(0xffffffff, value.index, off)};
            value = mtpPick(value, other);
        }
        if (lane == 0)
            shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}
__global__ void mtpGreedyPartialKernel(const float* probs, MtpMaxToken* partial, int width, int partitions) {
    int         row = blockIdx.y, tile = blockIdx.x;
    MtpMaxToken best{-INFINITY, INT_MAX};
    for (int col = tile * 2048 + threadIdx.x; col < min(width, (tile + 1) * 2048); col += 256) {
        best = mtpPick(best, MtpMaxToken{probs[row * width + col], col});
    }
    best = mtpReduce(best);
    if (threadIdx.x == 0)
        partial[row * partitions + tile] = best;
}
__global__ void mtpGreedyPackKernel(const MtpMaxToken* partial,
                                    float*             output,
                                    int64_t*           ids,
                                    int32_t*           ids32,
                                    const int64_t*     mapping,
                                    int                mapsize,
                                    int                width,
                                    int                partitions,
                                    const int32_t*     sequence_lengths,
                                    int32_t*           next_sequence_lengths,
                                    __nv_bfloat16*     hidden_to_zero,
                                    int32_t            hidden_width,
                                    bool               zero_tokens) {
    int         row = blockIdx.y, tile = blockIdx.x;
    MtpMaxToken best{-INFINITY, INT_MAX};
    for (int p = threadIdx.x; p < partitions; p += 256)
        best = mtpPick(best, partial[row * partitions + p]);
    best = mtpReduce(best);
    if (output != nullptr) {
        for (int col = tile * 2048 + threadIdx.x; col < min(width, (tile + 1) * 2048); col += 256)
            output[row * width + col] = col == best.index ? 1.f : 0.f;
    }
    for (int col = tile * 2048 + threadIdx.x; col < min(hidden_width, (tile + 1) * 2048); col += 256) {
        hidden_to_zero[row * hidden_width + col] = __float2bfloat16(0.f);
    }
    if (tile == 0 && threadIdx.x == 0) {
        int64_t id = mapping && best.index < mapsize ? mapping[best.index] : best.index;
        if (zero_tokens)
            id = 0;
        ids[row]   = id;
        ids32[row] = static_cast<int32_t>(id);
        if (next_sequence_lengths != nullptr) {
            next_sequence_lengths[row] = sequence_lengths[row] + 1;
        }
    }
}

__global__ void
mtpStreamInitKernel(int32_t* seed, int32_t* target, int32_t* propose, int64_t sequence_length, int32_t topk) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col == 0) {
        if (target != nullptr)
            target[0] = 0;
        if (propose != nullptr)
            propose[0] = 0;
    }
    if (col < topk) {
        int64_t valid = sequence_length < topk ? sequence_length : topk;
        seed[col]     = col < valid ? static_cast<int32_t>(sequence_length - valid + col) : -1;
    }
}

__global__ void mtpPostRejectionPrepareKernel(int32_t* accept_len,
                                              int32_t* tokens,
                                              int32_t* indices,
                                              int32_t* selected_tokens,
                                              int32_t  stride,
                                              int32_t  batch,
                                              bool     initialize_fake) {
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch) {
        if (initialize_fake) {
            accept_len[row] = stride;
            for (int32_t j = 0; j < stride; ++j) {
                tokens[row * stride + j] = 0;
            }
        }
        int32_t idx  = row * stride + accept_len[row] - 1;
        indices[row] = idx;
        if (selected_tokens != nullptr) {
            selected_tokens[row] = tokens[idx];
        }
    }
}

__global__ void mtpSelectedTokensKernel(const int32_t* tokens, const int32_t* indices, int32_t* output, int32_t batch) {
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch) {
        output[row] = tokens[indices[row]];
    }
}

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

bool tryMtpPublishState(const torch::Tensor&               accept_lengths,
                        const torch::Tensor&               accepted_tokens,
                        const torch::Tensor&               previous_lengths,
                        const torch::Tensor&               hidden_states,
                        int32_t                            tokens_per_batch,
                        torch::Tensor&                     next_lengths,
                        torch::Tensor&                     selected_tokens,
                        torch::Tensor&                     selected_hidden,
                        cudaStream_t                       stream,
                        const std::vector<const int32_t*>& previous_rows) {
    if (!accept_lengths.defined() || !accepted_tokens.defined() || !accept_lengths.is_cuda()
        || accept_lengths.scalar_type() != torch::kInt32 || !accept_lengths.is_contiguous()
        || accepted_tokens.device() != accept_lengths.device() || accepted_tokens.scalar_type() != torch::kInt32
        || !accepted_tokens.is_contiguous() || tokens_per_batch < 2)
        return false;
    const int64_t batch        = accept_lengths.numel();
    const bool    has_previous = previous_lengths.defined() || !previous_rows.empty();
    const bool    has_hidden   = hidden_states.defined();
    if (batch > 65535 || accepted_tokens.numel() > INT_MAX || accepted_tokens.numel() != batch * tokens_per_batch
        || (previous_lengths.defined()
            && (previous_lengths.device() != accept_lengths.device() || previous_lengths.scalar_type() != torch::kInt32
                || !previous_lengths.is_contiguous() || previous_lengths.numel() != batch))
        || (has_hidden
            && (hidden_states.device() != accept_lengths.device() || hidden_states.scalar_type() != torch::kBFloat16
                || !hidden_states.is_contiguous() || hidden_states.dim() != 2
                || hidden_states.size(0) != batch * tokens_per_batch || hidden_states.numel() > INT_MAX)))
        return false;
    if (!previous_rows.empty()) {
        if (previous_rows.size() != static_cast<size_t>(batch))
            return false;
        for (auto ptr : previous_rows)
            if (!ptr)
                return false;
    }
    const int32_t width = has_hidden ? static_cast<int32_t>(hidden_states.size(1)) : 0;
    selected_tokens     = torch::empty({batch}, accept_lengths.options());
    if (has_previous)
        next_lengths = torch::empty({batch}, accept_lengths.options());
    if (has_hidden)
        selected_hidden = torch::empty({batch, width}, hidden_states.options());
    if (batch == 0)
        return true;
    if (!previous_rows.empty()) {
        for (int32_t offset = 0; offset < batch; offset += 128) {
            const int32_t   count = std::min<int64_t>(batch - offset, 128);
            MtpPreviousRows previous{};
            std::copy_n(previous_rows.begin() + offset, count, previous.rows);
            mtpPublishStateRowsKernel<<<dim3(std::max((width + 255) / 256, 1), count), 256, 0, stream>>>(
                accept_lengths.data_ptr<int32_t>(),
                accepted_tokens.data_ptr<int32_t>(),
                previous,
                has_hidden ? reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()) : nullptr,
                next_lengths.data_ptr<int32_t>(),
                selected_tokens.data_ptr<int32_t>(),
                has_hidden ? reinterpret_cast<__nv_bfloat16*>(selected_hidden.data_ptr<at::BFloat16>()) : nullptr,
                tokens_per_batch,
                width,
                offset);
        }
        return true;
    }
    mtpPublishStateKernel<<<dim3(std::max((width + 255) / 256, 1), batch), 256, 0, stream>>>(
        accept_lengths.data_ptr<int32_t>(),
        accepted_tokens.data_ptr<int32_t>(),
        has_previous ? previous_lengths.data_ptr<int32_t>() : nullptr,
        has_hidden ? reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()) : nullptr,
        has_previous ? next_lengths.data_ptr<int32_t>() : nullptr,
        selected_tokens.data_ptr<int32_t>(),
        has_hidden ? reinterpret_cast<__nv_bfloat16*>(selected_hidden.data_ptr<at::BFloat16>()) : nullptr,
        tokens_per_batch,
        width);
    return true;
}

bool tryMtpGreedyFinalize(const torch::Tensor& draft,
                          const torch::Tensor& target,
                          torch::Tensor&       tokens,
                          torch::Tensor&       lengths,
                          torch::Tensor&       rows,
                          torch::Tensor&       total,
                          cudaStream_t         stream) {
    const torch::Tensor* tensors[] = {&draft, &target, &tokens, &lengths};
    for (const auto* tensor : tensors) {
        if (!tensor->defined() || !tensor->is_cuda() || tensor->scalar_type() != torch::kInt32
            || !tensor->is_contiguous() || tensor->device() != draft.device())
            return false;
    }
    if (draft.dim() != 2 || target.dim() != 2 || tokens.dim() != 2 || lengths.dim() != 1)
        return false;
    const auto batch  = draft.size(0);
    const auto steps  = draft.size(1);
    const auto stride = target.size(1);
    if (batch <= 0 || batch > 65535 || steps <= 0 || steps > 32 || stride <= 0 || stride > INT_MAX
        || target.size(0) != batch * (steps + 1) || target.numel() > INT_MAX || tokens.size(0) != batch
        || tokens.size(1) != steps + 1 || lengths.numel() != batch)
        return false;
    rows  = torch::empty({batch}, lengths.options().dtype(torch::kInt64));
    total = torch::empty({1}, rows.options());
    deepinferGreedyFinalizeRtp<<<1, 256, 0, stream>>>(draft.data_ptr<int32_t>(),
                                                      target.data_ptr<int32_t>(),
                                                      tokens.data_ptr<int32_t>(),
                                                      lengths.data_ptr<int32_t>(),
                                                      rows.data_ptr<int64_t>(),
                                                      total.data_ptr<int64_t>(),
                                                      static_cast<int>(batch),
                                                      static_cast<int>(steps),
                                                      static_cast<int>(stride));
    return true;
}

bool tryMtpAcceptMetrics(const torch::Tensor& lengths, torch::Tensor& rows, torch::Tensor& total, cudaStream_t stream) {
    if (!lengths.is_cuda() || lengths.scalar_type() != torch::kInt32 || !lengths.is_contiguous()
        || lengths.numel() > 65535)
        return false;
    rows  = torch::empty({lengths.numel()}, lengths.options().dtype(torch::kInt64));
    total = torch::empty({1}, rows.options());
    mtpAcceptMetricsKernel<<<1, 256, 0, stream>>>(lengths.data_ptr<int32_t>(),
                                                  rows.data_ptr<int64_t>(),
                                                  total.data_ptr<int64_t>(),
                                                  static_cast<int32_t>(lengths.numel()));
    return true;
}

bool tryMtpCleanRejectedTokens(torch::Tensor& tokens, cudaStream_t stream) {
    if (!tokens.is_cuda() || tokens.scalar_type() != torch::kInt32 || !tokens.is_contiguous()
        || tokens.numel() > INT_MAX)
        return false;
    if (tokens.numel())
        mtpCleanRejectedTokensKernel<<<(tokens.numel() + 255) / 256, 256, 0, stream>>>(tokens.data_ptr<int32_t>(),
                                                                                       tokens.numel());
    return true;
}

bool tryMtpGreedyPack(const torch::Tensor& probabilities,
                      const torch::Tensor& mapping,
                      torch::Tensor&       output_probabilities,
                      torch::Tensor&       output_tokens,
                      torch::Tensor&       output_tokens_int32,
                      const torch::Tensor& sequence_lengths,
                      const torch::Tensor& hidden_to_zero,
                      bool                 zero_tokens,
                      torch::Tensor&       next_sequence_lengths,
                      cudaStream_t         stream,
                      bool                 compact_point_mass) {
    const bool has_mapping = mapping.defined() && mapping.numel() > 0;
    if (!probabilities.is_cuda() || probabilities.scalar_type() != torch::kFloat32 || probabilities.dim() != 2
        || !probabilities.is_contiguous() || probabilities.size(1) <= 0 || probabilities.size(0) > 65535
        || probabilities.numel() > INT_MAX || probabilities.size(1) > INT_MAX - 2047
        || (has_mapping
            && (mapping.device() != probabilities.device() || mapping.scalar_type() != torch::kInt64
                || !mapping.is_contiguous() || mapping.numel() > INT_MAX)))
        return false;
    const bool has_sequence = sequence_lengths.defined();
    if ((has_sequence
         && (sequence_lengths.device() != probabilities.device() || sequence_lengths.scalar_type() != torch::kInt32
             || !sequence_lengths.is_contiguous() || sequence_lengths.numel() != probabilities.size(0)))
        || (zero_tokens
            && (!hidden_to_zero.defined() || hidden_to_zero.device() != probabilities.device()
                || hidden_to_zero.scalar_type() != torch::kBFloat16 || hidden_to_zero.dim() != 2
                || !hidden_to_zero.is_contiguous() || hidden_to_zero.size(0) != probabilities.size(0)
                || hidden_to_zero.size(1) > probabilities.size(1))))
        return false;
    next_sequence_lengths    = has_sequence ? torch::empty_like(sequence_lengths) : torch::Tensor();
    const int32_t batch      = static_cast<int32_t>(probabilities.size(0));
    const int32_t width      = static_cast<int32_t>(probabilities.size(1));
    const int32_t partitions = (width + 2047) / 2048;
    output_probabilities     = compact_point_mass ? torch::Tensor() : torch::empty_like(probabilities);
    output_tokens            = torch::empty({batch, 1}, probabilities.options().dtype(torch::kInt64));
    output_tokens_int32      = torch::empty({batch, 1}, probabilities.options().dtype(torch::kInt32));
    if (batch == 0)
        return true;
    auto scratch = torch::empty({static_cast<int64_t>(batch) * partitions * static_cast<int64_t>(sizeof(MtpMaxToken))},
                                probabilities.options().dtype(torch::kByte));
    auto partial = reinterpret_cast<MtpMaxToken*>(scratch.data_ptr<uint8_t>());
    mtpGreedyPartialKernel<<<dim3(partitions, batch), 256, 0, stream>>>(
        probabilities.data_ptr<float>(), partial, width, partitions);
    const int32_t output_partitions =
        compact_point_mass ? std::max<int32_t>(1, zero_tokens ? (hidden_to_zero.size(1) + 2047) / 2048 : 1) :
                             partitions;
    mtpGreedyPackKernel<<<dim3(output_partitions, batch), 256, 0, stream>>>(
        partial,
        compact_point_mass ? nullptr : output_probabilities.data_ptr<float>(),
        output_tokens.data_ptr<int64_t>(),
        output_tokens_int32.data_ptr<int32_t>(),
        has_mapping ? mapping.data_ptr<int64_t>() : nullptr,
        has_mapping ? static_cast<int32_t>(mapping.numel()) : 0,
        width,
        partitions,
        has_sequence ? sequence_lengths.data_ptr<int32_t>() : nullptr,
        has_sequence ? next_sequence_lengths.data_ptr<int32_t>() : nullptr,
        zero_tokens ? reinterpret_cast<__nv_bfloat16*>(hidden_to_zero.data_ptr<at::BFloat16>()) : nullptr,
        zero_tokens ? static_cast<int32_t>(hidden_to_zero.size(1)) : 0,
        zero_tokens);
    return true;
}

void invokeMtpStreamInit(torch::Tensor& seed,
                         torch::Tensor& target_token,
                         torch::Tensor& propose_token,
                         int64_t        sequence_length,
                         cudaStream_t   stream) {
    int64_t topk = 0;
    if (seed.defined()) {
        topk = seed.numel();
        checkCudaI32Vector(seed, "seed", topk);
    }
    if (target_token.defined())
        checkCudaI32Vector(target_token, "target_token", 1);
    if (propose_token.defined())
        checkCudaI32Vector(propose_token, "propose_token", 1);
    int64_t work = std::max<int64_t>(topk, 1);
    mtpStreamInitKernel<<<(work + 255) / 256, 256, 0, stream>>>(
        seed.defined() ? seed.data_ptr<int32_t>() : nullptr,
        target_token.defined() ? target_token.data_ptr<int32_t>() : nullptr,
        propose_token.defined() ? propose_token.data_ptr<int32_t>() : nullptr,
        std::max<int64_t>(sequence_length, 0),
        static_cast<int32_t>(topk));
}

void invokeMtpPostRejectionPrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& tokens,
                                   torch::Tensor&       indices,
                                   torch::Tensor&       selected_tokens,
                                   int32_t              tokens_per_batch,
                                   cudaStream_t         stream,
                                   bool                 initialize_fake) {
    const int64_t batch = accept_len.numel();
    checkCudaI32Vector(accept_len, "accept_len", batch);
    checkCudaI32Vector(indices, "indices", batch);
    checkCudaI32Vector(tokens, "tokens", batch * tokens_per_batch);
    if (selected_tokens.defined()) {
        checkCudaI32Vector(selected_tokens, "selected_tokens", batch);
    }
    if (batch == 0) {
        return;
    }
    mtpPostRejectionPrepareKernel<<<(batch + 255) / 256, 256, 0, stream>>>(
        accept_len.data_ptr<int32_t>(),
        tokens.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        selected_tokens.defined() ? selected_tokens.data_ptr<int32_t>() : nullptr,
        tokens_per_batch,
        static_cast<int32_t>(batch),
        initialize_fake);
}

void invokeMtpSelectedTokens(const torch::Tensor& tokens,
                             const torch::Tensor& indices,
                             torch::Tensor&       output,
                             cudaStream_t         stream) {
    const int64_t batch = indices.numel();
    checkCudaI32Vector(tokens, "tokens", 0);
    checkCudaI32Vector(indices, "indices", batch);
    checkCudaI32Vector(output, "output", batch);
    if (batch == 0) {
        return;
    }
    mtpSelectedTokensKernel<<<(batch + 255) / 256, 256, 0, stream>>>(tokens.data_ptr<int32_t>(),
                                                                     indices.data_ptr<int32_t>(),
                                                                     output.data_ptr<int32_t>(),
                                                                     static_cast<int32_t>(batch));
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
