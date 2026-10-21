#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>
#include <climits>
#include <c10/cuda/CUDAException.h>

namespace rtp_llm {

namespace {

constexpr int kMtpInputRowsPerLaunch = 128;

struct MtpSpecDecodeStateParams {
    const int32_t* accepted[kMtpInputRowsPerLaunch];
    const int32_t* lengths[kMtpInputRowsPerLaunch];
    int32_t        widths[kMtpInputRowsPerLaunch];
    const int32_t* columns[8];
    int32_t*       tokens;
    int32_t*       input_lengths;
    int32_t*       lm_indexes;
    int32_t        row_offset;
    int32_t        rows;
    int32_t        tokens_per_batch;
};

static_assert(sizeof(MtpSpecDecodeStateParams) < 4096, "MTP state parameters exceed the kernel argument budget");

__global__ void mtpSpecDecodeTokensMetadataFromStateKernel(MtpSpecDecodeStateParams params) {
    const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= params.rows * params.tokens_per_batch) {
        return;
    }
    const int32_t local_row = index / params.tokens_per_batch;
    const int32_t column    = index % params.tokens_per_batch;
    const int32_t row       = params.row_offset + local_row;
    const int32_t output    = row * params.tokens_per_batch + column;
    if (column == 0) {
        const int32_t accepted_index = *params.lengths[local_row] - 1;
        CUDA_KERNEL_ASSERT(accepted_index >= 0 && accepted_index < params.widths[local_row]);
        params.tokens[output]     = params.accepted[local_row][accepted_index];
        params.input_lengths[row] = params.tokens_per_batch;
    } else {
        params.tokens[output] = params.columns[column][row];
    }
    params.lm_indexes[output] = output;
}

bool isMtpI32Row(const torch::Tensor& tensor) {
    return tensor.defined() && tensor.is_cuda() && tensor.layout() == c10::kStrided && tensor.numel() > 0
           && tensor.scalar_type() == torch::kInt32 && (tensor.dim() == 1 || (tensor.dim() == 2 && tensor.size(0) == 1))
           && tensor.stride(-1) > 0;
}

bool isMtpI32Output(const torch::Tensor& tensor, int64_t numel, const torch::Device& device) {
    return tensor.defined() && tensor.device() == device && tensor.layout() == c10::kStrided
           && tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous() && tensor.numel() == numel;
}

bool sharesMtpStorage(const torch::Tensor& first, const torch::Tensor& second) {
    return first.defined() && second.defined() && first.storage().data_ptr().get() == second.storage().data_ptr().get();
}

bool aliasesMtpOutputs(const torch::Tensor& source,
                       const torch::Tensor& first,
                       const torch::Tensor& second,
                       const torch::Tensor& third) {
    return sharesMtpStorage(source, first) || sharesMtpStorage(source, second) || sharesMtpStorage(source, third);
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
                                            const int32_t* __restrict__ position_ids_in,
                                            int32_t* __restrict__ position_ids_out,
                                            int32_t token_stride,
                                            int32_t batch_size,
                                            int32_t total_tokens) {
    const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_tokens) {
        return;
    }
    int32_t first = 0;
    int32_t last  = batch_size - 1;
    while (first < last) {
        const int32_t middle = first + ((last - first) >> 1);
        if (batch_offsets[middle] <= index) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    const int32_t request = first;
    const int32_t begin   = request == 0 ? 0 : batch_offsets[request - 1];
    const int32_t offset  = index - begin;
    const int32_t length  = input_lengths[request];
    combo_tokens_out[index] = offset == length - 1 ?
        new_all_token_ids[request * token_stride + token_stride - 1] : combo_tokens_in[index + 1];
    if (position_ids_out != nullptr) {
        position_ids_out[index] =
            position_ids_in[offset == length - 1 ? index : index + 1];
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
    mtpTargetVerifyPrepareKernel<<<grid_size, block_size, 0, stream>>>(
        sequence_lengths.data_ptr<int32_t>(),
        input_lengths.data_ptr<int32_t>(),
        prefix_lengths.data_ptr<int32_t>(),
        sequence_lengths_plus_1.data_ptr<int32_t>(),
        lm_output_indexes.data_ptr<int32_t>(),
        tokens_per_batch,
        static_cast<int32_t>(batch_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

bool tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(const std::vector<torch::Tensor>& accepted_rows,
                                                          const std::vector<torch::Tensor>& accepted_length_rows,
                                                          const std::vector<torch::Tensor>& token_columns,
                                                          torch::Tensor&                    spec_tokens,
                                                          torch::Tensor&                    input_lengths,
                                                          torch::Tensor&                    lm_output_indexes,
                                                          int32_t                           tokens_per_batch,
                                                          cudaStream_t                      stream) {
    const int64_t batch = accepted_rows.size();
    if (batch <= 0 || tokens_per_batch <= 0 || tokens_per_batch > 8 || batch > INT_MAX / tokens_per_batch
        || accepted_length_rows.size() != accepted_rows.size()
        || token_columns.size() != static_cast<size_t>(tokens_per_batch) || !isMtpI32Row(accepted_rows.front())) {
        return false;
    }
    const auto device = accepted_rows.front().device();
    if (!isMtpI32Output(spec_tokens, batch * tokens_per_batch, device) || !isMtpI32Output(input_lengths, batch, device)
        || !isMtpI32Output(lm_output_indexes, batch * tokens_per_batch, device)
        || sharesMtpStorage(spec_tokens, input_lengths) || sharesMtpStorage(spec_tokens, lm_output_indexes)
        || sharesMtpStorage(input_lengths, lm_output_indexes)) {
        return false;
    }
    // Validate every chunk before writing any output, so a bad late row can
    // fall back to ATen without partially overwriting a previous chunk.
    for (int64_t row = 0; row < batch; ++row) {
        const auto& accepted = accepted_rows[row];
        const auto& length   = accepted_length_rows[row];
        if (!isMtpI32Row(accepted) || !accepted.is_contiguous() || accepted.device() != device
            || accepted.numel() > INT_MAX || !length.defined() || length.device() != device
            || length.layout() != c10::kStrided || length.scalar_type() != torch::kInt32 || length.numel() != 1
            || aliasesMtpOutputs(accepted, spec_tokens, input_lengths, lm_output_indexes)
            || aliasesMtpOutputs(length, spec_tokens, input_lengths, lm_output_indexes)) {
            return false;
        }
    }
    for (int32_t column = 1; column < tokens_per_batch; ++column) {
        if (!isMtpI32Output(token_columns[column], batch, device)
            || aliasesMtpOutputs(token_columns[column], spec_tokens, input_lengths, lm_output_indexes)) {
            return false;
        }
    }
    constexpr int threads = 256;
    for (int64_t offset = 0; offset < batch; offset += kMtpInputRowsPerLaunch) {
        MtpSpecDecodeStateParams params;
        params.row_offset       = static_cast<int32_t>(offset);
        params.rows             = static_cast<int32_t>(std::min<int64_t>(kMtpInputRowsPerLaunch, batch - offset));
        params.tokens_per_batch = tokens_per_batch;
        params.tokens           = spec_tokens.data_ptr<int32_t>();
        params.input_lengths    = input_lengths.data_ptr<int32_t>();
        params.lm_indexes       = lm_output_indexes.data_ptr<int32_t>();
        for (int32_t row = 0; row < params.rows; ++row) {
            params.accepted[row] = accepted_rows[offset + row].data_ptr<int32_t>();
            params.lengths[row]  = accepted_length_rows[offset + row].data_ptr<int32_t>();
            params.widths[row]   = static_cast<int32_t>(accepted_rows[offset + row].numel());
        }
        for (int32_t column = 1; column < tokens_per_batch; ++column) {
            params.columns[column] = token_columns[column].data_ptr<int32_t>();
        }
        const int blocks = (params.rows * tokens_per_batch + threads - 1) / threads;
        mtpSpecDecodeTokensMetadataFromStateKernel<<<blocks, threads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
}

void invokeMtpPrefillShiftAppendImpl(const torch::Tensor& combo_tokens_in,
                                     const torch::Tensor& input_lengths,
                                     const torch::Tensor& batch_offsets,
                                     const torch::Tensor& new_all_token_ids,
                                     torch::Tensor&       combo_tokens_out,
                                     const torch::Tensor* position_ids_in,
                                     torch::Tensor*       position_ids_out,
                                     int32_t              token_stride,
                                     cudaStream_t         stream) {
    const int64_t batch_size   = input_lengths.numel();
    const int64_t total_tokens = combo_tokens_in.numel();
    if (batch_size == 0 || total_tokens == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(batch_size <= INT_MAX && total_tokens <= INT_MAX,
                            "MTP Prefill token count exceeds int32 range");
    RTP_LLM_CHECK_WITH_INFO(token_stride > 0, "MTP Prefill sample stride must be positive");
    checkCudaI32Vector(combo_tokens_in, "combo_tokens_in", total_tokens);
    checkCudaI32Vector(combo_tokens_out, "combo_tokens_out", total_tokens);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(batch_offsets, "batch_offsets", batch_size);
    checkCudaI32Vector(new_all_token_ids, "new_all_token_ids", batch_size * token_stride);
    RTP_LLM_CHECK_WITH_INFO(combo_tokens_out.data_ptr<int32_t>() != combo_tokens_in.data_ptr<int32_t>(),
                            "MTP Prefill output must not alias input");
    RTP_LLM_CHECK_WITH_INFO((position_ids_in == nullptr) == (position_ids_out == nullptr),
                            "MTP Prefill position input and output must be provided together");
    if (position_ids_in != nullptr) {
        checkCudaI32Vector(*position_ids_in, "position_ids_in", total_tokens);
        checkCudaI32Vector(*position_ids_out, "position_ids_out", total_tokens);
        RTP_LLM_CHECK_WITH_INFO(position_ids_out->data_ptr<int32_t>() != position_ids_in->data_ptr<int32_t>(),
                                "MTP Prefill position output must not alias input");
    }
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_tokens + block_size - 1) / block_size);
    mtpPrefillShiftAppendKernel<<<grid_size, block_size, 0, stream>>>(
        combo_tokens_in.data_ptr<int32_t>(), input_lengths.data_ptr<int32_t>(),
        batch_offsets.data_ptr<int32_t>(), new_all_token_ids.data_ptr<int32_t>(),
        combo_tokens_out.data_ptr<int32_t>(),
        position_ids_in == nullptr ? nullptr : position_ids_in->data_ptr<int32_t>(),
        position_ids_out == nullptr ? nullptr : position_ids_out->data_ptr<int32_t>(),
        token_stride,
        static_cast<int32_t>(batch_size), static_cast<int32_t>(total_tokens));
}

void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream) {
    invokeMtpPrefillShiftAppendImpl(combo_tokens_in, input_lengths, batch_offsets,
                                    new_all_token_ids, combo_tokens_out, nullptr, nullptr,
                                    token_stride, stream);
}

void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 const torch::Tensor& position_ids_in,
                                 torch::Tensor&       position_ids_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream) {
    invokeMtpPrefillShiftAppendImpl(combo_tokens_in, input_lengths, batch_offsets,
                                    new_all_token_ids, combo_tokens_out, &position_ids_in,
                                    &position_ids_out, token_stride, stream);
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

}  // namespace rtp_llm
