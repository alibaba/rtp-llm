#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rtp_llm {

namespace {

constexpr int kRowsPerLaunch    = 64;
constexpr int kPrepareBlockSize = 256;

template<typename Row>
struct PrepareRows {
    Row rows[kRowsPerLaunch];
};

static_assert(std::is_trivially_copyable<MtpRoundHeadRow>::value, "CUDA launch rows must be trivially copyable");
static_assert(sizeof(PrepareRows<MtpRoundHeadRow>) + 64 <= 4096, "Keep launch parameters below the CUDA 4 KiB limit");

int prepareGridSize(int64_t work_items) {
    return static_cast<int>(std::min<int64_t>((work_items + kPrepareBlockSize - 1) / kPrepareBlockSize, 65535));
}

__global__ void mtpDSparkRoundHeadPrepareKernel(PrepareRows<MtpRoundHeadRow> rows,
                                                int32_t* __restrict__ anchors,
                                                int32_t* __restrict__ committed_ends,
                                                int32_t* __restrict__ combo_tokens,
                                                int32_t batch_size,
                                                int32_t width,
                                                int32_t mask_token) {
    const int64_t count = static_cast<int64_t>(batch_size) * width;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int32_t row_idx = static_cast<int32_t>(idx / width);
        if (idx % width != 0) {
            combo_tokens[idx] = mask_token;
            continue;
        }
        const auto row    = rows.rows[row_idx];
        int32_t    anchor = row.anchor;
        if (row.tokens != nullptr) {
            const int64_t token_idx = static_cast<int64_t>(*row.accept_len) - 1;
            CUDA_KERNEL_ASSERT(token_idx >= 0 && token_idx < row.token_count);
            anchor = row.tokens[token_idx * row.token_stride];
        }
        const int32_t end       = row.sequence != nullptr ? *row.sequence + row.sequence_bias : row.sequence_value;
        anchors[row_idx]        = anchor;
        committed_ends[row_idx] = end;
        combo_tokens[idx]       = anchor;
    }
}

__global__ void mtpDispatchStatePrepareRowsKernel(PrepareRows<MtpSequenceRow> rows,
                                                  const int32_t* __restrict__ accept_len,
                                                  int32_t* __restrict__ prev_seq_len,
                                                  int32_t* __restrict__ next_seq_len,
                                                  int64_t* __restrict__ hidden_idx,
                                                  int32_t batch_size) {
    const int32_t idx = threadIdx.x;
    if (idx < batch_size) {
        const auto    row      = rows.rows[idx];
        const int32_t previous = row.sequence_length != nullptr ? *row.sequence_length : row.host_sequence_length;
        const int32_t accepted = accept_len[idx];
        prev_seq_len[idx]      = previous;
        next_seq_len[idx]      = previous + accepted;
        hidden_idx[idx]        = static_cast<int64_t>(accepted) - 1;
    }
}

__global__ void mtpPositionGatherKernel(PrepareRows<const int32_t*> rows,
                                        int32_t* __restrict__ bases,
                                        int32_t batch_size,
                                        int64_t factor) {
    const int64_t count = batch_size * factor;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        bases[idx] = rows.rows[idx / factor][idx % factor];
    }
}

__global__ void mtpPositionExpandKernel(const int32_t* __restrict__ bases,
                                        int32_t* __restrict__ output,
                                        int64_t count,
                                        int64_t factor,
                                        int64_t width,
                                        int64_t row_stride) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row_step = idx / factor;
        output[idx] = bases[(row_step / width) * row_stride + idx % factor] + static_cast<int32_t>(row_step % width);
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

void checkCudaI32Vector(const torch::Tensor& tensor, const char* name, int64_t batch_size) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(
        tensor.numel() >= batch_size, "%s numel %ld is smaller than batch_size %ld", name, tensor.numel(), batch_size);
}

void checkCudaVector(const torch::Tensor& tensor, const char* name, int64_t batch_size, torch::ScalarType dtype) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined() && tensor.is_cuda(), "%s must be a CUDA tensor", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == dtype, "%s has an unexpected dtype", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == 1 && tensor.numel() == batch_size && tensor.is_contiguous(),
                            "%s must be a contiguous vector of length %ld",
                            name,
                            batch_size);
}

}  // namespace

void invokeMtpDSparkRoundHeadPrepare(const std::vector<MtpRoundHeadRow>& rows,
                                     torch::Tensor&                      anchors,
                                     torch::Tensor&                      committed_ends,
                                     torch::Tensor&                      combo_tokens,
                                     int32_t                             tokens_per_batch,
                                     int32_t                             mask_token,
                                     cudaStream_t                        stream) {
    const int64_t batch_size = static_cast<int64_t>(rows.size());
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch > 0, "tokens_per_batch must be positive");
    RTP_LLM_CHECK_WITH_INFO(batch_size <= std::numeric_limits<int64_t>::max() / tokens_per_batch,
                            "DSpARK combo size overflows int64");
    checkCudaVector(anchors, "anchors", batch_size, torch::kInt32);
    checkCudaVector(committed_ends, "committed_ends", batch_size, torch::kInt32);
    checkCudaI32Vector(combo_tokens, "combo_tokens", batch_size * tokens_per_batch);
    RTP_LLM_CHECK_WITH_INFO(combo_tokens.numel() == batch_size * tokens_per_batch
                                && (combo_tokens.dim() == 1
                                    || (combo_tokens.dim() == 2 && combo_tokens.size(0) == batch_size
                                        && combo_tokens.size(1) == tokens_per_batch)),
                            "combo_tokens must be [B * W] or [B, W]");
    RTP_LLM_CHECK_WITH_INFO(anchors.device() == committed_ends.device() && anchors.device() == combo_tokens.device(),
                            "DSpARK outputs must be on the same CUDA device");
    for (const auto& row : rows) {
        RTP_LLM_CHECK_WITH_INFO((row.tokens == nullptr) == (row.accept_len == nullptr),
                                "tokens and accept_len must both be present or both absent");
        RTP_LLM_CHECK_WITH_INFO(row.tokens == nullptr || (row.token_stride >= 0 && row.token_count > 0),
                                "device token row requires a nonnegative stride and positive token_count");
    }
    if (batch_size == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(anchors.device());
    for (int64_t offset = 0; offset < batch_size; offset += kRowsPerLaunch) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(kRowsPerLaunch, batch_size - offset));
        PrepareRows<MtpRoundHeadRow> params{};
        std::copy_n(rows.data() + offset, count, params.rows);
        mtpDSparkRoundHeadPrepareKernel<<<prepareGridSize(static_cast<int64_t>(count) * tokens_per_batch),
                                          kPrepareBlockSize,
                                          0,
                                          stream>>>(params,
                                                    anchors.data_ptr<int32_t>() + offset,
                                                    committed_ends.data_ptr<int32_t>() + offset,
                                                    combo_tokens.data_ptr<int32_t>() + offset * tokens_per_batch,
                                                    count,
                                                    tokens_per_batch,
                                                    mask_token);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void invokeMtpDispatchStatePrepareRows(const std::vector<MtpSequenceRow>& rows,
                                       const torch::Tensor&               accept_len,
                                       torch::Tensor&                     prev_seq_len,
                                       torch::Tensor&                     next_seq_len,
                                       torch::Tensor&                     hidden_idx,
                                       cudaStream_t                       stream) {
    const int64_t batch_size = static_cast<int64_t>(rows.size());
    checkCudaVector(accept_len, "accept_len", batch_size, torch::kInt32);
    checkCudaVector(prev_seq_len, "prev_seq_len", batch_size, torch::kInt32);
    checkCudaVector(next_seq_len, "next_seq_len", batch_size, torch::kInt32);
    checkCudaVector(hidden_idx, "hidden_idx", batch_size, torch::kInt64);
    RTP_LLM_CHECK_WITH_INFO(accept_len.device() == prev_seq_len.device() && accept_len.device() == next_seq_len.device()
                                && accept_len.device() == hidden_idx.device(),
                            "dispatch tensors must be on the same CUDA device");
    if (batch_size == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(accept_len.device());
    for (int64_t offset = 0; offset < batch_size; offset += kRowsPerLaunch) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(kRowsPerLaunch, batch_size - offset));
        PrepareRows<MtpSequenceRow> params{};
        std::copy_n(rows.data() + offset, count, params.rows);
        mtpDispatchStatePrepareRowsKernel<<<1, kRowsPerLaunch, 0, stream>>>(params,
                                                                            accept_len.data_ptr<int32_t>() + offset,
                                                                            prev_seq_len.data_ptr<int32_t>() + offset,
                                                                            next_seq_len.data_ptr<int32_t>() + offset,
                                                                            hidden_idx.data_ptr<int64_t>() + offset,
                                                                            count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void invokeMtpPositionGather(const std::vector<const int32_t*>& rows, torch::Tensor& bases, cudaStream_t stream) {
    const int64_t batch_size = static_cast<int64_t>(rows.size());
    checkCudaI32Vector(bases, "position bases", 0);
    RTP_LLM_CHECK_WITH_INFO(bases.dim() == 2 && bases.size(0) == batch_size && bases.size(1) > 0,
                            "position bases must be [B, F] with F > 0");
    for (const auto* row : rows) {
        RTP_LLM_CHECK_WITH_INFO(row != nullptr, "position source row must be present");
    }
    if (batch_size == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(bases.device());
    const int64_t              factor = bases.size(1);
    for (int64_t offset = 0; offset < batch_size; offset += kRowsPerLaunch) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(kRowsPerLaunch, batch_size - offset));
        PrepareRows<const int32_t*> params{};
        std::copy_n(rows.data() + offset, count, params.rows);
        mtpPositionGatherKernel<<<prepareGridSize(count * factor), kPrepareBlockSize, 0, stream>>>(
            params, bases.data_ptr<int32_t>() + offset * factor, count, factor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void invokeMtpPositionExpand(const torch::Tensor& bases, torch::Tensor& output, int64_t width, cudaStream_t stream) {
    RTP_LLM_CHECK_WITH_INFO(bases.defined() && bases.is_cuda() && bases.scalar_type() == torch::kInt32,
                            "position bases must be CUDA int32");
    RTP_LLM_CHECK_WITH_INFO(bases.dim() == 2 && bases.size(1) > 0 && bases.stride(1) == 1 && bases.stride(0) >= 0,
                            "position bases must be [B, F] with contiguous factors");
    RTP_LLM_CHECK_WITH_INFO(width > 0 && width <= std::numeric_limits<int32_t>::max(),
                            "position width must be positive and fit int32");
    RTP_LLM_CHECK_WITH_INFO(bases.numel() <= std::numeric_limits<int64_t>::max() / width,
                            "expanded position size overflows int64");
    checkCudaVector(output, "expanded positions", bases.numel() * width, torch::kInt32);
    RTP_LLM_CHECK_WITH_INFO(output.device() == bases.device(), "position tensors must be on the same CUDA device");
    if (output.numel() == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(bases.device());
    mtpPositionExpandKernel<<<prepareGridSize(output.numel()), kPrepareBlockSize, 0, stream>>>(
        bases.data_ptr<int32_t>(), output.data_ptr<int32_t>(), output.numel(), bases.size(1), width, bases.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

}  // namespace rtp_llm
