#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rtp_llm {

namespace {

constexpr int kRowsPerLaunch    = 64;
constexpr int kPrepareBlockSize = 256;

constexpr int kBudgetMaxBatch  = 128;
constexpr int kBudgetMaxGamma  = 8;
constexpr int kBudgetSortItems = 4;
constexpr int kBudgetCapacity  = kPrepareBlockSize * kBudgetSortItems;

struct DSparkBudgetedVerifyPrepare {
    const float*   rates;
    int32_t        order[kBudgetMaxBatch];
    const int32_t *anchors, *proposals, *history, *bases, *prefixes;
    int32_t *      valid_lengths, *tokens, *query_lengths, *lm_indexes, *output_history, *positions;
    int32_t *      owned_proposals, *dense_tokens, *dense_positions, *sequence_lengths_plus_1;
    int64_t*       packed_to_dense;
    int32_t        batch, gamma, factor, budget;
    int64_t        anchor_stride, proposal_row_stride, proposal_col_stride;
    int64_t        history_row_stride, history_col_stride, base_row_stride, base_col_stride, prefix_stride;
};

static_assert(sizeof(DSparkBudgetedVerifyPrepare) < 4096, "DSpARK budget launch arguments must fit legacy CUDA limit");

// RTP's default nvcc flags flush subnormals. The planner must retain the
// original Torch FP32 products and their ordering, including underflow ties.
__device__ __forceinline__ float budgetMultiply(float left, float right) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
}

__device__ __forceinline__ float budgetCleanRate(float value) {
    const uint32_t bits      = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    // Bit comparisons avoid the TU's ftz=true affecting subnormal validity.
    return magnitude <= 0x3f800000u && ((bits & 0x80000000u) == 0 || magnitude == 0) ? value : 0.0f;
}

__device__ __forceinline__ int32_t budgetVerifyToken(const DSparkBudgetedVerifyPrepare& p, int32_t row, int32_t step) {
    return step == 0 ? p.anchors[row * p.anchor_stride] :
                       p.proposals[row * p.proposal_row_stride + (step - 1) * p.proposal_col_stride];
}

using BudgetSort = cub::BlockRadixSort<uint64_t, kPrepareBlockSize, kBudgetSortItems, uint32_t>;
using BudgetScan = cub::BlockScan<int32_t, kPrepareBlockSize>;
union BudgetScratch {
    BudgetSort::TempStorage sort;
    BudgetScan::TempStorage scan;
};

__global__ void mtpDSparkBudgetedVerifyPrepareKernel(DSparkBudgetedVerifyPrepare p) {
    __shared__ BudgetScratch scratch;
    __shared__ uint32_t      scores[kBudgetCapacity];
    __shared__ int32_t       lengths[kBudgetMaxBatch];
    const int32_t            row = threadIdx.x;
    if (row < p.batch) {
        lengths[row] = 0;
        if (p.budget > 0) {
            float products[kBudgetMaxGamma];
#pragma unroll
            for (int pos = 0; pos < kBudgetMaxGamma; ++pos) {
                products[pos] = pos < p.gamma ? budgetCleanRate(p.rates[row * p.gamma + pos]) : 1.0f;
            }
            // This is Torch2.11 ScanUtils.cuh's innermost Sklansky association,
            // restricted to the first eight positions. In particular p3 is
            // (r3*r2)*(r1*r0), rather than sequential multiplication.
#pragma unroll
            for (int span = 1; span < kBudgetMaxGamma; span <<= 1) {
#pragma unroll
                for (int pos = 0; pos < kBudgetMaxGamma; ++pos) {
                    if (pos & span) {
                        const int source = (pos & ~(span * 2 - 1)) + span - 1;
                        products[pos]    = budgetMultiply(products[pos], products[source]);
                    }
                }
            }
            uint32_t prefix_min = 0x3f800000u;
#pragma unroll
            for (int pos = 0; pos < kBudgetMaxGamma; ++pos) {
                if (pos < p.gamma) {
                    const uint32_t product = __float_as_uint(products[pos]);
                    if ((product & 0x7fffffffu) < (prefix_min & 0x7fffffffu)) {
                        prefix_min = product;
                    }
                    scores[row * p.gamma + pos] = prefix_min;
                }
            }
        }
    }
    __syncthreads();
    if (p.budget > 0) {
        uint64_t keys[kBudgetSortItems];
        uint32_t candidates[kBudgetSortItems];
#pragma unroll
        for (int item = 0; item < kBudgetSortItems; ++item) {
            const int32_t  stable_index = threadIdx.x * kBudgetSortItems + item;
            const bool     live         = stable_index < p.batch * p.gamma;
            const int32_t  source       = live ? p.order[stable_index / p.gamma] * p.gamma + stable_index % p.gamma : 0;
            const uint32_t score        = live && (scores[source] & 0x7fffffffu) != 0 ? scores[source] : 0;
            keys[item] =
                live ? (static_cast<uint64_t>(score) << 32) | (0xffffffffu - static_cast<uint32_t>(stable_index)) : 0;
            candidates[item] = source;
        }
        BudgetSort(scratch.sort).SortDescending(keys, candidates);
#pragma unroll
        for (int item = 0; item < kBudgetSortItems; ++item) {
            if (threadIdx.x * kBudgetSortItems + item < p.budget) {
                atomicAdd(lengths + candidates[item] / p.gamma, 1);
            }
        }
    }
    __syncthreads();
    const int32_t query_length = row < p.batch ? lengths[row] + 1 : 0;
    int32_t       query_start;
    BudgetScan(scratch.scan).ExclusiveSum(query_length, query_start);
    __syncthreads();
    if (row >= p.batch) {
        return;
    }
    p.valid_lengths[row] = lengths[row];
    p.query_lengths[row] = query_length;
    p.sequence_lengths_plus_1[row] =
        static_cast<int32_t>(static_cast<uint32_t>(p.prefixes[row * p.prefix_stride]) + 1u);
    const int32_t width = p.gamma + 1;
    for (int32_t pos = 0; pos < width; ++pos) {
        const int32_t token               = budgetVerifyToken(p, row, pos);
        p.dense_tokens[row * width + pos] = token;
        if (p.owned_proposals && pos != 0) {
            p.owned_proposals[row * p.gamma + pos - 1] = token;
        }
        for (int32_t factor = 0; factor < p.factor; ++factor) {
            const auto base = static_cast<uint32_t>(p.bases[row * p.base_row_stride + factor * p.base_col_stride]);
            p.dense_positions[(row * width + pos) * p.factor + factor] = static_cast<int32_t>(base + pos);
        }
        if (pos >= query_length) {
            continue;
        }
        const int32_t packed      = query_start + pos;
        p.tokens[packed]          = token;
        p.lm_indexes[packed]      = packed;
        p.packed_to_dense[packed] = static_cast<int64_t>(row) * width + pos;
        for (int32_t factor = 0; factor < p.factor; ++factor) {
            p.positions[packed * p.factor + factor] = p.dense_positions[(row * width + pos) * p.factor + factor];
        }
        if (p.output_history) {
            for (int32_t col = 0; col < 4; ++col) {
                p.output_history[packed * 4 + col] =
                    col <= pos ? budgetVerifyToken(p, row, pos - col) :
                                 p.history[row * p.history_row_stride + (col - pos) * p.history_col_stride];
            }
        }
    }
}

struct DSparkVerifyPrepare {
    const int32_t *anchors, *proposals, *history, *bases;
    int32_t *      tokens, *lengths, *lm_indexes, *output_history, *positions;
    int64_t        batch, width, factor, anchor_stride, proposal_row_stride, proposal_col_stride;
    int64_t        history_row_stride, history_col_stride, base_row_stride, base_col_stride;
    int32_t*       owned_proposals = nullptr;
};

__device__ int32_t dsparkVerifyToken(const DSparkVerifyPrepare& p, int64_t row, int64_t step) {
    return step == 0 ? p.anchors[row * p.anchor_stride] :
                       p.proposals[row * p.proposal_row_stride + (step - 1) * p.proposal_col_stride];
}

__global__ void mtpDSparkVerifyPrepareKernel(DSparkVerifyPrepare p, int64_t work_items) {
    const int64_t tokens = p.batch * p.width;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < work_items;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (i < p.batch) {
            p.lengths[i] = p.width;
        }
        if (i < tokens) {
            const auto step  = i % p.width;
            const auto token = dsparkVerifyToken(p, i / p.width, step);
            p.tokens[i]      = token;
            p.lm_indexes[i]  = i;
            if (p.owned_proposals && step != 0) {
                p.owned_proposals[(i / p.width) * (p.width - 1) + step - 1] = token;
            }
        }
        if (p.output_history && i < tokens * 4) {
            const int64_t token = i / 4, col = i % 4, row = token / p.width, step = token % p.width;
            p.output_history[i] = col <= step ?
                                      dsparkVerifyToken(p, row, step - col) :
                                      p.history[row * p.history_row_stride + (col - step) * p.history_col_stride];
        }
        if (p.positions && i < tokens * p.factor) {
            const int64_t token = i / p.factor, col = i % p.factor;
            p.positions[i] = p.bases[(token / p.width) * p.base_row_stride + col * p.base_col_stride] + token % p.width;
        }
    }
}

template<typename Token, typename Length>
__global__ void mtpAdvanceEngramTokenWindowsKernel(const Token*  anchor,
                                                   const Token*  tokens,
                                                   const Length* lengths,
                                                   Token*        output,
                                                   int64_t       count,
                                                   int64_t       width,
                                                   int64_t       anchor_row_stride,
                                                   int64_t       anchor_col_stride,
                                                   int64_t       token_row_stride,
                                                   int64_t       token_col_stride,
                                                   int64_t       length_stride) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = idx / 4, col = idx % 4;
        const int64_t accepted = lengths[row * length_stride];
        CUDA_KERNEL_ASSERT(accepted >= 0 && accepted <= width);
        output[idx] = col < accepted ? tokens[row * token_row_stride + (accepted - 1 - col) * token_col_stride] :
                                       anchor[row * anchor_row_stride + (col - accepted) * anchor_col_stride];
    }
}

template<typename Row>
struct PrepareRows {
    Row rows[kRowsPerLaunch];
};

struct PageSnapshotRow {
    const int32_t* source[2] = {};
    int64_t        stride[2] = {};
    int64_t        width[2]  = {};
};

struct PageSnapshotGather {
    PrepareRows<PageSnapshotRow> rows;
    const int32_t*               host[2];
    int32_t*                     output[2];
    int64_t                      groups[2], width[2], host_group_stride[2], host_row_stride[2];
    int64_t                      batch, offset;
};

struct EngramUpdate {
    const int32_t* anchor            = nullptr;
    const int32_t* tokens            = nullptr;
    int32_t*       output            = nullptr;
    int64_t        anchor_row_stride = 0, anchor_col_stride = 0;
    int64_t        token_row_stride = 0, token_col_stride = 0, width = 0;
};
static_assert(sizeof(PageSnapshotGather) < 4096, "Keep snapshot launch parameters below the CUDA 4 KiB limit");

__global__ void mtpGatherPageSnapshotsKernel(PageSnapshotGather params) {
    const int     table = blockIdx.z;
    const int64_t group = blockIdx.x;
    if (group >= params.groups[table]) {
        return;
    }
    const auto  row    = params.rows.rows[blockIdx.y];
    const auto  index  = params.offset + blockIdx.y;
    const auto* source = row.source[table];
    const auto* host =
        params.host[table] + group * params.host_group_stride[table] + index * params.host_row_stride[table];
    auto* output = params.output[table] + (group * params.batch + index) * params.width[table];
    for (int64_t col = threadIdx.x; col < params.width[table]; col += blockDim.x) {
        output[col] = source && col < row.width[table] ? source[group * row.stride[table] + col] : host[col];
    }
}

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
                                                  int32_t      batch_size,
                                                  EngramUpdate history) {
    const int32_t idx = threadIdx.x;
    if (idx < batch_size) {
        const auto    row      = rows.rows[idx];
        const int32_t previous = row.sequence_length != nullptr ? *row.sequence_length : row.host_sequence_length;
        const int32_t accepted = accept_len[idx];
        prev_seq_len[idx]      = previous;
        next_seq_len[idx]      = previous + accepted;
        hidden_idx[idx]        = static_cast<int64_t>(accepted) - 1;
        if (history.output) {
            CUDA_KERNEL_ASSERT(accepted >= 0 && accepted <= history.width);
            for (int col = 0; col < 4; ++col) {
                history.output[idx * 4 + col] =
                    col < accepted ?
                        history
                            .tokens[idx * history.token_row_stride + (accepted - 1 - col) * history.token_col_stride] :
                        history.anchor[idx * history.anchor_row_stride + (col - accepted) * history.anchor_col_stride];
            }
        }
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

MtpDSparkVerifyInputs tryMtpDSparkVerifyPrepare(const torch::Tensor& anchors,
                                                const torch::Tensor& proposals,
                                                const torch::Tensor& history,
                                                const torch::Tensor& position_bases,
                                                cudaStream_t         stream,
                                                bool                 copy_proposals) {
    auto eligible = [&](const torch::Tensor& t) {
        return t.defined() && t.is_cuda() && t.scalar_type() == torch::kInt32
               && t.get_device() == at::cuda::current_device();
    };
    if (!eligible(anchors) || anchors.dim() != 1 || !eligible(proposals) || proposals.dim() != 2
        || proposals.size(0) != anchors.numel() || proposals.size(1) >= std::numeric_limits<int32_t>::max()) {
        return {};
    }
    const int64_t batch = anchors.numel(), width = proposals.size(1) + 1;
    if (batch > std::numeric_limits<int32_t>::max() / width) {
        return {};
    }
    if (history.defined()
        && (!eligible(history) || history.dim() != 2 || history.size(0) != batch || history.size(1) != 4)) {
        return {};
    }
    const bool has_positions = position_bases.defined() && position_bases.numel() > 0;
    if (has_positions && (!eligible(position_bases) || position_bases.dim() != 2 || position_bases.size(0) != batch)) {
        return {};
    }
    const int64_t factor = has_positions ? position_bases.size(1) : 0;
    const int64_t count  = batch * width;
    if (factor > std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, count)) {
        return {};
    }
    MtpDSparkVerifyInputs out;
    out.tokens        = torch::empty({count}, anchors.options());
    out.input_lengths = torch::empty({batch}, anchors.options());
    out.lm_indexes    = torch::empty({count}, anchors.options());
    if (copy_proposals) {
        out.proposals = torch::empty({batch, width - 1}, anchors.options());
    }
    if (history.defined()) {
        out.history = torch::empty({count, 4}, anchors.options());
    }
    if (has_positions) {
        out.positions = torch::empty({count * factor}, anchors.options());
    }
    if (batch == 0) {
        return out;
    }
    DSparkVerifyPrepare params{anchors.data_ptr<int32_t>(),
                               proposals.data_ptr<int32_t>(),
                               history.defined() ? history.data_ptr<int32_t>() : nullptr,
                               has_positions ? position_bases.data_ptr<int32_t>() : nullptr,
                               out.tokens.data_ptr<int32_t>(),
                               out.input_lengths.data_ptr<int32_t>(),
                               out.lm_indexes.data_ptr<int32_t>(),
                               out.history.defined() ? out.history.data_ptr<int32_t>() : nullptr,
                               out.positions.defined() ? out.positions.data_ptr<int32_t>() : nullptr,
                               batch,
                               width,
                               factor,
                               anchors.stride(0),
                               proposals.stride(0),
                               proposals.stride(1),
                               history.defined() ? history.stride(0) : 0,
                               history.defined() ? history.stride(1) : 0,
                               has_positions ? position_bases.stride(0) : 0,
                               has_positions ? position_bases.stride(1) : 0};
    params.owned_proposals   = out.proposals.defined() ? out.proposals.data_ptr<int32_t>() : nullptr;
    const int64_t work_items = count * std::max<int64_t>(history.defined() ? 4 : 1, factor);
    mtpDSparkVerifyPrepareKernel<<<prepareGridSize(work_items), kPrepareBlockSize, 0, stream>>>(params, work_items);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto cuda_stream = c10::cuda::getStreamFromExternal(stream, anchors.get_device());
    for (const auto& t : {anchors, proposals, history, position_bases}) {
        if (t.defined() && t.is_cuda()) {
            c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), cuda_stream);
        }
    }
    return out;
}

MtpDSparkBudgetedVerifyInputs tryMtpDSparkBudgetedVerifyPrepare(const torch::Tensor&        rates,
                                                                const std::vector<int64_t>& stable_order_host,
                                                                const torch::Tensor&        anchors,
                                                                const torch::Tensor&        proposals,
                                                                const torch::Tensor&        history,
                                                                const torch::Tensor&        position_bases,
                                                                const torch::Tensor&        prefix_lengths,
                                                                int64_t                     total_draft_budget,
                                                                cudaStream_t                stream,
                                                                bool                        copy_proposals,
                                                                bool                        allow_full_budget) {
    if (!rates.defined() || !rates.is_cuda() || rates.scalar_type() != torch::kFloat32 || rates.dim() != 2
        || !rates.is_contiguous()) {
        return {};
    }
    const int64_t batch = rates.size(0), gamma = rates.size(1);
    if (batch < 2 || batch > kBudgetMaxBatch || gamma < 1 || gamma > kBudgetMaxGamma || batch * gamma > kBudgetCapacity
        || total_draft_budget < 0 || total_draft_budget > batch * gamma
        || (total_draft_budget == batch * gamma && !allow_full_budget)) {
        return {};
    }
    auto eligible = [&](const torch::Tensor& tensor) {
        return tensor.defined() && tensor.is_cuda() && tensor.scalar_type() == torch::kInt32
               && tensor.device() == rates.device();
    };
    if (!eligible(anchors) || anchors.dim() != 1 || anchors.numel() != batch || !eligible(proposals)
        || proposals.dim() != 2 || proposals.size(0) != batch || proposals.size(1) != gamma || !eligible(prefix_lengths)
        || prefix_lengths.dim() != 1 || prefix_lengths.numel() != batch) {
        return {};
    }
    if (history.defined()
        && (!eligible(history) || history.dim() != 2 || history.size(0) != batch || history.size(1) != 4)) {
        return {};
    }
    const bool has_positions = position_bases.defined() && position_bases.numel() > 0;
    if (has_positions
        && (!eligible(position_bases) || position_bases.dim() != 2 || position_bases.size(0) != batch
            || position_bases.size(1) > 16)) {
        return {};
    }
    TORCH_CHECK(stable_order_host.size() == static_cast<size_t>(batch),
                "DSpARK budgeted verify requires one stable identity index per batch row");
    bool seen[kBudgetMaxBatch] = {};
    for (auto row : stable_order_host) {
        TORCH_CHECK(row >= 0 && row < batch && !seen[row],
                    "DSpARK budgeted verify stable identity indices must be a permutation");
        seen[row] = true;
    }
    c10::cuda::CUDAGuard          guard(rates.device());
    const int64_t                 factor = has_positions ? position_bases.size(1) : 0;
    const int64_t                 width = gamma + 1, packed_rows = batch + total_draft_budget;
    const auto                    options = anchors.options();
    MtpDSparkBudgetedVerifyInputs out;
    out.valid_draft_lengths     = torch::empty({batch}, options);
    out.packed_to_dense         = torch::empty({packed_rows}, options.dtype(torch::kInt64));
    out.dense_tokens            = torch::empty({batch, width}, options);
    out.sequence_lengths_plus_1 = torch::empty({batch}, options);
    out.packed.tokens           = torch::empty({packed_rows}, options);
    out.packed.input_lengths    = torch::empty({batch}, options);
    out.packed.lm_indexes       = torch::empty({packed_rows}, options);
    if (copy_proposals) {
        out.packed.proposals = torch::empty({batch, gamma}, options);
    }
    if (history.defined()) {
        out.packed.history = torch::empty({packed_rows, 4}, options);
    }
    if (has_positions) {
        out.packed.positions = torch::empty({packed_rows * factor}, options);
        out.dense_positions  = torch::empty({batch * width * factor}, options);
    }
    DSparkBudgetedVerifyPrepare params{};
    params.rates = rates.data_ptr<float>();
    for (int64_t row = 0; row < batch; ++row) {
        params.order[row] = static_cast<int32_t>(stable_order_host[row]);
    }
    params.anchors         = anchors.data_ptr<int32_t>();
    params.proposals       = proposals.data_ptr<int32_t>();
    params.history         = history.defined() ? history.data_ptr<int32_t>() : nullptr;
    params.bases           = has_positions ? position_bases.data_ptr<int32_t>() : nullptr;
    params.prefixes        = prefix_lengths.data_ptr<int32_t>();
    params.valid_lengths   = out.valid_draft_lengths.data_ptr<int32_t>();
    params.tokens          = out.packed.tokens.data_ptr<int32_t>();
    params.query_lengths   = out.packed.input_lengths.data_ptr<int32_t>();
    params.lm_indexes      = out.packed.lm_indexes.data_ptr<int32_t>();
    params.output_history  = out.packed.history.defined() ? out.packed.history.data_ptr<int32_t>() : nullptr;
    params.positions       = out.packed.positions.defined() ? out.packed.positions.data_ptr<int32_t>() : nullptr;
    params.owned_proposals = out.packed.proposals.defined() ? out.packed.proposals.data_ptr<int32_t>() : nullptr;
    params.dense_tokens    = out.dense_tokens.data_ptr<int32_t>();
    params.dense_positions = out.dense_positions.defined() ? out.dense_positions.data_ptr<int32_t>() : nullptr;
    params.sequence_lengths_plus_1 = out.sequence_lengths_plus_1.data_ptr<int32_t>();
    params.packed_to_dense         = out.packed_to_dense.data_ptr<int64_t>();
    params.batch                   = static_cast<int32_t>(batch);
    params.gamma                   = static_cast<int32_t>(gamma);
    params.factor                  = static_cast<int32_t>(factor);
    params.budget                  = static_cast<int32_t>(total_draft_budget);
    params.anchor_stride           = anchors.stride(0);
    params.proposal_row_stride     = proposals.stride(0);
    params.proposal_col_stride     = proposals.stride(1);
    params.history_row_stride      = history.defined() ? history.stride(0) : 0;
    params.history_col_stride      = history.defined() ? history.stride(1) : 0;
    params.base_row_stride         = has_positions ? position_bases.stride(0) : 0;
    params.base_col_stride         = has_positions ? position_bases.stride(1) : 0;
    params.prefix_stride           = prefix_lengths.stride(0);
    mtpDSparkBudgetedVerifyPrepareKernel<<<1, kPrepareBlockSize, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto cuda_stream = c10::cuda::getStreamFromExternal(stream, anchors.get_device());
    for (const auto& source : {rates, anchors, proposals, history, position_bases, prefix_lengths}) {
        if (source.defined() && source.is_cuda()) {
            c10::cuda::CUDACachingAllocator::recordStream(source.storage().data_ptr(), cuda_stream);
        }
    }
    return out;
}

bool tryMtpGatherPageSnapshots(const torch::Tensor&              physical_host,
                               const torch::Tensor&              kernel_host,
                               const std::vector<torch::Tensor>& physical_snapshots,
                               const std::vector<torch::Tensor>& kernel_snapshots,
                               torch::Tensor&                    physical_output,
                               torch::Tensor&                    kernel_output,
                               cudaStream_t                      stream) {
    const torch::Tensor               hosts[2]     = {physical_host, kernel_host};
    const std::vector<torch::Tensor>* snapshots[2] = {&physical_snapshots, &kernel_snapshots};
    for (int table = 0; table < 2; ++table) {
        const auto& host = hosts[table];
        if (!host.defined() || !host.device().is_cpu() || (host.numel() > 0 && !host.is_pinned())
            || host.scalar_type() != torch::kInt32 || host.dim() != 3 || host.stride(2) != 1
            || snapshots[table]->size() != static_cast<size_t>(host.size(1))) {
            return false;
        }
    }
    if (physical_host.size(1) != kernel_host.size(1)) {
        return false;
    }
    const auto device = c10::cuda::current_device();
    for (int table = 0; table < 2; ++table) {
        for (const auto& snapshot : *snapshots[table]) {
            if (snapshot.defined()
                && (!snapshot.is_cuda() || snapshot.get_device() != device || snapshot.scalar_type() != torch::kInt32
                    || snapshot.dim() != 3 || snapshot.size(0) != hosts[table].size(0) || snapshot.size(1) != 1
                    || snapshot.size(2) > hosts[table].size(2) || snapshot.stride(2) != 1)) {
                return false;
            }
        }
    }
    torch::Tensor outputs[2];
    auto          storage = torch::empty({physical_host.numel() + kernel_host.numel()},
                                physical_host.options().device(torch::kCUDA).pinned_memory(false));
    outputs[0]            = storage.narrow(0, 0, physical_host.numel()).view(physical_host.sizes());
    outputs[1]            = storage.narrow(0, physical_host.numel(), kernel_host.numel()).view(kernel_host.sizes());
    PageSnapshotGather params{};
    params.batch = physical_host.size(1);
    for (int table = 0; table < 2; ++table) {
        const auto& host                = hosts[table];
        params.host[table]              = host.data_ptr<int32_t>();
        params.output[table]            = outputs[table].data_ptr<int32_t>();
        params.groups[table]            = host.size(0);
        params.width[table]             = host.size(2);
        params.host_group_stride[table] = host.stride(0);
        params.host_row_stride[table]   = host.stride(1);
        for (const auto& snapshot : *snapshots[table]) {
            if (snapshot.defined()) {
                c10::cuda::CUDACachingAllocator::recordStream(snapshot.storage().data_ptr(),
                                                              at::cuda::getCurrentCUDAStream());
            }
        }
    }
    const auto groups = std::max(params.groups[0], params.groups[1]);
    if (groups > 0 && (params.width[0] > 0 || params.width[1] > 0)) {
        for (int64_t offset = 0; offset < params.batch; offset += kRowsPerLaunch) {
            const auto count = std::min<int64_t>(kRowsPerLaunch, params.batch - offset);
            params.offset    = offset;
            for (int64_t row = 0; row < count; ++row) {
                for (int table = 0; table < 2; ++table) {
                    const auto& snapshot                = (*snapshots[table])[offset + row];
                    params.rows.rows[row].source[table] = snapshot.defined() ? snapshot.data_ptr<int32_t>() : nullptr;
                    params.rows.rows[row].width[table]  = snapshot.defined() ? snapshot.size(2) : 0;
                    params.rows.rows[row].stride[table] = snapshot.defined() ? snapshot.stride(0) : 0;
                }
            }
            mtpGatherPageSnapshotsKernel<<<dim3(groups, count, 2), kPrepareBlockSize, 0, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        for (const auto& host : hosts) {
            if (host.numel()) {
                at::cuda::CachingHostAllocator_recordEvent(
                    host.data_ptr(), host.storage().data_ptr().get_context(), at::cuda::getCurrentCUDAStream());
            }
        }
    }
    physical_output = std::move(outputs[0]);
    kernel_output   = std::move(outputs[1]);
    return true;
}

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
                                       cudaStream_t                       stream,
                                       const torch::Tensor&               engram_anchor,
                                       const torch::Tensor&               accepted_tokens,
                                       const torch::Tensor&               engram_output) {
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
    EngramUpdate               history;
    if (engram_output.defined()) {
        RTP_LLM_CHECK_WITH_INFO(
            engram_anchor.defined() && accepted_tokens.defined() && engram_anchor.device() == accept_len.device()
                && accepted_tokens.device() == accept_len.device() && engram_output.device() == accept_len.device()
                && engram_anchor.scalar_type() == torch::kInt32 && accepted_tokens.scalar_type() == torch::kInt32
                && engram_output.scalar_type() == torch::kInt32 && engram_anchor.dim() == 2
                && engram_anchor.size(0) == batch_size && engram_anchor.size(1) == 4 && accepted_tokens.dim() == 2
                && accepted_tokens.size(0) == batch_size && engram_output.dim() == 2
                && engram_output.size(0) == batch_size && engram_output.size(1) == 4 && engram_output.is_contiguous(),
            "dispatch Engram tensors must be CUDA int32 [B,4]/[B,W]/[B,4]");
        history = {engram_anchor.data_ptr<int32_t>(),
                   accepted_tokens.data_ptr<int32_t>(),
                   engram_output.data_ptr<int32_t>(),
                   engram_anchor.stride(0),
                   engram_anchor.stride(1),
                   accepted_tokens.stride(0),
                   accepted_tokens.stride(1),
                   accepted_tokens.size(1)};
    }
    for (int64_t offset = 0; offset < batch_size; offset += kRowsPerLaunch) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(kRowsPerLaunch, batch_size - offset));
        PrepareRows<MtpSequenceRow> params{};
        std::copy_n(rows.data() + offset, count, params.rows);
        mtpDispatchStatePrepareRowsKernel<<<1, kRowsPerLaunch, 0, stream>>>(params,
                                                                            accept_len.data_ptr<int32_t>() + offset,
                                                                            prev_seq_len.data_ptr<int32_t>() + offset,
                                                                            next_seq_len.data_ptr<int32_t>() + offset,
                                                                            hidden_idx.data_ptr<int64_t>() + offset,
                                                                            count,
                                                                            history);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (history.output) {
            history.anchor += count * history.anchor_row_stride;
            history.tokens += count * history.token_row_stride;
            history.output += count * 4;
        }
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

__global__ void mtpSpecSamplerTokenMatrixKernel(const int32_t* __restrict__ history,
                                                const int32_t* __restrict__ draft,
                                                const int64_t* __restrict__ base_lengths,
                                                int32_t* __restrict__ output,
                                                int64_t work_items,
                                                int64_t width,
                                                int64_t rows,
                                                int64_t gamma) {
    const int64_t row_stride = width * rows;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < work_items;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t column = i % width;
        const int64_t batch  = i / row_stride;
        const int64_t base   = base_lengths[batch];
        int32_t       value  = 0;
        if (column < base) {
            value = history[batch * width + column];
        } else if (column < base + gamma) {
            value = draft[batch * gamma + (column - base)];
        }
        output[i] = value;
    }
}

void invokeMtpSpecSamplerTokenMatrix(const torch::Tensor& history,
                                     const torch::Tensor& draft,
                                     const torch::Tensor& base_lengths,
                                     torch::Tensor&       output,
                                     int64_t              gamma,
                                     cudaStream_t         stream) {
    RTP_LLM_CHECK_WITH_INFO(history.defined() && history.is_cuda() && history.scalar_type() == torch::kInt32
                                && history.dim() == 2 && history.is_contiguous(),
                            "sampler history must be contiguous CUDA int32 [B, width]");
    RTP_LLM_CHECK_WITH_INFO(draft.defined() && draft.is_cuda() && draft.scalar_type() == torch::kInt32
                                && draft.dim() == 2 && draft.is_contiguous() && draft.size(1) == gamma,
                            "sampler draft tokens must be contiguous CUDA int32 [B, gamma]");
    RTP_LLM_CHECK_WITH_INFO(base_lengths.defined() && base_lengths.is_cuda()
                                && base_lengths.scalar_type() == torch::kInt64 && base_lengths.dim() == 1
                                && base_lengths.is_contiguous() && base_lengths.numel() == history.size(0),
                            "sampler base lengths must be contiguous CUDA int64 [B]");
    RTP_LLM_CHECK_WITH_INFO(output.defined() && output.is_cuda() && output.scalar_type() == torch::kInt32
                                && output.dim() == 3 && output.is_contiguous() && output.size(0) == history.size(0)
                                && output.size(2) == history.size(1) && output.size(1) > 0,
                            "sampler token matrix must be contiguous CUDA int32 [B, rows, width]");
    RTP_LLM_CHECK_WITH_INFO(history.device() == output.device() && draft.device() == output.device()
                                && base_lengths.device() == output.device(),
                            "sampler token matrix tensors must share one CUDA device");
    if (output.numel() == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(output.device());
    mtpSpecSamplerTokenMatrixKernel<<<prepareGridSize(output.numel()), kPrepareBlockSize, 0, stream>>>(
        history.data_ptr<int32_t>(),
        draft.data_ptr<int32_t>(),
        base_lengths.data_ptr<int64_t>(),
        output.data_ptr<int32_t>(),
        output.numel(),
        history.size(1),
        output.size(1),
        gamma);
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

torch::Tensor tryMtpAdvanceEngramTokenWindows(const torch::Tensor& anchor,
                                              const torch::Tensor& tokens,
                                              const torch::Tensor& lengths) {
    if (!anchor.is_cuda() || tokens.device() != anchor.device() || lengths.device() != anchor.device()
        || anchor.dim() != 2 || anchor.size(1) != 4 || tokens.dim() != 2 || tokens.size(0) != anchor.size(0)
        || lengths.dim() != 1 || lengths.numel() != anchor.size(0) || tokens.scalar_type() != anchor.scalar_type()
        || (anchor.scalar_type() != torch::kInt32 && anchor.scalar_type() != torch::kInt64)
        || (lengths.scalar_type() != torch::kInt32 && lengths.scalar_type() != torch::kInt64)) {
        return {};
    }
    const c10::cuda::CUDAGuard device_guard(anchor.device());
    auto                       output = torch::empty({anchor.size(0), 4}, anchor.options());
    if (output.numel() == 0) {
        return output;
    }
    AT_DISPATCH_INDEX_TYPES(anchor.scalar_type(), "mtp_advance_engram", [&] {
        using Token = index_t;
        AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "mtp_advance_engram_lengths", [&] {
            mtpAdvanceEngramTokenWindowsKernel<Token, index_t>
                <<<prepareGridSize(output.numel()), kPrepareBlockSize, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    anchor.data_ptr<Token>(),
                    tokens.data_ptr<Token>(),
                    lengths.data_ptr<index_t>(),
                    output.data_ptr<Token>(),
                    output.numel(),
                    tokens.size(1),
                    anchor.stride(0),
                    anchor.stride(1),
                    tokens.stride(0),
                    tokens.stride(1),
                    lengths.stride(0));
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
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
