#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_token_gather.h"
#include <algorithm>
#include <climits>

namespace rtp_llm {
namespace {
struct MtpNextRowsParams { MtpNextInputRow rows[128]; };
__global__ void mtpPrepareNextInputsKernel(MtpNextRowsParams input, int32_t* output,
                                           int32_t batch, int32_t offset, int32_t count) {
    const int row = threadIdx.x;
    if (row < count) {
        const int dst = offset + row;
        const int committed = *input.rows[row].committed_length;
        output[dst] = *input.rows[row].token;
        output[batch + dst] = committed;
        output[2 * batch + dst] = committed - 1;
        output[3 * batch + dst] = dst;
    }
}

__global__ void mtpDecodeAttentionMetadataKernel(
    const int32_t* lengths, int32_t* cu, int32_t* cu_kv, int32_t* decode_cu, int32_t* lengths_plus_one, int32_t count) {
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row <= count) {
        cu[row]        = 0;
        cu_kv[row]     = 0;
        decode_cu[row] = row;
    }
    if (row < count)
        lengths_plus_one[row] = lengths[row] + 1;
}

__global__ void mtpHeadInputsKernel(
    const at::BFloat16* hidden, const int32_t* indices, at::BFloat16* selected, float* head_input, int32_t width) {
    int32_t row = blockIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        int64_t source   = static_cast<int64_t>(indices[row]) * width + col;
        int64_t dest     = static_cast<int64_t>(row) * width + col;
        auto    value    = hidden[source];
        selected[dest]   = value;
        head_input[dest] = static_cast<float>(value);
    }
}

constexpr int kMaxMtpTokenRows = 128;
struct MtpAcceptedTokenRows {
    const int32_t* tokens[kMaxMtpTokenRows]  = {};
    const int32_t* lengths[kMaxMtpTokenRows] = {};
};

__global__ void mtpAcceptedTokenGatherKernel(MtpAcceptedTokenRows input, int32_t* output, int32_t count) {
    int32_t row = threadIdx.x;
    if (row < count)
        output[row] = input.tokens[row][input.lengths[row][0] - 1];
}
}  // namespace

bool tryMtpPrepareNextInputs(const std::vector<MtpNextInputRow>& rows,
                              const torch::TensorOptions& options,
                              torch::Tensor& packed,
                              cudaStream_t stream) {
    if (rows.empty() || rows.size() > INT_MAX / 4 || options.device().type() != torch::kCUDA
        || options.dtype() != torch::TensorOptions().dtype(torch::kInt32).dtype())
        return false;
    for (const auto& row : rows)
        if (!row.token || !row.committed_length)
            return false;
    const int32_t batch = static_cast<int32_t>(rows.size());
    packed = torch::empty({4, batch}, options);
    for (int32_t offset = 0; offset < batch; offset += 128) {
        const int32_t count = std::min(batch - offset, 128);
        MtpNextRowsParams params{};
        std::copy_n(rows.begin() + offset, count, params.rows);
        mtpPrepareNextInputsKernel<<<1, 128, 0, stream>>>(params, packed.data_ptr<int32_t>(), batch, offset, count);
    }
    return true;
}

bool tryMtpDecodeAttentionMetadata(const torch::Tensor& lengths,
                                   torch::Tensor&       cu,
                                   torch::Tensor&       cu_kv,
                                   torch::Tensor&       decode_cu,
                                   torch::Tensor&       lengths_plus_one,
                                   cudaStream_t         stream) {
    if (!lengths.is_cuda() || lengths.scalar_type() != torch::kInt32 || !lengths.is_contiguous() || lengths.dim() != 1
        || lengths.numel() > 65535)
        return false;
    const int32_t count = static_cast<int32_t>(lengths.numel());
    cu                  = torch::empty({count + 1}, lengths.options());
    cu_kv               = torch::empty_like(cu);
    decode_cu           = torch::empty_like(cu);
    lengths_plus_one    = torch::empty_like(lengths);
    mtpDecodeAttentionMetadataKernel<<<(count + 256) / 256, 256, 0, stream>>>(lengths.data_ptr<int32_t>(),
                                                                              cu.data_ptr<int32_t>(),
                                                                              cu_kv.data_ptr<int32_t>(),
                                                                              decode_cu.data_ptr<int32_t>(),
                                                                              lengths_plus_one.data_ptr<int32_t>(),
                                                                              count);
    return true;
}

bool tryMtpHeadInputs(const torch::Tensor& hidden,
                      const torch::Tensor& indices,
                      torch::Tensor&       selected_hidden,
                      torch::Tensor&       head_input,
                      cudaStream_t         stream) {
    if (!hidden.is_cuda() || hidden.scalar_type() != torch::kBFloat16 || hidden.dim() != 2 || !hidden.is_contiguous()
        || indices.device() != hidden.device() || indices.scalar_type() != torch::kInt32 || indices.dim() != 1
        || !indices.is_contiguous() || indices.numel() > 65535 || hidden.size(1) <= 0 || hidden.size(1) > 1048576)
        return false;
    int64_t rows    = indices.numel();
    int32_t width   = static_cast<int32_t>(hidden.size(1));
    selected_hidden = torch::empty({rows, width}, hidden.options());
    head_input      = torch::empty({rows, width}, hidden.options().dtype(torch::kFloat32));
    if (rows) {
        mtpHeadInputsKernel<<<dim3((width + 255) / 256, rows), 256, 0, stream>>>(
            hidden.data_ptr<at::BFloat16>(),
            indices.data_ptr<int32_t>(),
            selected_hidden.data_ptr<at::BFloat16>(),
            head_input.data_ptr<float>(),
            width);
    }
    return true;
}

bool tryMtpAcceptedTokenGather(const std::vector<torch::Tensor>& tokens,
                               const std::vector<torch::Tensor>& lengths,
                               torch::Tensor&                    output,
                               cudaStream_t                      stream) {
    if (tokens.empty() || tokens.size() > kMaxMtpTokenRows || lengths.size() != tokens.size())
        return false;
    const auto           device = tokens.front().device();
    MtpAcceptedTokenRows input;
    for (size_t row = 0; row < tokens.size(); ++row) {
        const auto& t = tokens[row];
        const auto& l = lengths[row];
        if (!t.is_cuda() || !l.is_cuda() || t.device() != device || l.device() != device
            || t.scalar_type() != torch::kInt32 || l.scalar_type() != torch::kInt32 || !t.is_contiguous()
            || !l.is_contiguous() || t.numel() == 0 || l.numel() != 1)
            return false;
        input.tokens[row]  = t.data_ptr<int32_t>();
        input.lengths[row] = l.data_ptr<int32_t>();
    }
    output = torch::empty({static_cast<int64_t>(tokens.size())}, tokens.front().options());
    // Pointer arrays are copied as kernel parameters, avoiding H2D pointer
    // tensors and the token/length cat, int64 conversion and gather chain.
    mtpAcceptedTokenGatherKernel<<<1, kMaxMtpTokenRows, 0, stream>>>(input, output.data_ptr<int32_t>(), tokens.size());
    return true;
}
}  // namespace rtp_llm
