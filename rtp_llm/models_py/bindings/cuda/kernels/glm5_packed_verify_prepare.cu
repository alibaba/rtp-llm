#include "rtp_llm/models_py/bindings/cuda/kernels/glm5_packed_verify_prepare.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace rtp_llm {
namespace {

struct IntegerVector {
    const void* input;
    void* output;
    bool wide;
};

__device__ int64_t readInteger(const void* values, int64_t index, bool wide) {
    return wide ? static_cast<const int64_t*>(values)[index] : static_cast<const int32_t*>(values)[index];
}

__device__ void writeInteger(void* values, int64_t index, int64_t value, bool wide) {
    if (wide) {
        static_cast<int64_t*>(values)[index] = value;
    } else {
        static_cast<int32_t*>(values)[index] = static_cast<int32_t>(value);
    }
}

__global__ void packedVerifyPrepare(IntegerVector tokens,
                                    IntegerVector positions,
                                    IntegerVector types,
                                    IntegerVector caps,
                                    const void* rows,
                                    bool wide_rows,
                                    void* output_indexes,
                                    int64_t dense_count,
                                    int64_t packed_count,
                                    int64_t batch,
                                    int64_t rows_per_request) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < packed_count || i < batch;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        if (i < batch) {
            const int64_t cap = readInteger(caps.input, i, caps.wide);
            const bool valid = cap >= 0 && cap < rows_per_request
                               && (caps.wide || cap < INT32_MAX);
            CUDA_KERNEL_ASSERT(valid);
            if (valid) writeInteger(caps.output, i, cap + 1, caps.wide);
        }
        if (i < packed_count) {
            const int64_t row = readInteger(rows, i, wide_rows);
            const bool valid = row >= 0 && row < dense_count;
            CUDA_KERNEL_ASSERT(valid);
            if (!valid) continue;
            writeInteger(tokens.output, i, readInteger(tokens.input, row, tokens.wide), tokens.wide);
            writeInteger(output_indexes, i, i, caps.wide);
            if (positions.input) {
                writeInteger(positions.output, i, readInteger(positions.input, row, positions.wide), positions.wide);
            }
            if (types.input) {
                writeInteger(types.output, i, readInteger(types.input, row, types.wide), types.wide);
            }
        }
    }
}

bool supportedVector(const torch::Tensor& tensor, int device) {
    return tensor.defined() && tensor.is_cuda() && tensor.get_device() == device && tensor.dim() == 1
           && tensor.is_contiguous() && (tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64);
}

IntegerVector vectorPair(const torch::Tensor& input, const torch::Tensor& output) {
    if (!input.defined()) return {nullptr, nullptr, false};
    return {input.data_ptr(), output.data_ptr(), input.scalar_type() == torch::kInt64};
}

}  // namespace

bool tryGlm5PackedVerifyPrepare(const torch::Tensor& dense_tokens,
                                const torch::Tensor& caps,
                                const torch::Tensor& rows,
                                const torch::Tensor& position_ids,
                                const torch::Tensor& token_type_ids,
                                Glm5PackedVerifyInputs& output,
                                cudaStream_t stream) {
    if (!dense_tokens.defined() || !dense_tokens.is_cuda()) return false;
    const int device = dense_tokens.get_device();
    if (!supportedVector(dense_tokens, device) || !supportedVector(caps, device) || !supportedVector(rows, device)) {
        return false;
    }
    const int64_t dense_count = dense_tokens.numel();
    const int64_t packed_count = rows.numel();
    const int64_t batch = caps.numel();
    for (const auto& optional : {position_ids, token_type_ids}) {
        if (optional.defined() && (!supportedVector(optional, device) || optional.numel() != dense_count)) return false;
    }
    if (packed_count > dense_count || (batch == 0 && dense_count != 0)
        || (batch > 0 && (dense_count == 0 || dense_count % batch != 0))
        || (caps.scalar_type() == torch::kInt32 && packed_count > std::numeric_limits<int32_t>::max())) {
        return false;
    }
    const auto cuda_stream = c10::cuda::getStreamFromExternal(stream, device);
    c10::cuda::CUDAStreamGuard guard(cuda_stream);
    Glm5PackedVerifyInputs prepared;
    prepared.tokens = torch::empty({packed_count}, dense_tokens.options());
    prepared.lengths = torch::empty({batch}, caps.options());
    prepared.lm_output_indexes = torch::empty({packed_count}, caps.options());
    if (position_ids.defined()) prepared.position_ids = torch::empty({packed_count}, position_ids.options());
    if (token_type_ids.defined()) prepared.token_type_ids = torch::empty({packed_count}, token_type_ids.options());
    const int64_t work = std::max(packed_count, batch);
    if (work > 0) {
        for (const auto& source : {dense_tokens, caps, rows, position_ids, token_type_ids}) {
            if (source.defined()) c10::cuda::CUDACachingAllocator::recordStream(source.storage().data_ptr(), cuda_stream);
        }
        constexpr int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((work - 1) / threads + 1, 4096));
        packedVerifyPrepare<<<blocks, threads, 0, stream>>>(
            vectorPair(dense_tokens, prepared.tokens), vectorPair(position_ids, prepared.position_ids),
            vectorPair(token_type_ids, prepared.token_type_ids), vectorPair(caps, prepared.lengths),
            rows.data_ptr(), rows.scalar_type() == torch::kInt64, prepared.lm_output_indexes.data_ptr(),
            dense_count, packed_count, batch, dense_count / batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    output = std::move(prepared);
    return true;
}

}  // namespace rtp_llm
