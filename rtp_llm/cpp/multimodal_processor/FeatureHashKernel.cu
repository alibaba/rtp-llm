#include "rtp_llm/cpp/multimodal_processor/FeatureHashKernel.h"

#include "rtp_llm/cpp/multimodal_processor/FeatureHash.h"

namespace rtp_llm {
namespace {

constexpr int kFeatureHashThreads = 256;

__global__ void featureHashKernel(const uint8_t* input, int64_t num_rows, int64_t row_bytes, int32_t* output) {
    const int64_t row = blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    const uint8_t* row_data   = input + row * row_bytes;
    const uint64_t word_count = (row_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t       partial    = 0;
    for (uint64_t word_index = threadIdx.x; word_index < word_count; word_index += blockDim.x) {
        partial ^= featureHashWordContribution(
            loadFeatureHashWord(row_data, static_cast<uint64_t>(row_bytes), word_index), word_index);
    }

    __shared__ uint64_t block_hash[kFeatureHashThreads];
    block_hash[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_hash[threadIdx.x] ^= block_hash[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const uint64_t hash = featureHashFinalize(featureHashSeed(row_bytes) ^ block_hash[0]);
        output[row]         = featureHashToTokenId(hash);
    }
}

}  // namespace

cudaError_t
invokeFeatureHash(const void* input, int64_t num_rows, int64_t row_bytes, int32_t* output, cudaStream_t stream) {
    if (input == nullptr || output == nullptr || num_rows <= 0 || row_bytes <= 0) {
        return cudaErrorInvalidValue;
    }
    featureHashKernel<<<static_cast<unsigned int>(num_rows), kFeatureHashThreads, 0, stream>>>(
        static_cast<const uint8_t*>(input), num_rows, row_bytes, output);
    return cudaPeekAtLastError();
}

}  // namespace rtp_llm
