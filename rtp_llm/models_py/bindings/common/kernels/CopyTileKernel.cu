#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"

#include <cstdint>
#include <limits>

namespace rtp_llm {
namespace {

template<typename T>
__device__ __forceinline__ void copyTyped(const char* src, char* dst, size_t bytes) {
    const size_t count = bytes / sizeof(T);
    size_t       index = threadIdx.x;
#pragma unroll 4
    while (index < count) {
        reinterpret_cast<T*>(dst)[index] = reinterpret_cast<const T*>(src)[index];
        index += blockDim.x;
    }
    for (size_t offset = count * sizeof(T) + threadIdx.x; offset < bytes; offset += blockDim.x) {
        dst[offset] = src[offset];
    }
}

__device__ __forceinline__ void copyBytes(const char* src, char* dst, size_t bytes) {
    const auto alignment = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
    if ((alignment & 15) == 0) {
        copyTyped<int4>(src, dst, bytes);
    } else if ((alignment & 7) == 0) {
        copyTyped<int2>(src, dst, bytes);
    } else if ((alignment & 3) == 0) {
        copyTyped<unsigned int>(src, dst, bytes);
    } else if ((alignment & 1) == 0) {
        copyTyped<unsigned short>(src, dst, bytes);
    } else {
        copyTyped<char>(src, dst, bytes);
    }
}

__global__ void copyTilesKernel(const CopyTile* tiles, char* staging, bool scatter) {
    const auto tile   = tiles[blockIdx.x];
    auto*      device = static_cast<char*>(tile.device);
    auto*      packed = staging + tile.staging_offset;
    copyBytes(scatter ? packed : device, scatter ? device : packed, tile.bytes);
}

}  // namespace

cudaError_t
launchCopyTiles(const CopyTile* tiles, size_t count, void* staging, bool scatter, cudaStream_t stream, int threads) {
    if (count == 0) {
        return cudaSuccess;
    }
    if (tiles == nullptr || staging == nullptr || threads < 32 || threads > 1024 || threads % 32 != 0) {
        return cudaErrorInvalidValue;
    }
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }
    copyTilesKernel<<<static_cast<unsigned int>(count), threads, 0, stream>>>(
        tiles, static_cast<char*>(staging), scatter);
    return cudaGetLastError();
}

}  // namespace rtp_llm
