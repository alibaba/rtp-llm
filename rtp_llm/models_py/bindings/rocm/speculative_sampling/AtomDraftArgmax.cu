#include "AtomDraftArgmax.h"

#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

namespace rtp_llm {
namespace {

using Pair = unsigned long long;

// One LDS value per wave. MI308X uses wave64. Allow for wave32 on other targets.
#if defined(__gfx942__)
constexpr int kTileWaveSlots = 4;
#else
constexpr int kTileWaveSlots = 8;
#endif

__device__ Pair pack(float value, unsigned index) {
    unsigned bits = __float_as_uint(value);
    if ((bits << 1) == 0) {
        bits = 0;  // +0 and -0 tie.
    }
    // NaNs win; among equal values the earliest vocabulary index wins.
    unsigned key = isnan(value) ? 0xffffffffu : (bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u));
    return (Pair(key) << 32) | ~index;
}

__device__ Pair reduce_block(Pair value, Pair* wave_maxima) {
    const int lane = threadIdx.x % warpSize;
    for (int offset = warpSize / 2; offset; offset >>= 1) {
        value = max(value, __shfl_down(value, offset));
    }
    if (lane == 0) {
        wave_maxima[threadIdx.x / warpSize] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int wave = 1; wave < blockDim.x / warpSize; ++wave) {
            value = max(value, wave_maxima[wave]);
        }
    }
    return value;
}

__global__ void atom_draft_argmax_tiles(const float* logits, Pair* partial, int cols) {
    __shared__ Pair wave_maxima[kTileWaveSlots];
    Pair best = 0;
    for (int i = blockIdx.x * 2048 + threadIdx.x; i < min(cols, int(blockIdx.x + 1) * 2048); i += blockDim.x) {
        best = max(best, pack(logits[blockIdx.y * cols + i], i));
    }
    best = reduce_block(best, wave_maxima);
    if (threadIdx.x == 0) {
        partial[blockIdx.y * gridDim.x + blockIdx.x] = best;
    }
}

__global__ void atom_draft_argmax_finalize(const Pair* partial, int64_t* output, int tiles) {
    __shared__ Pair wave_maxima[4];  // 128 threads, at most four waves
    Pair best = threadIdx.x < tiles ? partial[blockIdx.x * tiles + threadIdx.x] : 0;
    best = reduce_block(best, wave_maxima);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = ~static_cast<unsigned>(best);
    }
}

}  // namespace

at::Tensor atomDecodeDraftArgmax(const at::Tensor& logits) {
    if (!logits.is_cuda() || logits.dim() != 2 || logits.size(0) != 2 || logits.size(1) != 248320
        || logits.scalar_type() != at::kFloat || !logits.is_contiguous()) {
        return at::argmax(logits, -1, true);
    }

    c10::hip::HIPGuard guard(logits.device().index());
    const int rows = logits.size(0);
    const int cols = logits.size(1);
    const int tiles = (cols + 2047) / 2048;
    auto partial = at::empty({rows, tiles}, logits.options().dtype(at::kLong));
    auto output = at::empty({rows}, logits.options().dtype(at::kLong));
    auto stream = c10::hip::getCurrentHIPStream();
    auto* tile_maxima = reinterpret_cast<Pair*>(partial.data_ptr<int64_t>());
    atom_draft_argmax_tiles<<<dim3(tiles, rows), 256, 0, stream>>>(logits.data_ptr<float>(), tile_maxima, cols);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    atom_draft_argmax_finalize<<<rows, 128, 0, stream>>>(tile_maxima, output.data_ptr<int64_t>(), tiles);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    return output.view({rows, 1});
}

}  // namespace rtp_llm
