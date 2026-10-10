#include "rtp_llm/models_py/bindings/cuda/LinearCheckpointCopy.h"
#include <cfloat>
#include <cuda_bf16.h>

namespace rtp_llm {
namespace {

// One FP32 scale per K channel; recurrence and conv history remain full precision.
// INT16 uses the same symmetric per-channel layout with a wider integer range.
template<bool Unpack, typename Quant>
__device__ void
quantizedCheckpointCopy(const LinearCheckpointDeviceTile& tile, int lane, int channel, size_t elements, size_t base) {
    constexpr float limit  = sizeof(Quant) == 2 ? 32767.f : 127.f;
    auto*           packed = reinterpret_cast<Quant*>(tile.packed);
    auto*           scales = reinterpret_cast<float*>(tile.packed + elements * sizeof(Quant));
    if constexpr (Unpack) {
        const float scale = scales[channel];
        for (int v = lane; v < tile.value_dim; v += 32) {
            tile.state[base + v] = static_cast<float>(packed[base + v]) * scale;
        }
    } else {
        float amax = 0.f;
        for (int v = lane; v < tile.value_dim; v += 32) {
            amax = fmaxf(amax, fabsf(tile.state[base + v]));
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        // Preserve small recurrent channels; only guard against FP32 underflow.
        const float scale = fmaxf(amax / limit, FLT_MIN);
        if (lane == 0) {
            scales[channel] = scale;
        }
        for (int v = lane; v < tile.value_dim; v += 32) {
            const int value  = __float2int_rn(fminf(limit, fmaxf(-limit, tile.state[base + v] / scale)));
            packed[base + v] = static_cast<Quant>(value);
        }
    }
}

// RTP stores recurrent states as [H, K, V]. Each warp reduces one contiguous
// V row; quantization is applied only when an idle checkpoint is inserted.
template<bool Unpack>
__global__ void checkpointCopy(const LinearCheckpointDeviceTile* tiles) {
    const auto tile     = tiles[blockIdx.y];
    const int  lane     = threadIdx.x % 32;
    const int  channel  = blockIdx.x * 4 + threadIdx.x / 32;
    const int  channels = tile.heads * tile.key_dim;
    if (channel >= channels) {
        return;
    }
    const size_t elements = static_cast<size_t>(channels) * tile.value_dim;
    const size_t base     = static_cast<size_t>(channel) * tile.value_dim;
    if (tile.dtype == LinearCheckpointDType::BF16) {
        auto* packed = reinterpret_cast<__nv_bfloat16*>(tile.packed);
        for (int v = lane; v < tile.value_dim; v += 32) {
            if constexpr (Unpack) {
                tile.state[base + v] = __bfloat162float(packed[base + v]);
            } else {
                packed[base + v] = __float2bfloat16_rn(tile.state[base + v]);
            }
        }
        return;
    }
    if (tile.dtype == LinearCheckpointDType::INT16) {
        quantizedCheckpointCopy<Unpack, int16_t>(tile, lane, channel, elements, base);
    } else {
        quantizedCheckpointCopy<Unpack, int8_t>(tile, lane, channel, elements, base);
    }
}

}  // namespace

void invokeLinearCheckpointCopy(
    const LinearCheckpointDeviceTile* tiles, int count, int max_channels, bool unpack, cudaStream_t stream) {
    const dim3 grid((max_channels + 3) / 4, count);
    if (unpack) {
        checkpointCopy<true><<<grid, 128, 0, stream>>>(tiles);
    } else {
        checkpointCopy<false><<<grid, 128, 0, stream>>>(tiles);
    }
}

}  // namespace rtp_llm
