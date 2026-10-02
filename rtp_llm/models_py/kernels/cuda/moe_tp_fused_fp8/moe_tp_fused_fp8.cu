/*
 * Copyright (c) 2026, Alibaba Group. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
// Correctness-first TP=2 FC2/finalize plus one-shot FP8 IPC all-reduce.
// Scalar, shared-staged and FP8 MMA compute variants are explicit benchmark APIs.
// None is registered as a serving backend.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fc2_reference.cuh"
#include "fc2_mma.cuh"
#include "fc2_staged.cuh"
#include "moe_tp_fused_fp8_transport.cuh"

namespace {
namespace fused = rtp_llm::moe_tp_fused_fp8;

constexpr int      kDefaultBlocks    = 16;
constexpr uint64_t kWireMagic        = 0x5254504D4F454631ULL;  // RTPMOEF1
constexpr uint32_t kWireVersion      = 4;
constexpr uint64_t kSpinLimit        = 1ULL << 26;
constexpr uint64_t kErrorPeerTimeout = 1ULL << 63;
constexpr uint64_t kErrorNumeric     = 1ULL << 62;

inline void cuda_check(cudaError_t status, char const* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

inline size_t div_up(size_t numerator, size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

struct IpcWireHandle {
    uint64_t           magic;
    uint32_t           version;
    int32_t            device;
    int32_t            rank;
    int32_t            blocks;
    uint64_t           max_numel;
    uint64_t           packet_count;
    cudaIpcMemHandle_t packets;
    cudaIpcMemHandle_t inbox;
    cudaIpcMemHandle_t ready;
    cudaIpcMemHandle_t ack;
    cudaIpcMemHandle_t error;
};
static_assert(std::is_trivially_copyable_v<IpcWireHandle>);

struct DeviceParams {
    fused::TransportWorkspace local;
    fused::TransportWorkspace peer;
    fused::Packet*            local_inbox;
    fused::Packet*            peer_inbox;
    uint64_t*                 completion;
    int                       active_blocks;
    uint64_t                  epoch;
    size_t                    packet_count;
    size_t                    numel;
    int                       rank;
};

__device__ __forceinline__ void set_timeout(uint64_t* error) {
    atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorPeerTimeout));
}

// All participating CTAs are resident.  Every current peer-inbox writer uses
// exactly this owner mapping: first = blockIdx * 16 + n * gridDim * 16.  Once
// CTA b has consumed all its groups, acknowledging ack[b * 16] lets only the
// same owner CTA overwrite those groups in a later call.  This is safe across
// sequential push/gather calls even when their message lengths differ, but is
// deliberately not a general protocol for a future writer with another group
// size, grid, or ownership mapping.
__device__ __forceinline__ bool deferred_final_ack(DeviceParams const& params, int* protocol_status) {
    constexpr size_t packets_per_group = fused::kPacketThreads / 32;
    const size_t     owner_first       = blockIdx.x * packets_per_group;
    __syncthreads();
    // Preserve the old system-scope retirement discipline.  Each thread's
    // fence precedes the CTA barrier so the leader publishes only after every
    // consuming lane has completed its local-inbox reads and output stores.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.peer.ack + owner_first);
    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.local.ack + owner_first, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status) {
        if (threadIdx.x == 0)
            set_timeout(params.local.error);
        return false;
    }
    return true;
}

__device__ __forceinline__ void write_packet(fused::Packet*       packet,
                                             __nv_bfloat16 const* source,
                                             size_t               numel,
                                             size_t               packet_index,
                                             float*               warp_maxima,
                                             uint64_t*            error) {
    const size_t base      = packet_index * fused::kPacketValues;
    float        local_max = 0.f;
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index = base + item;
        const float  value = index < numel ? __bfloat162float(source[index]) : 0.f;
        if (!isfinite(value))
            atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorNumeric));
        local_max = fused::max_abs(value, local_max);
    }
    const float max_abs = fused::block_max_abs(local_max, warp_maxima);
    const float scale   = max_abs == 0.f ? 0.f : 448.f / max_abs;
    if (threadIdx.x == 0 && !isfinite(scale))
        atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorNumeric));
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index   = base + item;
        const float  value   = index < numel ? __bfloat162float(source[index]) : 0.f;
        packet->values[item] = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
    }
    if (threadIdx.x == 0) {
        packet->scale       = scale;
        packet->reserved[0] = 0;
        packet->reserved[1] = 0;
        packet->reserved[2] = 0;
    }
}

__device__ __forceinline__ bool publish_reduce_and_ack(
    DeviceParams const& params, __nv_bfloat16* out, size_t packet_index, float* warp_maxima, int* protocol_status) {
    fused::Packet* local_packet = params.local.packets + packet_index;
    write_packet(local_packet, out, params.numel, packet_index, warp_maxima, params.local.error);
    // The system fence is issued by every packet writer. A block barrier alone
    // cannot make another thread's global stores system-visible to a PCIe peer.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ready + packet_index);
    __syncthreads();

    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ready + packet_index, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status) {
        if (threadIdx.x == 0)
            set_timeout(params.local.error);
        return false;
    }

    const fused::Packet* peer_packet = params.peer.packets + packet_index;
    const size_t         base        = packet_index * fused::kPacketValues;
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index = base + item;
        if (index < params.numel) {
            const float own   = local_packet->scale == 0.f ?
                                    static_cast<float>(local_packet->values[item]) :
                                    static_cast<float>(local_packet->values[item]) / local_packet->scale;
            const float other = peer_packet->scale == 0.f ?
                                    static_cast<float>(peer_packet->values[item]) :
                                    static_cast<float>(peer_packet->values[item]) / peer_packet->scale;
            // Rank ordering is part of the numeric contract, even though a
            // two-addend finite sum is usually commutative.
            out[index] = __float2bfloat16_rn(params.rank == 0 ? own + other : other + own);
            if (!isfinite(__bfloat162float(out[index])))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
        }
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ack + packet_index);
    __syncthreads();
    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ack + packet_index, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status && threadIdx.x == 0)
        set_timeout(params.local.error);
    return *protocol_status != 0;
}

// Keep the 496-value quantization scale unchanged, but amortize PCIe control
// traffic over up to 32 grid-strided packets. Their quantization boundaries
// stay unchanged. Only the first packet's flags represent the group.
constexpr size_t kPacketBatch = 32;

__device__ __forceinline__ bool publish_batch_reduce_and_ack(DeviceParams const& params,
                                                             __nv_bfloat16*      output,
                                                             size_t              first,
                                                             size_t              count,
                                                             float*              warp_maxima,
                                                             int*                protocol_status) {
    for (size_t offset = 0; offset < count; ++offset) {
        const size_t packet = first + offset * gridDim.x;
        write_packet(params.local.packets + packet, output, params.numel, packet, warp_maxima, params.local.error);
        __syncthreads();
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ready + first);
    __syncthreads();
    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ready + first, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status) {
        if (threadIdx.x == 0)
            set_timeout(params.local.error);
        return false;
    }
    // Fetch each peer packet with coalesced 16-byte loads, independently of the
    // scalar per-value dequantization. Staging the whole group exposes enough
    // outstanding remote work and avoids a remote byte load per output lane.
    __shared__ fused::Packet peer_staging[kPacketBatch];
    constexpr size_t         vectors_per_packet = sizeof(fused::Packet) / sizeof(uint4);
    auto*                    staged_vectors     = reinterpret_cast<uint4*>(peer_staging);
    for (size_t vector = threadIdx.x; vector < count * vectors_per_packet; vector += blockDim.x) {
        const size_t offset       = vector / vectors_per_packet;
        const size_t within       = vector % vectors_per_packet;
        const auto*  peer_vectors = reinterpret_cast<const uint4*>(params.peer.packets + first + offset * gridDim.x);
        staged_vectors[vector]    = peer_vectors[within];
    }
    __syncthreads();
    for (size_t offset = 0; offset < count; ++offset) {
        const size_t packet       = first + offset * gridDim.x;
        const auto*  local_packet = params.local.packets + packet;
        const auto*  peer_packet  = peer_staging + offset;
        const size_t base         = packet * fused::kPacketValues;
        for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
            const size_t index = base + item;
            if (index < params.numel) {
                const float own   = local_packet->scale == 0.f ?
                                        static_cast<float>(local_packet->values[item]) :
                                        static_cast<float>(local_packet->values[item]) / local_packet->scale;
                const float other = peer_packet->scale == 0.f ?
                                        static_cast<float>(peer_packet->values[item]) :
                                        static_cast<float>(peer_packet->values[item]) / peer_packet->scale;
                output[index]     = __float2bfloat16_rn(params.rank == 0 ? own + other : other + own);
                if (!isfinite(__bfloat162float(output[index])))
                    atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                             static_cast<unsigned long long>(kErrorNumeric));
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ack + first);
    __syncthreads();
    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ack + first, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status && threadIdx.x == 0)
        set_timeout(params.local.error);
    return *protocol_status != 0;
}

__global__ void
one_shot_all_reduce_batch_kernel(DeviceParams params, __nv_bfloat16 const* input, __nv_bfloat16* output) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    const size_t packets_per_cta = params.packet_count / gridDim.x;
    const size_t batch           = packets_per_cta < 1 ? 1 : min(kPacketBatch, packets_per_cta);
    for (size_t first = blockIdx.x; first < params.packet_count; first += gridDim.x * batch) {
        const size_t count = min(batch, (params.packet_count - first + gridDim.x - 1) / gridDim.x);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t base = (first + offset * gridDim.x) * fused::kPacketValues;
            for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x)
                if (base + item < params.numel)
                    output[base + item] = input[base + item];
        }
        __syncthreads();
        if (!publish_batch_reduce_and_ack(params, output, first, count, warp_maxima, &protocol_status))
            return;
    }
}

union WarpPacketBf16 {
    uint4         vectors[2];
    __nv_bfloat16 values[16];
};

union WarpPacketFp8 {
    uint4         vector;
    __nv_fp8_e4m3 values[16];
    float         metadata[4];
};

// Exact instruction sequence emitted by Triton 3.6.0 for
// ``tl.sigmoid(gate) * shared + experts`` on SM120 in the existing
// SigmoidGateScaleAdd kernel.  Keep this inline sequence rather than using
// expf/__exp2f or fmaf: the experiment compares packet bytes to that Triton
// implementation, including ex2.approx/div.full and the BF16 rounding point.
__device__ __forceinline__ float triton_sigmoid_f32(float gate) {
    float result;
    asm volatile("{ .reg .f32 neg, scaled, exponent, denominator;\n\t"
                 "sub.f32 neg, 0f00000000, %1;\n\t"
                 "mul.f32 scaled, neg, 0f3FB8AA3B;\n\t"
                 "ex2.approx.f32 exponent, scaled;\n\t"
                 "add.f32 denominator, exponent, 0f3F800000;\n\t"
                 "div.full.f32 %0, 0f3F800000, denominator;\n\t"
                 "}"
                 : "=f"(result)
                 : "f"(gate));
    return result;
}

__device__ __forceinline__ __nv_bfloat16 triton_gate_scale_add_from_sigmoid_bf16(float sigmoid,
                                                                                 float shared,
                                                                                 float experts) {
    union {
        unsigned short bits;
        __nv_bfloat16  value;
    } result;
    asm volatile("{ .reg .f32 sum;\n\t"
                 "fma.rn.f32 sum, %1, %2, %3;\n\t"
                 "cvt.rn.bf16.f32 %0, sum;\n\t"
                 "}"
                 : "=h"(result.bits)
                 : "f"(sigmoid), "f"(shared), "f"(experts));
    return result.value;
}

__device__ __forceinline__ __nv_bfloat16 triton_sigmoid_gate_scale_add_bf16(float gate, float shared, float experts) {
    return triton_gate_scale_add_from_sigmoid_bf16(triton_sigmoid_f32(gate), shared, experts);
}

__global__ void gated_local_warp_kernel(__nv_bfloat16 const* experts,
                                        __nv_bfloat16 const* shared,
                                        __nv_bfloat16 const* gate,
                                        __nv_bfloat16*       output,
                                        size_t               numel,
                                        int                  hidden_size) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < numel; index += gridDim.x * blockDim.x) {
        output[index] = triton_sigmoid_gate_scale_add_bf16(__bfloat162float(gate[index / hidden_size]),
                                                           __bfloat162float(shared[index]),
                                                           __bfloat162float(experts[index]));
    }
}

// One warp owns one unchanged 496-value packet: 31 lanes each hold 16
// values, and lane 31 holds the scale/padding. All warps in a CTA quantize
// concurrently, amortizing ready/ack over 16 contiguous packets. This is a
// separate transport candidate; the scalar and batch protocols stay intact.
__global__ void
one_shot_all_reduce_warp_kernel(DeviceParams params, __nv_bfloat16 const* input, __nv_bfloat16* output) {
    constexpr size_t packets_per_group = fused::kPacketThreads / 32;
    const int        lane              = threadIdx.x & 31;
    const int        warp              = threadIdx.x / 32;
    const bool       input_aligned     = (reinterpret_cast<uintptr_t>(input) & (alignof(uint4) - 1)) == 0;
    const bool       output_aligned    = (reinterpret_cast<uintptr_t>(output) & (alignof(uint4) - 1)) == 0;
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t first = blockIdx.x * packets_per_group; first < params.packet_count;
         first += gridDim.x * packets_per_group) {
        const size_t  packet = first + warp;
        const size_t  base   = packet * fused::kPacketValues + lane * 16;
        WarpPacketFp8 own;
        float         scale = 0.f;
        if (packet < params.packet_count) {
            WarpPacketBf16 source;
            source.vectors[0] = make_uint4(0, 0, 0, 0);
            source.vectors[1] = make_uint4(0, 0, 0, 0);
            if (input_aligned && lane < 31 && base + 16 <= params.numel) {
                source.vectors[0] = reinterpret_cast<const uint4*>(input + base)[0];
                source.vectors[1] = reinterpret_cast<const uint4*>(input + base)[1];
            } else if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item)
                    if (base + item < params.numel)
                        source.values[item] = input[base + item];
            }
            float local_max = 0.f;
#pragma unroll
            for (int item = 0; item < 16; ++item) {
                const float value = __bfloat162float(source.values[item]);
                if (!isfinite(value))
                    atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                             static_cast<unsigned long long>(kErrorNumeric));
                local_max = fused::max_abs(value, local_max);
            }
            const float max_abs = fused::warp_max(local_max);
            scale               = max_abs == 0.f ? 0.f : 448.f / max_abs;
            if (!isfinite(scale))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
            own.vector = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float value = __bfloat162float(source.values[item]);
                    own.values[item]  = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
                }
            } else {
                own.metadata[0] = scale;
            }
            reinterpret_cast<uint4*>(params.local.packets + packet)[lane] = own.vector;
        }
        // Every writer fences its own payload before the CTA publishes.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.local.ready + first);
            protocol_status = fused::wait_epoch(params.peer.ready + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
        if (packet < params.packet_count) {
            WarpPacketFp8 peer;
            peer.vector            = reinterpret_cast<const uint4*>(params.peer.packets + packet)[lane];
            const float peer_scale = __shfl_sync(0xffffffff, peer.metadata[0], 31);
            if (lane < 31) {
                WarpPacketBf16 result;
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float a       = scale == 0.f ? static_cast<float>(own.values[item]) :
                                                         static_cast<float>(own.values[item]) / scale;
                    const float b       = peer_scale == 0.f ? static_cast<float>(peer.values[item]) :
                                                              static_cast<float>(peer.values[item]) / peer_scale;
                    result.values[item] = __float2bfloat16_rn(params.rank == 0 ? a + b : b + a);
                    if (!isfinite(__bfloat162float(result.values[item])))
                        atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                                 static_cast<unsigned long long>(kErrorNumeric));
                }
                if (output_aligned && base + 16 <= params.numel) {
                    reinterpret_cast<uint4*>(output + base)[0] = result.vectors[0];
                    reinterpret_cast<uint4*>(output + base)[1] = result.vectors[1];
                } else {
#pragma unroll
                    for (int item = 0; item < 16; ++item)
                        if (base + item < params.numel)
                            output[base + item] = result.values[item];
                }
            }
        }
        __syncthreads();
        __threadfence_system();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.local.ack + first);
            protocol_status = fused::wait_epoch(params.peer.ack + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
    }
}

// Push variant of the warp transport.  Each rank posts its packet to the
// peer's inbox, then consumes the peer's posted packet from local memory.
// Existing packets_ remains an own-packet debug mirror; old pull protocol
// fields remain unchanged and are only interpreted in the reverse direction.
template<bool DeferredAck>
__global__ void
one_shot_all_reduce_push_warp_kernel(DeviceParams params, __nv_bfloat16 const* input, __nv_bfloat16* output) {
    constexpr size_t packets_per_group = fused::kPacketThreads / 32;
    const int        lane              = threadIdx.x & 31;
    const int        warp              = threadIdx.x / 32;
    const bool       input_aligned     = (reinterpret_cast<uintptr_t>(input) & (alignof(uint4) - 1)) == 0;
    const bool       output_aligned    = (reinterpret_cast<uintptr_t>(output) & (alignof(uint4) - 1)) == 0;
    __shared__ int   protocol_status;
    if constexpr (DeferredAck) {
        if (blockIdx.x >= params.active_blocks)
            return;
    }
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t first = blockIdx.x * packets_per_group; first < params.packet_count;
         first += gridDim.x * packets_per_group) {
        const size_t  packet = first + warp;
        const size_t  base   = packet * fused::kPacketValues + lane * 16;
        WarpPacketFp8 own;
        float         scale = 0.f;
        if (packet < params.packet_count) {
            WarpPacketBf16 source;
            source.vectors[0] = make_uint4(0, 0, 0, 0);
            source.vectors[1] = make_uint4(0, 0, 0, 0);
            if (input_aligned && lane < 31 && base + 16 <= params.numel) {
                source.vectors[0] = reinterpret_cast<uint4 const*>(input + base)[0];
                source.vectors[1] = reinterpret_cast<uint4 const*>(input + base)[1];
            } else if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item)
                    if (base + item < params.numel)
                        source.values[item] = input[base + item];
            }
            float local_max = 0.f;
#pragma unroll
            for (int item = 0; item < 16; ++item) {
                const float value = __bfloat162float(source.values[item]);
                if (!isfinite(value))
                    atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                             static_cast<unsigned long long>(kErrorNumeric));
                local_max = fused::max_abs(value, local_max);
            }
            const float max_abs = fused::warp_max(local_max);
            scale               = max_abs == 0.f ? 0.f : 448.f / max_abs;
            if (!isfinite(scale))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
            own.vector = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float value = __bfloat162float(source.values[item]);
                    own.values[item]  = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
                }
            } else {
                own.metadata[0] = scale;
            }
            // Keep a locally owned debug copy and post the identical vector to
            // peer inbox.  Each warp is a contiguous 512-byte packet store.
            reinterpret_cast<uint4*>(params.local.packets + packet)[lane] = own.vector;
            reinterpret_cast<uint4*>(params.peer_inbox + packet)[lane]    = own.vector;
        }
        // A CTA leader may only publish ready after every lane's remote store
        // is system-visible. A leader-only fence would not order other lanes.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.peer.ready + first);
            protocol_status = fused::wait_epoch(params.local.ready + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
        if (packet < params.packet_count) {
            // Every consuming lane acquires the locally owned ready word before
            // reading its inbox vector; do not rely on leader acquire crossing
            // a CTA barrier for system-scope visibility.
            (void)fused::load_acquire_sys(params.local.ready + first);
            WarpPacketFp8 peer;
            peer.vector            = reinterpret_cast<uint4 const*>(params.local_inbox + packet)[lane];
            const float peer_scale = __shfl_sync(0xffffffff, peer.metadata[0], 31);
            if (lane < 31) {
                WarpPacketBf16 result;
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float a       = scale == 0.f ? static_cast<float>(own.values[item]) :
                                                         static_cast<float>(own.values[item]) / scale;
                    const float b       = peer_scale == 0.f ? static_cast<float>(peer.values[item]) :
                                                              static_cast<float>(peer.values[item]) / peer_scale;
                    result.values[item] = __float2bfloat16_rn(params.rank == 0 ? a + b : b + a);
                    if (!isfinite(__bfloat162float(result.values[item])))
                        atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                                 static_cast<unsigned long long>(kErrorNumeric));
                }
                if (output_aligned && base + 16 <= params.numel) {
                    reinterpret_cast<uint4*>(output + base)[0] = result.vectors[0];
                    reinterpret_cast<uint4*>(output + base)[1] = result.vectors[1];
                } else {
#pragma unroll
                    for (int item = 0; item < 16; ++item)
                        if (base + item < params.numel)
                            output[base + item] = result.values[item];
                }
            }
        }
        // Keep a CTA barrier after consuming this group.  The deferred
        // variant replaces the per-group ack below with one final per-CTA ack.
        __syncthreads();
        if constexpr (!DeferredAck) {
            __threadfence_system();
            if (threadIdx.x == 0) {
                fused::store_release_sys(params.epoch, params.peer.ack + first);
                protocol_status = fused::wait_epoch(params.local.ack + first, params.epoch, kSpinLimit) ? 1 : 0;
            }
            __syncthreads();
            if (!protocol_status) {
                if (threadIdx.x == 0)
                    set_timeout(params.local.error);
                return;
            }
        }
    }
    if constexpr (DeferredAck)
        (void)deferred_final_ack(params, &protocol_status);
}

// Experimental whole-batch Pure-TP finalization.  It mirrors Triton's
// ep_gather routing order (slot 0..7, fp32 fma, then one BF16 round), then
// applies the existing gate PTX and immediately posts the FP8 packet.  The
// routed FC2 producer remains DeepGEMM; no GEMM epilogue is assumed here.
template<bool DeferredAck>
__global__ __launch_bounds__(fused::kPacketThreads,
                             2) void gather_gate_push_warp_kernel(DeviceParams         params,
                                                                  __nv_bfloat16 const* down_output,
                                                                  int64_t const*       topk_ids,
                                                                  float const*         topk_weights,
                                                                  int64_t const*       output_index,
                                                                  size_t               down_rows,
                                                                  __nv_bfloat16 const* shared,
                                                                  __nv_bfloat16 const* gate,
                                                                  __nv_bfloat16*       output) {
    constexpr size_t packets_per_group   = fused::kPacketThreads / 32;
    constexpr int    hidden_size         = fused::kFc2ReferenceHiddenSize;
    constexpr int    topk                = 8;
    const int        lane                = threadIdx.x & 31;
    const int        warp                = threadIdx.x / 32;
    const bool       down_output_aligned = (reinterpret_cast<uintptr_t>(down_output) & (alignof(uint4) - 1)) == 0;
    const bool       output_aligned      = (reinterpret_cast<uintptr_t>(output) & (alignof(uint4) - 1)) == 0;
    __shared__ int   protocol_status;
    if constexpr (DeferredAck) {
        if (blockIdx.x >= params.active_blocks)
            return;
    }
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t first = blockIdx.x * packets_per_group; first < params.packet_count;
         first += gridDim.x * packets_per_group) {
        const size_t  packet = first + warp;
        const size_t  base   = packet * fused::kPacketValues + lane * 16;
        WarpPacketFp8 own;
        float         scale = 0.f;
        if (packet < params.packet_count) {
            WarpPacketBf16 source;
            source.vectors[0] = make_uint4(0, 0, 0, 0);
            source.vectors[1] = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
                // H=2048 and every lane begins at a multiple of 16, hence one
                // gate scalar serves all 16 output elements in this lane.  The
                // same divisibility also makes a complete lane's down row
                // slice one aligned 32-byte vector pair when the tensor base
                // itself is aligned; offset-one test views take the scalar
                // fallback below.
                float        accumulator[16]{};
                const bool   has_values   = base < params.numel;
                const size_t token        = base / hidden_size;
                const size_t hidden       = base % hidden_size;
                const size_t token_offset = token * topk;
                const float  sigmoid      = has_values ? triton_sigmoid_f32(__bfloat162float(gate[token])) : 0.f;
#pragma unroll
                for (int slot = 0; slot < topk; ++slot) {
                    // Keep ep_gather's predicate and its slot-ascending FP32
                    // FMA sequence.  Metadata is read once per slot/lane,
                    // rather than once for each of the lane's 16 elements.
                    const int64_t expert_id  = has_values ? topk_ids[token_offset + slot] : -1;
                    const int64_t source_row = has_values ? output_index[token_offset + slot] : -1;
                    const float   weight     = has_values ? topk_weights[token_offset + slot] : 0.f;
                    if (expert_id < 0 || source_row < 0 || static_cast<size_t>(source_row) >= down_rows)
                        continue;
                    WarpPacketBf16 values;
                    const size_t   down_offset = static_cast<size_t>(source_row) * hidden_size + hidden;
                    if (down_output_aligned && base + 16 <= params.numel) {
                        values.vectors[0] = reinterpret_cast<uint4 const*>(down_output + down_offset)[0];
                        values.vectors[1] = reinterpret_cast<uint4 const*>(down_output + down_offset)[1];
                    } else {
#pragma unroll
                        for (int item = 0; item < 16; ++item) {
                            const size_t index = base + item;
                            values.values[item] =
                                index < params.numel ? down_output[down_offset + item] : __float2bfloat16_rn(0.f);
                        }
                    }
#pragma unroll
                    for (int item = 0; item < 16; ++item)
                        accumulator[item] = __fmaf_rn(__bfloat162float(values.values[item]), weight, accumulator[item]);
                }
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const size_t index = base + item;
                    if (index < params.numel) {
                        const __nv_bfloat16 gathered = __float2bfloat16_rn(accumulator[item]);
                        source.values[item]          = triton_gate_scale_add_from_sigmoid_bf16(
                            sigmoid, __bfloat162float(shared[index]), __bfloat162float(gathered));
                    }
                }
            }
            float local_max = 0.f;
#pragma unroll
            for (int item = 0; item < 16; ++item) {
                const float value = __bfloat162float(source.values[item]);
                if (!isfinite(value))
                    atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                             static_cast<unsigned long long>(kErrorNumeric));
                local_max = fused::max_abs(value, local_max);
            }
            const float max_abs = fused::warp_max(local_max);
            scale               = max_abs == 0.f ? 0.f : 448.f / max_abs;
            if (!isfinite(scale))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
            own.vector = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float value = __bfloat162float(source.values[item]);
                    own.values[item]  = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
                }
            } else {
                own.metadata[0] = scale;
            }
            reinterpret_cast<uint4*>(params.local.packets + packet)[lane] = own.vector;
            reinterpret_cast<uint4*>(params.peer_inbox + packet)[lane]    = own.vector;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.peer.ready + first);
            protocol_status = fused::wait_epoch(params.local.ready + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
        if (packet < params.packet_count) {
            (void)fused::load_acquire_sys(params.local.ready + first);
            WarpPacketFp8 peer;
            peer.vector            = reinterpret_cast<uint4 const*>(params.local_inbox + packet)[lane];
            const float peer_scale = __shfl_sync(0xffffffff, peer.metadata[0], 31);
            if (lane < 31) {
                WarpPacketBf16 result;
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float a       = scale == 0.f ? static_cast<float>(own.values[item]) :
                                                         static_cast<float>(own.values[item]) / scale;
                    const float b       = peer_scale == 0.f ? static_cast<float>(peer.values[item]) :
                                                              static_cast<float>(peer.values[item]) / peer_scale;
                    result.values[item] = __float2bfloat16_rn(params.rank == 0 ? a + b : b + a);
                    if (!isfinite(__bfloat162float(result.values[item])))
                        atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                                 static_cast<unsigned long long>(kErrorNumeric));
                }
                if (output_aligned && base + 16 <= params.numel) {
                    reinterpret_cast<uint4*>(output + base)[0] = result.vectors[0];
                    reinterpret_cast<uint4*>(output + base)[1] = result.vectors[1];
                } else {
#pragma unroll
                    for (int item = 0; item < 16; ++item)
                        if (base + item < params.numel)
                            output[base + item] = result.values[item];
                }
            }
        }
        __syncthreads();
        if constexpr (!DeferredAck) {
            __threadfence_system();
            if (threadIdx.x == 0) {
                fused::store_release_sys(params.epoch, params.peer.ack + first);
                protocol_status = fused::wait_epoch(params.local.ack + first, params.epoch, kSpinLimit) ? 1 : 0;
            }
            __syncthreads();
            if (!protocol_status) {
                if (threadIdx.x == 0)
                    set_timeout(params.local.error);
                return;
            }
        }
    }
    if constexpr (DeferredAck)
        (void)deferred_final_ack(params, &protocol_status);
}

// Like the warp transport, but form each local BF16 value with the exact
// existing Triton shared-gate sequence before deriving the packet FP8 scale.
// A CTA consumes all 16 source packets before it can overwrite an exactly
// aliased experts/output tensor, so exact in-place experts/output is safe.
__global__ void gated_all_reduce_warp_kernel(DeviceParams         params,
                                             __nv_bfloat16 const* experts,
                                             __nv_bfloat16 const* shared,
                                             __nv_bfloat16 const* gate,
                                             __nv_bfloat16*       output,
                                             int                  hidden_size) {
    constexpr size_t packets_per_group = fused::kPacketThreads / 32;
    const int        lane              = threadIdx.x & 31;
    const int        warp              = threadIdx.x / 32;
    const bool       output_aligned    = (reinterpret_cast<uintptr_t>(output) & (alignof(uint4) - 1)) == 0;
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t first = blockIdx.x * packets_per_group; first < params.packet_count;
         first += gridDim.x * packets_per_group) {
        const size_t  packet = first + warp;
        const size_t  base   = packet * fused::kPacketValues + lane * 16;
        WarpPacketFp8 own;
        float         scale = 0.f;
        if (packet < params.packet_count) {
            WarpPacketBf16 source;
            source.vectors[0] = make_uint4(0, 0, 0, 0);
            source.vectors[1] = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
                // Packet/lane bases are multiples of 16 and H is a multiple
                // of 16, so these 16 values are always in one token row.
                // Compute Triton's BF16 gate scalar once per lane, not once
                // per element, while retaining the exact PTX instruction path.
                const float gate_value = base < params.numel ? __bfloat162float(gate[base / hidden_size]) : 0.f;
                const float sigmoid    = triton_sigmoid_f32(gate_value);
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const size_t index = base + item;
                    if (index < params.numel) {
                        source.values[item] = triton_gate_scale_add_from_sigmoid_bf16(
                            sigmoid, __bfloat162float(shared[index]), __bfloat162float(experts[index]));
                    }
                }
            }
            float local_max = 0.f;
#pragma unroll
            for (int item = 0; item < 16; ++item) {
                const float value = __bfloat162float(source.values[item]);
                if (!isfinite(value))
                    atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                             static_cast<unsigned long long>(kErrorNumeric));
                local_max = fused::max_abs(value, local_max);
            }
            const float max_abs = fused::warp_max(local_max);
            scale               = max_abs == 0.f ? 0.f : 448.f / max_abs;
            if (!isfinite(scale))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
            own.vector = make_uint4(0, 0, 0, 0);
            if (lane < 31) {
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float value = __bfloat162float(source.values[item]);
                    own.values[item]  = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
                }
            } else {
                own.metadata[0] = scale;
            }
            reinterpret_cast<uint4*>(params.local.packets + packet)[lane] = own.vector;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.local.ready + first);
            protocol_status = fused::wait_epoch(params.peer.ready + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
        if (packet < params.packet_count) {
            WarpPacketFp8 peer;
            peer.vector            = reinterpret_cast<const uint4*>(params.peer.packets + packet)[lane];
            const float peer_scale = __shfl_sync(0xffffffff, peer.metadata[0], 31);
            if (lane < 31) {
                WarpPacketBf16 result;
#pragma unroll
                for (int item = 0; item < 16; ++item) {
                    const float a       = scale == 0.f ? static_cast<float>(own.values[item]) :
                                                         static_cast<float>(own.values[item]) / scale;
                    const float b       = peer_scale == 0.f ? static_cast<float>(peer.values[item]) :
                                                              static_cast<float>(peer.values[item]) / peer_scale;
                    result.values[item] = __float2bfloat16_rn(params.rank == 0 ? a + b : b + a);
                    if (!isfinite(__bfloat162float(result.values[item])))
                        atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                                 static_cast<unsigned long long>(kErrorNumeric));
                }
                if (output_aligned && base + 16 <= params.numel) {
                    reinterpret_cast<uint4*>(output + base)[0] = result.vectors[0];
                    reinterpret_cast<uint4*>(output + base)[1] = result.vectors[1];
                } else {
#pragma unroll
                    for (int item = 0; item < 16; ++item)
                        if (base + item < params.numel)
                            output[base + item] = result.values[item];
                }
            }
        }
        __syncthreads();
        __threadfence_system();
        if (threadIdx.x == 0) {
            fused::store_release_sys(params.epoch, params.local.ack + first);
            protocol_status = fused::wait_epoch(params.peer.ack + first, params.epoch, kSpinLimit) ? 1 : 0;
        }
        __syncthreads();
        if (!protocol_status) {
            if (threadIdx.x == 0)
                set_timeout(params.local.error);
            return;
        }
    }
}

__global__ void one_shot_all_reduce_kernel(DeviceParams params, __nv_bfloat16 const* input, __nv_bfloat16* output) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t packet = blockIdx.x; packet < params.packet_count; packet += gridDim.x) {
        const size_t base = packet * fused::kPacketValues;
        for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
            const size_t index = base + item;
            if (index < params.numel)
                output[index] = input[index];
        }
        __syncthreads();
        if (!publish_reduce_and_ack(params, output, packet, warp_maxima, &protocol_status))
            return;
    }
}

__global__ void local_fc2_kernel(fused::Fc2ReferenceParams fc2, __nv_bfloat16* output, size_t numel) {
    if (reinterpret_cast<uint64_t const*>(fc2.error_flags) != nullptr
        && fused::load_acquire_sys(reinterpret_cast<uint64_t const*>(fc2.error_flags)) != 0)
        return;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < numel;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        output[index] = fused::fc2_token_h_reference(
            fc2, index / fused::kFc2ReferenceHiddenSize, index % fused::kFc2ReferenceHiddenSize);
    }
}

// Packet layouts are identical between local controls and fused candidates.
template<int Compute>
__device__ __forceinline__ void
compute_fc2_packet(fused::Fc2ReferenceParams fc2, __nv_bfloat16* output, size_t base, size_t numel) {
    if constexpr (Compute == 1) {
        fused::fc2_packet_mma(fc2, output, base, numel);
    } else if constexpr (Compute == 2) {
        fused::fc2_packet_staged(fc2, output, base, numel);
    } else {
        for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
            const size_t index = base + item;
            if (index < numel)
                output[index] = fused::fc2_token_h_reference(
                    fc2, index / fused::kFc2ReferenceHiddenSize, index % fused::kFc2ReferenceHiddenSize);
        }
    }
}

template<int Compute>
__global__ void local_fc2_packet_kernel(fused::Fc2ReferenceParams fc2, __nv_bfloat16* output, size_t numel) {
    __shared__ int good;
    if (threadIdx.x == 0)
        good = !fc2.error_flags || fused::load_acquire_sys(reinterpret_cast<uint64_t const*>(fc2.error_flags)) == 0;
    __syncthreads();
    if (!good)
        return;
    const size_t packets = (numel + fused::kPacketValues - 1) / fused::kPacketValues;
    for (size_t packet = blockIdx.x; packet < packets; packet += gridDim.x) {
        compute_fc2_packet<Compute>(fc2, output, packet * fused::kPacketValues, numel);
        __syncthreads();
    }
}

// Profiling is a separate instantiation and is never used for latency samples.
// Each CTA records [smid, packets, FC2 cycles, communication cycles, start ns, end ns].
template<bool Profile = false, int Compute = 0>
__global__ void
fused_fc2_kernel(DeviceParams params, fused::Fc2ReferenceParams fc2, __nv_bfloat16* output, uint64_t* stats = nullptr) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    uint64_t started = 0, compute_cycles = 0, communication_cycles = 0, packets_done = 0;
    uint32_t smid = 0;
    if constexpr (Profile) {
        if (threadIdx.x == 0) {
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
        }
    }
    for (size_t packet = blockIdx.x; packet < params.packet_count; packet += gridDim.x) {
        uint64_t before = 0;
        if constexpr (Profile) {
            if (threadIdx.x == 0)
                before = clock64();
        }
        const size_t base = packet * fused::kPacketValues;
        compute_fc2_packet<Compute>(fc2, output, base, params.numel);
        __syncthreads();
        if constexpr (Profile) {
            if (threadIdx.x == 0) {
                compute_cycles += clock64() - before;
                before = clock64();
            }
        }
        if (!publish_reduce_and_ack(params, output, packet, warp_maxima, &protocol_status))
            return;
        if constexpr (Profile) {
            if (threadIdx.x == 0) {
                communication_cycles += clock64() - before;
                ++packets_done;
            }
        }
    }
    if constexpr (Profile) {
        if (threadIdx.x == 0) {
            uint64_t ended;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ended));
            uint64_t* row = stats + blockIdx.x * 6;
            row[0]        = smid;
            row[1]        = packets_done;
            row[2]        = compute_cycles;
            row[3]        = communication_cycles;
            row[4]        = started;
            row[5]        = ended;
        }
    }
}

// The producer still computes packet by packet, but publication and retirement
// happen once per up-to-16 KiB group. Retaining the local-MMA grid stride
// preserves inter-CTA weight locality. Peer staging uses 16 KiB shared memory.
__global__ void fused_fc2_batch_kernel(DeviceParams params, fused::Fc2ReferenceParams fc2, __nv_bfloat16* output) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    const size_t packets_per_cta = params.packet_count / gridDim.x;
    const size_t batch           = packets_per_cta < 1 ? 1 : min(kPacketBatch, packets_per_cta);
    for (size_t first = blockIdx.x; first < params.packet_count; first += gridDim.x * batch) {
        const size_t count = min(batch, (params.packet_count - first + gridDim.x - 1) / gridDim.x);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t packet = first + offset * gridDim.x;
            compute_fc2_packet<1>(fc2, output, packet * fused::kPacketValues, params.numel);
            __syncthreads();
        }
        if (!publish_batch_reduce_and_ack(params, output, first, count, warp_maxima, &protocol_status))
            return;
    }
}

class MoeTpFusedFp8 {
public:
    MoeTpFusedFp8(uint64_t max_numel, int device_index, int rank, int blocks = kDefaultBlocks):
        max_numel_(max_numel),
        device_(device_index),
        rank_(rank),
        blocks_(blocks),
        packet_capacity_(div_up(max_numel, fused::kPacketValues)) {
        TORCH_CHECK(max_numel_ > 0, "max_numel must be positive");
        TORCH_CHECK(rank_ >= 0 && rank_ < fused::kTpSize, "rank must be 0 or 1");
        TORCH_CHECK(blocks_ >= 0, "blocks must be nonnegative (zero selects the resident grid)");
        c10::cuda::CUDAGuard guard(device_);
        ensure_resident_grid();
        try {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&packets_), packet_capacity_ * sizeof(fused::Packet)),
                       "cudaMalloc(packets)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&inbox_), packet_capacity_ * sizeof(fused::Packet)),
                       "cudaMalloc(inbox)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&completion_), blocks_ * sizeof(uint64_t)),
                       "cudaMalloc(completion)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ready_), packet_capacity_ * sizeof(uint64_t)),
                       "cudaMalloc(ready)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ack_), packet_capacity_ * sizeof(uint64_t)),
                       "cudaMalloc(ack)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&error_), sizeof(uint64_t)), "cudaMalloc(error)");
            cuda_check(cudaMemsetAsync(ready_, 0, packet_capacity_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(ready)");
            cuda_check(cudaMemsetAsync(ack_, 0, packet_capacity_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(ack)");
            cuda_check(cudaMemsetAsync(error_, 0, sizeof(uint64_t), current_stream()), "cudaMemsetAsync(error)");
            cuda_check(cudaMemsetAsync(completion_, 0, blocks_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(completion)");
        } catch (...) {
            release_noexcept();
            throw;
        }
    }

    ~MoeTpFusedFp8() {
        release_noexcept();
    }

    py::bytes get_ipc_handle() {
        ensure_open();
        c10::cuda::CUDAGuard guard(device_);
        IpcWireHandle        wire{};
        wire.magic        = kWireMagic;
        wire.version      = kWireVersion;
        wire.device       = device_;
        wire.rank         = rank_;
        wire.blocks       = blocks_;
        wire.max_numel    = max_numel_;
        wire.packet_count = packet_capacity_;
        cuda_check(cudaIpcGetMemHandle(&wire.packets, packets_), "cudaIpcGetMemHandle(packets)");
        cuda_check(cudaIpcGetMemHandle(&wire.inbox, inbox_), "cudaIpcGetMemHandle(inbox)");
        cuda_check(cudaIpcGetMemHandle(&wire.ready, ready_), "cudaIpcGetMemHandle(ready)");
        cuda_check(cudaIpcGetMemHandle(&wire.ack, ack_), "cudaIpcGetMemHandle(ack)");
        cuda_check(cudaIpcGetMemHandle(&wire.error, error_), "cudaIpcGetMemHandle(error)");
        return py::bytes(reinterpret_cast<char const*>(&wire), sizeof(wire));
    }

    void open_peer(py::bytes raw) {
        ensure_open();
        const std::string bytes = raw;
        TORCH_CHECK(bytes.size() == sizeof(IpcWireHandle), "invalid fused MoE IPC handle size");
        IpcWireHandle wire{};
        std::memcpy(&wire, bytes.data(), sizeof(wire));
        TORCH_CHECK(wire.magic == kWireMagic && wire.version == kWireVersion, "invalid fused MoE IPC wire version");
        TORCH_CHECK(wire.rank == 1 - rank_ && wire.max_numel == max_numel_ && wire.blocks == blocks_
                        && wire.packet_count == packet_capacity_,
                    "peer fused MoE workspace mismatch");
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        try {
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_packets_), wire.packets, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(packets; requires CUDA IPC P2P)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_inbox_), wire.inbox, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(inbox; requires CUDA IPC P2P)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_ready_), wire.ready, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(ready; requires CUDA IPC P2P)");
            cuda_check(
                cudaIpcOpenMemHandle(reinterpret_cast<void**>(&peer_ack_), wire.ack, cudaIpcMemLazyEnablePeerAccess),
                "cudaIpcOpenMemHandle(ack; requires CUDA IPC P2P)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_error_), wire.error, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(error; requires CUDA IPC P2P)");
        } catch (...) {
            close_peer_noexcept();
            throw;
        }
    }

    void close_peer() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
    }

    template<int PacketBatch = 1>
    void all_reduce(torch::Tensor input, torch::Tensor output) {
        validate_bf16_pair(input, output, "all_reduce");
        if constexpr (PacketBatch == 32)
            ensure_compute_resident<3>();
        if constexpr (PacketBatch == 16) {
            ensure_warp_resident();
            const auto in_begin  = reinterpret_cast<uintptr_t>(input.data_ptr());
            const auto out_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
            TORCH_CHECK(in_begin == out_begin
                            || std::max(in_begin, out_begin)
                                   >= std::min(in_begin + input.nbytes(), out_begin + output.nbytes()),
                        "warp transport only permits disjoint or exactly aliased input/output");
        }
        launch_common(input.numel(), [&](DeviceParams params, cudaStream_t stream) {
            if constexpr (PacketBatch == 16)
                one_shot_all_reduce_warp_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(
                    params, bf16_ptr(input), bf16_ptr(output));
            else if constexpr (PacketBatch == 32)
                one_shot_all_reduce_batch_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(
                    params, bf16_ptr(input), bf16_ptr(output));
            else
                one_shot_all_reduce_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(
                    params, bf16_ptr(input), bf16_ptr(output));
            cuda_check(cudaGetLastError(), "one_shot_all_reduce_kernel launch");
        });
    }

    void all_reduce_push_warp(torch::Tensor input, torch::Tensor output) {
        validate_bf16_pair(input, output, "push warp all_reduce");
        ensure_push_warp_resident();
        const auto in_begin  = reinterpret_cast<uintptr_t>(input.data_ptr());
        const auto out_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
        TORCH_CHECK(in_begin == out_begin
                        || std::max(in_begin, out_begin)
                               >= std::min(in_begin + input.nbytes(), out_begin + output.nbytes()),
                    "push warp transport only permits disjoint or exactly aliased input/output");
        launch_common(
            input.numel(),
            [&](DeviceParams params, cudaStream_t stream) {
                one_shot_all_reduce_push_warp_kernel<false>
                    <<<blocks_, fused::kPacketThreads, 0, stream>>>(params, bf16_ptr(input), bf16_ptr(output));
                cuda_check(cudaGetLastError(), "one_shot_all_reduce_push_warp_kernel launch");
            },
            /*published_inbox=*/true);
    }

    void all_reduce_push_warp_deferred(torch::Tensor input, torch::Tensor output) {
        validate_bf16_pair(input, output, "deferred push warp all_reduce");
        ensure_push_warp_deferred_resident();
        const auto in_begin  = reinterpret_cast<uintptr_t>(input.data_ptr());
        const auto out_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
        TORCH_CHECK(in_begin == out_begin
                        || std::max(in_begin, out_begin)
                               >= std::min(in_begin + input.nbytes(), out_begin + output.nbytes()),
                    "deferred push warp transport only permits disjoint or exactly aliased input/output");
        launch_common(
            input.numel(),
            [&](DeviceParams params, cudaStream_t stream) {
                one_shot_all_reduce_push_warp_kernel<true>
                    <<<blocks_, fused::kPacketThreads, 0, stream>>>(params, bf16_ptr(input), bf16_ptr(output));
                cuda_check(cudaGetLastError(), "one_shot_all_reduce_push_warp_deferred_kernel launch");
            },
            /*published_inbox=*/true);
    }

    // Experimental only: consume DeepGEMM's per-routed-row BF16 FC2 output,
    // reproduce ep_gather + shared gate locally, and use the push transport.
    // This does not register a serving backend or change router behavior.
    void gather_gate_push(torch::Tensor down_output,
                          torch::Tensor topk_ids,
                          torch::Tensor topk_weights,
                          torch::Tensor output_index,
                          torch::Tensor shared,
                          torch::Tensor gate,
                          torch::Tensor output) {
        const size_t tokens =
            validate_gather_gate_push(down_output, topk_ids, topk_weights, output_index, shared, gate, output);
        ensure_gather_gate_push_resident();
        launch_common(
            output.numel(),
            [&](DeviceParams params, cudaStream_t stream) {
                gather_gate_push_warp_kernel<false>
                    <<<blocks_, fused::kPacketThreads, 0, stream>>>(params,
                                                                    bf16_ptr(down_output),
                                                                    topk_ids.data_ptr<int64_t>(),
                                                                    topk_weights.data_ptr<float>(),
                                                                    output_index.data_ptr<int64_t>(),
                                                                    down_output.size(0),
                                                                    bf16_ptr(shared),
                                                                    bf16_ptr(gate),
                                                                    bf16_ptr(output));
                cuda_check(cudaGetLastError(), "gather_gate_push_warp_kernel launch");
            },
            /*published_inbox=*/true);
        (void)tokens;
    }

    void gather_gate_push_deferred(torch::Tensor down_output,
                                   torch::Tensor topk_ids,
                                   torch::Tensor topk_weights,
                                   torch::Tensor output_index,
                                   torch::Tensor shared,
                                   torch::Tensor gate,
                                   torch::Tensor output) {
        (void)validate_gather_gate_push(down_output, topk_ids, topk_weights, output_index, shared, gate, output);
        ensure_gather_gate_push_deferred_resident();
        launch_common(
            output.numel(),
            [&](DeviceParams params, cudaStream_t stream) {
                gather_gate_push_warp_kernel<true>
                    <<<blocks_, fused::kPacketThreads, 0, stream>>>(params,
                                                                    bf16_ptr(down_output),
                                                                    topk_ids.data_ptr<int64_t>(),
                                                                    topk_weights.data_ptr<float>(),
                                                                    output_index.data_ptr<int64_t>(),
                                                                    down_output.size(0),
                                                                    bf16_ptr(shared),
                                                                    bf16_ptr(gate),
                                                                    bf16_ptr(output));
                cuda_check(cudaGetLastError(), "gather_gate_push_deferred_kernel launch");
            },
            /*published_inbox=*/true);
    }

    // Experimental only: encode sigmoid(gate) * shared + experts directly to
    // the unchanged warp-packet transport.  This is intentionally separate
    // from the serving TP all-reduce and the established all_reduce API.
    void gated_all_reduce_warp(torch::Tensor experts, torch::Tensor shared, torch::Tensor gate, torch::Tensor output) {
        const int hidden_size = validate_gated_warp(experts, shared, gate, output, /*require_peer=*/true);
        ensure_gated_warp_resident();
        launch_common(experts.numel(), [&](DeviceParams params, cudaStream_t stream) {
            gated_all_reduce_warp_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(
                params, bf16_ptr(experts), bf16_ptr(shared), bf16_ptr(gate), bf16_ptr(output), hidden_size);
            cuda_check(cudaGetLastError(), "gated_all_reduce_warp_kernel launch");
        });
    }

    // Debug oracle for the exact Triton gate arithmetic before FP8 encoding or
    // communication.  It shares the native validation with the fused path.
    void gated_local_warp(torch::Tensor experts, torch::Tensor shared, torch::Tensor gate, torch::Tensor output) {
        const int hidden_size = validate_gated_warp(experts, shared, gate, output, /*require_peer=*/false);
        bind_stream();
        c10::cuda::CUDAGuard guard(device_);
        gated_local_warp_kernel<<<blocks_, fused::kPacketThreads, 0, current_stream()>>>(
            bf16_ptr(experts), bf16_ptr(shared), bf16_ptr(gate), bf16_ptr(output), experts.numel(), hidden_size);
        cuda_check(cudaGetLastError(), "gated_local_warp_kernel launch");
    }

    template<int Compute = 0>
    void local_fc2(torch::Tensor activation_fp8,
                   torch::Tensor activation_scale,
                   torch::Tensor weight_fp8,
                   torch::Tensor weight_scale,
                   torch::Tensor route_ids,
                   torch::Tensor route_weights,
                   py::object    gated_shared,
                   torch::Tensor output) {
        const auto   fc2   = validate_fc2(activation_fp8,
                                      activation_scale,
                                      weight_fp8,
                                      weight_scale,
                                      route_ids,
                                      route_weights,
                                      gated_shared,
                                      output,
                                      /*require_peer=*/false);
        const size_t numel = output.numel();
        bind_stream();
        c10::cuda::CUDAGuard guard(device_);
        if constexpr (Compute == 0)
            local_fc2_kernel<<<blocks_, fused::kPacketThreads, 0, current_stream()>>>(fc2, bf16_ptr(output), numel);
        else
            local_fc2_packet_kernel<Compute>
                <<<blocks_, fused::kPacketThreads, 0, current_stream()>>>(fc2, bf16_ptr(output), numel);
        cuda_check(cudaGetLastError(), "local_fc2_kernel launch");
    }

    template<int Compute = 0>
    void fused_fc2(torch::Tensor activation_fp8,
                   torch::Tensor activation_scale,
                   torch::Tensor weight_fp8,
                   torch::Tensor weight_scale,
                   torch::Tensor route_ids,
                   torch::Tensor route_weights,
                   py::object    gated_shared,
                   torch::Tensor output) {
        const auto fc2 = validate_fc2(activation_fp8,
                                      activation_scale,
                                      weight_fp8,
                                      weight_scale,
                                      route_ids,
                                      route_weights,
                                      gated_shared,
                                      output,
                                      /*require_peer=*/true);
        ensure_compute_resident<Compute>();
        launch_common(output.numel(), [&](DeviceParams params, cudaStream_t stream) {
            if constexpr (Compute == 3)
                fused_fc2_batch_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(params, fc2, bf16_ptr(output));
            else
                fused_fc2_kernel<false, Compute>
                    <<<blocks_, fused::kPacketThreads, 0, stream>>>(params, fc2, bf16_ptr(output));
            cuda_check(cudaGetLastError(), "fused_fc2_kernel launch");
        });
    }

    torch::Tensor copy_local_packets() {
        ensure_open();
        TORCH_CHECK(last_packets_ != 0, "no fused/all_reduce invocation has published packets");
        c10::cuda::CUDAGuard guard(device_);
        auto                 options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_);
        auto result = torch::empty({static_cast<int64_t>(last_packets_ * sizeof(fused::Packet))}, options);
        cuda_check(cudaMemcpyAsync(result.data_ptr(),
                                   packets_,
                                   last_packets_ * sizeof(fused::Packet),
                                   cudaMemcpyDeviceToDevice,
                                   current_stream()),
                   "cudaMemcpyAsync(copy_local_packets)");
        return result;
    }

    torch::Tensor copy_local_inbox() {
        ensure_open();
        TORCH_CHECK(last_push_ && last_packets_ != 0, "no push invocation has published an inbox");
        c10::cuda::CUDAGuard guard(device_);
        auto                 options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_);
        auto result = torch::empty({static_cast<int64_t>(last_packets_ * sizeof(fused::Packet))}, options);
        cuda_check(cudaMemcpyAsync(result.data_ptr(),
                                   inbox_,
                                   last_packets_ * sizeof(fused::Packet),
                                   cudaMemcpyDeviceToDevice,
                                   current_stream()),
                   "cudaMemcpyAsync(copy_local_inbox)");
        return result;
    }

    // Debug-only asynchronous device scalar. Callers must first synchronize the
    // context's dedicated stream, then may consume this tensor on their current
    // stream. It deliberately does not bind that debug copy to the launch stream.
    torch::Tensor error_status() {
        ensure_open();
        c10::cuda::CUDAGuard guard(device_);
        auto result = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA, device_));
        cuda_check(
            cudaMemcpyAsync(result.data_ptr(), error_, sizeof(uint64_t), cudaMemcpyDeviceToDevice, current_stream()),
            "cudaMemcpyAsync(error_status)");
        return result;
    }

    int blocks() const {
        return blocks_;
    }

    py::dict launch_info() const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        cudaFuncAttributes   attributes{};
        int                  active = 0, profile_active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaFuncGetAttributes(&attributes, fused_fc2_kernel<false>), "cudaFuncGetAttributes");
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fused_fc2_kernel<false>, fused::kPacketThreads, 0),
            "fused occupancy");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &profile_active, fused_fc2_kernel<true>, fused::kPacketThreads, 0),
                   "profile occupancy");
        py::dict info;
        info["sm_count"]                    = properties.multiProcessorCount;
        info["max_resident_blocks"]         = active * properties.multiProcessorCount;
        info["profile_max_resident_blocks"] = profile_active * properties.multiProcessorCount;
        info["active_blocks_per_sm"]        = active;
        info["regs_per_thread"]             = attributes.numRegs;
        info["static_shared_bytes"]         = attributes.sharedSizeBytes;
        info["threads_per_block"]           = fused::kPacketThreads;
        info["max_threads_per_sm"]          = properties.maxThreadsPerMultiProcessor;
        info["blocks"]                      = blocks_;
        append_compute_info<1>(info, "mma", properties.multiProcessorCount);
        append_compute_info<2>(info, "staged", properties.multiProcessorCount);
        append_compute_info<3>(info, "mma_batch", properties.multiProcessorCount);
        int warp_active = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, one_shot_all_reduce_warp_kernel), "warp transport attributes");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &warp_active, one_shot_all_reduce_warp_kernel, fused::kPacketThreads, 0),
                   "warp transport occupancy");
        info["warp_max_resident_blocks"] = warp_active * properties.multiProcessorCount;
        info["warp_regs_per_thread"]     = attributes.numRegs;
        info["warp_static_shared_bytes"] = attributes.sharedSizeBytes;
        int gated_warp_active            = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, gated_all_reduce_warp_kernel), "gated warp transport attributes");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &gated_warp_active, gated_all_reduce_warp_kernel, fused::kPacketThreads, 0),
                   "gated warp transport occupancy");
        info["gated_warp_max_resident_blocks"] = gated_warp_active * properties.multiProcessorCount;
        info["gated_warp_regs_per_thread"]     = attributes.numRegs;
        info["gated_warp_static_shared_bytes"] = attributes.sharedSizeBytes;
        int push_warp_active                   = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, one_shot_all_reduce_push_warp_kernel<false>),
                   "push warp transport attributes");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &push_warp_active, one_shot_all_reduce_push_warp_kernel<false>, fused::kPacketThreads, 0),
                   "push warp transport occupancy");
        info["push_warp_max_resident_blocks"] = push_warp_active * properties.multiProcessorCount;
        info["push_warp_regs_per_thread"]     = attributes.numRegs;
        info["push_warp_static_shared_bytes"] = attributes.sharedSizeBytes;
        int push_warp_deferred_active         = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, one_shot_all_reduce_push_warp_kernel<true>),
                   "deferred push warp transport attributes");
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &push_warp_deferred_active, one_shot_all_reduce_push_warp_kernel<true>, fused::kPacketThreads, 0),
            "deferred push warp transport occupancy");
        info["push_warp_deferred_max_resident_blocks"] = push_warp_deferred_active * properties.multiProcessorCount;
        info["push_warp_deferred_regs_per_thread"]     = attributes.numRegs;
        info["push_warp_deferred_static_shared_bytes"] = attributes.sharedSizeBytes;
        int gather_gate_push_active                    = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, gather_gate_push_warp_kernel<false>),
                   "gather gate push transport attributes");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &gather_gate_push_active, gather_gate_push_warp_kernel<false>, fused::kPacketThreads, 0),
                   "gather gate push transport occupancy");
        info["gather_gate_push_max_resident_blocks"] = gather_gate_push_active * properties.multiProcessorCount;
        info["gather_gate_push_regs_per_thread"]     = attributes.numRegs;
        info["gather_gate_push_static_shared_bytes"] = attributes.sharedSizeBytes;
        int gather_gate_push_deferred_active         = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, gather_gate_push_warp_kernel<true>),
                   "deferred gather gate push attributes");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &gather_gate_push_deferred_active, gather_gate_push_warp_kernel<true>, fused::kPacketThreads, 0),
                   "deferred gather gate push occupancy");
        info["gather_gate_push_deferred_max_resident_blocks"] =
            gather_gate_push_deferred_active * properties.multiProcessorCount;
        info["gather_gate_push_deferred_regs_per_thread"]     = attributes.numRegs;
        info["gather_gate_push_deferred_static_shared_bytes"] = attributes.sharedSizeBytes;
        return info;
    }

    void profile_fc2(torch::Tensor activation_fp8,
                     torch::Tensor activation_scale,
                     torch::Tensor weight_fp8,
                     torch::Tensor weight_scale,
                     torch::Tensor route_ids,
                     torch::Tensor route_weights,
                     py::object    gated_shared,
                     torch::Tensor output,
                     torch::Tensor stats) {
        const auto fc2 = validate_fc2(activation_fp8,
                                      activation_scale,
                                      weight_fp8,
                                      weight_scale,
                                      route_ids,
                                      route_weights,
                                      gated_shared,
                                      output,
                                      true);
        TORCH_CHECK(stats.is_cuda() && stats.device().index() == device_ && stats.is_contiguous()
                        && stats.scalar_type() == at::kUInt64 && stats.dim() == 2 && stats.size(0) == blocks_
                        && stats.size(1) == 6,
                    "stats must be uint64 [blocks,6] on the context device");
        c10::cuda::CUDAGuard guard(device_);
        int                  active = 0;
        cudaDeviceProp       properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fused_fc2_kernel<true>, fused::kPacketThreads, 0),
            "profile occupancy");
        TORCH_CHECK(blocks_ <= active * properties.multiProcessorCount, "profile grid exceeds resident capacity");
        launch_common(output.numel(), [&](DeviceParams params, cudaStream_t stream) {
            fused_fc2_kernel<true><<<blocks_, fused::kPacketThreads, 0, stream>>>(
                params, fc2, bf16_ptr(output), stats.data_ptr<uint64_t>());
            cuda_check(cudaGetLastError(), "profile fused_fc2_kernel launch");
        });
    }

    void close() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        free_local_checked();
        bound_stream_ = nullptr;
        stream_bound_ = false;
        last_packets_ = 0;
        last_push_    = false;
    }

private:
    static bool overlaps(torch::Tensor const& left, torch::Tensor const& right) {
        const auto left_begin  = reinterpret_cast<uintptr_t>(left.data_ptr());
        const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
        return std::max(left_begin, right_begin) < std::min(left_begin + left.nbytes(), right_begin + right.nbytes());
    }

    int validate_gated_warp(torch::Tensor const& experts,
                            torch::Tensor const& shared,
                            torch::Tensor const& gate,
                            torch::Tensor const& output,
                            bool                 require_peer) const {
        if (require_peer)
            ensure_peer_open();
        else
            ensure_open();
        validate_bf16_output(experts, "gated warp experts");
        validate_bf16_output(shared, "gated warp shared");
        validate_bf16_output(gate, "gated warp gate");
        validate_bf16_output(output, "gated warp output");
        TORCH_CHECK(experts.dim() == 2 && experts.size(0) > 0 && experts.size(1) > 0
                        && experts.size(1) <= std::numeric_limits<int>::max()
                        && (experts.size(1) == fused::kFc2ReferenceHiddenSize || experts.size(1) % 16 == 0),
                    "gated warp experts must be BF16 [T,H], H=2048 or H divisible by 16");
        TORCH_CHECK(shared.sizes() == experts.sizes() && output.sizes() == experts.sizes(),
                    "gated warp shared/output must match experts [T,H]");
        TORCH_CHECK(gate.dim() == 2 && gate.size(0) == experts.size(0) && gate.size(1) == 1,
                    "gated warp gate must be BF16 [T,1]");
        TORCH_CHECK(experts.numel() > 0 && static_cast<uint64_t>(experts.numel()) <= max_numel_,
                    "gated warp output must fit the context workspace");
        TORCH_CHECK((experts.data_ptr() == output.data_ptr()) || !overlaps(experts, output),
                    "gated warp experts/output must be disjoint or exactly aliased");
        TORCH_CHECK(!overlaps(shared, output) && !overlaps(gate, output) && !overlaps(experts, shared)
                        && !overlaps(experts, gate) && !overlaps(shared, gate),
                    "gated warp experts/shared/gate must have disjoint allocations; only experts/output may alias");
        return static_cast<int>(experts.size(1));
    }

    size_t validate_gather_gate_push(torch::Tensor const& down_output,
                                     torch::Tensor const& topk_ids,
                                     torch::Tensor const& topk_weights,
                                     torch::Tensor const& output_index,
                                     torch::Tensor const& shared,
                                     torch::Tensor const& gate,
                                     torch::Tensor const& output) const {
        ensure_peer_open();
        auto check_cuda_contiguous = [&](torch::Tensor const& tensor, char const* name) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device().index() == device_ && tensor.is_contiguous(),
                        name,
                        " must be a contiguous CUDA tensor on the context device");
        };
        check_cuda_contiguous(down_output, "gather gate push down_output");
        check_cuda_contiguous(topk_ids, "gather gate push topk_ids");
        check_cuda_contiguous(topk_weights, "gather gate push topk_weights");
        check_cuda_contiguous(output_index, "gather gate push output_index");
        validate_bf16_output(shared, "gather gate push shared");
        validate_bf16_output(gate, "gather gate push gate");
        validate_bf16_output(output, "gather gate push output");
        TORCH_CHECK(down_output.scalar_type() == at::kBFloat16 && down_output.dim() == 2 && down_output.size(0) > 0
                        && down_output.size(1) == fused::kFc2ReferenceHiddenSize,
                    "gather gate push down_output must be BF16 [rows,2048]");
        TORCH_CHECK(output.dim() == 2, "gather gate push output must be rank-2");
        const int64_t tokens = output.size(0);
        TORCH_CHECK(tokens > 0 && output.size(1) == fused::kFc2ReferenceHiddenSize
                        && static_cast<uint64_t>(output.numel()) <= max_numel_,
                    "gather gate push output must be nonempty BF16 [T,2048] within workspace");
        TORCH_CHECK(topk_ids.scalar_type() == at::kLong && output_index.scalar_type() == at::kLong
                        && topk_weights.scalar_type() == at::kFloat && topk_ids.dim() == 2 && topk_ids.size(0) == tokens
                        && topk_ids.size(1) == 8 && output_index.sizes() == topk_ids.sizes()
                        && topk_weights.sizes() == topk_ids.sizes(),
                    "gather gate push ids/index must be int64 [T,8] and weights float32 [T,8]");
        TORCH_CHECK(shared.sizes() == output.sizes() && gate.dim() == 2 && gate.size(0) == tokens && gate.size(1) == 1,
                    "gather gate push shared must be BF16 [T,2048] and gate BF16 [T,1]");
        TORCH_CHECK(!overlaps(output, down_output) && !overlaps(output, topk_ids) && !overlaps(output, topk_weights)
                        && !overlaps(output, output_index) && !overlaps(output, shared) && !overlaps(output, gate),
                    "gather gate push output must not overlap any input");
        TORCH_CHECK(!overlaps(down_output, topk_ids) && !overlaps(down_output, topk_weights)
                        && !overlaps(down_output, output_index) && !overlaps(down_output, shared)
                        && !overlaps(down_output, gate) && !overlaps(topk_ids, topk_weights)
                        && !overlaps(topk_ids, output_index) && !overlaps(topk_ids, shared) && !overlaps(topk_ids, gate)
                        && !overlaps(topk_weights, output_index) && !overlaps(topk_weights, shared)
                        && !overlaps(topk_weights, gate) && !overlaps(output_index, shared)
                        && !overlaps(output_index, gate) && !overlaps(shared, gate),
                    "gather gate push inputs must have disjoint allocations");
        return static_cast<size_t>(tokens);
    }

    void ensure_warp_resident() const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        int                  active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active, one_shot_all_reduce_warp_kernel, fused::kPacketThreads, 0),
                   "warp transport occupancy");
        TORCH_CHECK(active > 0 && blocks_ <= active * properties.multiProcessorCount,
                    "warp transport grid exceeds resident capacity");
    }

    void ensure_gated_warp_resident() const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        int                  active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active, gated_all_reduce_warp_kernel, fused::kPacketThreads, 0),
                   "gated warp transport occupancy");
        TORCH_CHECK(active > 0 && blocks_ <= active * properties.multiProcessorCount,
                    "gated warp transport grid exceeds resident capacity");
    }

    void ensure_push_warp_resident() const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        int                  active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active, one_shot_all_reduce_push_warp_kernel<false>, fused::kPacketThreads, 0),
                   "push warp transport occupancy");
        TORCH_CHECK(active > 0 && blocks_ <= active * properties.multiProcessorCount,
                    "push warp transport grid exceeds resident capacity");
    }

    void ensure_push_warp_deferred_resident() const {
        ensure_resident(one_shot_all_reduce_push_warp_kernel<true>, "deferred push warp transport");
    }

    void ensure_gather_gate_push_resident() const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        int                  active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active, gather_gate_push_warp_kernel<false>, fused::kPacketThreads, 0),
                   "gather gate push transport occupancy");
        TORCH_CHECK(active > 0 && blocks_ <= active * properties.multiProcessorCount,
                    "gather gate push transport grid exceeds resident capacity");
    }

    void ensure_gather_gate_push_deferred_resident() const {
        ensure_resident(gather_gate_push_warp_kernel<true>, "deferred gather gate push transport");
    }

    template<class Kernel>
    void ensure_resident(Kernel kernel, char const* name) const {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp       properties{};
        int                  active = 0;
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, fused::kPacketThreads, 0), name);
        TORCH_CHECK(
            active > 0 && blocks_ <= active * properties.multiProcessorCount, name, " grid exceeds resident capacity");
    }

    cudaStream_t current_stream() const {
        return at::cuda::getCurrentCUDAStream(device_).stream();
    }

    void ensure_open() const {
        TORCH_CHECK(packets_ && inbox_ && completion_ && ready_ && ack_ && error_, "MoeTpFusedFp8 context is closed");
    }

    void ensure_peer_open() const {
        ensure_open();
        TORCH_CHECK(peer_packets_ && peer_inbox_ && peer_ready_ && peer_ack_ && peer_error_,
                    "open_peer must be called first");
    }

    void bind_stream() {
        const cudaStream_t stream = current_stream();
        if (stream_bound_) {
            TORCH_CHECK(bound_stream_ == stream, "MoeTpFusedFp8 context may only use one CUDA stream");
        } else {
            bound_stream_ = stream;
            stream_bound_ = true;
        }
        cudaStreamCaptureStatus status{};
        cuda_check(cudaStreamIsCapturing(stream, &status), "cudaStreamIsCapturing");
        TORCH_CHECK(status == cudaStreamCaptureStatusNone, "MoeTpFusedFp8 does not support CUDA graph capture");
    }

    template<int Compute>
    static const void* compute_kernel() {
        if constexpr (Compute == 3)
            return reinterpret_cast<const void*>(fused_fc2_batch_kernel);
        else
            return reinterpret_cast<const void*>(fused_fc2_kernel<false, Compute>);
    }

    template<int Compute>
    void append_compute_info(py::dict& info, std::string const& prefix, int sm_count) const {
        cudaFuncAttributes attributes{};
        int                active = 0;
        cuda_check(cudaFuncGetAttributes(&attributes, compute_kernel<Compute>()), "compute attributes");
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, compute_kernel<Compute>(), fused::kPacketThreads, 0),
            "compute occupancy");
        if constexpr (Compute == 3) {
            int transport_active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &transport_active, one_shot_all_reduce_batch_kernel, fused::kPacketThreads, 0),
                       "batch transport occupancy");
            active = std::min(active, transport_active);
        }
        info[py::str(prefix + "_max_resident_blocks")] = active * sm_count;
        info[py::str(prefix + "_regs_per_thread")]     = attributes.numRegs;
        info[py::str(prefix + "_static_shared_bytes")] = attributes.sharedSizeBytes;
    }

    template<int Compute>
    void ensure_compute_resident() const {
        c10::cuda::CUDAGuard guard(device_);
        int                  active = 0;
        cudaDeviceProp       properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, compute_kernel<Compute>(), fused::kPacketThreads, 0),
            "compute occupancy");
        if constexpr (Compute == 3) {
            int transport_active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &transport_active, one_shot_all_reduce_batch_kernel, fused::kPacketThreads, 0),
                       "batch transport occupancy");
            active = std::min(active, transport_active);
        }
        TORCH_CHECK(active > 0 && blocks_ <= active * properties.multiProcessorCount,
                    "selected FC2 compute grid exceeds resident capacity");
    }

    void ensure_resident_grid() {
        int active = 0;
        cuda_check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fused_fc2_kernel<false>, fused::kPacketThreads, 0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(fused_fc2_kernel)");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        const int64_t limit = static_cast<int64_t>(active) * properties.multiProcessorCount;
        TORCH_CHECK(limit > 0, "fused MoE kernel has no resident launch configuration");
        if (blocks_ == 0)
            blocks_ = static_cast<int>(std::min<uint64_t>(limit, packet_capacity_));
        TORCH_CHECK(blocks_ <= limit, "fused MoE blocks=", blocks_, " exceeds resident limit=", limit);
    }

    fused::Fc2ReferenceParams validate_fc2(torch::Tensor activation_fp8,
                                           torch::Tensor activation_scale,
                                           torch::Tensor weight_fp8,
                                           torch::Tensor weight_scale,
                                           torch::Tensor route_ids,
                                           torch::Tensor route_weights,
                                           py::object    gated_shared,
                                           torch::Tensor output,
                                           bool          require_peer) const {
        if (require_peer)
            ensure_peer_open();
        else
            ensure_open();
        validate_bf16_output(output, "fused/local FC2 output");
        auto check_cuda_contiguous = [&](torch::Tensor const& tensor, char const* name) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device().index() == device_ && tensor.is_contiguous(),
                        name,
                        " must be a contiguous CUDA tensor on the context device");
        };
        check_cuda_contiguous(activation_fp8, "activation_fp8");
        check_cuda_contiguous(activation_scale, "activation_scale");
        check_cuda_contiguous(weight_fp8, "weight_fp8");
        check_cuda_contiguous(weight_scale, "weight_scale");
        check_cuda_contiguous(route_ids, "route_ids");
        check_cuda_contiguous(route_weights, "route_weights");
        TORCH_CHECK(activation_fp8.scalar_type() == at::kFloat8_e4m3fn
                        && weight_fp8.scalar_type() == at::kFloat8_e4m3fn,
                    "activation_fp8 and weight_fp8 must be Float8_e4m3fn");
        TORCH_CHECK(activation_scale.scalar_type() == at::kFloat && weight_scale.scalar_type() == at::kFloat
                        && route_weights.scalar_type() == at::kFloat && route_ids.scalar_type() == at::kInt,
                    "scales/route_weights must be float32 and route_ids int32");
        TORCH_CHECK(activation_fp8.dim() == 3 && activation_fp8.size(1) == fused::kFc2ReferenceTopK
                        && activation_fp8.size(2) == fused::kFc2ReferenceIntermediateSize,
                    "activation_fp8 must be [T,8,256]");
        const int tokens = activation_fp8.size(0);
        TORCH_CHECK(activation_scale.dim() == 3 && activation_scale.size(0) == tokens
                        && activation_scale.size(1) == fused::kFc2ReferenceTopK
                        && activation_scale.size(2) == fused::kFc2ReferenceKBlocks,
                    "activation_scale must be [T,8,2]");
        TORCH_CHECK(weight_fp8.dim() == 3 && weight_fp8.size(1) == fused::kFc2ReferenceHiddenSize
                        && weight_fp8.size(2) == fused::kFc2ReferenceIntermediateSize,
                    "weight_fp8 must be [E,2048,256]");
        const int experts = weight_fp8.size(0);
        TORCH_CHECK(experts > 0 && experts <= fused::kFc2ReferenceExperts, "weight expert count must be in [1,256]");
        TORCH_CHECK(weight_scale.dim() == 3 && weight_scale.size(0) == experts
                        && weight_scale.size(1) == fused::kFc2ReferenceHBlocks
                        && weight_scale.size(2) == fused::kFc2ReferenceKBlocks,
                    "weight_scale must be [E,16,2]");
        TORCH_CHECK(route_ids.dim() == 2 && route_ids.size(0) == tokens && route_ids.size(1) == fused::kFc2ReferenceTopK
                        && route_weights.dim() == 2 && route_weights.size(0) == tokens
                        && route_weights.size(1) == fused::kFc2ReferenceTopK,
                    "route ids/weights must be [T,8]");
        TORCH_CHECK(output.dim() == 2 && output.size(0) == tokens && output.size(1) == fused::kFc2ReferenceHiddenSize
                        && static_cast<uint64_t>(output.numel()) <= max_numel_,
                    "output must be [T,2048] and fit the context workspace");
        __nv_bfloat16 const* shared_ptr = nullptr;
        if (!gated_shared.is_none()) {
            auto shared = gated_shared.cast<torch::Tensor>();
            check_cuda_contiguous(shared, "gated_shared");
            TORCH_CHECK(shared.scalar_type() == at::kBFloat16 && shared.sizes() == output.sizes(),
                        "gated_shared must be BF16 [T,2048]");
            shared_ptr = bf16_ptr(shared);
        }
        return {reinterpret_cast<__nv_fp8_e4m3 const*>(activation_fp8.data_ptr()),
                activation_scale.data_ptr<float>(),
                reinterpret_cast<__nv_fp8_e4m3 const*>(weight_fp8.data_ptr()),
                weight_scale.data_ptr<float>(),
                route_ids.data_ptr<int32_t>(),
                route_weights.data_ptr<float>(),
                shared_ptr,
                reinterpret_cast<uint32_t*>(error_),
                tokens,
                experts};
    }

    void validate_bf16_output(torch::Tensor const& output, char const* name) const {
        TORCH_CHECK(output.is_cuda() && output.device().index() == device_ && output.is_contiguous()
                        && output.scalar_type() == at::kBFloat16,
                    name,
                    " must be contiguous BF16 CUDA tensor on context device");
    }

    void validate_bf16_pair(torch::Tensor const& input, torch::Tensor const& output, char const* name) const {
        ensure_peer_open();
        validate_bf16_output(input, name);
        validate_bf16_output(output, name);
        TORCH_CHECK(input.numel() == output.numel() && input.numel() > 0
                        && static_cast<uint64_t>(input.numel()) <= max_numel_,
                    name,
                    " requires matching nonempty tensors within max_numel");
    }

    template<class Launch>
    void launch_common(size_t numel, Launch&& launch, bool published_inbox = false) {
        ensure_peer_open();
        bind_stream();
        c10::cuda::CUDAGuard guard(device_);
        const size_t         packets = div_up(numel, fused::kPacketValues);
        TORCH_CHECK(packets <= packet_capacity_, "workspace packet capacity exceeded");
        const uint64_t epoch = ++epoch_;
        TORCH_CHECK(epoch != 0, "fused MoE epoch overflow");
        last_packets_ = packets;
        last_push_    = published_inbox;
        DeviceParams params{{packets_, ready_, ack_, error_},
                            {peer_packets_, peer_ready_, peer_ack_, peer_error_},
                            inbox_,
                            peer_inbox_,
                            completion_,
                            static_cast<int>(std::min<size_t>(blocks_, div_up(packets, fused::kPacketThreads / 32))),
                            epoch,
                            packets,
                            numel,
                            rank_};
        launch(params, current_stream());
    }

    static __nv_bfloat16 const* bf16_ptr(torch::Tensor const& tensor) {
        return reinterpret_cast<__nv_bfloat16 const*>(tensor.data_ptr());
    }
    static __nv_bfloat16* bf16_ptr(torch::Tensor& tensor) {
        return reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr());
    }

    void close_peer_impl() {
        if (peer_packets_)
            cuda_check(cudaIpcCloseMemHandle(peer_packets_), "cudaIpcCloseMemHandle(packets)");
        if (peer_inbox_)
            cuda_check(cudaIpcCloseMemHandle(peer_inbox_), "cudaIpcCloseMemHandle(inbox)");
        if (peer_ready_)
            cuda_check(cudaIpcCloseMemHandle(peer_ready_), "cudaIpcCloseMemHandle(ready)");
        if (peer_ack_)
            cuda_check(cudaIpcCloseMemHandle(peer_ack_), "cudaIpcCloseMemHandle(ack)");
        if (peer_error_)
            cuda_check(cudaIpcCloseMemHandle(peer_error_), "cudaIpcCloseMemHandle(error)");
        peer_packets_ = nullptr;
        peer_inbox_   = nullptr;
        peer_ready_   = nullptr;
        peer_ack_     = nullptr;
        peer_error_   = nullptr;
    }

    void close_peer_noexcept() noexcept {
        if (peer_packets_)
            cudaIpcCloseMemHandle(peer_packets_);
        if (peer_inbox_)
            cudaIpcCloseMemHandle(peer_inbox_);
        if (peer_ready_)
            cudaIpcCloseMemHandle(peer_ready_);
        if (peer_ack_)
            cudaIpcCloseMemHandle(peer_ack_);
        if (peer_error_)
            cudaIpcCloseMemHandle(peer_error_);
        peer_packets_ = nullptr;
        peer_inbox_   = nullptr;
        peer_ready_   = nullptr;
        peer_ack_     = nullptr;
        peer_error_   = nullptr;
    }

    void free_local_checked() {
        if (packets_) {
            cuda_check(cudaFree(packets_), "cudaFree(packets)");
            packets_ = nullptr;
        }
        if (inbox_) {
            cuda_check(cudaFree(inbox_), "cudaFree(inbox)");
            inbox_ = nullptr;
        }
        if (completion_) {
            cuda_check(cudaFree(completion_), "cudaFree(completion)");
            completion_ = nullptr;
        }
        if (ready_) {
            cuda_check(cudaFree(ready_), "cudaFree(ready)");
            ready_ = nullptr;
        }
        if (ack_) {
            cuda_check(cudaFree(ack_), "cudaFree(ack)");
            ack_ = nullptr;
        }
        if (error_) {
            cuda_check(cudaFree(error_), "cudaFree(error)");
            error_ = nullptr;
        }
    }

    void release_noexcept() noexcept {
        int        old_device  = -1;
        const bool have_device = cudaGetDevice(&old_device) == cudaSuccess;
        if (have_device && old_device != device_)
            cudaSetDevice(device_);
        close_peer_noexcept();
        if (packets_)
            cudaFree(packets_);
        if (inbox_)
            cudaFree(inbox_);
        if (completion_)
            cudaFree(completion_);
        if (ready_)
            cudaFree(ready_);
        if (ack_)
            cudaFree(ack_);
        if (error_)
            cudaFree(error_);
        packets_    = nullptr;
        inbox_      = nullptr;
        completion_ = nullptr;
        ready_      = nullptr;
        ack_        = nullptr;
        error_      = nullptr;
        if (have_device && old_device != device_)
            cudaSetDevice(old_device);
    }

    uint64_t       max_numel_;
    int            device_;
    int            rank_;
    int            blocks_;
    size_t         packet_capacity_;
    fused::Packet* packets_{};
    fused::Packet* inbox_{};
    uint64_t*      completion_{};
    uint64_t*      ready_{};
    uint64_t*      ack_{};
    uint64_t*      error_{};
    fused::Packet* peer_packets_{};
    fused::Packet* peer_inbox_{};
    uint64_t*      peer_ready_{};
    uint64_t*      peer_ack_{};
    uint64_t*      peer_error_{};
    uint64_t       epoch_{};
    size_t         last_packets_{};
    bool           last_push_{};
    cudaStream_t   bound_stream_{};
    bool           stream_bound_{};
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MoeTpFusedFp8>(m, "MoeTpFusedFp8")
        .def(py::init<uint64_t, int, int, int>(),
             py::arg("max_numel"),
             py::arg("device_index"),
             py::arg("rank"),
             py::arg("blocks") = kDefaultBlocks)
        .def("get_ipc_handle", &MoeTpFusedFp8::get_ipc_handle)
        .def("open_peer", &MoeTpFusedFp8::open_peer)
        .def("close_peer", &MoeTpFusedFp8::close_peer)
        .def("all_reduce", &MoeTpFusedFp8::all_reduce<1>)
        .def("all_reduce_batch", &MoeTpFusedFp8::all_reduce<32>)
        .def("all_reduce_warp", &MoeTpFusedFp8::all_reduce<16>)
        .def("all_reduce_push_warp", &MoeTpFusedFp8::all_reduce_push_warp)
        .def("all_reduce_push_warp_deferred", &MoeTpFusedFp8::all_reduce_push_warp_deferred)
        .def("gather_gate_push", &MoeTpFusedFp8::gather_gate_push)
        .def("gather_gate_push_deferred", &MoeTpFusedFp8::gather_gate_push_deferred)
        .def("gated_all_reduce_warp", &MoeTpFusedFp8::gated_all_reduce_warp)
        .def("gated_local_warp", &MoeTpFusedFp8::gated_local_warp)
        .def("local_fc2", &MoeTpFusedFp8::local_fc2<0>)
        .def("fused_fc2", &MoeTpFusedFp8::fused_fc2<0>)
        .def("local_fc2_mma_batch", &MoeTpFusedFp8::local_fc2<1>)
        .def("fused_fc2_mma_batch", &MoeTpFusedFp8::fused_fc2<3>)
        .def("local_fc2_mma", &MoeTpFusedFp8::local_fc2<1>)
        .def("fused_fc2_mma", &MoeTpFusedFp8::fused_fc2<1>)
        .def("local_fc2_staged", &MoeTpFusedFp8::local_fc2<2>)
        .def("fused_fc2_staged", &MoeTpFusedFp8::fused_fc2<2>)
        .def("copy_local_packets", &MoeTpFusedFp8::copy_local_packets)
        .def("copy_local_inbox", &MoeTpFusedFp8::copy_local_inbox)
        .def("debug_packet_bytes", &MoeTpFusedFp8::copy_local_packets)
        .def("error_status", &MoeTpFusedFp8::error_status)
        .def("blocks", &MoeTpFusedFp8::blocks)
        .def("launch_info", &MoeTpFusedFp8::launch_info)
        .def("profile_fc2", &MoeTpFusedFp8::profile_fc2)
        .def("close", &MoeTpFusedFp8::close);
}
