/*
 * Copyright (c) 2026, Alibaba Group. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace rtp_llm::moe_tp_fused_fp8 {

constexpr int kTpSize        = 2;
constexpr int kPacketValues  = 496;
constexpr int kPacketBytes   = 512;
constexpr int kPacketThreads = 512;

// The byte layout is deliberately stable across processes.  Values occupy the
// first 496 bytes; only ``scale`` has semantic meaning in the 16-byte tail.
// The remaining bytes are zeroed before publish so a peer never observes stale
// protocol data when the final packet is short.
struct alignas(16) Packet {
    __nv_fp8_e4m3 values[kPacketValues];
    float         scale;
    uint32_t      reserved[3];
};
static_assert(sizeof(Packet) == kPacketBytes);
static_assert(alignof(Packet) == 16);

struct TransportWorkspace {
    Packet*   packets;
    uint64_t* ready;
    uint64_t* ack;
    uint64_t* error;
};

__device__ __forceinline__ void store_release_sys(uint64_t value, uint64_t* address) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b64 [%1], %0;" : : "l"(value), "l"(address) : "memory");
#else
    __threadfence_system();
    *address = value;
#endif
}

__device__ __forceinline__ uint64_t load_acquire_sys(uint64_t const* address) {
    uint64_t value;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
#else
    value = *address;
#endif
    return value;
}

__device__ __forceinline__ float max_abs(float a, float b) {
    a = fabsf(a);
    return a > b ? a : b;
}

__device__ __forceinline__ float warp_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 16));
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 8));
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 4));
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
    return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
}

__device__ __forceinline__ float block_max_abs(float value, float* warp_maxima) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value          = warp_max(value);
    if (lane == 0)
        warp_maxima[warp] = value;
    __syncthreads();
    value = lane < (blockDim.x >> 5) ? warp_maxima[lane] : 0.f;
    if (warp == 0)
        value = warp_max(value);
    if (threadIdx.x == 0)
        warp_maxima[0] = value;
    __syncthreads();
    return warp_maxima[0];
}

// Returns false only after a bounded wait.  The caller writes its local error
// word and returns from the kernel; it must not select a rank-local fallback.
__device__ __forceinline__ bool wait_epoch(uint64_t const* word, uint64_t epoch, uint64_t spin_limit) {
    for (uint64_t spin = 0; spin < spin_limit; ++spin) {
        if (load_acquire_sys(word) == epoch)
            return true;
        if ((spin & 0x3ff) == 0)
            __nanosleep(128);
    }
    return false;
}

}  // namespace rtp_llm::moe_tp_fused_fp8
