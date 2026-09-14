#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdint>
#include <hip/hip_runtime.h>

namespace rtp_llm {

using SyncEpoch = uint32_t;

namespace details {

// Ranks use the same ordered barrier sequence per block. A peer may already
// have published the next epoch when we read its slot, so equality is not
// sufficient. Compare modulo 2^32, assuming the epoch distance is below 2^31.
// Unsigned addition and subtraction also keep both counter boundaries defined.
__host__ __device__ constexpr bool syncEpochReached(SyncEpoch observed, SyncEpoch expected) {
    return SyncEpoch(observed - expected) < (SyncEpoch{1} << 31);
}

template<bool RELAXED = true>
__device__ __forceinline__ void st_flag(SyncEpoch* addr, SyncEpoch flag) {
    __scoped_atomic_store_n(addr, flag, RELAXED ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

template<bool RELAXED = true>
__device__ __forceinline__ SyncEpoch ld_flag(SyncEpoch* addr) {
    SyncEpoch flag;
    flag = __scoped_atomic_load_n(addr, RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    return flag;
}

}  // namespace details

template<int NRanks>
struct CommDeviceMeta {
    void* barrier_flag_ptrs[NRanks];
    void* sync_clock;
    int   rank;
    int   nranks;
};

template<int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(CommDeviceMeta<NRanks>& meta) {
        flag_ptr = static_cast<SyncEpoch*>(meta.sync_clock) + blockIdx.x;
        int rank = meta.rank;
        if (threadIdx.x < NRanks) {
            int target_rank = threadIdx.x;
            target_flag     = static_cast<SyncEpoch*>(meta.barrier_flag_ptrs[target_rank]) + blockIdx.x * NRanks + rank;
            current_flag    = static_cast<SyncEpoch*>(meta.barrier_flag_ptrs[rank]) + blockIdx.x * NRanks + target_rank;
        }
        flag = *flag_ptr;
    }

    template<bool RELAXED = true, bool FINAL = true>
    __device__ __forceinline__ void sync() {
        __syncthreads();
        flag += SyncEpoch{1};
        if (threadIdx.x < NRanks) {
            // This is a generation barrier: peer slots are never cleared after
            // initialization. A fast rank may publish a later generation while
            // a slow rank still waits for this one; syncEpochReached accepts the
            // bounded-ahead value, so that newer arrival cannot be erased.
            details::st_flag<RELAXED>(target_flag, flag);
            while (!details::syncEpochReached(details::ld_flag<RELAXED>(current_flag), flag)) {}
        }
        __syncthreads();
        if constexpr (FINAL) {
            if (threadIdx.x == 0) {
                *flag_ptr = flag;
            }
        }
    }

    SyncEpoch* flag_ptr;
    SyncEpoch* target_flag;
    SyncEpoch* current_flag;
    SyncEpoch  flag;
};

}  // namespace rtp_llm
