// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// Adapted from atrex trtllm_all_reduce_fusion for rtp-llm ROCm backend.

#include <iostream>
#include <functional>
#include <limits>
#include <vector>
#include <array>
#include <tuple>

#include "rtp_llm/cpp/kernels/rocm/hip_float8_impl.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "rtp_llm/cpp/kernels/rocm/trtllm_allreduce_fusion.h"

using namespace std;
using namespace at;

#define NBLOCKS_PER_GPU 256

namespace rtp_llm {

namespace cg = cooperative_groups;
using __bfloat16 = __hip_bfloat16;

static_assert(sizeof(void*) == sizeof(fptr_t));

namespace details {

static constexpr int kBytesPerAccess = 16;

template <bool RELAXED = true>
__device__ __forceinline__ void st_flag(int *addr, int flag) {
    __scoped_atomic_store_n(addr, flag,
                            RELAXED ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
}

template <bool RELAXED = true>
__device__ __forceinline__ int ld_flag(int *addr) {
    int flag;
    flag = __scoped_atomic_load_n(addr,
                                  RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM);
    return flag;
}

} // namespace details

#define gpuSuccess hipSuccess
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuStream_t hipStream_t
#define gpuLaunchCooperativeKernel hipLaunchCooperativeKernel
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuDeviceptr_t hipDeviceptr_t
#define gpuPointerGetAttribute hipPointerGetAttribute
#define gpuStreamCaptureStatus hipStreamCaptureStatus
#define gpuStreamIsCapturing hipStreamIsCapturing
#define gpuStreamCaptureStatusActive hipStreamCaptureStatusActive

namespace kernel_utils {

struct alignas(1) fp8e4m3fn {
    enum { max_value = 240 };
    struct from_bits_t {};
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fn() = default;
    __host__ __device__ constexpr fp8e4m3fn(const fp8e4m3fn &) = default;
    __host__ __device__ constexpr fp8e4m3fn(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fn(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fn(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, false, true>(v);
    }

    explicit __host__ __device__ fp8e4m3fn(double v) : fp8e4m3fn(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, false>(data);
    }
};

struct alignas(1) fp8e4m3fnuz {
    enum { max_value = 120 };
    struct from_bits_t {};
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fnuz() = default;
    __host__ __device__ constexpr fp8e4m3fnuz(const fp8e4m3fnuz &) = default;
    __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fnuz(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, true, true>(v);
    }

    explicit __host__ __device__ fp8e4m3fnuz(double v) : fp8e4m3fnuz(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, true>(data);
    }
};

template <typename T, int WARP_SIZE, typename func_t>
__device__ __forceinline__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val = fn(val, __shfl_xor(val, offset, WARP_SIZE));
    }
    return val;
}

template <typename T, int WARP_SIZE, int BLOCK_SIZE, typename func_t>
__device__ __forceinline__ T block_reduce(T val, func_t fn) {
    static __shared__ T shared[BLOCK_SIZE / WARP_SIZE];
    const int tid = threadIdx.x;
    const int w_tid = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    val = warp_reduce<T, WARP_SIZE, func_t>(val, fn);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = shared[w_tid];
    __syncthreads();
    val = warp_reduce<T, BLOCK_SIZE / WARP_SIZE, func_t>(val, fn);
    return val;
}

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) vec_t {
    T data[VEC_SIZE];
    __device__ __forceinline__ T &operator[](int i) { return data[i]; }
    __device__ __forceinline__ T const &operator[](int i) const { return data[i]; }
    __device__ __forceinline__ void load(const T *ptr) {
        *this = *reinterpret_cast<vec_t<T, VEC_SIZE> *>(const_cast<T *>(ptr));
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, VEC_SIZE> *>(ptr) = *this;
    }
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] =
                __builtin_nontemporal_load((uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t *>(&data)[i],
                                        (uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void volatile_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] = __scoped_atomic_load_n(
                (uint32_t *)ptr + i, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void volatile_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __scoped_atomic_store_n((uint32_t *)ptr + i,
                                    reinterpret_cast<uint32_t *>(&data)[i],
                                    __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = val;
        }
    }
    template <typename VT>
    __device__ __forceinline__ void cast_fill(VT val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            *reinterpret_cast<VT *>(&data[i]) = val;
        }
    }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE> &self,
                                         const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[i] = (float)self[i] + (float)other[i];
    }
}

template <typename T, int VEC_SIZE, int NRanks>
__device__ __forceinline__ void vec_add_r_(vec_t<T, VEC_SIZE> (&self)[NRanks]) {
    vec_t<float, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] = (float)self[0][i];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            acc[i] += (float)self[r][i];
        }
    }
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[0][i] = (T)acc[i];
    }
}

} // namespace kernel_utils

using namespace kernel_utils;

#define WARP_SIZE 32
#define MAX_RANKS 8

template <int NRanks>
struct CommDeviceMeta {
    void *barrier_flag_ptrs[NRanks];
    void *sync_clock;
    int rank;
    int nranks;
};

struct CommMeta {
    void *barrier_flag_ptrs[MAX_RANKS];
    void *sync_clock;
    int rank;
    int nranks;
};

struct CommPtrs {
    void *data_ptrs[MAX_RANKS];
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(CommDeviceMeta<NRanks> &meta) {
        flag_ptr = ((int *)meta.sync_clock) + blockIdx.x;
        int rank = meta.rank;
        if (threadIdx.x < NRanks) {
            int target_rank = threadIdx.x;
            target_flag = reinterpret_cast<int *>(meta.barrier_flag_ptrs[target_rank]) + blockIdx.x * NRanks + rank;
            current_flag = reinterpret_cast<int *>(meta.barrier_flag_ptrs[rank]) + blockIdx.x * NRanks + target_rank;
        }
        flag = *flag_ptr;
    }

    template <bool RELAXED = true, bool FINAL = true>
    __device__ __forceinline__ void sync() {
        __syncthreads();
        flag += 1;
        if (threadIdx.x < NRanks) {
            details::st_flag<RELAXED>(target_flag, flag);
            while (details::ld_flag<RELAXED>(current_flag) < flag) {
            }
        }
        __syncthreads();
        if constexpr (FINAL) {
            if (threadIdx.x == 0) {
                *flag_ptr = flag;
            }
        }
    }

    int *flag_ptr;
    int *target_flag;
    int *current_flag;
    int flag;
};

enum QuantType {
    NONE = 0,
    FP8E4M3FN = 1,
    FP8E4M3FNUZ = 2,
};

template <typename T>
struct AllReduceFusionParams {
    int nranks;
    int rank;
    int size;
    int hidden_dim;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    float rms_eps;
    QuantType quant_type;
    void *scale_out;
};

template <typename T, int VEC_SIZE, typename QuantT>
__device__ __forceinline__ vec_t<QuantT, VEC_SIZE> convert_to_fp8(vec_t<T, VEC_SIZE> &in_vec, float scale) {
    vec_t<QuantT, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float out = static_cast<float>(in_vec[i]) / scale;
        out_vec[i] = static_cast<QuantT>(out);
    }
    return out_vec;
}

template <typename T, int VEC_SIZE, typename OutT, int BLOCK_SIZE>
__device__ __forceinline__ vec_t<OutT, VEC_SIZE> rms_norm(AllReduceFusionParams<T> const &m_params,
                                                          vec_t<T, VEC_SIZE> const &residual, vec_t<T, VEC_SIZE> const &gamma) {
    __shared__ float s_val;
    vec_t<OutT, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]);
        acc += v * v;
    }
    acc = block_reduce<float, WARP_SIZE, BLOCK_SIZE>(acc, std::plus<float>());
    if (threadIdx.x == 0) {
        s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float out = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]) * s_val * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]);
        norm_out[i] = static_cast<OutT>(out);
    }
    return norm_out;
}

template <typename T, int VEC_SIZE, int BLOCK_SIZE>
__device__ __forceinline__ float reduce_abs_max(vec_t<T, VEC_SIZE> const &data) {
    __shared__ float s_val;
    auto fn = [](float a, float b) { return a > b ? a : b; };
    float acc = -1.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&data)[i]);
        acc = fn(acc, std::abs(v));
    }
    acc = block_reduce<float, WARP_SIZE, BLOCK_SIZE>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE, bool STORE = true, int BLOCK_SIZE = 0, int QUANT_TYPE = 0>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
    if constexpr (STORE)
        rms_in.store(reinterpret_cast<T *>(params.residual_out) + idx);

    if constexpr (QUANT_TYPE == QuantType::NONE) {
        auto val = rms_norm<T, VEC_SIZE, T, BLOCK_SIZE>(params, rms_in, rms_weight);
        val.store(reinterpret_cast<T *>(params.norm_out) + idx);
    } else {
        auto val = rms_norm<T, VEC_SIZE, float, BLOCK_SIZE>(params, rms_in, rms_weight);
        float scale = reduce_abs_max<float, VEC_SIZE, BLOCK_SIZE>(val);
        if constexpr (QUANT_TYPE == QuantType::FP8E4M3FN) {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fn::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fn>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fn *>(params.norm_out) + idx);
        } else {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fnuz::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fnuz>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fnuz *>(params.norm_out) + idx);
        }
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(params.scale_out)[tidx] = scale;
    }
}

// ============================================================================
// 1-stage kernels
// ============================================================================

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_1stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *__restrict__ cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    SyncComm<NRanks> comm(meta);
    comm.sync();
    using vec_t_ = vec_t<T, VEC_SIZE>;
    using acc_vec_t_ = vec_t<float, VEC_SIZE>;
    int tidx = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int idx = tidx * params.hidden_dim + access_id_in_token;

    acc_vec_t_ acc;
    auto vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        acc.data[v] = vec.data[v];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
        vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[r]) + idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            acc.data[v] += (float)vec.data[v];
        }
    }
    auto res = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.residual_in) + idx);
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        acc.data[v] += (float)res.data[v];
    }
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        vec.data[v] = (T)acc.data[v];
    }
    *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.residual_out) + idx) = vec;
    auto gamma = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
    epilogue<T, VEC_SIZE, false, BLOCK_SIZE, QUANT_TYPE>(params, vec, gamma, idx, tidx);
}

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_kernel_1stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *__restrict__ cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    SyncComm<NRanks> comm(meta);
    comm.sync();
    using vec_t_ = vec_t<T, VEC_SIZE>;
    using acc_vec_t_ = vec_t<float, VEC_SIZE>;
    int tidx = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int idx = tidx * params.hidden_dim + access_id_in_token;

    acc_vec_t_ acc;
    auto vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        acc.data[v] = vec.data[v];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
        vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[r]) + idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            acc.data[v] += (float)vec.data[v];
        }
    }
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        vec.data[v] = (T)acc.data[v];
    }
    *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.residual_out) + idx) = vec;
}

// ============================================================================
// 1-stage launchers
// ============================================================================

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_1stage_launcher(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    TORCH_CHECK(token_num <= NBLOCKS_PER_GPU);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_1stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_kernel_1stage_launcher(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    TORCH_CHECK(token_num <= NBLOCKS_PER_GPU);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(token_num);
    allreduce_kernel_1stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

// ============================================================================
// 2-stage kernels
// ============================================================================

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_2stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int WARP_SIZE_ = BLOCK_SIZE / NRanks;
    SyncComm<NRanks> comm(meta);

    __shared__ T shared[NRanks * WARP_SIZE_ * VEC_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE_;
    int lane_id = threadIdx.x % WARP_SIZE_;

    vec_t<T, VEC_SIZE> val;
    vec_t<float, VEC_SIZE> acc;

    comm.template sync<true, false>();

    for (int idx = ((blockIdx.x * NRanks + params.rank) * WARP_SIZE_ + lane_id) * VEC_SIZE;
         idx < params.size;
         idx += gridDim.x * NRanks * WARP_SIZE_ * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
        val.store(&shared[0] + threadIdx.x * VEC_SIZE);
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc.data[v] = (float)val.data[v];
            }
#pragma unroll
            for (int r = 1; r < NRanks; ++r) {
                val.load(&shared[0] + (r * WARP_SIZE_ + lane_id) * VEC_SIZE);
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    acc.data[v] += (float)val.data[v];
                }
            }
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                val.data[v] = (T)acc.data[v];
            }
            val.store(&shared[0] + lane_id * VEC_SIZE);
        }
        __syncthreads();
        val.load(&shared[0] + lane_id * VEC_SIZE);
        val.store(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
    }

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
    comm.template sync<false, true>();
    for (int idx = blockIdx.x * params.hidden_dim + access_id_in_token, tidx = blockIdx.x;
         idx < params.size;
         idx += gridDim.x * params.hidden_dim, tidx += gridDim.x) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
        vec_t<T, VEC_SIZE> res;
        res.load(reinterpret_cast<T *>(params.residual_in) + idx);
        vec_add_<T, VEC_SIZE>(val, res);
        epilogue<T, VEC_SIZE, true, BLOCK_SIZE, QUANT_TYPE>(params, val, gamma, idx, tidx);
    }
}

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_kernel_2stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int WARP_SIZE_ = BLOCK_SIZE / NRanks;
    SyncComm<NRanks> comm(meta);

    __shared__ T shared[NRanks * WARP_SIZE_ * VEC_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE_;
    int lane_id = threadIdx.x % WARP_SIZE_;

    vec_t<T, VEC_SIZE> val;
    vec_t<float, VEC_SIZE> acc;

    comm.template sync<true, false>();

    for (int idx = ((blockIdx.x * NRanks + params.rank) * WARP_SIZE_ + lane_id) * VEC_SIZE;
         idx < params.size;
         idx += gridDim.x * NRanks * WARP_SIZE_ * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
        val.store(&shared[0] + threadIdx.x * VEC_SIZE);
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc.data[v] = (float)val.data[v];
            }
#pragma unroll
            for (int r = 1; r < NRanks; ++r) {
                val.load(&shared[0] + (r * WARP_SIZE_ + lane_id) * VEC_SIZE);
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    acc.data[v] += (float)val.data[v];
                }
            }
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                val.data[v] = (T)acc.data[v];
            }
            val.store(&shared[0] + lane_id * VEC_SIZE);
        }
        __syncthreads();
        val.load(&shared[0] + lane_id * VEC_SIZE);
        val.store(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
    }

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    comm.template sync<false, true>();
    for (int idx = blockIdx.x * params.hidden_dim + access_id_in_token, tidx = blockIdx.x;
         idx < params.size;
         idx += gridDim.x * params.hidden_dim, tidx += gridDim.x) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
        val.store(reinterpret_cast<T *>(params.residual_out) + idx);
    }
}

// ============================================================================
// 2-stage launchers
// ============================================================================

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_2stage_launcher(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    dim3 threadsPerBlock(BLOCK_SIZE);
    token_num = std::min(token_num, NBLOCKS_PER_GPU);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_2stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_kernel_2stage_launcher(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    dim3 threadsPerBlock(BLOCK_SIZE);
    token_num = std::min(token_num, NBLOCKS_PER_GPU);
    dim3 numBlocks(token_num);
    allreduce_kernel_2stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

// ============================================================================
// Dispatch logic: 1-stage vs 2-stage
// ============================================================================

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_launcher_(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_num = params.size / params.hidden_dim;
    TORCH_CHECK(params.size % params.hidden_dim == 0);
    TORCH_CHECK(params.hidden_dim % VEC_SIZE == 0);
    TORCH_CHECK(params.hidden_dim == HIDDEN_DIM);
    auto bytes = params.size * sizeof(T);
    bool use_1s = token_num <= (NBLOCKS_PER_GPU / 4);
    use_1s = use_1s && ((NRanks <= 2) || (NRanks <= 4 && bytes < 160 * 1024) || (NRanks <= 8 && bytes < 80 * 1024));
    if (use_1s) {
        allreduce_fusion_kernel_1stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    } else {
        allreduce_fusion_kernel_2stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    }
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_kernel_launcher_(
    AllReduceFusionParams<T> const &params, CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs, gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_num = params.size / params.hidden_dim;
    TORCH_CHECK(params.size % params.hidden_dim == 0);
    TORCH_CHECK(params.hidden_dim % VEC_SIZE == 0);
    TORCH_CHECK(params.hidden_dim == HIDDEN_DIM);
    auto bytes = params.size * sizeof(T);
    bool use_1s = token_num <= (NBLOCKS_PER_GPU / 4);
    use_1s = use_1s && ((NRanks <= 2) || (NRanks <= 4 && bytes < 160 * 1024) || (NRanks <= 8 && bytes < 80 * 1024));
    if (use_1s) {
        allreduce_kernel_1stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    } else {
        allreduce_kernel_2stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    }
}

// ============================================================================
// Hidden dim dispatch
// ============================================================================

template <typename T, int NRanks, int QUANT_TYPE>
void allreduce_fusion_kernel_launcher_hd(AllReduceFusionParams<T> const &params,
                                         CommDeviceMeta<NRanks> const &meta,
                                         CommPtrs *cptrs, gpuStream_t stream) {
    switch (params.hidden_dim) {
    case 4096: allreduce_fusion_kernel_launcher_<T, NRanks, 4096, QUANT_TYPE>(params, meta, cptrs, stream); return;
    case 2048: allreduce_fusion_kernel_launcher_<T, NRanks, 2048, QUANT_TYPE>(params, meta, cptrs, stream); return;
    case 1024: allreduce_fusion_kernel_launcher_<T, NRanks, 1024, QUANT_TYPE>(params, meta, cptrs, stream); return;
    default: TORCH_CHECK(false, "Unsupported hidden_dim: ", params.hidden_dim);
    }
}

template <typename T, int NRanks, int QUANT_TYPE>
void allreduce_kernel_launcher_hd(AllReduceFusionParams<T> const &params,
                                  CommDeviceMeta<NRanks> const &meta,
                                  CommPtrs *cptrs, gpuStream_t stream) {
    switch (params.hidden_dim) {
    case 4096: allreduce_kernel_launcher_<T, NRanks, 4096, QUANT_TYPE>(params, meta, cptrs, stream); return;
    case 2048: allreduce_kernel_launcher_<T, NRanks, 2048, QUANT_TYPE>(params, meta, cptrs, stream); return;
    case 1024: allreduce_kernel_launcher_<T, NRanks, 1024, QUANT_TYPE>(params, meta, cptrs, stream); return;
    default: TORCH_CHECK(false, "Unsupported hidden_dim: ", params.hidden_dim);
    }
}

// ============================================================================
// Quant type dispatch
// ============================================================================

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      CommDeviceMeta<NRanks> const &meta,
                                      CommPtrs *cptrs, gpuStream_t stream) {
    switch (params.quant_type) {
    case QuantType::NONE:       allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::NONE>(params, meta, cptrs, stream); return;
    case QuantType::FP8E4M3FN:  allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FN>(params, meta, cptrs, stream); return;
    case QuantType::FP8E4M3FNUZ: allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FNUZ>(params, meta, cptrs, stream); return;
    default: TORCH_CHECK(false, "Unsupported quant_type");
    }
}

template <typename T, int NRanks>
void allreduce_kernel_launcher(AllReduceFusionParams<T> const &params,
                               CommDeviceMeta<NRanks> const &meta,
                               CommPtrs *cptrs, gpuStream_t stream) {
    switch (params.quant_type) {
    case QuantType::NONE:       allreduce_kernel_launcher_hd<T, NRanks, QuantType::NONE>(params, meta, cptrs, stream); return;
    case QuantType::FP8E4M3FN:  allreduce_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FN>(params, meta, cptrs, stream); return;
    case QuantType::FP8E4M3FNUZ: allreduce_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FNUZ>(params, meta, cptrs, stream); return;
    default: TORCH_CHECK(false, "Unsupported quant_type");
    }
}

// ============================================================================
// Top-level impl functions
// ============================================================================

template <typename T>
void allreduce_rms_fusion_impl(CommMeta meta, CommPtrs *cptrs, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = meta.nranks;
    params.rank = meta.rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.allreduce_in = allreduce_in;
    params.residual_in = residual_in;
    params.residual_out = residual_out;
    params.norm_out = norm_out;
    params.rms_gamma = rms_gamma;
    params.rms_eps = eps;
    params.scale_out = scale_out;
    params.quant_type = (QuantType)quant_type;

#define DISPATCH_NRANKS(NRANKS)                                                    \
    {                                                                              \
        CommDeviceMeta<NRANKS> dmeta;                                              \
        for (int i = 0; i < NRANKS; ++i) {                                         \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                \
        }                                                                          \
        dmeta.sync_clock = meta.sync_clock;                                        \
        dmeta.rank = meta.rank;                                                    \
        dmeta.nranks = meta.nranks;                                                \
        allreduce_fusion_kernel_launcher<T, NRANKS>(params, dmeta, cptrs, stream); \
    }

    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        TORCH_CHECK(false, "Unsupported nranks: ", nranks);
    }

#undef DISPATCH_NRANKS
}

template <typename T>
void allreduce_impl(CommMeta meta, CommPtrs *cptrs, int size,
                    int hidden_dim, void *allreduce_in,
                    void *allreduce_out,
                    gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = meta.nranks;
    params.rank = meta.rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.allreduce_in = allreduce_in;
    params.residual_out = allreduce_out;
    params.quant_type = QuantType::NONE;

#define DISPATCH_NRANKS(NRANKS)                                                    \
    {                                                                              \
        CommDeviceMeta<NRANKS> dmeta;                                              \
        for (int i = 0; i < NRANKS; ++i) {                                         \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                \
        }                                                                          \
        dmeta.sync_clock = meta.sync_clock;                                        \
        dmeta.rank = meta.rank;                                                    \
        dmeta.nranks = meta.nranks;                                                \
        allreduce_kernel_launcher<T, NRANKS>(params, dmeta, cptrs, stream);        \
    }

    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        TORCH_CHECK(false, "Unsupported nranks: ", nranks);
    }

#undef DISPATCH_NRANKS
}

// ============================================================================
// IPC utilities
// ============================================================================

namespace ipc_details {

Tensor get_handle(void *ptr) {
    gpuIpcMemHandle_t handle;
    TORCH_CHECK(gpuIpcGetMemHandle(&handle, ptr) == gpuSuccess);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
    return data_handle;
}

void open_handles(int rank, std::vector<Tensor> &handles, void *ptr, std::vector<void *> &ipc_ptrs) {
    std::vector<gpuIpcMemHandle_t> ipc_handles;
    int world_size = handles.size();
    ipc_handles.reserve(world_size);
    ipc_ptrs.resize(world_size);
    for (auto &handle : handles) {
        gpuIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
        ipc_handles.push_back(ipc_handle);
    }
    for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
            TORCH_CHECK(
                gpuIpcOpenMemHandle((void **)&ipc_ptrs[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
        } else {
            ipc_ptrs[i] = ptr;
        }
    }
}

void create_base_ptr(void **base_ptr, void *ptr) {
    if (gpuPointerGetAttribute(base_ptr, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, (gpuDeviceptr_t)ptr) != gpuSuccess) {
        throw std::runtime_error("failed to get pointer attr");
    }
}

} // namespace ipc_details

// ============================================================================
// CommWorkspace
// ============================================================================

class CommWorkspace {
public:
    CommWorkspace(
        int64_t device_id, int64_t rank, int64_t world_size,
        int64_t size_in_bytes, int64_t comm_ptrs_buf_len,
        int64_t max_thread_blocks = NBLOCKS_PER_GPU, bool round_robin = true) {
        TORCH_CHECK(rank < world_size);
        gpuSetDevice(device_id);
        device_id_ = device_id;
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        comm_ptrs_buf_len_ = comm_ptrs_buf_len;
        max_thread_blocks_ = max_thread_blocks;
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMalloc(&comm_ptrs_, comm_ptrs_buf_len_ * sizeof(CommPtrs));
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
        used_comm_ptrs_ = 0;
        round_robin_ = round_robin;
    }

    ~CommWorkspace() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
        gpuFree(comm_ptrs_);
    }

    Tensor get_barrier_handle() {
        return ipc_details::get_handle(barrier_flags_);
    }

    Tensor get_data_handle() {
        return ipc_details::get_handle(data_);
    }

    void open_barrier_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, barrier_flags_, ipc_barrier_flags_);
    }

    void open_data_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, data_, ipc_data_);
        CommPtrs *cptrs = new CommPtrs[comm_ptrs_buf_len_];
        for (int i = 0; i < comm_ptrs_buf_len_; ++i) {
            for (int j = 0; j < world_size_; ++j) {
                int r = round_robin_ ? ((rank_ + j) % world_size_) : j;
                cptrs[i].data_ptrs[j] = ipc_data_[r];
            }
        }
        gpuMemcpy(comm_ptrs_, cptrs, comm_ptrs_buf_len_ * sizeof(CommPtrs), gpuMemcpyHostToDevice);
        used_comm_ptrs_ = 1;
        delete[] cptrs;
    }

    std::tuple<CommMeta, CommPtrs *> get_comm_data(const Tensor &input, gpuStream_t stream) {
        int64_t size = input.numel() * input.element_size();
        void *ptr = (void *)input.data_ptr();

        CommMeta meta;
        for (int r = 0; r < world_size_; ++r) {
            meta.barrier_flag_ptrs[r] = ipc_barrier_flags_[r];
        }
        meta.sync_clock = sync_clock_;
        meta.rank = rank_;
        meta.nranks = world_size_;

        CommPtrs *cptrs;
        auto it = ptr_to_comm_ptrs_.find(ptr);
        if (it != ptr_to_comm_ptrs_.end()) {
            cptrs = it->second;
        } else {
            gpuStreamCaptureStatus status;
            gpuStreamIsCapturing(stream, &status);
            int remaining = comm_ptrs_buf_len_ - used_comm_ptrs_ - unregistered_ptrs_.size();
            if (status == gpuStreamCaptureStatusActive && size < 1024 * 4096 * 16 && remaining > 0) {
                unregistered_ptrs_.push_back(ptr);
                cptrs = comm_ptrs_ + used_comm_ptrs_ + unregistered_ptrs_.size() - 1;
            } else {
                cptrs = comm_ptrs_ + 0;
                gpuMemcpyAsync(data_, ptr, size, gpuMemcpyDeviceToDevice, stream);
            }
        }

        return {meta, cptrs};
    }

    void capture_clear() {
        unregistered_ptrs_.clear();
        unregistered_base_ptrs_.clear();
    }

    std::vector<Tensor> get_captured_handles() {
        int num_datas = unregistered_ptrs_.size();
        std::vector<Tensor> ipc_handles;
        ipc_handles.reserve(num_datas);
        for (int i = 0; i < num_datas; ++i) {
            void *ptr = unregistered_ptrs_[i];
            void *base_ptr;
            ipc_details::create_base_ptr(&base_ptr, ptr);
            ipc_handles.push_back(ipc_details::get_handle(base_ptr));
            unregistered_base_ptrs_.push_back(base_ptr);
        }
        return ipc_handles;
    }

    Tensor get_captured_offsets() {
        int num_datas = unregistered_ptrs_.size();
        std::vector<int64_t> offsets;
        offsets.reserve(num_datas);
        for (int i = 0; i < num_datas; ++i) {
            void *ptr = unregistered_ptrs_[i];
            void *base_ptr = unregistered_base_ptrs_[i];
            int64_t offset = ((char *)ptr) - ((char *)base_ptr);
            offsets.push_back(offset);
        }
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto t = torch::tensor(offsets, options);
        return t;
    }

    void open_captured_handles(std::vector<Tensor> &handles, std::vector<int64_t> &offsets, int64_t ptr_idx) {
        void *ptr = unregistered_ptrs_[ptr_idx];
        void *base_ptr = unregistered_base_ptrs_[ptr_idx];
        std::vector<void *> ipc_data;
        ipc_details::open_handles(rank_, handles, base_ptr, ipc_data);
        CommPtrs cptrs;
        for (size_t i = 0; i < offsets.size(); ++i) {
            ipc_data[i] = (void *)((char *)ipc_data[i] + offsets[i]);
        }
        for (size_t i = 0; i < offsets.size(); ++i) {
            int r = round_robin_ ? ((rank_ + i) % world_size_) : i;
            cptrs.data_ptrs[i] = ipc_data[r];
        }
        gpuMemcpy(comm_ptrs_ + used_comm_ptrs_, &cptrs, sizeof(CommPtrs), gpuMemcpyHostToDevice);
        ptr_to_comm_ptrs_[ptr] = comm_ptrs_ + used_comm_ptrs_;
        used_comm_ptrs_++;
    }

private:
    int device_id_;
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int comm_ptrs_buf_len_;
    int max_thread_blocks_;
    bool round_robin_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> ipc_barrier_flags_;
    std::vector<void *> ipc_data_;
    std::vector<void *> unregistered_ptrs_;
    std::vector<void *> unregistered_base_ptrs_;
    CommPtrs *comm_ptrs_;
    int used_comm_ptrs_;
    std::unordered_map<void *, CommPtrs *> ptr_to_comm_ptrs_;
};

// ============================================================================
// Public API functions
// ============================================================================

fptr_t init_ar_fusion(int64_t device_id, int64_t rank, int64_t world_size,
                      int64_t max_size_in_bytes, int64_t comm_ptrs_buf_len) {
    switch (world_size) {
    case 8: case 4: case 2: break;
    default: throw std::invalid_argument("world size is not supported");
    }
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");
    return (fptr_t) new CommWorkspace(device_id, rank, world_size, max_size_in_bytes, comm_ptrs_buf_len);
}

void destroy_ar_fusion(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    delete ptr;
}

Tensor get_ar_fusion_barrier_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_barrier_handle();
}

Tensor get_ar_fusion_data_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_data_handle();
}

void open_ar_fusion_barrier_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_barrier_handles(handles);
}

void open_ar_fusion_data_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_data_handles(handles);
}

void ar_fusion_capture_clear(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->capture_clear();
}

std::vector<Tensor> get_ar_fusion_captured_handles(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_captured_handles();
}

Tensor get_ar_fusion_captured_offsets(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_captured_offsets();
}

void open_ar_fusion_captured_handles(fptr_t fptr, std::vector<Tensor> handles,
                                     std::vector<int64_t> offsets, int64_t ptr_idx) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_captured_handles(handles, offsets, ptr_idx);
}

template <typename T>
struct KernelElementType { using type = T; };

template <>
struct KernelElementType<c10::Half> { using type = __half; };

template <>
struct KernelElementType<c10::BFloat16> { using type = __bfloat16; };

void allreduce_rms(fptr_t fptr, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type) {
    TORCH_CHECK(allreduce_in.is_contiguous() && residual_in.is_contiguous() && rms_gamma.is_contiguous());
    TORCH_CHECK(residual_out.is_contiguous() && norm_out.is_contiguous() && scale_out.is_contiguous());
    auto dev = allreduce_in.device();
    c10::DeviceGuard dev_guard(dev);
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    auto comm_data = ptr->get_comm_data(allreduce_in, stream);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, allreduce_in.scalar_type(), "allreduce_rms", [&] {
            using k_scalar_t = typename KernelElementType<scalar_t>::type;
            allreduce_rms_fusion_impl<k_scalar_t>(
                std::get<0>(comm_data), std::get<1>(comm_data),
                size, hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)residual_in.data_ptr<scalar_t>(),
                (void *)residual_out.data_ptr<scalar_t>(),
                (void *)norm_out.data_ptr(),
                (void *)rms_gamma.data_ptr<scalar_t>(),
                eps, quant_type,
                (void *)scale_out.data_ptr<float>(),
                stream);
        });
}

void allreduce(fptr_t fptr, Tensor &allreduce_in, Tensor &allreduce_out) {
    TORCH_CHECK(allreduce_in.is_contiguous() && allreduce_out.is_contiguous());
    auto dev = allreduce_in.device();
    c10::DeviceGuard dev_guard(dev);
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    auto comm_data = ptr->get_comm_data(allreduce_in, stream);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, allreduce_in.scalar_type(), "allreduce", [&] {
            using k_scalar_t = typename KernelElementType<scalar_t>::type;
            allreduce_impl<k_scalar_t>(
                std::get<0>(comm_data), std::get<1>(comm_data),
                size, hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)allreduce_out.data_ptr<scalar_t>(),
                stream);
        });
}

} // namespace rtp_llm
