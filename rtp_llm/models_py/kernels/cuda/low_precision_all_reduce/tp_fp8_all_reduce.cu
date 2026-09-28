/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2026, Alibaba Group.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Derived from TensorRT-LLM customLowPrecisionAllReduceKernels.cu at
 * https://github.com/NVIDIA/TensorRT-LLM/blob/c2220eef33f407fd3d805a6ab8a2725f54bdd3c4/
 * cpp/tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.cu
 * (commit c2220eef33f407fd3d805a6ab8a2725f54bdd3c4). This is the TP=2,
 * BF16-to-FP8-E4M3 two-shot path only; TensorRT-LLM strategy selection,
 * static process-wide buffers, and TP=4/8 hierarchical paths are omitted.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
constexpr int      kTp                  = 2;
constexpr int      kThreads             = 512;
constexpr int      kWarps               = 16;
constexpr int      kDefaultBlocks       = 16;
constexpr size_t   kPayloadEltsPerBlock = 16 * 31 * 16;           // 7936 FP8 values.
constexpr size_t   kBufferEltsPerRound  = 16 * 512;               // 8192 bytes incl. scales.
constexpr uint64_t kWireMagic           = 0x5254504650384152ULL;  // RTPFP8AR
constexpr uint32_t kWireVersion         = 2;

inline void cuda_check(cudaError_t status, char const* op) {
    TORCH_CHECK(status == cudaSuccess, op, ": ", cudaGetErrorString(status));
}

struct IpcWireHandle {
    uint64_t           magic;
    uint32_t           version;
    int32_t            device;
    int32_t            rank;
    int32_t            blocks;
    uint64_t           max_numel;
    uint64_t           comm_bytes;
    uint64_t           barrier_words;
    cudaIpcMemHandle_t comm;
    cudaIpcMemHandle_t barrier;
};
static_assert(std::is_trivially_copyable_v<IpcWireHandle>);

union PackedBf16 {
    int4          packed;
    __nv_bfloat16 values[8];
};
union PackedFp8 {
    int4          packed;
    __nv_fp8_e4m3 values[16];
};

// Match TensorRT-LLM cuda_max(cuda_abs(candidate), running), including
// its NaN propagation order. Non-finite values do not alter control flow.
__device__ __forceinline__ float upstream_max_abs(float candidate, float running) {
    candidate = fabsf(candidate);
    return candidate > running ? candidate : running;
}
__device__ __forceinline__ float upstream_max(float left, float right) {
    return left > right ? left : right;
}
__device__ float warp_max(float value) {
    value = upstream_max(__shfl_xor_sync(0xffffffff, value, 16), value);
    value = upstream_max(__shfl_xor_sync(0xffffffff, value, 8), value);
    value = upstream_max(__shfl_xor_sync(0xffffffff, value, 4), value);
    value = upstream_max(__shfl_xor_sync(0xffffffff, value, 2), value);
    return upstream_max(__shfl_xor_sync(0xffffffff, value, 1), value);
}
__device__ __forceinline__ void st_release(uint64_t value, uint64_t* addr) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(value), "l"(addr));
#else
    __threadfence_system();
    *addr = value;
#endif
}
__device__ __forceinline__ uint64_t ld_acquire(uint64_t* addr) {
    uint64_t value;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(value) : "l"(addr));
#else
    value = *addr;
#endif
    return value;
}

// TensorRT-LLM's initial multi_gpu_barrier specialized to two ranks. The
// upstream code used volatile stores/loads here; use system release/acquire so
// peer preprocessing writes are published before its two-shot read.
__device__ __forceinline__ void initial_barrier(uint64_t* barriers[kTp], uint64_t flag, int rank, int tidx, int bidx) {
    if (tidx < kTp) {
        if (bidx == 0)
            st_release(flag, barriers[tidx] + rank);
        while (ld_acquire(barriers[rank] + tidx) != flag) {}
    }
    __syncthreads();
}

__device__ __forceinline__ void
block_peer_barrier(uint64_t* barriers[kTp], uint64_t flag, int rank, size_t offset, int tidx) {
    if (tidx < kTp) {
        st_release(flag, barriers[tidx] + offset + rank);
        while (ld_acquire(barriers[rank] + offset + tidx) != flag) {}
    }
    __syncthreads();
}

// TensorRT-LLM lowPrecisionPreprocessKernel<2, bf16, fp8>, preserving its
// 31 lanes of values + one lane containing a float scale per warp.
__global__ void preprocess_tp2_bf16_fp8(__nv_bfloat16 const* __restrict__ input,
                                        size_t elts_per_rank_in,
                                        size_t elts_per_rank_out,
                                        __nv_fp8_e4m3* __restrict__ output) {
    constexpr int in_per_lane  = 8;
    constexpr int out_per_lane = 16;
    constexpr int in_per_warp  = 31 * out_per_lane;  // 496 bf16 values
    constexpr int out_per_warp = 32 * out_per_lane;  // 512 fp8 bytes
    const int     target_rank  = blockIdx.x / (gridDim.x / kTp);
    const int     local_bid    = blockIdx.x % (gridDim.x / kTp);
    input += elts_per_rank_in * target_rank;
    output += elts_per_rank_out * target_rank;
    const int    lane      = threadIdx.x & 31;
    const int    warp      = threadIdx.x >> 5;
    const size_t start_in  = in_per_warp * kWarps * local_bid + warp * in_per_warp;
    const size_t start_out = out_per_warp * kWarps * local_bid + warp * out_per_warp;
    PackedBf16   values[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int lane_offset = lane * in_per_lane + 32 * in_per_lane * r;
        if (lane_offset < in_per_warp && start_in + lane_offset < elts_per_rank_in)
            values[r].packed = *reinterpret_cast<int4 const*>(input + start_in + lane_offset);
        else
#pragma unroll
            for (int j = 0; j < in_per_lane; ++j)
                values[r].values[j] = __float2bfloat16(0.f);
    }
    float scale = 0.f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < in_per_lane; ++j)
            scale = upstream_max_abs(__bfloat162float(values[r].values[j]), scale);
    scale = warp_max(scale);
    if (scale != 0.f)
        scale = 448.f / scale;
    PackedFp8 out[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int lane_offset = lane * in_per_lane + 32 * in_per_lane * r;
        if (lane_offset < in_per_warp) {
#pragma unroll
            for (int j = 0; j < in_per_lane; ++j) {
                float v          = __bfloat162float(values[r].values[j]);
                out[r].values[j] = static_cast<__nv_fp8_e4m3>(scale == 0.f ? v : v * scale);
            }
        } else if (lane_offset == in_per_warp) {
            *reinterpret_cast<float*>(&out[r]) = scale;
        }
        *reinterpret_cast<int2*>(output + start_out + lane * in_per_lane + 32 * in_per_lane * r) =
            *reinterpret_cast<int2*>(&out[r]);
    }
}

__device__ void first_stage_tp2(int rank, size_t per_rank, __nv_fp8_e4m3* input[kTp], float* smem) {
    const int    lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t begin       = (blockIdx.x * kWarps + warp) * 16 * 32 + lane * 16;
    const size_t rank_offset = per_rank * rank;
    float*       scales      = &smem[kTp * warp];
    for (size_t off = begin; off < per_rank; off += gridDim.x * blockDim.x * 16) {
        PackedFp8 values[kTp];
        float     sums[16] = {};
#pragma unroll
        for (int i = 0; i < kTp; ++i)
            values[i].packed = *reinterpret_cast<int4 const*>(input[i] + rank_offset + off);
        if (lane == 31) {
#pragma unroll
            for (int i = 0; i < kTp; ++i)
                scales[i] = *reinterpret_cast<float*>(&values[i]);
        }
        __syncwarp();
        if (lane < 31) {
#pragma unroll
            for (int i = 0; i < kTp; ++i)
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    sums[j] += scales[i] == 0.f ? static_cast<float>(values[i].values[j]) :
                                                  static_cast<float>(values[i].values[j]) / scales[i];
        }
        float scale = 0.f;
        if (lane < 31) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                scale = upstream_max_abs(sums[j], scale);
        }
        scale = warp_max(scale);
        if (scale != 0.f)
            scale = 448.f / scale;
        PackedFp8 result;
        if (lane < 31) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                result.values[j] = static_cast<__nv_fp8_e4m3>(scale == 0.f ? sums[j] : sums[j] * scale);
        } else
            *reinterpret_cast<float*>(&result) = scale;
        *reinterpret_cast<int4*>(input[0] + rank_offset + off) = result.packed;
    }
}

__device__ void second_stage_tp2(size_t         in_per_rank,
                                 size_t         out_per_rank,
                                 __nv_fp8_e4m3* input[kTp],
                                 __nv_bfloat16* output,
                                 float*         smem,
                                 int            dst[kTp]) {
    const int    lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t in_start  = (blockIdx.x * kWarps + warp) * 16 * 32 + lane * 16;
    const size_t out_start = (blockIdx.x * kWarps + warp) * 31 * 16 + lane * 16;
    float*       scales    = &smem[kTp * warp];
    PackedFp8    values[kTp];
    for (size_t in_off = in_start, out_off = out_start; in_off < in_per_rank;
         in_off += gridDim.x * kWarps * 16 * 32, out_off += gridDim.x * kWarps * 31 * 16) {
#pragma unroll
        for (int i = 0; i < kTp; ++i)
            values[i].packed = *reinterpret_cast<int4 const*>(input[i] + dst[i] * in_per_rank + in_off);
        if (lane == 31) {
#pragma unroll
            for (int i = 0; i < kTp; ++i)
                scales[i] = *reinterpret_cast<float*>(&values[i]);
        }
        __syncwarp();
        if (lane < 31 && out_off < out_per_rank) {
#pragma unroll
            for (int i = 0; i < kTp; ++i) {
                const size_t dst_off = dst[i] * out_per_rank + out_off;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    float v             = static_cast<float>(values[i].values[j]);
                    output[dst_off + j] = __float2bfloat16(scales[i] == 0.f ? v : v / scales[i]);
                }
            }
        }
        // Lane 31 publishes the next iteration's scale. On independent-thread
        // scheduling it must not overwrite shared scales while other lanes still
        // dequantize this iteration.
        __syncwarp();
    }
}

struct KernelParams {
    __nv_bfloat16 const* in;
    __nv_bfloat16*       out;
    __nv_fp8_e4m3*       comm[kTp];
    uint64_t*            barriers[kTp];
    size_t               out_per_rank;
    size_t               buffer_per_rank;
    int                  rank;
    uint64_t             flag;
};
__global__ void two_shot_tp2(KernelParams p) {
    extern __shared__ float smem[];
    initial_barrier(p.barriers, p.flag, p.rank, threadIdx.x, blockIdx.x);
    __nv_fp8_e4m3* src[kTp];
    int            dst[kTp];
#pragma unroll
    for (int i = 0; i < kTp; ++i) {
        int r  = (p.rank + i) % kTp;
        src[i] = p.comm[r];
        dst[i] = r;
    }
    first_stage_tp2(p.rank, p.buffer_per_rank, src, smem);
    __syncthreads();
    const size_t pre_offset = kTp + blockIdx.x * kTp;
    block_peer_barrier(p.barriers, p.flag, p.rank, pre_offset, threadIdx.x);
    second_stage_tp2(p.buffer_per_rank, p.out_per_rank, src, p.out, smem + kTp * kWarps, dst);
    // Do not let thread 0/1 publish completion while another warp still reads
    // its FP8/scale packet in second_stage_tp2.
    __syncthreads();
    // Upstream supplies ping/pong buffers but has no post-allgather fence. This
    // fence makes reuse safe even when a peer's dedicated stream is behind.
    const size_t post_offset = kTp + gridDim.x * kTp + blockIdx.x * kTp;
    block_peer_barrier(p.barriers, p.flag, p.rank, post_offset, threadIdx.x);
}

class TpFp8AllReduce {
public:
    TpFp8AllReduce(uint64_t max_numel, int device_index, int rank, int blocks = kDefaultBlocks):
        max_numel_(max_numel), device_(device_index), rank_(rank), blocks_(blocks) {
        TORCH_CHECK(max_numel_ > 0 && max_numel_ % (16 * kTp) == 0, "max_numel must be positive and divisible by 32");
        TORCH_CHECK(rank_ >= 0 && rank_ < kTp, "rank must be 0 or 1");
        TORCH_CHECK(blocks_ > 0, "blocks must be positive");
        c10::cuda::CUDAGuard guard(device_);
        const int            dynamic_smem_bytes   = kWarps * kTp * sizeof(float) * 2;
        int                  active_blocks_per_sm = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active_blocks_per_sm, two_shot_tp2, kThreads, dynamic_smem_bytes),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor(two_shot_tp2)");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        const int64_t resident_limit = static_cast<int64_t>(active_blocks_per_sm) * properties.multiProcessorCount;
        TORCH_CHECK(blocks_ <= resident_limit,
                    "TP FP8 allreduce blocks=",
                    blocks_,
                    " exceeds concurrently resident grid limit=",
                    resident_limit,
                    "; this kernel uses device-wide peer barriers");
        barrier_words_   = kTp + 2 * static_cast<size_t>(blocks_) * kTp;
        rounds_capacity_ = div_up(div_up(max_numel_, kTp), kPayloadEltsPerBlock);
        comm_bytes_      = 2 * kTp * kBufferEltsPerRound * rounds_capacity_;
        try {
            cuda_check(cudaMalloc(&comm_, comm_bytes_), "cudaMalloc(comm)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&barrier_), barrier_words_ * sizeof(uint64_t)),
                       "cudaMalloc(barrier)");
            cuda_check(cudaMemsetAsync(barrier_, 0, barrier_words_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(barrier)");
        } catch (...) {
            // cudaMalloc/cudaMemset failures must not strand the first allocation.
            if (barrier_)
                cudaFree(barrier_);
            if (comm_)
                cudaFree(comm_);
            barrier_ = nullptr;
            comm_    = nullptr;
            throw;
        }
    }

    ~TpFp8AllReduce() {
        release_noexcept();
    }

    py::bytes get_ipc_handle() {
        ensure_open();
        c10::cuda::CUDAGuard guard(device_);
        IpcWireHandle        wire{};
        wire.magic         = kWireMagic;
        wire.version       = kWireVersion;
        wire.device        = device_;
        wire.rank          = rank_;
        wire.blocks        = blocks_;
        wire.max_numel     = max_numel_;
        wire.comm_bytes    = comm_bytes_;
        wire.barrier_words = barrier_words_;
        cuda_check(cudaIpcGetMemHandle(&wire.comm, comm_), "cudaIpcGetMemHandle(comm)");
        cuda_check(cudaIpcGetMemHandle(&wire.barrier, barrier_), "cudaIpcGetMemHandle(barrier)");
        return py::bytes(reinterpret_cast<char const*>(&wire), sizeof(wire));
    }

    void open_peer(py::bytes raw) {
        ensure_open();
        std::string bytes = raw;
        TORCH_CHECK(bytes.size() == sizeof(IpcWireHandle), "invalid TP FP8 allreduce IPC handle size");
        IpcWireHandle wire{};
        std::memcpy(&wire, bytes.data(), sizeof(wire));
        TORCH_CHECK(wire.magic == kWireMagic && wire.version == kWireVersion, "invalid TP FP8 allreduce IPC handle");
        TORCH_CHECK(wire.max_numel == max_numel_ && wire.comm_bytes == comm_bytes_
                        && wire.barrier_words == barrier_words_ && wire.blocks == blocks_,
                    "peer TP FP8 allreduce workspace mismatch");
        // CUDA indices are process-local under CUDA_VISIBLE_DEVICES. Rank is the
        // stable TP identity; the Python communicator owns topology validation.
        TORCH_CHECK(wire.rank == 1 - rank_, "peer TP FP8 allreduce rank mismatch");
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        try {
            // LazyEnablePeerAccess requests CUDA IPC/P2P mapping. CUDA returns an
            // error here when the selected GPUs or driver topology cannot provide it.
            cuda_check(cudaIpcOpenMemHandle(&peer_comm_, wire.comm, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(comm; requires CUDA IPC peer access)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_barrier_), wire.barrier, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(barrier; requires CUDA IPC peer access)");
        } catch (...) {
            close_peer_noexcept();
            throw;
        }
    }

    // The Python lifecycle calls this on both ranks after its own peer barrier,
    // before either owner frees local IPC allocations.
    void close_peer() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
    }

    void all_reduce(torch::Tensor input, torch::Tensor output) {
        ensure_open();
        TORCH_CHECK(peer_comm_ && peer_barrier_, "open_peer must be called before all_reduce");
        TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
        TORCH_CHECK(input.device().index() == device_ && output.device().index() == device_,
                    "tensor device does not match context");
        TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                    "TP FP8 allreduce supports BF16 only");
        TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input and output must be contiguous");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(int4) == 0
                        && reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(int4) == 0,
                    "TP FP8 allreduce input and output must be 16-byte aligned");
        TORCH_CHECK(input.numel() == output.numel() && input.numel() > 0
                        && static_cast<uint64_t>(input.numel()) <= max_numel_,
                    "invalid TP FP8 allreduce tensor size");
        TORCH_CHECK(input.numel() % (16 * kTp) == 0, "TP FP8 allreduce numel must be divisible by 32");

        c10::cuda::CUDAGuard guard(device_);
        cudaStream_t         stream = current_stream();
        if (stream_bound_) {
            TORCH_CHECK(bound_stream_ == stream,
                        "TpFp8AllReduce context may only be used on one dedicated CUDA stream");
        } else {
            bound_stream_ = stream;
            stream_bound_ = true;
        }

        const size_t total           = input.numel();
        const size_t per_rank        = total / kTp;
        const size_t rounds          = div_up(per_rank, kPayloadEltsPerBlock);
        const size_t buffer_per_rank = kBufferEltsPerRound * rounds;
        TORCH_CHECK(rounds <= rounds_capacity_, "TP FP8 allreduce workspace too small");

        const uint64_t flag = ++flag_;
        // Slots have fixed capacity offsets. A short invocation must not select an
        // offset that overlaps a preceding larger ping/pong invocation.
        const size_t slot_bytes       = kTp * kBufferEltsPerRound * rounds_capacity_;
        const size_t ping_pong_offset = (flag % 2 == 0) ? 0 : slot_bytes;
        auto*        local_comm = reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<char*>(comm_) + ping_pong_offset);
        auto* peer_comm = reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<char*>(peer_comm_) + ping_pong_offset);
        __nv_fp8_e4m3* comm[kTp] = {
            rank_ == 0 ? local_comm : peer_comm,
            rank_ == 0 ? peer_comm : local_comm,
        };
        uint64_t* barriers[kTp] = {
            rank_ == 0 ? barrier_ : peer_barrier_,
            rank_ == 0 ? peer_barrier_ : barrier_,
        };

        auto* input_ptr  = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr<at::BFloat16>());
        auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        preprocess_tp2_bf16_fp8<<<rounds * kTp, kThreads, 0, stream>>>(
            input_ptr, per_rank, buffer_per_rank, local_comm);
        cuda_check(cudaGetLastError(), "preprocess_tp2_bf16_fp8 launch");
        KernelParams params{input_ptr,
                            output_ptr,
                            {comm[0], comm[1]},
                            {barriers[0], barriers[1]},
                            per_rank,
                            buffer_per_rank,
                            rank_,
                            flag};
        two_shot_tp2<<<blocks_, kThreads, kWarps * kTp * sizeof(float) * 2, stream>>>(params);
        cuda_check(cudaGetLastError(), "two_shot_tp2 launch");
    }

    void close() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        if (comm_) {
            cuda_check(cudaFree(comm_), "cudaFree(comm)");
            comm_ = nullptr;
        }
        if (barrier_) {
            cuda_check(cudaFree(barrier_), "cudaFree(barrier)");
            barrier_ = nullptr;
        }
        bound_stream_ = nullptr;
        stream_bound_ = false;
    }

private:
    static size_t div_up(size_t a, size_t b) {
        return (a + b - 1) / b;
    }

    cudaStream_t current_stream() const {
        return at::cuda::getCurrentCUDAStream(device_).stream();
    }

    void ensure_open() const {
        TORCH_CHECK(comm_ && barrier_, "TpFp8AllReduce context is closed");
    }

    void close_peer_impl() {
        if (peer_comm_) {
            cuda_check(cudaIpcCloseMemHandle(peer_comm_), "cudaIpcCloseMemHandle(comm)");
            peer_comm_ = nullptr;
        }
        if (peer_barrier_) {
            cuda_check(cudaIpcCloseMemHandle(peer_barrier_), "cudaIpcCloseMemHandle(barrier)");
            peer_barrier_ = nullptr;
        }
    }

    void close_peer_noexcept() noexcept {
        if (peer_comm_)
            cudaIpcCloseMemHandle(peer_comm_);
        if (peer_barrier_)
            cudaIpcCloseMemHandle(peer_barrier_);
        peer_comm_    = nullptr;
        peer_barrier_ = nullptr;
    }

    void release_noexcept() noexcept {
        int        previous_device = -1;
        const bool have_previous   = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool switched        = have_previous && device_ >= 0 && previous_device != device_;
        if (switched)
            cudaSetDevice(device_);
        close_peer_noexcept();
        if (comm_)
            cudaFree(comm_);
        if (barrier_)
            cudaFree(barrier_);
        comm_    = nullptr;
        barrier_ = nullptr;
        if (switched)
            cudaSetDevice(previous_device);
    }

    uint64_t     max_numel_;
    int          device_;
    int          rank_;
    int          blocks_;
    size_t       barrier_words_{};
    size_t       rounds_capacity_{};
    size_t       comm_bytes_{};
    void*        comm_{};
    uint64_t*    barrier_{};
    void*        peer_comm_{};
    uint64_t*    peer_barrier_{};
    uint64_t     flag_{};
    cudaStream_t bound_stream_{};
    bool         stream_bound_{};
};
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<TpFp8AllReduce>(m, "TpFp8AllReduce")
        .def(py::init<uint64_t, int, int, int>(),
             py::arg("max_numel"),
             py::arg("device_index"),
             py::arg("rank"),
             py::arg("blocks") = kDefaultBlocks)
        .def("get_ipc_handle", &TpFp8AllReduce::get_ipc_handle)
        .def("open_peer", &TpFp8AllReduce::open_peer)
        .def("close_peer", &TpFp8AllReduce::close_peer)
        .def("all_reduce", &TpFp8AllReduce::all_reduce)
        .def("close", &TpFp8AllReduce::close);
}
