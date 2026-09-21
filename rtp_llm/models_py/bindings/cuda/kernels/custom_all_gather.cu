// Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. The full license is in the repository LICENSE.
//
// BF16 staging/direct and multicast semaphore/PDL protocols adapted from
// SGLang revision efa7be2091282e96318f0d9d22d9f78dde099848:
// python/sglang/kernels/jit/csrc/kimi_k3/comm/sp_collective.cuh and
// python/sglang/kernels/jit/include/sgl_kernel/distributed/{communicator,ptx}.cuh.
// The FP8 pair traversal and lossless marker encoding extend those protocols.
// This port uses only PyTorch/CUDA, with no SGLang or TVM runtime dependency.
#include "rtp_llm/models_py/bindings/cuda/kernels/custom_all_gather.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace rtp_llm {
namespace {

constexpr uint32_t kWorld = 8;

struct alignas(16) Vector {
    uint32_t words[4];
};

// Match the upstream flag/counter layout. Only the first two words are used;
// neighboring blocks own separate cache lines.
struct alignas(128) Semaphore {
    uint32_t flag;
    uint32_t counter;
};
static_assert(sizeof(Semaphore) == 128);

struct Params {
    const uint8_t*  input;
    const uint32_t* scales;
    uint8_t*        output;
    uint32_t*       output_scales;
    uint8_t*        workspace;
    uint8_t*        workspace_mc;
    uint32_t*       counters;
    Semaphore*      sem_local;
    Semaphore*      sem_mc;
    uint8_t*        output_mc;
    uint32_t*       output_scales_mc;
    int64_t         slot_bytes;
    uint32_t        num_counters;
    uint32_t        rank;
    uint32_t        rows;
    uint32_t        padded_rows;
    uint32_t        output_rows;
    uint32_t        groups;
    uint32_t        value_vecs;
    uint32_t        total_vecs;
};

template<bool Peer>
__device__ __forceinline__ void load_vector(Vector& v, const void* base, uint32_t index) {
    const void* ptr = static_cast<const uint8_t*>(base) + int64_t(index) * 16;
    if constexpr (Peer) {
        asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.words[0]), "=r"(v.words[1]), "=r"(v.words[2]), "=r"(v.words[3])
                     : "l"(ptr));
    } else {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.words[0]), "=r"(v.words[1]), "=r"(v.words[2]), "=r"(v.words[3])
                     : "l"(ptr));
    }
}

__device__ __forceinline__ void store_vector(void* base, uint32_t index, const Vector& v) {
    void* ptr = static_cast<uint8_t*>(base) + int64_t(index) * 16;
    asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
                 :
                 : "r"(v.words[0]), "r"(v.words[1]), "r"(v.words[2]), "r"(v.words[3]), "l"(ptr));
}

__device__ __forceinline__ void multicast_vector(void* base, uint32_t index, const Vector& v) {
#if __CUDA_ARCH__ >= 900
    void* ptr = static_cast<uint8_t*>(base) + int64_t(index) * 16;
    // PTX's v4.f32 store transports raw bits; these are bitcasts, not FP
    // conversions, including for FP8 payload bytes and packed scale words.
    asm volatile("multimem.st.weak.global.v4.f32 [%4], {%0, %1, %2, %3};"
                 :
                 : "f"(__uint_as_float(v.words[0])),
                   "f"(__uint_as_float(v.words[1])),
                   "f"(__uint_as_float(v.words[2])),
                   "f"(__uint_as_float(v.words[3])),
                   "l"(ptr));
#else
    asm volatile("trap;");
#endif
}

__device__ __forceinline__ void multicast_word(uint32_t* ptr, uint32_t value) {
#if __CUDA_ARCH__ >= 900
    asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
#else
    asm volatile("trap;");
#endif
}

template<bool Acquire>
__device__ __forceinline__ uint32_t load_word(const uint32_t* ptr) {
    uint32_t value;
    if constexpr (Acquire) {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    } else {
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    }
    return value;
}

__device__ __forceinline__ void wait_primary() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void trigger_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Thread 0 alone owns the reservation and arrival count. The reservation must
// precede the PDL wait, but signalling must follow it: signalling promises
// that this rank's preceding producer grid flushed.
struct MulticastBarrier {
    Semaphore* local;
    Semaphore* mc;
    uint32_t   window;

    __device__ __forceinline__ MulticastBarrier(Semaphore* local_base, Semaphore* mc_base):
        local(local_base + blockIdx.x), mc(mc_base + blockIdx.x), window(0) {
        if (threadIdx.x == 0)
            window = atomicAdd(&local->counter, 2 * kWorld);
    }

    template<bool ReleaseAcquire>
    __device__ __forceinline__ void arrive(uint32_t n) const {
        if (threadIdx.x != 0)
            return;
#if __CUDA_ARCH__ >= 900
        if constexpr (ReleaseAcquire) {
            asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(&mc->flag), "r"(1u) : "memory");
        } else {
            asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" ::"l"(&mc->flag), "r"(1u) : "memory");
        }
        const uint32_t current = window + n * kWorld;
        while (load_word<ReleaseAcquire>(&local->flag) - current < kWorld) {}
#else
        asm volatile("trap;");
#endif
    }
};

__device__ __forceinline__ bool bump_inactive_counters(const Params& p, uint32_t phase) {
    if (blockIdx.x + 1 != gridDim.x)
        return false;
    __syncthreads();
    // The final block flips itself and all inactive counters so later calls
    // can change the work grid without selecting inconsistent phases.
    for (uint32_t i = blockIdx.x + threadIdx.x; i < p.num_counters; i += blockDim.x)
        p.counters[i] = phase ^ 1;
    trigger_dependents();
    return true;
}

__global__ void all_gather_staging_kernel(const __grid_constant__ Params p) {
    wait_primary();
    const uint32_t phase = p.counters[blockIdx.x] & 1;
    if (bump_inactive_counters(p, phase))
        return;
    const uint32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t step         = (gridDim.x - 1) * blockDim.x;
    const int64_t  phase_offset = phase * kWorld * p.slot_bytes;
    auto*          producer     = p.workspace_mc + phase_offset + p.rank * p.slot_bytes;
    for (uint32_t vid = tid; vid < p.value_vecs; vid += step) {
        Vector value;
        load_vector<false>(value, p.input, vid);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (value.words[j] == 0)
                value.words[j] = 0x8000u;
        }
        multicast_vector(producer, vid, value);
    }
    trigger_dependents();

    const Vector zero = {};
    for (uint32_t vid = tid; vid < kWorld * p.value_vecs; vid += step) {
        const uint32_t src_rank  = vid / p.value_vecs;
        const uint32_t local_vid = vid - src_rank * p.value_vecs;
        auto*          src       = p.workspace + phase_offset + src_rank * p.slot_bytes;
        Vector         value;
        do {
            load_vector<true>(value, src, local_vid);
        } while (value.words[0] == 0 || value.words[1] == 0 || value.words[2] == 0 || value.words[3] == 0);
        store_vector(p.output, vid, value);
        store_vector(src, local_vid, zero);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        p.counters[blockIdx.x] = phase ^ 1;
}

template<bool AlignedScales>
__global__ void all_gather_fp8_staging_kernel(const __grid_constant__ Params p) {
    wait_primary();
    const uint32_t phase = p.counters[blockIdx.x] & 1;
    if (bump_inactive_counters(p, phase))
        return;
    const uint32_t lanes         = blockDim.x / kWorld;
    const uint32_t tid           = blockIdx.x * lanes + threadIdx.x % lanes;
    const uint32_t step          = (gridDim.x - 1) * lanes;
    const int64_t  phase_offset  = phase * kWorld * p.slot_bytes;
    const int64_t  meta_offset   = (p.slot_bytes / 20) * 16;
    auto*          producer      = p.workspace_mc + phase_offset + p.rank * p.slot_bytes;
    auto*          producer_meta = reinterpret_cast<uint32_t*>(producer + meta_offset);
    // One subgroup produces; eight subgroups consume the corresponding
    // vectors from every rank. Four reserved sign bits are stored separately,
    // with a fifth marker bit, so every published word is independently ready.
    if (threadIdx.x < lanes) {
        for (uint32_t vid = tid; vid < p.total_vecs; vid += step) {
            Vector value;
            if (vid < p.value_vecs)
                load_vector<false>(value, p.input, vid);
            else
                load_vector<false>(value, p.scales, vid - p.value_vecs);
            uint32_t meta = 16;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                meta |= (value.words[j] >> 31) << j;
                value.words[j] |= 0x80000000u;
            }
            multicast_vector(producer, vid, value);
            multicast_word(producer_meta + vid, meta);
        }
    }
    trigger_dependents();
    const uint32_t src_rank = threadIdx.x / lanes;
    auto*          src      = p.workspace + phase_offset + src_rank * p.slot_bytes;
    auto*          src_meta = reinterpret_cast<uint32_t*>(src + meta_offset);
    const Vector   zero     = {};
    for (uint32_t vid = tid; vid < p.total_vecs; vid += step) {
        Vector value;
        do {
            load_vector<true>(value, src, vid);
        } while ((value.words[0] & value.words[1] & value.words[2] & value.words[3] & 0x80000000u) == 0);
        uint32_t meta;
        do {
            meta = load_word<false>(src_meta + vid);
        } while (meta == 0);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
            value.words[j] = (value.words[j] & 0x7fffffffu) | (((meta >> j) & 1u) << 31);
        if (vid < p.value_vecs) {
            store_vector(p.output, src_rank * p.value_vecs + vid, value);
        } else {
            const uint32_t scale_vid = vid - p.value_vecs;
            const uint32_t group     = scale_vid / (p.padded_rows / 4);
            const uint32_t row       = (scale_vid % (p.padded_rows / 4)) * 4;
            const uint32_t dst       = group * p.output_rows + src_rank * p.rows + row;
            if constexpr (AlignedScales) {
                store_vector(p.output_scales, dst / 4, value);
            } else {
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j)
                    if (row + j < p.rows)
                        p.output_scales[dst + j] = value.words[j];
            }
        }
        // Reclaim payload AND metadata locally. The other phase protects a
        // peer still draining this round; PDL waits order local invocations.
        store_vector(src, vid, zero);
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(src_meta + vid), "r"(0u) : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0)
        p.counters[blockIdx.x] = phase ^ 1;
}

template<bool FP8, bool AlignedScales = true>
__global__ void all_gather_direct_kernel(const __grid_constant__ Params p) {
    const MulticastBarrier barrier(p.sem_local, p.sem_mc);
    wait_primary();
    barrier.arrive<false>(0);
    __syncthreads();
    const uint32_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t step     = gridDim.x * blockDim.x;
    const uint32_t dst_bias = p.rank * p.value_vecs;
    for (uint32_t vid = tid; vid < p.value_vecs; vid += step) {
        Vector value;
        load_vector<false>(value, p.input, vid);
        multicast_vector(p.output_mc, dst_bias + vid, value);
    }
    if constexpr (FP8) {
        if constexpr (AlignedScales) {
            const uint32_t rows4 = p.rows / 4;
            const uint32_t count = p.groups * rows4;
            for (uint32_t vid = tid; vid < count; vid += step) {
                const uint32_t group = vid / rows4;
                const uint32_t row4  = vid - group * rows4;
                const uint32_t dst   = group * (p.output_rows / 4) + p.rank * rows4 + row4;
                Vector         value;
                load_vector<false>(value, p.scales, vid);
                multicast_vector(p.output_scales_mc, dst, value);
            }
        } else {
            const uint32_t count = p.groups * p.rows;
            for (uint32_t i = tid; i < count; i += step) {
                const uint32_t group = i / p.rows;
                const uint32_t row   = i - group * p.rows;
                const uint32_t dst   = group * p.output_rows + p.rank * p.rows + row;
                multicast_word(p.output_scales_mc + dst, p.scales[group * p.padded_rows + row]);
            }
        }
    }
    // Values and scales share one entry/exit protocol.
    trigger_dependents();
    __syncthreads();
    barrier.arrive<true>(1);
}

void check_tensor(const torch::Tensor& tensor, const torch::Tensor& input, at::ScalarType type, const char* name) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.scalar_type() == type && tensor.is_contiguous(),
                "Custom all-gather ",
                name,
                " has an invalid device, dtype or layout");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "Custom all-gather ",
                name,
                " must have a 16-byte aligned address");
}

void check_mc(int64_t address, int64_t bytes, uint64_t alignment, const char* name) {
    TORCH_CHECK(address > 0 && uint64_t(address) % alignment == 0
                    && uint64_t(address) <= std::numeric_limits<uintptr_t>::max() - uint64_t(bytes),
                "Custom all-gather ",
                name,
                " must be a nonzero, aligned multicast mapping of the local buffer");
}

void check_indices(int64_t count, int64_t step) {
    TORCH_CHECK(count > 0 && count <= int64_t(std::numeric_limits<uint32_t>::max()) - step,
                "Custom all-gather exceeds 32-bit indexing range");
}

Params make_params(const torch::Tensor& input, torch::Tensor& output, int64_t rank, bool fp8) {
    const auto type = fp8 ? at::kByte : at::kBFloat16;
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.scalar_type() == type && input.is_contiguous(),
                "Custom all-gather requires a contiguous CUDA ",
                fp8 ? "uint8" : "BF16",
                " [rows,K] input");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0
                    && input.size(0) <= std::numeric_limits<uint32_t>::max() / kWorld,
                "Custom all-gather requires nonempty shards within the row indexing range");
    check_tensor(input, input, type, "input");
    check_tensor(output, input, type, "output");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == kWorld * input.size(0) && output.size(1) == input.size(1),
                "Custom all-gather output must contain eight rank-contiguous row shards");
    TORCH_CHECK(input.nbytes() % 16 == 0, "Custom all-gather requires 16-byte aligned shards");
    TORCH_CHECK(rank >= 0 && rank < kWorld, "Custom all-gather requires TP8 ranks");
    check_indices(output.nbytes() / 16, 1);
    Params p{};
    p.input       = static_cast<const uint8_t*>(input.data_ptr());
    p.output      = static_cast<uint8_t*>(output.data_ptr());
    p.rank        = rank;
    p.rows        = input.size(0);
    p.output_rows = output.size(0);
    p.value_vecs  = input.nbytes() / 16;
    p.total_vecs  = p.value_vecs;
    return p;
}

void bind_scales(Params& p, const torch::Tensor& input, const torch::Tensor& scales, torch::Tensor& output_scales) {
    TORCH_CHECK(input.size(1) <= std::numeric_limits<uint32_t>::max() && input.size(1) % 128 == 0,
                "Custom FP8 all-gather requires K divisible by 128 within 32-bit indexing range");
    const int64_t padded_rows = (input.size(0) + 3) / 4 * 4;
    const int64_t groups      = (input.size(1) + 511) / 512;
    check_tensor(scales, input, at::kInt, "scales");
    check_tensor(output_scales, input, at::kInt, "output scales");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == groups && scales.size(1) == padded_rows,
                "Custom FP8 all-gather scales must be int32[ceil(K/512), align(local_rows,4)]");
    TORCH_CHECK(output_scales.dim() == 2 && output_scales.size(0) == groups && output_scales.size(1) == p.output_rows,
                "Custom FP8 all-gather output scales must be int32[ceil(K/512), global_rows]");
    check_indices(output_scales.numel(), 1);
    check_indices(scales.numel(), 1);
    check_indices(int64_t(p.value_vecs) + scales.numel() / 4, 1);
    p.scales        = reinterpret_cast<const uint32_t*>(scales.data_ptr<int32_t>());
    p.output_scales = reinterpret_cast<uint32_t*>(output_scales.data_ptr<int32_t>());
    p.padded_rows   = padded_rows;
    p.groups        = groups;
    p.total_vecs    = int64_t(p.value_vecs) + scales.numel() / 4;
}

int check_launch(const Params& p, int64_t blocks, int64_t threads, bool staging, bool fp8) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "Custom all-gather is enabled only on SM100/SM103");
    const int sms = properties->multiProcessorCount;
    TORCH_CHECK(blocks > 0 && blocks <= sms - int64_t(staging),
                "Custom all-gather grid must fit concurrently on the GPU");
    const int64_t minimum_threads = staging && fp8 ? 128 : 32;
    const int64_t maximum_threads = staging ? 512 : 1024;
    TORCH_CHECK(threads >= minimum_threads && threads <= maximum_threads && threads % minimum_threads == 0,
                "Invalid custom all-gather block size");
    const int64_t step = blocks * threads;
    check_indices(int64_t(kWorld) * p.value_vecs, step);
    check_indices(p.total_vecs, step);
    if (fp8) {
        check_indices(int64_t(p.groups) * p.output_rows, step);
        check_indices(int64_t(p.groups) * p.padded_rows, step);
    }
    return sms;
}

void bind_staging(Params&              p,
                  const torch::Tensor& input,
                  torch::Tensor&       workspace,
                  torch::Tensor&       counters,
                  int64_t              workspace_mc_ptr,
                  int                  sms,
                  bool                 fp8) {
    check_tensor(workspace, input, at::kByte, "workspace");
    check_tensor(counters, input, at::kInt, "counters");
    TORCH_CHECK(workspace.dim() == 1 && workspace.numel() > 0 && workspace.numel() % (2 * kWorld) == 0,
                "Custom all-gather workspace must be uint8[16 * slot_bytes]");
    TORCH_CHECK(counters.dim() == 1 && counters.numel() == sms, "Custom all-gather counters must be int32[SM_count]");
    const int64_t slot_bytes = workspace.numel() / (2 * kWorld);
    if (fp8) {
        TORCH_CHECK(slot_bytes % 80 == 0 && slot_bytes / 20 >= p.total_vecs,
                    "Custom FP8 all-gather encoded slot must be a multiple of 80 bytes and fit payload plus metadata");
    } else {
        TORCH_CHECK(slot_bytes % 16 == 0 && slot_bytes / 16 >= p.value_vecs,
                    "Custom all-gather workspace slot is too small or misaligned");
    }
    check_mc(workspace_mc_ptr, workspace.nbytes(), 16, "workspace address");
    p.workspace    = workspace.data_ptr<uint8_t>();
    p.workspace_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(workspace_mc_ptr));
    p.counters     = reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>());
    p.slot_bytes   = slot_bytes;
    p.num_counters = sms;
}

void bind_direct(Params&              p,
                 const torch::Tensor& input,
                 torch::Tensor&       output,
                 torch::Tensor&       semaphores,
                 int64_t              output_mc_ptr,
                 int64_t              semaphore_mc_ptr,
                 int                  sms) {
    check_tensor(semaphores, input, at::kByte, "semaphores");
    TORCH_CHECK(semaphores.dim() == 2 && semaphores.size(0) == sms && semaphores.size(1) == sizeof(Semaphore)
                    && reinterpret_cast<uintptr_t>(semaphores.data_ptr()) % alignof(Semaphore) == 0,
                "Custom all-gather semaphores must be uint8[SM_count,128] with a 128-byte aligned address");
    check_mc(output_mc_ptr, output.nbytes(), 16, "output address");
    check_mc(semaphore_mc_ptr, semaphores.nbytes(), 128, "semaphore address");
    p.output_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_mc_ptr));
    p.sem_local = reinterpret_cast<Semaphore*>(semaphores.data_ptr<uint8_t>());
    p.sem_mc    = reinterpret_cast<Semaphore*>(static_cast<uintptr_t>(semaphore_mc_ptr));
}

template<typename Kernel>
void launch(Kernel kernel, const Params& p, int64_t blocks, int64_t threads) {
    cudaLaunchAttribute attribute{};
    attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(blocks);
    config.blockDim = dim3(threads);
    config.stream   = at::cuda::getCurrentCUDAStream();
    config.attrs    = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

}  // namespace

void custom_all_gather_staging(const torch::Tensor& input,
                               torch::Tensor&       output,
                               torch::Tensor&       workspace,
                               torch::Tensor&       counters,
                               int64_t              workspace_mc_ptr,
                               int64_t              rank,
                               int64_t              blocks,
                               int64_t              threads) {
    auto                       p = make_params(input, output, rank, false);
    const c10::cuda::CUDAGuard guard(input.device());
    const int                  sms = check_launch(p, blocks, threads, true, false);
    bind_staging(p, input, workspace, counters, workspace_mc_ptr, sms, false);
    launch(all_gather_staging_kernel, p, blocks + 1, threads);
}

void custom_all_gather_direct(const torch::Tensor& input,
                              torch::Tensor&       output,
                              torch::Tensor&       semaphores,
                              int64_t              output_mc_ptr,
                              int64_t              semaphore_mc_ptr,
                              int64_t              rank,
                              int64_t              blocks,
                              int64_t              threads) {
    auto                       p = make_params(input, output, rank, false);
    const c10::cuda::CUDAGuard guard(input.device());
    const int                  sms = check_launch(p, blocks, threads, false, false);
    bind_direct(p, input, output, semaphores, output_mc_ptr, semaphore_mc_ptr, sms);
    launch(all_gather_direct_kernel<false>, p, blocks, threads);
}

void custom_all_gather_fp8_staging(const torch::Tensor& values,
                                   const torch::Tensor& scales,
                                   torch::Tensor&       output_values,
                                   torch::Tensor&       output_scales,
                                   torch::Tensor&       workspace,
                                   torch::Tensor&       counters,
                                   int64_t              workspace_mc_ptr,
                                   int64_t              rank,
                                   int64_t              blocks,
                                   int64_t              threads) {
    auto p = make_params(values, output_values, rank, true);
    bind_scales(p, values, scales, output_scales);
    const c10::cuda::CUDAGuard guard(values.device());
    const int                  sms = check_launch(p, blocks, threads, true, true);
    bind_staging(p, values, workspace, counters, workspace_mc_ptr, sms, true);
    if (p.rows % 4 == 0)
        launch(all_gather_fp8_staging_kernel<true>, p, blocks + 1, threads);
    else
        launch(all_gather_fp8_staging_kernel<false>, p, blocks + 1, threads);
}

void custom_all_gather_fp8_direct(const torch::Tensor& values,
                                  const torch::Tensor& scales,
                                  torch::Tensor&       output_values,
                                  torch::Tensor&       output_scales,
                                  torch::Tensor&       semaphores,
                                  int64_t              output_values_mc_ptr,
                                  int64_t              output_scales_mc_ptr,
                                  int64_t              semaphore_mc_ptr,
                                  int64_t              rank,
                                  int64_t              blocks,
                                  int64_t              threads) {
    auto p = make_params(values, output_values, rank, true);
    bind_scales(p, values, scales, output_scales);
    const c10::cuda::CUDAGuard guard(values.device());
    const int                  sms = check_launch(p, blocks, threads, false, true);
    bind_direct(p, values, output_values, semaphores, output_values_mc_ptr, semaphore_mc_ptr, sms);
    check_mc(output_scales_mc_ptr, output_scales.nbytes(), 16, "output scales address");
    p.output_scales_mc = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(output_scales_mc_ptr));
    if (p.rows % 4 == 0)
        launch(all_gather_direct_kernel<true, true>, p, blocks, threads);
    else
        launch(all_gather_direct_kernel<true, false>, p, blocks, threads);
}

}  // namespace rtp_llm
