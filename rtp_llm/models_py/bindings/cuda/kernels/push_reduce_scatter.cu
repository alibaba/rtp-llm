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
// Adapted from SGLang, revision
// 2d216a11f8c2ed125e46de581ed255bc138fffe1:
// python/sglang/kernels/jit/csrc/kimi_k3/comm/sp_collective.cuh
// reduce_scatter_res_kernel<8, false, true>, and include/sgl_kernel/{vec,
// distributed/communicator,distributed/ptx}.cuh.
// Preserve its 16-byte Lamport payload, two phases, counter bumper, PDL and
// rank-ordered FP32 accumulation. Only the host binding is changed to PyTorch.
#include "rtp_llm/models_py/bindings/cuda/kernels/push_reduce_scatter.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace rtp_llm {
namespace {

constexpr uint32_t kWorld = 8;

struct Params {
    const uint8_t* input;
    uint8_t*       output;
    uint8_t*       peers[kWorld];
    uint32_t*      counters;
    int64_t        stride_bytes;
    uint32_t       num_counters;
    uint32_t       rank;
    uint32_t       local_vecs;
};

union alignas(16) Vector {
    uint32_t       words[4];
    __nv_bfloat162 pairs[4];
};

template<bool Peer>
__device__ __forceinline__ void load(Vector& v, const uint8_t* base, uint32_t index) {
    const void* ptr = base + int64_t(index) * 16;
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

template<bool Peer>
__device__ __forceinline__ void store(uint8_t* base, uint32_t index, const Vector& v) {
    void* ptr = base + int64_t(index) * 16;
    if constexpr (Peer) {
        asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
                     :
                     : "r"(v.words[0]), "r"(v.words[1]), "r"(v.words[2]), "r"(v.words[3]), "l"(ptr));
    } else {
        asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
                     :
                     : "r"(v.words[0]), "r"(v.words[1]), "r"(v.words[2]), "r"(v.words[3]), "l"(ptr));
    }
}

__device__ __forceinline__ void trigger_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__global__ void push_reduce_scatter_kernel(const __grid_constant__ Params params) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const uint32_t bx    = blockIdx.x;
    const uint32_t phase = params.counters[bx] & 1;
    // Keep inactive counters in the same phase when M changes the launch grid.
    if (bx + 1 == gridDim.x) {
        __syncthreads();
        for (uint32_t i = bx + threadIdx.x; i < params.num_counters; i += blockDim.x) {
            params.counters[i] = phase ^ 1;
        }
        trigger_dependents();
        return;
    }
    const uint32_t tid             = bx * blockDim.x + threadIdx.x;
    const uint32_t step            = (gridDim.x - 1) * blockDim.x;
    const int64_t  phase_offset    = phase * kWorld * params.stride_bytes;
    const int64_t  producer_offset = phase_offset + params.rank * params.stride_bytes;
    for (uint32_t vid = tid; vid < kWorld * params.local_vecs; vid += step) {
        const uint32_t dst_rank  = vid / params.local_vecs;
        const uint32_t local_vid = vid - dst_rank * params.local_vecs;
        Vector         v;
        load<false>(v, params.input, vid);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Two +zero BF16 values would be indistinguishable from empty.
            if (v.words[j] == 0)
                v.words[j] = 0x8000u;
        }
        store<true>(params.peers[dst_rank] + producer_offset, local_vid, v);
    }
    trigger_dependents();

    auto*        poll_base = params.peers[params.rank] + phase_offset;
    const Vector zero      = {};
    for (uint32_t vid = tid; vid < params.local_vecs; vid += step) {
        Vector values[kWorld];
        while (true) {
            bool empty = false;
#pragma unroll
            for (uint32_t rank = 0; rank < kWorld; ++rank) {
                load<true>(values[rank], poll_base + rank * params.stride_bytes, vid);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    empty |= values[rank].words[j] == 0;
            }
            if (!empty)
                break;
        }
        float2 acc[4];
#pragma unroll
        for (uint32_t rank = 0; rank < kWorld; ++rank) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto pair = __bfloat1622float2(values[rank].pairs[j]);
                acc[j].x        = rank == 0 ? pair.x : acc[j].x + pair.x;
                acc[j].y        = rank == 0 ? pair.y : acc[j].y + pair.y;
            }
        }
        Vector out;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out.pairs[j] = __float22bfloat162_rn(acc[j]);
        store<false>(params.output, vid, out);
#pragma unroll
        for (uint32_t rank = 0; rank < kWorld; ++rank) {
            store<false>(poll_base + rank * params.stride_bytes, vid, zero);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        params.counters[bx] = phase ^ 1;
}

}  // namespace

void push_reduce_scatter(const torch::Tensor&              input,
                         torch::Tensor&                    output,
                         const std::vector<torch::Tensor>& peers,
                         torch::Tensor&                    counters,
                         int64_t                           rank,
                         int64_t                           blocks,
                         int64_t                           threads) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
                "Push RS requires contiguous CUDA BF16 [M,N]");
    TORCH_CHECK(output.device() == input.device() && output.dim() == 2 && output.scalar_type() == at::kBFloat16
                    && output.is_contiguous() && output.size(1) == input.size(1)
                    && input.size(0) == kWorld * output.size(0),
                "Push RS output must be a matching TP8 row shard");
    TORCH_CHECK(output.numel() > 0 && output.numel() % 8 == 0, "Push RS requires nonempty 16-byte aligned shards");
    // The upstream producer loop uses uint32 vector indices; avoid wraparound.
    TORCH_CHECK(input.numel() / 8 <= std::numeric_limits<uint32_t>::max() - 65536,
                "Push RS input exceeds vector indexing range");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0
                    && reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
                "Push RS tensor addresses must be 16-byte aligned");
    TORCH_CHECK(peers.size() == kWorld && rank >= 0 && rank < kWorld, "Push RS requires TP8");
    TORCH_CHECK(counters.device() == input.device() && counters.scalar_type() == at::kInt && counters.dim() == 1
                    && counters.is_contiguous(),
                "invalid Push RS counters");
    TORCH_CHECK(blocks > 0 && blocks < counters.numel() && threads >= 32 && threads <= 512 && threads % 32 == 0,
                "invalid Push RS launch configuration");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto*                properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "Push RS is enabled only on SM100/SM103");
    TORCH_CHECK(blocks < properties->multiProcessorCount, "Push RS grid must fit concurrently on the GPU");
    const int64_t bytes = peers[0].numel();
    TORCH_CHECK(bytes % (2 * kWorld * 16) == 0 && bytes / (2 * kWorld) >= output.nbytes(),
                "Push RS workspace is too small or misaligned");
    Params params{};
    params.input        = static_cast<const uint8_t*>(input.data_ptr());
    params.output       = static_cast<uint8_t*>(output.data_ptr());
    params.counters     = reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>());
    params.stride_bytes = bytes / (2 * kWorld);
    params.num_counters = counters.numel();
    params.rank         = rank;
    params.local_vecs   = output.numel() / 8;
    for (uint32_t i = 0; i < kWorld; ++i) {
        TORCH_CHECK(peers[i].device() == input.device() && peers[i].scalar_type() == at::kByte
                        && peers[i].is_contiguous() && peers[i].numel() == bytes
                        && reinterpret_cast<uintptr_t>(peers[i].data_ptr()) % 16 == 0,
                    "invalid Push RS peer buffer");
        params.peers[i] = peers[i].data_ptr<uint8_t>();
    }
    cudaLaunchAttribute attribute{};
    attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(blocks + 1);
    config.blockDim = dim3(threads);
    config.stream   = at::cuda::getCurrentCUDAStream();
    config.attrs    = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, push_reduce_scatter_kernel, params));
}

}  // namespace rtp_llm
