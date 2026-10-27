#include "rtp_llm/models_py/bindings/cuda/kernels/int8_lm_head.h"

#include <cuda_bf16.h>

namespace rtp_llm {

namespace {

constexpr int kThreads    = 256;
constexpr int kMaxM       = 8;
constexpr int kChunkElems = 16;  // one uint4 of int8 weights per lane per step
constexpr int kMaxK       = 8192;

__device__ __forceinline__ float warpSum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

// One warp produces one output row at a time (grid-stride over rows); x stays in shared memory.
// Each lane streams 16 int8 weights per step with a 128-bit load, so the kernel is DRAM bound.
template<int M>
__global__ void __launch_bounds__(kThreads) int8LmHeadGemvKernel(const int8_t* __restrict__ Q,
                                                                 const float* __restrict__ S,
                                                                 const __nv_bfloat16* __restrict__ x,
                                                                 float* __restrict__ y,
                                                                 int N,
                                                                 int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16*                                xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    {
        const uint4* src = reinterpret_cast<const uint4*>(x);
        uint4*       dst = reinterpret_cast<uint4*>(xs);
        const int    n16 = M * K / 8;
        for (int i = threadIdx.x; i < n16; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    const int lane     = threadIdx.x & 31;
    const int warp     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int n_warps  = (gridDim.x * blockDim.x) >> 5;
    const int n_chunks = K / kChunkElems;

    for (int row = warp; row < N; row += n_warps) {
        const uint4* wp = reinterpret_cast<const uint4*>(Q + static_cast<size_t>(row) * K);
        float        acc[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] = 0.f;
        }
#pragma unroll 4
        for (int c = lane; c < n_chunks; c += 32) {
            const uint4    w     = __ldcs(wp + c);
            const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
            float          wf[kChunkElems];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int v   = static_cast<int>(ws[j]);
                wf[4 * j + 0] = static_cast<float>((v << 24) >> 24);
                wf[4 * j + 1] = static_cast<float>((v << 16) >> 24);
                wf[4 * j + 2] = static_cast<float>((v << 8) >> 24);
                wf[4 * j + 3] = static_cast<float>(v >> 24);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4*   xp    = reinterpret_cast<const uint4*>(xs + m * K + c * kChunkElems);
                const uint4    xa    = xp[0];
                const uint4    xb    = xp[1];
                const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float          a     = acc[m];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[j]));
                    a               = fmaf(wf[2 * j], xf.x, a);
                    a               = fmaf(wf[2 * j + 1], xf.y, a);
                }
                acc[m] = a;
            }
        }
        const float s = __ldg(S + row);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float v = warpSum(acc[m]);
            if (lane == 0) {
                y[static_cast<size_t>(m) * N + row] = v * s;
            }
        }
    }
}

__global__ void
quantizeLmHeadInt8Kernel(const __nv_bfloat16* __restrict__ W, int8_t* __restrict__ Q, float* __restrict__ S, int K) {
    const size_t         row = blockIdx.x;
    const __nv_bfloat16* src = W + row * K;
    __shared__ float     red[32];
    float                m = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(__bfloat162float(src[k])));
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 16));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 8));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        if (threadIdx.x == 0) {
            red[0] = v;
        }
    }
    __syncthreads();
    const float scale = red[0] > 0.f ? red[0] / 127.f : 1.f;
    const float inv   = 1.f / scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        Q[row * K + k] = static_cast<int8_t>(__float2int_rn(__bfloat162float(src[k]) * inv));
    }
    if (threadIdx.x == 0) {
        S[row] = scale;
    }
}

template<int M>
cudaError_t
launchGemv(const int8_t* Q, const float* S, const __nv_bfloat16* x, float* y, int N, int K, cudaStream_t stream) {
    const size_t smem = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
    cudaError_t  err  = cudaSuccess;
    if (smem > 48 * 1024) {
        // Must precede the occupancy query, which otherwise reports 0 blocks for > 48 KiB.
        err = cudaFuncSetAttribute(
            int8LmHeadGemvKernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (err != cudaSuccess) {
            return err;
        }
    }
    static int grid = 0;
    if (grid == 0) {
        int dev = 0, sms = 0, blocks_per_sm = 0;
        err = cudaGetDevice(&dev);
        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        }
        if (err == cudaSuccess) {
            err =
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, int8LmHeadGemvKernel<M>, kThreads, smem);
        }
        if (err != cudaSuccess) {
            return err;
        }
        grid = sms * (blocks_per_sm > 0 ? blocks_per_sm : 1);
    }
    int8LmHeadGemvKernel<M><<<grid, kThreads, smem, stream>>>(Q, S, x, y, N, K);
    return cudaGetLastError();
}

}  // namespace

void invokeQuantizeLmHeadInt8(const __nv_bfloat16* W, int8_t* Q, float* S, int64_t N, int64_t K, cudaStream_t stream) {
    quantizeLmHeadInt8Kernel<<<static_cast<unsigned>(N), 256, 0, stream>>>(W, Q, S, static_cast<int>(K));
}

bool int8LmHeadGemvSupported(int64_t M, int64_t K) {
    if (M < 1 || M > kMaxM || K % 256 != 0 || K > kMaxK) {
        return false;
    }
    // x[M, K] is staged in dynamic shared memory; larger shapes keep the bf16 path.
    int dev = 0, max_smem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess
        || cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return static_cast<size_t>(M) * K * sizeof(__nv_bfloat16) <= static_cast<size_t>(max_smem);
}

cudaError_t invokeInt8LmHeadGemv(const int8_t*        Q,
                                 const float*         S,
                                 const __nv_bfloat16* x,
                                 float*               y,
                                 int64_t              M,
                                 int64_t              N,
                                 int64_t              K,
                                 cudaStream_t         stream) {
    const int n = static_cast<int>(N);
    const int k = static_cast<int>(K);
    switch (M) {
        case 1:
            return launchGemv<1>(Q, S, x, y, n, k, stream);
        case 2:
            return launchGemv<2>(Q, S, x, y, n, k, stream);
        case 3:
            return launchGemv<3>(Q, S, x, y, n, k, stream);
        case 4:
            return launchGemv<4>(Q, S, x, y, n, k, stream);
        case 5:
            return launchGemv<5>(Q, S, x, y, n, k, stream);
        case 6:
            return launchGemv<6>(Q, S, x, y, n, k, stream);
        case 7:
            return launchGemv<7>(Q, S, x, y, n, k, stream);
        case 8:
            return launchGemv<8>(Q, S, x, y, n, k, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace rtp_llm
