// Candidate-only FP32 K2048 selector over default-disabled native policy hooks.
#include "topk_v3.cuh"
#include "_v41_candidate_topk_sb.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace v41_candidate_topk {
template <int Vecs, bool Aligned>
__global__ __launch_bounds__(1024, 2) void select_ids(
    const float* scores, int32_t* output, int32_t* row_status,
    int width, int out_width, int64_t row_stride, int64_t out_stride) {
    using Impl = device::topk::TopKRegister<Vecs>;
    struct SbScratch { int values[32]; };
    __shared__ device::topk::MaxSmem<typename Impl::Smem, SbScratch> scratch;
    __shared__ device::topk::CandidateContext candidate;
    const int tx = threadIdx.x;
    const int r = blockIdx.x;
    const float* row = scores + static_cast<int64_t>(r) * row_stride;
    int32_t* dst = output + static_cast<int64_t>(r) * out_stride;
    if (tx == 0) { candidate.status = 0; row_status[r] = 0; }
    for (int col = tx; col < out_width; col += blockDim.x) dst[col] = -1;
    __syncthreads();
    device::topk::TopKProblem problem{row, dst, nullptr, nullptr, 2048u,
                                     static_cast<uint32_t>(width), 0u};
    Impl::template forward<false, Aligned, true>(problem, &scratch, &candidate);
    // Small-tie paths have partially active warps; join before reusing scratch.
    __syncthreads();
    if (candidate.status < 0) {
        sb_fallback(row, dst, width, out_width, reinterpret_cast<int*>(&scratch));
    } else if (candidate.status != 7) {
        for (int col = tx; col < 2048; col += blockDim.x) {
            const int32_t idx = dst[col];
            dst[col] = idx >= 0 && idx < width && row[idx] > -CUDART_INF_F ? idx : -1;
        }
    }
    if (tx == 0) row_status[r] = candidate.status;
}
}  // namespace v41_candidate_topk

extern "C" int v41_candidate_topk_launch(const float* scores, int32_t* output,
    int32_t* status, int rows, int width, int out_width,
    int64_t row_stride, int64_t out_stride, cudaStream_t stream) {
    if (!scores || !output || !status || rows < 1 || rows > 8192 ||
        width <= 2048 || width > 16384 || out_width < 2048 || out_width > 4096 ||
        row_stride < width || out_stride < out_width ||
        row_stride > INT64_MAX / rows / 4 || out_stride > INT64_MAX / rows / 4)
        return static_cast<int>(cudaErrorInvalidValue);
    const bool aligned = (reinterpret_cast<uintptr_t>(scores) % 16 == 0) && (row_stride % 4 == 0);
    if (width <= 8192) {
        if (aligned) v41_candidate_topk::select_ids<2, true><<<rows,1024,0,stream>>>(scores,output,status,width,out_width,row_stride,out_stride);
        else v41_candidate_topk::select_ids<2, false><<<rows,1024,0,stream>>>(scores,output,status,width,out_width,row_stride,out_stride);
    } else {
        if (aligned) v41_candidate_topk::select_ids<4, true><<<rows,1024,0,stream>>>(scores,output,status,width,out_width,row_stride,out_stride);
        else v41_candidate_topk::select_ids<4, false><<<rows,1024,0,stream>>>(scores,output,status,width,out_width,row_stride,out_stride);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int v41_candidate_topk_prepare(void** context) {
    cudaFuncAttributes attributes;
    cudaError_t error;
#define PREPARE(V, A) \
    error = cudaFuncGetAttributes(&attributes, v41_candidate_topk::select_ids<V, A>); \
    if (error != cudaSuccess) return static_cast<int>(error)
    PREPARE(2, true);
    PREPARE(2, false);
    PREPARE(4, true);
    PREPARE(4, false);
#undef PREPARE
    return static_cast<int>(cuCtxGetCurrent(reinterpret_cast<CUcontext*>(context)));
}

extern "C" int v41_candidate_topk_is_current(void* expected) {
    CUcontext current = nullptr;
    return cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr &&
           current == reinterpret_cast<CUcontext>(expected);
}
