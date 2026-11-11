// Recovered from the fixed SDK 3580269309 FP32 GEMVT ELF.
// K4096 only. Arithmetic: 64 lanes/output, four serial FMA/lane per K256,
// down-shuffle 16,8,4,2,1 in each warp, then warp0 + warp1.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <vector>
namespace {
// One 64-thread group computes four columns, reusing each x float4.
__global__ void row_gemvt_cols4(const float* x, const float* w, float* out, int m, int n) {
    const int lane = threadIdx.x, first_col = 4 * blockIdx.x, first = 4 * blockIdx.y;
    float     acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
            acc[r][c] = 0.0f;
    }
    for (int k = 4 * lane; k < 4096; k += 256) {
        float4 weight[4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            weight[c] = *reinterpret_cast<const float4*>(w + (first_col + c) * 4096 + k);
#pragma unroll
        for (int r = 0; r < 4; ++r)
            if (first + r < m) {
                const float4 value = *reinterpret_cast<const float4*>(x + (first + r) * 4096 + k);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[r][c] = __fmaf_rn(weight[c].x, value.x, acc[r][c]);
                    acc[r][c] = __fmaf_rn(weight[c].y, value.y, acc[r][c]);
                    acc[r][c] = __fmaf_rn(weight[c].z, value.z, acc[r][c]);
                    acc[r][c] = __fmaf_rn(weight[c].w, value.w, acc[r][c]);
                }
            }
    }
    __shared__ float partials[4][4][2];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                acc[r][c] = __fadd_rn(acc[r][c], __shfl_down_sync(0xffffffff, acc[r][c], offset));
            if ((lane & 31) == 0)
                partials[r][c][lane / 32] = acc[r][c];
        }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
            if (first + r < m) {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    out[(first + r) * n + first_col + c] = __fmul_rn(1.0f, __fadd_rn(acc[r][c], partials[r][c][1]));
            }
    }
}
// Private fixed-output entry; no production dispatch modification.
void forward_out(const at::Tensor& x, const at::Tensor& w, at::Tensor out) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && out.device() == x.device(), "same CUDA device required");
    TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat
                    && x.dim() == 2 && w.dim() == 2 && out.dim() == 2 && x.is_contiguous() && w.is_contiguous()
                    && out.is_contiguous(),
                "contiguous FP32 matrices required");
    TORCH_CHECK(x.size(0) == 24 && x.size(1) == 4096 && w.size(1) == 4096 && (w.size(0) == 1024 || w.size(0) == 2048)
                    && out.size(0) == 24 && out.size(1) == w.size(0),
                "B8 Q3 main compressor only");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
                    && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "aligned vectors required");
    TORCH_CHECK(!out.is_alias_of(x) && !out.is_alias_of(w), "output must not alias operands");
    const c10::cuda::CUDAGuard guard(x.device());
    auto                       stream = c10::cuda::getCurrentCUDAStream();
    row_gemvt_cols4<<<dim3(w.size(0) / 4, 6), 64, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), 24, w.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> attributes() {
    cudaFuncAttributes a{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(row_gemvt_cols4)));
    return {a.maxThreadsPerBlock,
            a.numRegs,
            static_cast<int64_t>(a.sharedSizeBytes),
            static_cast<int64_t>(a.localSizeBytes)};
}
}  // namespace
TORCH_LIBRARY(fp32_cols4_probe, m) {
    m.def("forward_out(Tensor x, Tensor w, Tensor(a!) out) -> ()");
    m.def("attributes() -> int[]", &attributes);
}
TORCH_LIBRARY_IMPL(fp32_cols4_probe, CUDA, m) {
    m.impl("forward_out", forward_out);
}
