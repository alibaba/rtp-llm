// Recovered from the fixed SDK 3580269309 FP32 GEMVT ELF.
// K4096 only. Arithmetic: 64 lanes/output, four serial FMA/lane per K256,
// down-shuffle 16,8,4,2,1 in each warp, then warp0 + warp1.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
namespace {
template<int Rows>
__global__ void row_gemvt(const float* x, const float* w, float* out, int m, int n) {
    const int lane = threadIdx.x, column = 2 * blockIdx.x + threadIdx.y;
    const int first = blockIdx.y * Rows;
    float     acc[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r)
        acc[r] = 0.0f;
    for (int k = 4 * lane; k < 4096; k += 256) {
        const float4 weight = *reinterpret_cast<const float4*>(w + column * 4096 + k);
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (first + r < m) {
                const float4 value = *reinterpret_cast<const float4*>(x + (first + r) * 4096 + k);
                acc[r]             = __fmaf_rn(weight.x, value.x, acc[r]);
                acc[r]             = __fmaf_rn(weight.y, value.y, acc[r]);
                acc[r]             = __fmaf_rn(weight.z, value.z, acc[r]);
                acc[r]             = __fmaf_rn(weight.w, value.w, acc[r]);
            }
    }
    __shared__ float partials[Rows][2][2];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            acc[r] = __fadd_rn(acc[r], __shfl_down_sync(0xffffffff, acc[r], offset));
        if ((lane & 31) == 0)
            partials[r][threadIdx.y][lane / 32] = acc[r];
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (first + r < m)
                out[(first + r) * n + column] = __fmul_rn(1.0f, __fadd_rn(acc[r], partials[r][threadIdx.y][1]));
    }
}
at::Tensor forward(const at::Tensor& x, const at::Tensor& w, int64_t rows_per_cta) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && x.scalar_type() == at::kFloat
                    && w.scalar_type() == at::kFloat && x.dim() == 2 && w.dim() == 2 && x.is_contiguous()
                    && w.is_contiguous() && x.size(1) == 4096 && w.size(1) == 4096 && x.size(0) > 0 && x.size(0) <= 32
                    && (w.size(0) == 256 || w.size(0) == 512 || w.size(0) == 1024 || w.size(0) == 2048),
                "qualified FP32 GEMVT geometry");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
                    && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "aligned vectors");
    const c10::cuda::CUDAGuard guard(x.device());
    auto                       out    = at::empty({x.size(0), w.size(0)}, x.options());
    auto                       stream = c10::cuda::getCurrentCUDAStream();
    int                        m = x.size(0), n = w.size(0);
    dim3                       block(64, 2);
    if (rows_per_cta == 1)
        row_gemvt<1><<<dim3(n / 2, m), block, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), m, n);
    else if (rows_per_cta == 2)
        row_gemvt<2><<<dim3(n / 2, (m + 1) / 2), block, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), m, n);
    else if (rows_per_cta == 4)
        row_gemvt<4><<<dim3(n / 2, (m + 3) / 4), block, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), m, n);
    else
        TORCH_CHECK(false, "rows_per_cta must be 1,2,4");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}  // namespace
TORCH_LIBRARY(fp32_row_probe, m) {
    m.def("forward(Tensor x, Tensor w, int rows_per_cta) -> Tensor");
}
TORCH_LIBRARY_IMPL(fp32_row_probe, CUDA, m) {
    m.impl("forward", forward);
}
