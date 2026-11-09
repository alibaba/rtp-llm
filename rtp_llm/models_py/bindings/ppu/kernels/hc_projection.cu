#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
namespace {
template<int Rows>
__global__ void hc_partial(const float* x, const float* w, float* parts, int m) {
    const int lane = threadIdx.x, col = blockIdx.x, slice = blockIdx.y, first = blockIdx.z * Rows;
    float     acc[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r)
        acc[r] = 0.f;
    for (int k = slice * 256 + lane * 2; k < 16384; k += 4096) {
        const float2 weight = *reinterpret_cast<const float2*>(w + col * 16384 + k);
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (first + r < m) {
                const float2 a = *reinterpret_cast<const float2*>(x + (first + r) * 16384 + k);
                acc[r]         = __fmaf_rn(weight.x, a.x, acc[r]);
                acc[r]         = __fmaf_rn(weight.y, a.y, acc[r]);
            }
    }
    __shared__ float partial[Rows][4];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int d = 16; d; d /= 2)
            acc[r] = __fadd_rn(acc[r], __shfl_down_sync(0xffffffff, acc[r], d));
        if ((lane & 31) == 0)
            partial[r][lane / 32] = acc[r];
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (first + r < m) {
                float a                                      = __fadd_rn(acc[r], partial[r][1]);
                a                                            = __fadd_rn(a, partial[r][2]);
                a                                            = __fadd_rn(a, partial[r][3]);
                parts[((first + r) * 24 + col) * 16 + slice] = a;
            }
    }
}
__global__ void hc_reduce(const float* p, float* y, int m) {
    int   item = blockIdx.x * 8 + threadIdx.x / 16, slice = threadIdx.x % 16;
    float a = __fadd_rn(p[item * 16 + slice], 0.f);
#pragma unroll
    for (int d = 8; d; d /= 2)
        a = __fadd_rn(a, __shfl_xor_sync(0xffffffff, a, d));
    if (slice == 0)
        y[item] = __fmul_rn(1.f, a);
}
at::Tensor forward(const at::Tensor& x, const at::Tensor& w, int64_t r) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && x.scalar_type() == at::kFloat
                    && w.scalar_type() == at::kFloat && x.dim() == 2 && w.dim() == 2 && x.is_contiguous()
                    && w.is_contiguous() && x.size(1) == 16384 && w.size(0) == 24 && w.size(1) == 16384 && x.size(0) > 0
                    && x.size(0) <= 32,
                "HC FP32 geometry");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0
                    && reinterpret_cast<uintptr_t>(w.data_ptr()) % 8 == 0,
                "aligned vectors");
    c10::cuda::CUDAGuard guard(x.device());
    int                  m      = x.size(0);
    auto                 tmp    = at::empty({m, 24, 16}, x.options());
    auto                 out    = at::empty({m, 24}, x.options());
    auto                 stream = c10::cuda::getCurrentCUDAStream();
    if (r == 1)
        hc_partial<1>
            <<<dim3(24, 16, m), 128, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), tmp.data_ptr<float>(), m);
    else if (r == 2)
        hc_partial<2><<<dim3(24, 16, (m + 1) / 2), 128, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), tmp.data_ptr<float>(), m);
    else if (r == 4)
        hc_partial<4><<<dim3(24, 16, (m + 3) / 4), 128, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), tmp.data_ptr<float>(), m);
    else
        TORCH_CHECK(false, "rows 1/2/4");
    hc_reduce<<<m * 3, 128, 0, stream>>>(tmp.data_ptr<float>(), out.data_ptr<float>(), m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}  // namespace
TORCH_LIBRARY(hc_projection_probe, m) {
    m.def("forward(Tensor x, Tensor w, int rows_per_cta) -> Tensor");
}
TORCH_LIBRARY_IMPL(hc_projection_probe, CUDA, m) {
    m.impl("forward", forward);
}
