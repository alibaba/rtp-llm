#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

// Private Prefill-only entry points. Keep hc_rms_tail and its Decode ABI intact.
namespace {
void check_rms_bf16_prefill(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 4096
                    && (x.size(0) == 16384 || x.size(0) == 32768) && x.is_contiguous(),
                "HC RMS BF16 Prefill x layout");
}
__global__ void rms_square_bf16_prefill_kernel(const c10::BFloat16* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>(x[i]);
        out[i]  = __fmul_rn(v, v);
    }
}
template<class Weight>
__global__ void
rms_tail_bf16_prefill_kernel(const c10::BFloat16* x, const float* inv, const Weight* w, c10::BFloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v          = static_cast<float>(x[i]);
        float normalized = __fmul_rn(v, inv[i / 4096]);
        float weighted   = __fmul_rn(static_cast<float>(w[i % 4096]), normalized);
        out[i]           = c10::BFloat16(weighted);
    }
}
at::Tensor rms_square_bf16_prefill(const at::Tensor& x) {
    check_rms_bf16_prefill(x);
    const c10::cuda::CUDAGuard guard(x.device());
    auto                       out    = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    auto                       stream = c10::cuda::getCurrentCUDAStream();
    const int                  n      = x.numel();
    rms_square_bf16_prefill_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        x.data_ptr<c10::BFloat16>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
at::Tensor rms_tail_bf16_prefill(const at::Tensor& x, const at::Tensor& inv, const at::Tensor& w) {
    check_rms_bf16_prefill(x);
    TORCH_CHECK(inv.device() == x.device() && inv.scalar_type() == at::kFloat && inv.dim() == 2
                    && inv.size(0) == x.size(0) && inv.size(1) == 1 && inv.is_contiguous(),
                "HC RMS BF16 Prefill inv layout");
    TORCH_CHECK(w.device() == x.device() && w.dim() == 1 && w.numel() == 4096 && w.is_contiguous()
                    && (w.scalar_type() == at::kFloat || w.scalar_type() == at::kBFloat16),
                "HC RMS BF16 Prefill weight layout");
    const c10::cuda::CUDAGuard guard(x.device());
    auto                       out    = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
    auto                       stream = c10::cuda::getCurrentCUDAStream();
    const int                  n      = x.numel();
    if (w.scalar_type() == at::kFloat)
        rms_tail_bf16_prefill_kernel<float><<<(n + 255) / 256, 256, 0, stream>>>(
            x.data_ptr<c10::BFloat16>(), inv.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<c10::BFloat16>(), n);
    else
        rms_tail_bf16_prefill_kernel<c10::BFloat16><<<(n + 255) / 256, 256, 0, stream>>>(x.data_ptr<c10::BFloat16>(),
                                                                                         inv.data_ptr<float>(),
                                                                                         w.data_ptr<c10::BFloat16>(),
                                                                                         out.data_ptr<c10::BFloat16>(),
                                                                                         n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(rtp_llm_ppu, m) {
    m.def("hc_rms_square_bf16_prefill(Tensor x) -> Tensor");
    m.def("hc_rms_tail_bf16_prefill(Tensor x, Tensor inv, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(rtp_llm_ppu, CUDA, m) {
    m.impl("hc_rms_square_bf16_prefill", rms_square_bf16_prefill);
    m.impl("hc_rms_tail_bf16_prefill", rms_tail_bf16_prefill);
}
