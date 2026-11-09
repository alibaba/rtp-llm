#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace {
template<class Weight>
__global__ void rms_tail(const float* x, const float* inv, const Weight* w, c10::BFloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Match two separate FP32 Torch MulFunctor results, then its BF16 cast.
        float normalized = __fmul_rn(x[i], inv[i / 4096]);
        float weighted   = __fmul_rn(static_cast<float>(w[i % 4096]), normalized);
        out[i]           = c10::BFloat16(weighted);
    }
}
at::Tensor tail(const at::Tensor& x, const at::Tensor& inv, const at::Tensor& w) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 2 && x.size(1) == 4096 && x.size(0) > 0
                    && (x.size(0) <= 48 || x.size(0) == 16384 || x.size(0) == 32768) && x.is_contiguous(),
                "HC RMS x layout");
    TORCH_CHECK(inv.device() == x.device() && inv.scalar_type() == at::kFloat && inv.numel() == x.size(0)
                    && inv.is_contiguous(),
                "HC RMS inv layout");
    TORCH_CHECK(w.device() == x.device() && w.dim() == 1 && w.numel() == 4096 && w.is_contiguous()
                    && (w.scalar_type() == at::kFloat || w.scalar_type() == at::kBFloat16),
                "HC RMS weight layout");
    const c10::cuda::CUDAGuard guard(x.device());
    auto                       out    = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
    auto                       stream = c10::cuda::getCurrentCUDAStream();
    const int                  n      = x.numel();
    if (w.scalar_type() == at::kFloat)
        rms_tail<float><<<(n + 255) / 256, 256, 0, stream>>>(
            x.data_ptr<float>(), inv.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<c10::BFloat16>(), n);
    else
        rms_tail<c10::BFloat16><<<(n + 255) / 256, 256, 0, stream>>>(
            x.data_ptr<float>(), inv.data_ptr<float>(), w.data_ptr<c10::BFloat16>(), out.data_ptr<c10::BFloat16>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(rtp_llm_ppu, m) {
    m.def("hc_rms_tail(Tensor x, Tensor inv, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(rtp_llm_ppu, CUDA, m) {
    m.impl("hc_rms_tail", tail);
}
