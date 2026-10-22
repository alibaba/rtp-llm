#include "rtp_llm/models_py/bindings/cuda/V41FrontGemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/v41_front_gemm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace torch_ext {
std::vector<at::Tensor> v41_front_gemm(const at::Tensor& x, const at::Tensor& a,
    const at::Tensor& sa, const at::Tensor& b, const at::Tensor& sb, const at::Tensor& head) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 5120, "expected CUDA X[M,5120]");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 10 && props->minor == 3 && props->multiProcessorCount == 148,
                "V41 front GEMM requires SM103 with 148 SMs");
    const auto m = x.size(0);
    TORCH_CHECK(m == 128 || m == 256 || m == 384 || m == 512, "unsupported M");
    for (const auto* t : {&x, &a, &sa, &b, &sb, &head}) {
        TORCH_CHECK(t->device() == x.device() && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                    "operands must be on the same CUDA device and 16-byte aligned");
    }
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && head.scalar_type() == at::kBFloat16,
                "X and head weight must remain BF16");
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b.scalar_type() == at::kFloat8_e4m3fn,
                "Q-A/KV operands must be E4M3");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({m,5120}) && b.sizes() == at::IntArrayRef({1792,5120})
                && head.sizes() == at::IntArrayRef({32,5120}), "invalid projection dimensions");
    TORCH_CHECK(x.is_contiguous() && a.is_contiguous() && b.is_contiguous() && head.is_contiguous(),
                "matrix operands must be contiguous");
    TORCH_CHECK(sa.scalar_type() == at::kInt && sb.scalar_type() == at::kInt
                && sa.sizes() == at::IntArrayRef({m,40}) && sb.sizes() == at::IntArrayRef({1792,40})
                && sa.strides() == at::IntArrayRef({1,m}) && sb.strides() == at::IntArrayRef({1,1792}),
                "expected packed MN-major UE8M0 scales");
    auto out = at::empty({m,1792}, x.options());
    auto head_out = at::empty({m,32}, x.options());
    rtp_llm::invokeV41FrontGemm(a.data_ptr(), sa.data_ptr(), b.data_ptr(), sb.data_ptr(),
        x.data_ptr(), head.data_ptr(), out.data_ptr(), head_out.data_ptr(), m,
        c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return {out, head_out};
}
}  // namespace torch_ext
