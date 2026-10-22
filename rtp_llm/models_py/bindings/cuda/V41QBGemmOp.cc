#include "rtp_llm/models_py/bindings/cuda/V41QBGemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/v41_qb_gemm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace torch_ext {
std::vector<at::Tensor> v41_qb_gemm(const at::Tensor& a, const at::Tensor& sa,
    const at::Tensor& b, const at::Tensor& sb, const at::Tensor& weights,
    const at::Tensor& frequencies, const at::Tensor& positions) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.size(1) == 1280, "expected CUDA A[M,1280]");
    const c10::cuda::CUDAGuard guard(a.device());
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 10 && props->minor == 3 && props->multiProcessorCount == 148,
                "V41 QB GEMM requires SM103 with 148 SMs");
    const auto m = a.size(0);
    TORCH_CHECK(m == 256 || m == 384 || m == 512, "unsupported M");
    for (const auto* t : {&a, &sa, &b, &sb, &weights, &frequencies, &positions}) {
        TORCH_CHECK(t->device() == a.device(), "operands must share a CUDA device");
    }
    for (const auto* t : {&a, &sa, &b, &sb}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "unaligned TMA operand");
    }
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b.scalar_type() == at::kFloat8_e4m3fn
                && b.sizes() == at::IntArrayRef({36864,1280}) && a.is_contiguous() && b.is_contiguous(),
                "expected contiguous E4M3 QB operands");
    TORCH_CHECK(sa.scalar_type() == at::kInt && sb.scalar_type() == at::kInt
                && sa.sizes() == at::IntArrayRef({m,10}) && sb.sizes() == at::IntArrayRef({36864,10})
                && sa.strides() == at::IntArrayRef({1,m}) && sb.strides() == at::IntArrayRef({1,36864}),
                "expected packed MN-major UE8M0 scales");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && weights.is_contiguous()
                && weights.sizes() == at::IntArrayRef({m,32}), "expected raw BF16 head weights");
    TORCH_CHECK(frequencies.scalar_type() == at::kComplexFloat && frequencies.is_contiguous()
                && frequencies.dim() == 2 && frequencies.size(0) > 0 && frequencies.size(1) == 32,
                "expected full complex64 frequency table");
    TORCH_CHECK((positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong)
                && positions.is_contiguous() && positions.dim() == 1 && positions.numel() == m,
                "expected int32/int64 token positions");
    // Only Main Q is materialized. The Indexer slice is deliberately not exposed.
    auto output = at::empty({m,36864}, a.options().dtype(at::kBFloat16));
    auto payload = at::empty({m,32,64}, a.options().dtype(at::kChar));
    auto sf = at::empty({m,32}, a.options().dtype(at::kInt));
    auto scaled_weights = at::empty_like(weights);
    rtp_llm::invokeV41QBGemm(a.data_ptr(),sa.data_ptr(),b.data_ptr(),sb.data_ptr(),output.data_ptr(),
        payload.data_ptr(),sf.data_ptr(),scaled_weights.data_ptr(),weights.data_ptr(),
        frequencies.data_ptr(),positions.data_ptr(),positions.scalar_type() == at::kLong,
        m,frequencies.size(0),c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
    return {output.narrow(1,0,32768),payload,sf,scaled_weights};
}
}  // namespace torch_ext
