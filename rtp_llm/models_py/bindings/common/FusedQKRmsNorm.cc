#include <torch/extension.h>
#include <ATen/ATen.h>
#include "rtp_llm/models_py/bindings/common/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_qk_rmsnorm_v2.h"
#endif
#include <vector>
#include <cstdint>
#include <iostream>
#include <type_traits>
#if USING_CUDA && defined(ENABLE_BF16)
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#endif
using namespace std;
namespace th = torch;
using namespace rtp_llm;
namespace rtp_llm {

void FusedQKRMSNorm(at::Tensor&   input,
                    at::Tensor&   q_gamma,
                    at::Tensor&   k_gamma,
                    const double  layernorm_eps,
                    const int64_t q_group_num,
                    const int64_t k_group_num,
                    const int64_t m,
                    const int64_t n,
                    const int64_t norm_size) {
    CHECK_INPUT(input);
    CHECK_INPUT(q_gamma);
    CHECK_INPUT(k_gamma);
    auto device = input.device();
    CHECK_EQ(q_gamma.device(), device);
    CHECK_EQ(k_gamma.device(), device);
    CHECK_DIM(2, input);    // input: (batch_size, hidden_size)
    CHECK_DIM(1, q_gamma);  // weight: (hidden_size)
    CHECK_DIM(1, k_gamma);  // weight: (hidden_size)

    StreamType stream = GET_CURRENT_STREAM();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeFusedQkRmsNorm(static_cast<c_type*>(input.data_ptr()),
                             static_cast<c_type*>(q_gamma.data_ptr()),
                             static_cast<c_type*>(nullptr),
                             static_cast<c_type*>(k_gamma.data_ptr()),
                             static_cast<c_type*>(nullptr),
                             float(layernorm_eps),
                             q_group_num,
                             k_group_num,
                             m,
                             n,
                             norm_size,
                             stream);
        return true;
    });
}

#if USING_ROCM
void FusedQKRMSNormV2(at::Tensor&   input,
                      at::Tensor&   q_gamma,
                      at::Tensor&   k_gamma,
                      const double  layernorm_eps,
                      const int64_t q_group_num,
                      const int64_t k_group_num,
                      const int64_t m,
                      const int64_t n,
                      const int64_t norm_size) {
    CHECK_INPUT(input);
    CHECK_INPUT(q_gamma);
    CHECK_INPUT(k_gamma);
    auto device = input.device();
    CHECK_EQ(q_gamma.device(), device);
    CHECK_EQ(k_gamma.device(), device);
    CHECK_DIM(2, input);
    CHECK_DIM(1, q_gamma);
    CHECK_DIM(1, k_gamma);

    StreamType stream = GET_CURRENT_STREAM();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeFusedQkRmsNormV2(static_cast<c_type*>(input.data_ptr()),
                               static_cast<c_type*>(q_gamma.data_ptr()),
                               static_cast<c_type*>(nullptr),
                               static_cast<c_type*>(k_gamma.data_ptr()),
                               static_cast<c_type*>(nullptr),
                               float(layernorm_eps),
                               q_group_num,
                               k_group_num,
                               m,
                               n,
                               norm_size,
                               stream);
        return true;
    });
}
#endif

#if USING_CUDA && defined(ENABLE_BF16)
void FusedQKRMSNormCudaVec8(at::Tensor&       input,
                            const at::Tensor& q_gamma,
                            const at::Tensor& k_gamma,
                            double            layernorm_eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(q_gamma);
    CHECK_INPUT(k_gamma);
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && q_gamma.scalar_type() == at::kBFloat16
                    && k_gamma.scalar_type() == at::kBFloat16,
                "fused_qk_rmsnorm_cuda_vec8 requires BF16 input and weights");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096 && input.size(0) >= 1 && input.size(0) <= 2048
                    && input.stride(0) == 4096 && input.stride(1) == 1,
                "fused_qk_rmsnorm_cuda_vec8 requires packed [M,4096], 1 <= M <= 2048");
    TORCH_CHECK(q_gamma.dim() == 1 && k_gamma.dim() == 1 && q_gamma.numel() == 128 && k_gamma.numel() == 128
                    && q_gamma.stride(0) == 1 && k_gamma.stride(0) == 1,
                "fused_qk_rmsnorm_cuda_vec8 requires contiguous 128-element weights");
    TORCH_CHECK(q_gamma.device() == input.device() && k_gamma.device() == input.device(),
                "fused_qk_rmsnorm_cuda_vec8 requires a common CUDA device");
    TORCH_CHECK(std::isfinite(layernorm_eps) && layernorm_eps > 0.0,
                "fused_qk_rmsnorm_cuda_vec8 requires positive finite epsilon");
    const uintptr_t begin = reinterpret_cast<uintptr_t>(input.data_ptr());
    const uintptr_t end   = begin + input.numel() * sizeof(__nv_bfloat16);
    TORCH_CHECK(begin % 16 == 0, "fused_qk_rmsnorm_cuda_vec8 requires 16-byte input alignment");
    for (const at::Tensor* weight : {&q_gamma, &k_gamma}) {
        const uintptr_t wbegin = reinterpret_cast<uintptr_t>(weight->data_ptr());
        TORCH_CHECK(wbegin % 16 == 0, "fused_qk_rmsnorm_cuda_vec8 requires 16-byte weight alignment");
        TORCH_CHECK(!(wbegin < end && begin < wbegin + 128 * sizeof(__nv_bfloat16)),
                    "fused_qk_rmsnorm_cuda_vec8 weights must not overlap input");
    }
    c10::cuda::CUDAGuard guard(input.device());
    const auto*          properties = at::cuda::getDeviceProperties(input.get_device());
    TORCH_CHECK(properties->major == 12 && properties->minor == 0, "fused_qk_rmsnorm_cuda_vec8 supports SM120 only");
    // Match FI's double->float conversion, including positive finite doubles
    // which round to zero or infinity in FP32. No additional precision policy.
    const cudaError_t status =
        invokeFusedQkRmsNormCudaVec8(reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
                                     reinterpret_cast<const __nv_bfloat16*>(q_gamma.data_ptr()),
                                     reinterpret_cast<const __nv_bfloat16*>(k_gamma.data_ptr()),
                                     static_cast<uint32_t>(input.size(0)),
                                     static_cast<float>(layernorm_eps),
                                     input.get_device(),
                                     properties->multiProcessorCount,
                                     at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    TORCH_CHECK(status == cudaSuccess, "fused_qk_rmsnorm_cuda_vec8 launch failed: ", cudaGetErrorString(status));
    const cudaError_t launch_status = cudaPeekAtLastError();
    TORCH_CHECK(
        launch_status == cudaSuccess, "fused_qk_rmsnorm_cuda_vec8 launch failed: ", cudaGetErrorString(launch_status));
}
#endif  // USING_CUDA && ENABLE_BF16

}  // namespace rtp_llm
