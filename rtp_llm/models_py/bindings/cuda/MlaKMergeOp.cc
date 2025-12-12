#include "rtp_llm/models_py/bindings/cuda/MlaKMergeOp.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <cuda_runtime.h>

namespace rtp_llm {

void MlaKMerge(torch::Tensor k_out, torch::Tensor k_nope, torch::Tensor k_pe) {
    TORCH_CHECK(k_out.is_cuda(), "k_out must be on CUDA");
    TORCH_CHECK(k_nope.is_cuda(), "k_nope must be on CUDA");
    TORCH_CHECK(k_pe.is_cuda(), "k_pe must be on CUDA");

    TORCH_CHECK(k_out.dim() == 3, "k_out must be 3D: [token_num, head_num, nope_head_dim + rope_head_dim]");
    TORCH_CHECK(k_nope.dim() == 3, "k_nope must be 3D: [token_num, head_num, nope_head_dim]");
    TORCH_CHECK(k_pe.dim() == 3, "k_pe must be 3D: [token_num, 1, rope_head_dim]");

    StreamType stream = GET_CURRENT_STREAM();

    const int     num_tokens      = k_out.size(0);
    const int64_t k_stride_0      = k_out.stride(0);
    const int     k_stride_1      = k_out.stride(1);
    const int64_t k_nope_stride_0 = k_nope.stride(0);
    const int     k_nope_stride_1 = k_nope.stride(1);
    const int64_t k_rope_stride_0 = k_pe.stride(0);

    // Dispatch based on dtype
    if (k_out.dtype() == torch::kBFloat16) {
        invokeMlaKMerge<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
                                       reinterpret_cast<__nv_bfloat16*>(k_nope.data_ptr()),
                                       reinterpret_cast<__nv_bfloat16*>(k_pe.data_ptr()),
                                       num_tokens,
                                       k_stride_0,
                                       k_stride_1,
                                       k_nope_stride_0,
                                       k_nope_stride_1,
                                       k_rope_stride_0,
                                       stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", k_out.dtype());
    }
}

}  // namespace rtp_llm
