#include "rtp_llm/models_py/bindings/cuda/MlaQMergeOp.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

void MlaQMerge(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.is_cuda(), "a must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && out.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(a.size(2) == 512 && b.size(2) == 64 && out.size(2) == 576,
                "concat_mla_absorb_q: last dim must be 512, 64, 576");
    TORCH_CHECK(a.stride(2) == 1 && b.stride(2) == 1 && out.stride(2) == 1, "last dim must be contiguous");
    TORCH_CHECK(a.dtype() == at::kBFloat16 && b.dtype() == at::kBFloat16 && out.dtype() == at::kBFloat16,
                "concat_mla_absorb_q: dtype must be bfloat16");
    TORCH_CHECK(a.size(0) * a.size(1) == b.size(0) * b.size(1) && a.size(1) == b.size(1), "batch/head mismatch");

    StreamType stream    = GET_CURRENT_STREAM();
    const int  num_items = a.size(0) * a.size(1);
    const int  dim_1     = a.size(1);

    // Same arg order as sglang concat_mla_absorb_q: a, b, out, num_items, dim_1, strides..., stream
    rtp_llm::invokeMlaQMerge<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
                                            reinterpret_cast<__nv_bfloat16*>(b.data_ptr()),
                                            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                            num_items,
                                            dim_1,
                                            a.stride(0),
                                            static_cast<int>(a.stride(1)),
                                            b.stride(0),
                                            static_cast<int>(b.stride(1)),
                                            out.stride(0),
                                            static_cast<int>(out.stride(1)),
                                            stream);
}

}  // namespace rtp_llm
