#include "rtp_llm/models_py/bindings/cuda/kernels/engram/launch.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/engram/impls/gate_update.cuh"
namespace rtp_llm {
void engramGateUpdate(const at::Tensor& hp,
                      const at::Tensor& kv,
                      const at::Tensor& qw,
                      const at::Tensor& kw,
                      const at::Tensor& mask,
                      const at::Tensor& updated,
                      double            eps) {
    const c10::cuda::CUDAGuard guard(hp.device());
    const int                  m = engram::rows(hp);
    using namespace engram;
    check(hp, hp, at::kBFloat16, {m, 4, 5120});
    check(kv, hp, at::kBFloat16, {m, 25600});
    check(qw, hp, at::kBFloat16, {4, 5120});
    check(kw, hp, at::kBFloat16, {4, 5120});
    TORCH_CHECK(mask.is_cuda() && mask.device() == hp.device() && mask.scalar_type() == at::kBool && mask.numel() == m
                    && mask.is_contiguous(),
                "Engram mask must be contiguous bool with one value per row");
    check(updated, hp, at::kBFloat16, {m, 4, 5120});
    TORCH_CHECK(updated.data_ptr() != hp.data_ptr(), "Gate+Update requires independently owned output");
    launch(engram_gate_update_kernel,
           128,
           0,
           1,
           dim3(m, 4, 1),
           reinterpret_cast<const nv_bfloat16*>(hp.data_ptr()),
           reinterpret_cast<const nv_bfloat16*>(kv.data_ptr()),
           reinterpret_cast<const nv_bfloat16*>(qw.data_ptr()),
           reinterpret_cast<const nv_bfloat16*>(kw.data_ptr()),
           mask.data_ptr<bool>(),
           reinterpret_cast<nv_bfloat16*>(updated.data_ptr()),
           float(eps));
}
}  // namespace rtp_llm
