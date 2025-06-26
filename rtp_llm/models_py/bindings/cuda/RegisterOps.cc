#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/cuda/RtpNorm.h"
#include "rtp_llm/models_py/bindings/cuda/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/cuda/FusedQKRmsNorm.h"

#include "3rdparty/flashinfer/flashinfer.h"

#include <torch/library.h>

using namespace rtp_llm;

namespace torch_ext {

// TODO(wangyin.yx): does this torch library register has a better way to organize?
TORCH_LIBRARY_FRAGMENT(rtp_llm_ops, m) {
    m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
    m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

    m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, int cuda_stream) -> ()");
    m.impl("fused_add_rmsnorm", torch::kCUDA, &fused_add_rmsnorm);

    m.def("silu_and_mul(Tensor! output, Tensor input, int cuda_stream) -> ()");
    m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

    m.def("fused_qk_rmsnorm(Tensor! IO, Tensor q_gamma, Tensor k_gamma, float layernorm_eps, int q_group_num, int k_group_num, int m, int n, int norm_size, int cuda_stream) -> ()");
    m.impl("fused_qk_rmsnorm", torch::kCUDA, &FusedQKRMSNorm);

    m.def("layernorm(Tensor! output, Tensor input, Tensor weight, Tensor beta, float eps, int cuda_stream) -> ()");
    m.impl("layernorm", torch::kCUDA, &layernorm);

    m.def("fused_add_layernorm(Tensor! input, Tensor! residual, Tensor bias, Tensor weight, Tensor beta, float eps, int cuda_stream) -> ()");
    m.impl("fused_add_layernorm", torch::kCUDA, &fused_add_layernorm);

    m.def("embedding(Tensor! output, Tensor input, Tensor weight, int cuda_stream) -> ()");
    m.impl("embedding", torch::kCUDA, &embedding);
}

void registerPyModuleOps(py::module &m) {
    registerFlashInferOp(m);

    // TODO(wangyin.yx): organize this attn param def.
    register_attn_params(m);
}

}

