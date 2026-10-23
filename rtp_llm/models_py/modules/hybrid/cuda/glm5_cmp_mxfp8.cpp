#include <torch/extension.h>

namespace rtp_llm {

void glm5CmpMxfp8IndexerQ(const torch::Tensor& projected,
                          const torch::Tensor& cos_sin,
                          const torch::Tensor& positions,
                          const torch::Tensor& gate_raw,
                          const torch::Tensor& q_fp8,
                          const torch::Tensor& q_sf,
                          const torch::Tensor& head_weights,
                          std::uintptr_t       notify_event,
                          bool                 pdl);

void glm5CmpMxfp8IndexerK(const torch::Tensor& activation,
                          const torch::Tensor& activation_scale,
                          const torch::Tensor& weight,
                          const torch::Tensor& weight_scale,
                          const torch::Tensor& norm_weight,
                          const torch::Tensor& norm_bias,
                          const torch::Tensor& cos_sin,
                          const torch::Tensor& positions,
                          const torch::Tensor& slot_mapping,
                          const torch::Tensor& cache,
                          double               epsilon,
                          bool                 pdl);

}  // namespace rtp_llm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("indexer_q_rope_quant", &rtp_llm::glm5CmpMxfp8IndexerQ);
    module.def("indexer_k_cache", &rtp_llm::glm5CmpMxfp8IndexerK);
}
