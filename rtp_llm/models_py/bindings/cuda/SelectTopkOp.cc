#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

#include <cuda_bf16.h>

namespace rtp_llm {

SelectTopkOp::SelectTopkOp(const ModelConfig& model_config):
    expert_num_(model_config.expert_num), moe_k_(model_config.moe_k), has_moe_norm_(model_config.has_moe_norm) {}

namespace {

// TOPK_T: expert_ids dtype (int32/int64). IN_T: router_logits storage dtype
// (float, or __nv_bfloat16 to skip the per-layer router_logits.float() cast
// kernel; the kernel widens bf16 to fp32 at load and is bit-exact with the
// fp32 path on the .float() cast).
template<typename TOPK_T, typename IN_T>
void invokeSelectTopkRouting(IN_T const*                     router_logits,
                             float*                          expert_scales,
                             float*                          softmax_out,
                             TOPK_T*                         expert_ids,
                             int32_t*                        source_rows,
                             int64_t const                   token_num,
                             int const                       num_expert,
                             int const                       top_k,
                             tensorrt_llm::kernels::MOEExpertScaleNormalizationMode normalization_mode,
                             cudaStream_t                    stream) {
    tensorrt_llm::kernels::invokeSelectExpertsForTokens<TOPK_T, IN_T>(router_logits,
                                                                      router_logits,
                                                                      expert_scales,
                                                                      nullptr,  // sparse_mixer_out
                                                                      softmax_out,
                                                                      expert_ids,
                                                                      source_rows,
                                                                      token_num,
                                                                      num_expert,
                                                                      top_k,
                                                                      0,
                                                                      num_expert,
                                                                      0,
                                                                      normalization_mode,
                                                                      stream);
}

}  // namespace

void SelectTopkOp::forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales) {
    const auto token_num      = router_logits.sizes()[0];
    const auto num_expert     = expert_num_;
    const auto top_k          = moe_k_;
    auto normalization_mode   = has_moe_norm_ ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                                tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    const auto topk_t         = expert_ids.dtype();
    const auto router_t       = router_logits.dtype();
    const auto softmax_out    = torch::empty({token_num, num_expert}, router_logits.options().dtype(torch::kFloat32));
    const auto source_rows    = torch::empty({token_num, top_k}, router_logits.options().dtype(torch::kInt32));
    StreamType current_stream = GET_CURRENT_STREAM();
    router_logits             = router_logits.contiguous();

    if (topk_t != torch::kInt64 && topk_t != torch::kInt32) {
        throw std::runtime_error("Unimplemented dtype for SelectTopkOp topk ids: " + std::string(topk_t.name()));
    }

    if (router_t == torch::kFloat32) {
        if (topk_t == torch::kInt64) {
            invokeSelectTopkRouting<int64_t, float>(router_logits.data_ptr<float>(),
                                                    expert_scales.data_ptr<float>(),
                                                    softmax_out.data_ptr<float>(),
                                                    expert_ids.data_ptr<int64_t>(),
                                                    source_rows.data_ptr<int32_t>(),
                                                    token_num,
                                                    num_expert,
                                                    top_k,
                                                    normalization_mode,
                                                    current_stream);
        } else {
            invokeSelectTopkRouting<int32_t, float>(router_logits.data_ptr<float>(),
                                                    expert_scales.data_ptr<float>(),
                                                    softmax_out.data_ptr<float>(),
                                                    expert_ids.data_ptr<int32_t>(),
                                                    source_rows.data_ptr<int32_t>(),
                                                    token_num,
                                                    num_expert,
                                                    top_k,
                                                    normalization_mode,
                                                    current_stream);
        }
    } else if (router_t == torch::kBFloat16) {
        const auto* router_bf16 =
            reinterpret_cast<const __nv_bfloat16*>(router_logits.data_ptr<at::BFloat16>());
        if (topk_t == torch::kInt64) {
            invokeSelectTopkRouting<int64_t, __nv_bfloat16>(router_bf16,
                                                            expert_scales.data_ptr<float>(),
                                                            softmax_out.data_ptr<float>(),
                                                            expert_ids.data_ptr<int64_t>(),
                                                            source_rows.data_ptr<int32_t>(),
                                                            token_num,
                                                            num_expert,
                                                            top_k,
                                                            normalization_mode,
                                                            current_stream);
        } else {
            invokeSelectTopkRouting<int32_t, __nv_bfloat16>(router_bf16,
                                                            expert_scales.data_ptr<float>(),
                                                            softmax_out.data_ptr<float>(),
                                                            expert_ids.data_ptr<int32_t>(),
                                                            source_rows.data_ptr<int32_t>(),
                                                            token_num,
                                                            num_expert,
                                                            top_k,
                                                            normalization_mode,
                                                            current_stream);
        }
    } else {
        throw std::runtime_error("Unimplemented dtype for SelectTopkOp router logits: " + std::string(router_t.name()));
    }
}

void registerSelectTopkOp(const py::module& m) {
    pybind11::class_<SelectTopkOp>(m, "SelectTopkOp")
        .def(pybind11::init<const ModelConfig&>(), py::arg("model_config"))
        .def("forward",
             &SelectTopkOp::forward,
             py::arg("router_logits"),
             py::arg("expert_ids"),
             py::arg("expert_scales"));
}

}  // namespace rtp_llm