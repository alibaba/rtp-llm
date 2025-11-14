#include "rtp_llm/models_py/bindings/cuda/FusedMoEOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include <cstdint>

namespace rtp_llm {

nvinfer1::DataType nvinfer1DtypeConvert(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf:
            return nvinfer1::DataType::kHALF;
        case at::kBFloat16:
            return nvinfer1::DataType::kBF16;
        case at::kFloat:
            return nvinfer1::DataType::kFLOAT;
        case at::kChar:
            return nvinfer1::DataType::kINT8;
        default:
            throw std::runtime_error("Unimplemented dtype for nvinfer1DtypeConvert");
    }
    return nvinfer1::DataType::kFLOAT;
}

FusedMoEOp::FusedMoEOp(const ModelConfig& model_config, const ParallelismConfig& parallelism_config):
    expert_num_(model_config.expert_num),
    moe_k_(model_config.moe_k),
    moe_normalize_expert_scale_(model_config.moe_normalize_expert_scale),
    has_moe_norm_(model_config.has_moe_norm),
    activation_type_(model_config.activation_type),
    ep_size_(parallelism_config.ep_size),
    ep_rank_(parallelism_config.ep_rank),
    moe_plugin_(std::make_unique<trt_plugins::MixtureOfExpertsPlugin>()) {}
void FusedMoEOp::forward(torch::Tensor hidden_states,
                         torch::Tensor up_proj,
                         torch::Tensor down_proj,
                         torch::Tensor expert_scales,
                         torch::Tensor expert_ids,
                         torch::Tensor outputs) {
    const auto type                   = hidden_states.scalar_type();
    const auto weight_type            = down_proj.scalar_type();
    const auto token_num              = hidden_states.sizes()[0];
    const auto hidden_dim             = hidden_states.sizes()[1];
    const auto num_expert             = expert_num_;
    const auto top_k                  = moe_k_;
    bool       is_gated_activation    = isGatedActivation(activation_type_);
    const auto moe_inter_size         = is_gated_activation ? up_proj.sizes()[1] / 2 : up_proj.sizes()[1];
    const auto normalize_expert_scale = moe_normalize_expert_scale_;
    auto normalization_mode = has_moe_norm_ ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                              tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    auto group_size         = 0;
    // TODO group_size
    if (token_num == 0) {
        return;
    }
    moe_plugin_->init(num_expert,
                      top_k,
                      normalize_expert_scale,
                      hidden_dim,
                      moe_inter_size,
                      activation_type_,
                      nvinfer1DtypeConvert(type),
                      nvinfer1DtypeConvert(weight_type),
                      group_size > 0,
                      group_size,
                      normalization_mode,
                      ep_size_,
                      ep_rank_);
    const auto new_ws_size = moe_plugin_->getWorkspaceSize(token_num);
    const auto new_worksapce =
        torch::zeros({static_cast<int64_t>(new_ws_size)}, hidden_states.options().dtype(torch::kUInt8));
    auto fc2_result =
        torch::zeros({token_num, top_k, hidden_dim}, hidden_states.options().dtype(hidden_states.dtype()));
    const auto new_expanded_source_row_to_dest =
        torch::zeros({top_k, token_num}, hidden_states.options().dtype(torch::kInt32));
    cudaStream_t stream = 0;
    if (hidden_states.scalar_type() == at::kBFloat16) {
        moe_plugin_->enqueue(hidden_states.data_ptr<at::BFloat16>(),
                             nullptr,  // gate->data<float>(),
                             nullptr,  // gate_with_bias->data<float>(),
                             up_proj.data_ptr<at::BFloat16>(),
                             nullptr,
                             nullptr,
                             nullptr,
                             down_proj.data_ptr<at::BFloat16>(),
                             nullptr,
                             nullptr,
                             nullptr,
                             token_num,
                             new_worksapce.data_ptr<uint8_t>(),
                             // output
                             outputs.data_ptr<at::BFloat16>(),
                             fc2_result.data_ptr<at::BFloat16>(),
                             nullptr,  // finished
                             expert_scales.data_ptr<float>(),
                             new_expanded_source_row_to_dest.data_ptr<int32_t>(),
                             expert_ids.data_ptr<int32_t>(),
                             stream);
    } else if (hidden_states.scalar_type() == at::kHalf) {
        moe_plugin_->enqueue(hidden_states.data_ptr<at::Half>(),
                             nullptr,  // gate->data<float>(),
                             nullptr,  // gate_with_bias->data<float>(),
                             up_proj.data_ptr<at::Half>(),
                             nullptr,
                             nullptr,
                             nullptr,
                             down_proj.data_ptr<at::Half>(),
                             nullptr,
                             nullptr,
                             nullptr,
                             token_num,
                             new_worksapce.data_ptr<uint8_t>(),
                             // output
                             outputs.data_ptr<at::Half>(),
                             fc2_result.data_ptr<at::Half>(),
                             nullptr,  // finished
                             expert_scales.data_ptr<float>(),
                             new_expanded_source_row_to_dest.data_ptr<int32_t>(),
                             expert_ids.data_ptr<int32_t>(),
                             stream);
    }
}

void registerFusedMoEOp(const py::module& m) {
    pybind11::class_<FusedMoEOp>(m, "FusedMoEOp")
        .def(pybind11::init<const ModelConfig&, const ParallelismConfig&>(),
             pybind11::arg("model_config"),
             pybind11::arg("parallelism_config"))
        .def("forward",
             &FusedMoEOp::forward,
             pybind11::arg("hidden_states"),
             pybind11::arg("up_proj"),
             pybind11::arg("down_proj"),
             pybind11::arg("expert_scales"),
             pybind11::arg("expert_ids"),
             pybind11::arg("outputs"));
}

}  // namespace rtp_llm
