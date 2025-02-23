#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput CudaDevice::moeFfnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden = params.input;
    const auto& weights = params.weights;
    const auto type = hidden.type();
    const auto weight_type = weights.moe_down_weight->kernel->type();
    const auto token_num = hidden.shape()[0];
    const auto hidden_dim = hidden.shape()[1];
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k = moe_conf.top_k;
    const auto moe_inter_size = moe_conf.moe_inter_padding_size;
    const auto normalize_expert_scale = moe_conf.normalize_expert_scale;
    // TODO group_size
    auto group_size = 0;
    if (params.weights.moe_gate_weight->kernel->isQBuffer()) {
        if ( dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zerosData() != nullptr) {
            group_size = params.weights.moe_gate_weight->kernel->shape()[1]
                         / dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zeros().shape()[1];
        }
    }

    BufferPtr output = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({type, {token_num, hidden_dim}});
    }

    const auto gate = gemm({hidden, *params.weights.moe_gating_weight->kernel,
                            nullopt, nullptr, DataType::TYPE_FP32});

    const auto fc2_result = allocateBuffer({type, {token_num, top_k, hidden_dim}}, {"moe_fc2_result"});
    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {pad_to_multiple_of_16(token_num * top_k)}}, {"moe_expert_scale"});
    const auto expanded_source_row_to_dest = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}}, {"moe_expand_src_to_dst"});
    const auto expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}}, {"moe_expert_for_src"});

    auto normalization_mode = moe_conf.has_moe_norm
                            ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE
                            : tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;

    auto gate_with_bias = gate;
    at::Tensor gate_with_bias_tensor; // hold the tensor to prevent it from being released
    prepareMoEGate(params, gate, gate_with_bias_tensor, gate_with_bias);

    printBufferData(*gate, "MOE gate");

    moe_plugin_->init(num_expert,
                      top_k,
                      normalize_expert_scale,
                      hidden_dim,
                      moe_inter_size,
                      params.configs.activation_type,
                      nvinfer1DtypeConvert(type),
                      nvinfer1DtypeConvert(weight_type),
                      group_size > 0,
                      group_size,
                      normalization_mode,
                      moe_conf.ep_size,
                      moe_conf.ep_rank);
    const auto ws_size   = moe_plugin_->getWorkspaceSize(token_num);
    const auto worksapce = allocateBuffer({DataType::TYPE_BYTES, {ws_size}}, {"moe_workspace"});

    moe_plugin_->enqueue(
        hidden.data(),
        gate->data<float>(),
        gate_with_bias->data<float>(),
        weights.moe_gate_weight->kernel->data(),
        BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
        BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
        OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        weights.moe_down_weight->kernel->data(),
        BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_down_weight->kernel),
        BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_down_weight->kernel),
        OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_down_weight->bias),
        token_num,
        worksapce->data<char>(),
        // output
        output->data(),
        fc2_result->data(),
        nullptr,   // finished
        expert_scales->data(),
        expanded_source_row_to_dest->data<int>(),
        expert_for_source_row->data<int>(),
        stream_
    );
    printBufferData(*expanded_source_row_to_dest, "expanded_source_row_to_dest");
    printBufferData(*expert_for_source_row, "expert_for_source_row");
    printBufferData(*output, "moe_ffn_out");

    return FfnLayerOutput({move(output)});
}

void CudaDevice::prepareMoEGate(const FfnLayerParams& params,
                                BufferPtr             gate,
                                torch::Tensor&        gate_with_bias_tensor,
                                BufferPtr&            gate_with_bias) {
    auto const& moe_conf   = params.configs.moe_configs.value();
    const auto& hidden     = params.input;
    const auto  token_num  = hidden.shape()[0];
    const auto  num_expert = params.weights.moe_gating_weight->kernel->shape()[1];

    if (moe_conf.scoring_func == 1) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(DataType::TYPE_FP32, invokeSigmoid, gate->data(), gate->size(), 1.0f, stream_);
    }
    if (params.weights.e_score_correction_bias) {
        const int n_routed_experts = num_expert;
        const int n_group          = moe_conf.n_group;
        const int topk_group       = moe_conf.topk_group;

        torch::Tensor gate_tensor = Buffer2torchTensor(gate, false);
        torch::Tensor e_score_correction_bias_tensor =
            Buffer2torchTensor(params.weights.e_score_correction_bias, false).to(torch::kFloat32);
        auto scores_for_choice = gate_tensor.add(e_score_correction_bias_tensor);
        auto reshaped_scores   = scores_for_choice.view({(int)token_num, n_group, -1});
        auto topk_result       = reshaped_scores.topk(2, /*dim=*/-1);
        auto group_scores      = std::get<0>(topk_result).sum(-1);
        auto group_topk_result = group_scores.topk(
            /*k=*/topk_group,
            /*dim=*/-1,
            /*largest=*/true,
            /*sorted=*/false);
        auto group_idx  = std::get<1>(group_topk_result);
        auto group_mask = torch::zeros_like(group_scores);
        group_mask.scatter_(
            /*dim=*/1,
            /*index=*/group_idx,
            /*src=*/1.0f);
        int64_t experts_per_group = n_routed_experts / n_group;
        auto    score_mask =
            group_mask.unsqueeze(-1).expand({(int)token_num, n_group, experts_per_group}).reshape({(int)token_num, -1});
        gate_with_bias_tensor = scores_for_choice.masked_fill(torch::logical_not(score_mask.to(torch::kBool)), 0.0);
        gate_with_bias        = torchTensor2Buffer(gate_with_bias_tensor);
        printBufferData(*gate_with_bias, "gate_with_bias");
    }
}

} // namespace fastertransformer
