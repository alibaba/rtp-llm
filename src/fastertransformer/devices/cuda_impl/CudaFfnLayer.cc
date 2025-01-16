#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

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



    const auto fc2_result = allocateBuffer({type, {token_num, top_k, hidden_dim}});
    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expanded_source_row_to_dest = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});

    auto normalization_mode = moe_conf.has_moe_norm
                            ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE
                            : tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;

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
    const auto worksapce = allocateBuffer({DataType::TYPE_BYTES, {ws_size}});

    moe_plugin_->enqueue(
        hidden.data(),
        gate->data<float>(),
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

} // namespace fastertransformer
