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
    const auto token_num = hidden.shape()[0];
    const auto hidden_dim = hidden.shape()[1];
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k = moe_conf.top_k;
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

    initMoeRunner(type, weights.moe_up_weight->kernel->type());
    const auto ws_size = moe_runner_->getWorkspaceSize(
        token_num, hidden_dim, moe_conf.moe_inter_padding_size, num_expert,
        top_k, params.configs.activation_type, {});

    const auto worksapce = allocateBuffer({DataType::TYPE_BYTES, {ws_size}});

    const auto fc2_result = allocateBuffer({type, {token_num, top_k, hidden_dim}});
    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expanded_source_row_to_dest = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});

    tensorrt_llm::kernels::MOEParallelismConfig parallelism_config;
    auto normalization_mode = moe_conf.has_moe_norm
                            ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE
                            : tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    moe_runner_->runMoe(
        hidden.data(),
        gate->data<float>(),
        weights.moe_gate_weight->kernel->data(),
        BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
        BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
        OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        params.configs.activation_type,
        weights.moe_down_weight->kernel->data(),
        BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_down_weight->kernel),
        BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_down_weight->kernel),
        OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_down_weight->bias),
        weights.moe_up_weight->kernel->data(),
        BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_up_weight->kernel),
        BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_up_weight->kernel),
        OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_up_weight->bias),
        token_num,
        hidden_dim,
        moe_conf.moe_inter_padding_size,
        num_expert,
        top_k,
        group_size,
        moe_conf.normalize_expert_scale,
        worksapce->data<char>(),
        // output
        output->data(),
        fc2_result->data(),
        nullptr,   // finished
        token_num, // num_not_finished
        expert_scales->data(),
        expanded_source_row_to_dest->data<int>(),
        expert_for_source_row->data<int>(),
        parallelism_config,
        normalization_mode,
        stream_
    );
    printBufferData(*output, "moe_ffn_out");

    return FfnLayerOutput({move(output)});
}

} // namespace fastertransformer
