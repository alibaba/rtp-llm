#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput CudaDevice::moeFfnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(moe_runner_, "moe runner not initialized");
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden = params.input;
    const auto type = hidden.type();
    const auto token_num = hidden.shape()[0];
    const auto hidden_dim = hidden.shape()[1];
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k = moe_conf.top_k;
    // For fp16 / bf16 weights:
    // moe up, gate weights expected shape: [num_expert, expert_hidden_size, hidden_dim]
    // moe down weights expected shape: [num_expert, hidden_dim, expert_hidden_size]
    // For int8 quantized weights:
    // moe up, gate weights expected shape: [num_expert, hidden_dim, expert_hidden_size]
    // moe down weights expected shape: [num_expert, expert_hidden_size, hidden_dim]
    // This difference is caused by kernel implementation
    // TODO(wangyin.yx): add device level weights preprocessor to handle this
    const auto expert_hidden_size = params.weights.up_weight->kernel->isQBuffer()
                                  ? params.weights.up_weight->kernel->shape()[2]
                                  : params.weights.up_weight->kernel->shape()[1];

    printf("hidden: %s\n", hidden.debugString().c_str());
    printf("moe gate weight: %s\n", params.weights.moe_gating_weight->kernel->debugString().c_str());

    const auto output = allocateBuffer({type, {token_num, hidden_dim}});

    const auto gate = gemm({hidden, *params.weights.moe_gating_weight->kernel,
                            nullopt, nullptr, DataType::TYPE_FP32});

    printf("gate: %s\n", gate->debugString().c_str());
    printf("gate weights: %s\n", params.weights.gate_weight->kernel->debugString().c_str());
    printf("up weights: %s\n", params.weights.up_weight->kernel->debugString().c_str());
    printf("down weights: %s\n", params.weights.down_weight->kernel->debugString().c_str());
    printf("gate is q buf: %d\n", params.weights.gate_weight->kernel->isQBuffer());

    const auto ws_size = moe_runner_->getWorkspaceSize(
        token_num, hidden_dim, expert_hidden_size, num_expert,
        top_k, params.configs.activation_type, {});
    printf("moe ws size: %d\n", ws_size);

    const auto worksapce = allocateBuffer({DataType::TYPE_INT8, {ws_size}});

    const auto fc2_result = allocateBuffer({type, {token_num, top_k, expert_hidden_size}});
    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expanded_source_row_to_dest = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});
    const auto expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {pad_to_multiple_of_16(token_num * top_k)}});

    const auto& weights = params.weights;

#define GET_SCALE_IF_Q_BUFFER(buf) \
    ((buf)->isQBuffer() ? dynamic_cast<const QBuffer*>(buf.get())->scalesData() : nullptr)
#define GET_BIAS_OR_NULL(buf) \
    ((buf) ? buf->data() : nullptr)

    tensorrt_llm::kernels::MOEParallelismConfig parallelism_config;
    auto normalization_mode = moe_conf.has_moe_norm
                            ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE
                            : tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;

    moe_runner_->runMoe(
        hidden.data(),
        gate->data<float>(),
        weights.gate_weight->kernel->data(),
        GET_SCALE_IF_Q_BUFFER(weights.gate_weight->kernel),
        GET_BIAS_OR_NULL(weights.gate_weight->bias),
        params.configs.activation_type,
        weights.down_weight->kernel->data(),
        GET_SCALE_IF_Q_BUFFER(weights.down_weight->kernel),
        GET_BIAS_OR_NULL(weights.down_weight->bias),
        weights.up_weight->kernel->data(),
        GET_SCALE_IF_Q_BUFFER(weights.up_weight->kernel),
        GET_BIAS_OR_NULL(weights.up_weight->bias),
        token_num,
        hidden_dim,
        expert_hidden_size,
        num_expert,
        top_k,
        moe_conf.normalize_expert_scale,
        (char*)worksapce->data(),
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
