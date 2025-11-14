#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace rtp_llm;

namespace unittest {

class ROCmFfnMoeBf16Op: public torch::jit::CustomClassHolder {

public:
    ROCmFfnMoeBf16Op(int64_t ep_rank, int64_t ep_size);

    void forward(torch::Tensor                input,
                 torch::Tensor                w1,
                 torch::Tensor                w2,
                 torch::Tensor                gating_weight,
                 std::optional<torch::Tensor> e_score_correction_bias,
                 int64_t                      topk,
                 int64_t                      num_expert_group,
                 int64_t                      topk_group,
                 torch::Tensor                output);

private:
    DeviceBase* device_ = nullptr;
    int64_t ep_rank_ = 0;
    int64_t ep_size_ = 1;
    int64_t dp_rank_ = 0;
    int64_t dp_size_ = 1;
};

ROCmFfnMoeBf16Op::ROCmFfnMoeBf16Op(int64_t ep_rank, int64_t ep_size) {
    ep_rank_ = ep_rank;
    ep_size_ = ep_size;
    dp_rank_ = ep_rank;
    dp_size_ = ep_size;
    ParallelismConfig parallelism_config;
    parallelism_config.dp_size = ep_size;
    parallelism_config.dp_rank = ep_rank;
    parallelism_config.ep_size = ep_size;
    parallelism_config.ep_rank = ep_rank;
    parallelism_config.nccl_ip = "localhost";
    parallelism_config.dp_tp_nccl_port = 50049;
    ModelConfig model_config;
    EPLBConfig eplb_config;
    FMHAConfig fmha_config;
    DeviceResourceConfig device_resource_config;
    MoeConfig moe_config;
    SpeculativeExecutionConfig sp_config;
    MiscellaneousConfig misc_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    HWKernelConfig hw_kernel_config;
    ConcurrencyConfig concurrency_config;
    FfnDisAggregateConfig ffn_disaggregate_config;
    RuntimeConfig runtime_config;
    DeviceFactory::initDevices(
        parallelism_config,
        model_config,
        eplb_config,
        fmha_config,
        device_resource_config,
        moe_config,
        sp_config,
        misc_config,
        profiling_debug_logging_config,
        hw_kernel_config,
        concurrency_config,
        ffn_disaggregate_config,
        runtime_config);
    device_ = DeviceFactory::getDefaultDevice();
}

void ROCmFfnMoeBf16Op::forward(torch::Tensor                input,
                               torch::Tensor                w1,
                               torch::Tensor                w2,
                               torch::Tensor                gating_weight,
                               std::optional<torch::Tensor> e_score_correction_bias,
                               int64_t                      topk,
                               int64_t                      num_expert_group,
                               int64_t                      topk_group,
                               torch::Tensor                output) {

    size_t  num_expert = static_cast<size_t>(gating_weight.size(0));
    int64_t model_dim  = input.size(1);

    // TODO: config ep parameters here
    MoeConfigs moe_configs({.expert_num             = num_expert,
                            .top_k                  = static_cast<size_t>(topk),
                            .normalize_expert_scale = false,  // FIXME(liyangcheng.lyc): has_moe_norm?
                            .has_moe_norm           = false,
                            .ep_rank                = static_cast<size_t>(ep_rank_),
                            .ep_size                = static_cast<size_t>(ep_size_),
                            .dp_rank                = static_cast<size_t>(dp_rank_),
                            .dp_size                = static_cast<size_t>(dp_size_),
                            .scoring_func           = 1,  // FIXME(liyangcheng.lyc): useless now
                            .topk_group             = static_cast<int>(topk_group),
                            .n_group                = static_cast<int>(num_expert_group)});

    FfnConfigs ffn_configs({.activation_type = ActivationType::Swiglu, .moe_configs = moe_configs});

    BufferPtr input_buffer = torchTensor2Buffer(input);

    BufferPtr gate_weight_buffer = torchTensor2Buffer(w1);
    BufferPtr down_weight_buffer = torchTensor2Buffer(w2);

    torch::Tensor gating_weight_t      = gating_weight.transpose(0, 1).contiguous();
    BufferPtr     gating_weight_buffer = torchTensor2Buffer(gating_weight_t);

    BufferPtr e_score_correction_bias_buffer =
        e_score_correction_bias.has_value() ? torchTensor2Buffer(e_score_correction_bias.value()) : nullptr;

    FfnLayerWeights weights;
    weights.moe_gate_weight         = std::make_shared<DenseWeights>(DenseWeights(gate_weight_buffer));
    weights.moe_down_weight         = std::make_shared<DenseWeights>(DenseWeights(down_weight_buffer));
    weights.moe_gating_weight       = std::make_shared<const DenseWeights>(DenseWeights(gating_weight_buffer));
    weights.e_score_correction_bias = e_score_correction_bias_buffer;

    FfnLayerParams ffn_layer_params(
        *input_buffer, ffn_configs, weights, std::nullopt, QScheme::NoQuantize, DataType::TYPE_INVALID, nullptr);

    FfnLayerOutput ffn_output = device_->ffnLayer(ffn_layer_params);

    BufferPtr output_buffer = torchTensor2Buffer(output);
    device_->copy({*output_buffer, *(ffn_output.hidden_states)});
}

}  // namespace unittest

static auto ROCmFfnMoeBf16Op = torch::jit::class_<unittest::ROCmFfnMoeBf16Op>("unittest", "ROCmFfnMoeBf16Op")
                                   .def(torch::jit::init<int64_t, int64_t>())
                                   .def("forward", &unittest::ROCmFfnMoeBf16Op::forward);