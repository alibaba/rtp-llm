#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/Weights.h"

using namespace rtp_llm;

namespace unittest {

class ROCmFfnMoeFp8Test : public torch::jit::CustomClassHolder {

public:
    ROCmFfnMoeFp8Test(int64_t ep_rank, int64_t ep_size);

    void forward(torch::Tensor input,
                 torch::Tensor w1,
                 torch::Tensor w2,
                 torch::Tensor fc1_scale,
                 torch::Tensor fc2_scale,
                 torch::Tensor gating_weight,
                 torch::Tensor e_score_correction_bias,
                 int64_t topk,
                 int64_t num_expert_group,
                 int64_t topk_group,
                 torch::Tensor output);

private:
    DeviceBase* device_ = nullptr;
    GptInitParameter params_;
};


ROCmFfnMoeFp8Test::ROCmFfnMoeFp8Test(int64_t ep_rank, int64_t ep_size) {
    // TODO: add ep parameters here
    params_ = GptInitParameter();
    params_.dp_size_ = ep_size;
    params_.dp_rank_ = ep_rank;
    params_.ep_size_ = ep_size;
    params_.ep_rank_ = ep_rank;
    params_.nccl_ip_ = "localhost";
    params_.dp_tp_nccl_port_ = 50049;

    params_.expert_num_ = 128;
    params_.phy_exp_num_ = 128;
    params_.max_generate_batch_size_ = 128;

    params_.moe_config.use_deepep_moe = true;
    params_.moe_config.use_deepep_low_latency = true;
    params_.device_resource_config.enable_comm_overlap = false;
    params_.hidden_size_ = 4096;
    params_.update_from_env_for_test();
    
    DeviceFactory::initDevices(params_);
    device_ = DeviceFactory::getDefaultDevice();
}


void ROCmFfnMoeFp8Test::forward(torch::Tensor input,
                              torch::Tensor w1,
                              torch::Tensor w2,
                              torch::Tensor fc1_scale,
                              torch::Tensor fc2_scale,
                              torch::Tensor gating_weight,
                              torch::Tensor e_score_correction_bias,
                              int64_t topk,
                              int64_t num_expert_group,
                              int64_t topk_group,
                              torch::Tensor output) {
    
    size_t num_expert = static_cast<size_t>(gating_weight.size(0));
    int64_t model_dim = input.size(1);

    // TODO: config ep parameters here
    MoeConfigs moe_configs({
        .expert_num=num_expert,
        .top_k=static_cast<size_t>(topk),
        .normalize_expert_scale=false, // FIXME(liyangcheng.lyc): has_moe_norm?
        .moe_inter_padding_size=model_dim,
        .has_moe_norm=true,
        .ep_rank=static_cast<size_t>(params_.ep_rank_),
        .ep_size=static_cast<size_t>(params_.ep_size_),
        .dp_rank=static_cast<size_t>(params_.dp_rank_),
        .dp_size=static_cast<size_t>(params_.dp_size_),
        .scoring_func=1, // FIXME(liyangcheng.lyc): useless now
        .topk_group=static_cast<int>(topk_group),
        .n_group=static_cast<int>(num_expert_group)
    });

    FfnConfigs ffn_configs({
        .activation_type=ActivationType::Swiglu,
        .moe_configs=moe_configs
    });

    BufferPtr input_buffer = torchTensor2Buffer(input);

    BufferPtr w1_buffer = torchTensor2Buffer(w1);
    BufferPtr w2_buffer = torchTensor2Buffer(w2);
    BufferPtr fc1_scale_buffer = torchTensor2Buffer(fc1_scale);
    BufferPtr fc2_scale_buffer = torchTensor2Buffer(fc2_scale);

    MemoryType zeros_type = fc1_scale_buffer->where();
    BufferPtr gate_weight_buffer = BufferPtr(new QBuffer(std::move(w1_buffer), std::move(fc1_scale_buffer), std::move(BufferPtr(new Buffer(zeros_type, DataType::TYPE_INVALID, {0}, nullptr)))));
    BufferPtr down_weight_buffer = BufferPtr(new QBuffer(std::move(w2_buffer), std::move(fc2_scale_buffer), std::move(BufferPtr(new Buffer(zeros_type, DataType::TYPE_INVALID, {0}, nullptr)))));

    torch::Tensor gating_weight_t = gating_weight.transpose(0, 1).contiguous();
    BufferPtr gating_weight_buffer = torchTensor2Buffer(gating_weight_t);

    BufferPtr e_score_correction_bias_buffer = torchTensor2Buffer(e_score_correction_bias);

    FfnLayerWeights weights;
    weights.moe_gate_weight = std::make_shared<DenseWeights>(DenseWeights(gate_weight_buffer));
    weights.moe_down_weight = std::make_shared<DenseWeights>(DenseWeights(down_weight_buffer));
    weights.moe_gating_weight = std::make_shared<const DenseWeights>(DenseWeights(gating_weight_buffer));
    weights.e_score_correction_bias = e_score_correction_bias_buffer;

    FfnLayerParams ffn_layer_params(*input_buffer, ffn_configs, weights, std::nullopt, QScheme::Qfp8PerToken, DataType::TYPE_FP16);

    FfnLayerOutput ffn_output = device_->ffnLayer(ffn_layer_params);

    BufferPtr output_buffer = torchTensor2Buffer(output);
    device_->copy({*output_buffer, *(ffn_output.hidden_states)});
}

} // namespace unittest


static auto ROCmFfnMoeFp8Test = torch::jit::class_<unittest::ROCmFfnMoeFp8Test>("unittest", "ROCmFfnMoeFp8Test")
    .def(torch::jit::init<int64_t, int64_t>())
    .def("forward", &unittest::ROCmFfnMoeFp8Test::forward);