#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/kernels/rocm/masked_silu_and_mul/mask_kernel.h"

#include "deepgemm.h"
#include "activation.h"

// aiter kernels
// ========= 修改头文件
#include "ck_tile/host.hpp"

namespace rtp_llm {

void quantize_3d(ROCmDevice* device, const BufferPtr& input, QBufferPtr& q_input, torch::Tensor& input_tensor, torch::Tensor& scale_tensor)
{
    std::vector<size_t> origin_shape = input->shape();
    std::vector<size_t> reshaped_dims = {origin_shape[0] * origin_shape[1], origin_shape[2]};
    input->updateShape(reshaped_dims);
    q_input = std::dynamic_pointer_cast<QBuffer>(
        device->quantize(QuantizeParams(*input, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerToken, 128, 0)));
    
    input_tensor = Buffer2torchTensor(q_input->kernel(), false);
    scale_tensor = Buffer2torchTensor(q_input->scales(), false);
    input_tensor = input_tensor.contiguous().view({
        static_cast<int64_t>(origin_shape[0]),
        static_cast<int64_t>(origin_shape[1]),
        static_cast<int64_t>(origin_shape[2])
    });
    auto input_tensor_shape = input_tensor.sizes();
    scale_tensor = scale_tensor.contiguous().view({
        static_cast<int64_t>(origin_shape[0]),
        static_cast<int64_t>(origin_shape[1]),
        1
    });
    auto scale_tensor_shape = scale_tensor.sizes();
    return;
}

FfnLayerOutput ROCmDevice::deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    // get model info
    const auto&  moe_conf            = params.configs.moe_configs.value();
    const auto&  weights             = params.weights;

    const size_t num_experts          = moe_conf.expert_num;
    const size_t num_experts_per_rank = num_experts / moe_conf.ep_size;

    // moe_gate_weight : [num_experts_per_rank, inter_dim * 2, hidden_dim]
	RTP_LLM_CHECK_WITH_INFO(weights.moe_gate_weight->kernel->shape().size() == 3, "moe gate weight kernel shape size must be 3");
    // moe_down_weight : [num_experts_per_rank, hidden_dim, inter_dim]
    RTP_LLM_CHECK_WITH_INFO(weights.moe_down_weight->kernel->shape().size() == 3, "moe down weight kernel shape size must be 3");
    const size_t    hidden_dim           = static_cast<size_t>(weights.moe_down_weight->kernel->shape()[1]);
    const size_t    inter_dim            = static_cast<size_t>(weights.moe_down_weight->kernel->shape()[2]);
    
    bool         is_gated_activation  = isGatedActivation(params.configs.activation_type);
    auto         deep_ep_ll_output    = gate_outputs.deep_ep_ll_output;
    
    // get input info
    // rank_hidden : [num_experts_per_rank, num_token, hidden_dim]
    BufferPtr rank_hidden = torchTensor2Buffer(deep_ep_ll_output->packed_recv_x);
    // masked_m : [num_experts_per_rank]
    BufferPtr masked_m = torchTensor2Buffer(deep_ep_ll_output->packed_recv_count);
    std::vector<size_t> hidden_shape = rank_hidden->shape();
    RUNTIME_ASSERT_OP_ARG(hidden_shape.size() == 3 && hidden_shape[0] == num_experts_per_rank,
                    "hidden_shape dims should be 3 and dim 0 should be num_experts_per_rank");
    const size_t num_token = rank_hidden->shape()[1];

    
    // get w1 and w2
    torch::Tensor                w1_tensor, w2_tensor;
    std::optional<torch::Tensor> w1_scale_tensor, w2_scale_tensor;
    QBufferPtr                   q_hidden;
    if (params.qscheme == QScheme::NoQuantize) {
        w1_tensor = Buffer2torchTensor(*(weights.moe_gate_weight->kernel), false);
        w2_tensor = Buffer2torchTensor(*(weights.moe_down_weight->kernel), false);
    } else {
        const QBuffer& qmoe_gate_weight = reinterpret_cast<const QBuffer&>(*(weights.moe_gate_weight->kernel));
        const QBuffer& qmoe_down_weight = reinterpret_cast<const QBuffer&>(*(weights.moe_down_weight->kernel));
        
        Buffer w1 = qmoe_gate_weight.kernel();
        Buffer w1_scale = qmoe_gate_weight.scales();
        Buffer w2 = qmoe_down_weight.kernel();
        Buffer w2_scale = qmoe_down_weight.scales();

        w1_tensor = Buffer2torchTensor(w1, false);
        w1_scale_tensor = Buffer2torchTensor(w1_scale, false);
        w2_tensor = Buffer2torchTensor(w2, false);
        w2_scale_tensor = Buffer2torchTensor(w2_scale, false);
    }
    
    // preallocate output
    BufferPtr fc1_result;
    if (is_gated_activation) {
        fc1_result = allocateBuffer(
            {DataType::TYPE_BF16, {num_experts_per_rank, num_token, inter_dim * 2}}, {"fc1_result"});
    } else {
        fc1_result = allocateBuffer({DataType::TYPE_BF16, {num_experts_per_rank, num_token, inter_dim}},
                                    {"fc1_result"});
    }
    torch::Tensor fc1_result_tensor = Buffer2torchTensor(fc1_result, false);

    BufferPtr output = allocateBuffer({DataType::TYPE_BF16, {num_experts_per_rank, num_token, hidden_dim}});
    torch::Tensor output_tensor = Buffer2torchTensor(output, false);
    if (num_token == 0) {
        return {output};
    }

    // bf16 input
    if (params.qscheme == QScheme::NoQuantize) {
        ::deepgemm(deep_ep_ll_output->packed_recv_x,
                        w1_tensor,
                        fc1_result_tensor,
                        deep_ep_ll_output->packed_recv_count,
                        std::nullopt,
                        std::nullopt);
        // 2. activation: silu and mul
        BufferPtr fc1_activation = allocateBuffer(
            {DataType::TYPE_BF16, {num_experts_per_rank, num_token, inter_dim}}, {"fc1_activation"});
        torch::Tensor fc1_activation_tensor = Buffer2torchTensor(fc1_activation, false);
        aiter::silu_and_mul(fc1_activation_tensor, fc1_result_tensor);
        // 3. second gemm
        ::deepgemm(fc1_activation_tensor,
                        w2_tensor,
                        output_tensor,
                        deep_ep_ll_output->packed_recv_count,
                        std::nullopt,
                        std::nullopt);
    } else if (params.qscheme == QScheme::Qfp8PerToken) {   // fp8 input
        // input quantization
        QBufferPtr      q_hidden;
        torch::Tensor   hidden_tensor;
        torch::Tensor   scale_tensor;

        if (deep_ep_ll_output->packed_recv_x_scales.has_value()) {
            const char* env_fp8_cast_level = std::getenv("ACCL_FP8_CAST_LEVEL");
            RUNTIME_ASSERT_OP_ARG(env_fp8_cast_level != nullptr, "env ACCL_FP8_CAST_LEVEL must be set when doing quant in low latency dispatch");
            RUNTIME_ASSERT_OP_ARG(0 == std::strcmp(env_fp8_cast_level, "2"), "env ACCL_FP8_CAST_LEVEL must be set to 2 when using PTPC quant in ROCm");

            hidden_tensor = deep_ep_ll_output->packed_recv_x;
            scale_tensor = deep_ep_ll_output->packed_recv_x_scales.value();
        } else {
            quantize_3d(this, rank_hidden, q_hidden, hidden_tensor, scale_tensor);
        }

        ::deepgemm(hidden_tensor,
                         w1_tensor,
                         fc1_result_tensor,
                         deep_ep_ll_output->packed_recv_count,
                         scale_tensor,
                         w1_scale_tensor);
        torch::Tensor fc1_act_tensor;
        torch::Tensor fc1_act_scale_tensor;
        bool fuse_silu_and_mul = true;
        if (fuse_silu_and_mul) {
            BufferPtr fc1_activation = allocateBuffer(
                {DataType::TYPE_FP8_E4M3, {num_experts_per_rank, num_token, inter_dim}}, {"fc1_activation"});
            BufferPtr fc1_activation_scale = allocateBuffer(
                {DataType::TYPE_FP32, {num_experts_per_rank, num_token, 1}}, {"fc1_activation_scale"});
            launch_doActivationMaskedKernelHIP(static_cast<fp8_e4m3_t*>(fc1_activation->data()),                                       
                                                static_cast<float*>(fc1_activation_scale->data()),
                                                static_cast<const hip_bfloat16*>(fc1_result->data()),
                                                num_experts_per_rank,
                                                num_token,
                                                inter_dim,
                                                is_gated_activation,
                                                static_cast<const int*>(masked_m->data()),
                                                stream_);
            fc1_act_tensor = Buffer2torchTensor(fc1_activation, false);
            fc1_act_scale_tensor = Buffer2torchTensor(fc1_activation_scale, false);
        } else {
            BufferPtr fc1_activation = allocateBuffer(
                {DataType::TYPE_BF16, {num_experts_per_rank, num_token, inter_dim}}, {"fc1_activation"});
            torch::Tensor fc1_activation_tensor = Buffer2torchTensor(fc1_activation, false);
            aiter::silu_and_mul(fc1_activation_tensor, fc1_result_tensor);
            // activation quantization
            QBufferPtr q_fc1_activation;
            quantize_3d(this, fc1_activation, q_fc1_activation, fc1_act_tensor, fc1_act_scale_tensor);
        }
        ::deepgemm(fc1_act_tensor,
                        w2_tensor,
                        output_tensor,
                        deep_ep_ll_output->packed_recv_count,
                        fc1_act_scale_tensor,
                        w2_scale_tensor);
    }
    return {output};   
}

}  // namespace rtp_llm
