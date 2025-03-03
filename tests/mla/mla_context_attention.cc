#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace fastertransformer;

namespace unittest {
class MlaContextAttnOp: public torch::jit::CustomClassHolder {
public:
    MlaContextAttnOp(int64_t head_num,
                     int64_t nope_head_dim,
                     int64_t rope_head_dim,
                     int64_t v_head_dim,
                     int64_t q_lora_rank,
                     int64_t kv_lora_rank,
                     int64_t hidden_size,
                     double  softmax_extra_scale);
    torch::Tensor forward(torch::Tensor q,
                          torch::Tensor kv_a,
                          torch::Tensor k_rope,
                          torch::Tensor k_nope_weight,
                          torch::Tensor v_weight,
                          torch::Tensor seq_len);

private:
    CudaDevice       device       = CudaDevice({});
    AttentionConfigs attn_configs = AttentionConfigs({});
};

MlaContextAttnOp::MlaContextAttnOp(int64_t head_num,
                                   int64_t nope_head_dim,
                                   int64_t rope_head_dim,
                                   int64_t v_head_dim,
                                   int64_t q_lora_rank,
                                   int64_t kv_lora_rank,
                                   int64_t hidden_size,
                                   double  softmax_extra_scale) {
    device.init();
    attn_configs = AttentionConfigs({
        static_cast<size_t>(head_num),
        static_cast<size_t>(head_num),
        static_cast<size_t>(nope_head_dim + rope_head_dim),
        static_cast<size_t>(hidden_size),
        RopeConfig(),
        64,
        AttentionMaskType::causalMask,
        1.0f,
        true,
        false,
        true,
        true,
        static_cast<size_t>(q_lora_rank),
        static_cast<size_t>(kv_lora_rank),
        static_cast<size_t>(nope_head_dim),
        static_cast<size_t>(rope_head_dim),
        static_cast<size_t>(v_head_dim),
        static_cast<float>(softmax_extra_scale),
        KvCacheDataType::BASE,
    });
}

torch::Tensor MlaContextAttnOp::forward(torch::Tensor q,
                                        torch::Tensor kv_a,
                                        torch::Tensor k_rope,
                                        torch::Tensor k_nope_weight,
                                        torch::Tensor v_weight,
                                        torch::Tensor seq_len) {
    size_t token_num       = q.size(0);
    auto   q_b             = torchTensor2Buffer(q);
    auto   kv_a_b          = torchTensor2Buffer(kv_a);
    auto   k_rope_b        = torchTensor2Buffer(k_rope);
    auto   k_nope_weight_b = torchTensor2Buffer(k_nope_weight);
    auto   v_weight_b      = torchTensor2Buffer(v_weight);
    auto   datatype        = kv_a_b->type();

    size_t               batch_size = seq_len.size(0);
    std::vector<int32_t> cu_seqlens_data(batch_size + 1, 0);
    int                  total_size  = 0;
    int                  max_seq_len = 0;
    for (int i = 0; i < batch_size; i++) {
        int cur_seq_len = seq_len[i].item<int>();
        total_size += cur_seq_len;
        cu_seqlens_data[i + 1] = total_size;
        max_seq_len            = std::max(max_seq_len, cur_seq_len);
    }

    BufferPtr sequence_lengths;
    BufferPtr kv_cache_block_id;
    BufferPtr input_lengths;

    auto device_prep_params = DevicePrepParams({
        attn_configs,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id,
        datatype,
        batch_size,
        0,
    });

    device.prepareModelRun(device_prep_params);
    auto output =
        device.allocateBuffer({datatype, {token_num, attn_configs.head_num * attn_configs.size_per_head}}, {"output"});

    auto k_nope_w = std::make_shared<DenseWeights>(k_nope_weight_b);
    auto v_w      = std::make_shared<DenseWeights>(v_weight_b);

    auto attn_layer_weight          = AttentionLayerWeights();
    attn_layer_weight.k_nope_weight = k_nope_w;
    attn_layer_weight.v_weight      = v_w;
    auto attn_common_inputs         = AttentionCommonInputs();
    attn_common_inputs.cu_seqlens =
        device.clone({*vector2Buffer(cu_seqlens_data), AllocationType::DEVICE, {"cu_seqlens"}});
    attn_common_inputs.context_batch_size  = batch_size;
    attn_common_inputs.decoder_batch_size  = 0;
    attn_common_inputs.context_max_seq_len = token_num;

    auto mla_params = MlaAttentionModuleParams{
        0, *q_b, *kv_a_b, *k_rope_b, output, attn_common_inputs, attn_layer_weight, attn_configs, QScheme::NoQuantize};

    device.mlaContextAttention(mla_params);

    auto output_t = Buffer2torchTensor(*output, false);
    return output_t.detach().clone();
}

}  // namespace unittest

static auto MergeTransposeTHS =
    torch::jit::class_<unittest::MlaContextAttnOp>("unittest", "MlaContextAttnOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("forward", &unittest::MlaContextAttnOp::forward);
