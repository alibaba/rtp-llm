#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace rtp_llm;

namespace unittest {
class MlaAttnLayerOp: public torch::jit::CustomClassHolder {
public:
    MlaAttnLayerOp(int64_t head_num,
                   int64_t nope_head_dim,
                   int64_t rope_head_dim,
                   int64_t v_head_dim,
                   int64_t q_lora_rank,
                   int64_t kv_lora_rank,
                   int64_t hidden_size,
                   double  softmax_extra_scale);
    torch::Tensor    forward(torch::Tensor              hidden,
                             std::vector<torch::Tensor> weights,
                             torch::Tensor              ckv_cache,
                             torch::Tensor              kpe_caches,
                             torch::Tensor              prefix_lengths_t,
                             torch::Tensor              sequence_length_t,
                             torch::Tensor              kvcache_block_id,
                             int64_t                    page_size);
    void             reset_head_num(int64_t head_num);
    AttentionConfigs attn_configs = AttentionConfigs({});
    DeviceBase*      device_;
};
MlaAttnLayerOp::MlaAttnLayerOp(int64_t head_num,
                               int64_t nope_head_dim,
                               int64_t rope_head_dim,
                               int64_t v_head_dim,
                               int64_t q_lora_rank,
                               int64_t kv_lora_rank,
                               int64_t hidden_size,
                               double  softmax_extra_scale) {
    rtp_llm::initLogger();
    GptInitParameter gpt_init_parameter;
    gpt_init_parameter.update_from_env_for_test();
    rtp_llm::DeviceFactory::initDevices(gpt_init_parameter);
    device_      = rtp_llm::DeviceFactory::getDefaultDevice();
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
        static_cast<size_t>(q_lora_rank),
        static_cast<size_t>(kv_lora_rank),
        static_cast<size_t>(nope_head_dim),
        static_cast<size_t>(rope_head_dim),
        static_cast<size_t>(v_head_dim),
        static_cast<float>(softmax_extra_scale),
        KvCacheDataType::BASE,
    });
}
torch::Tensor MlaAttnLayerOp::forward(torch::Tensor              hidden,
                                      std::vector<torch::Tensor> weights,
                                      torch::Tensor              ckv_cache,
                                      torch::Tensor              kpe_caches,
                                      torch::Tensor              prefix_lengths_t,
                                      torch::Tensor              sequence_length_t,
                                      torch::Tensor              kvcache_block_id,
                                      int64_t                    page_size) {
    try {
        attn_configs.tokens_per_block      = page_size;
        BufferPtr prefix_lengths_host      = torchTensor2Buffer(prefix_lengths_t);
        BufferPtr input_lengths_host       = torchTensor2Buffer(sequence_length_t);
        BufferPtr sequence_lengths_host    = torchTensor2Buffer(sequence_length_t);
        BufferPtr kvcache_block_id_host    = torchTensor2Buffer(kvcache_block_id);
        BufferPtr kvcache_block_id_device  = device_->clone({*kvcache_block_id_host});
        auto      decode_flash_infer_attn  = FlashInferAttnParams::prepare(device_,
                                                                     attn_configs,
                                                                     nullptr,
                                                                     sequence_lengths_host,
                                                                     input_lengths_host,
                                                                     kvcache_block_id_host,
                                                                     kvcache_block_id_device,
                                                                     DataType::TYPE_BF16);
        auto      context_flash_infer_attn = FlashInferAttnParams::prepare(device_,
                                                                      attn_configs,
                                                                      prefix_lengths_host,
                                                                      sequence_lengths_host,
                                                                      input_lengths_host,
                                                                      kvcache_block_id_host,
                                                                      kvcache_block_id_device,
                                                                      DataType::TYPE_BF16);

        size_t token_num              = hidden.size(0);
        auto   hidden_b               = torchTensor2Buffer(hidden);
        auto   kc_weight_b            = torchTensor2Buffer(weights[0]);
        auto   vc_t_weight_b          = torchTensor2Buffer(weights[1]);
        auto   q_a_norm_weight_gamma  = torchTensor2Buffer(weights[2]);
        auto   q_a_norm_weight_beta   = torchTensor2Buffer(weights[3]);
        auto   mla_fusedqkrope_w      = torchTensor2Buffer(weights[4]);
        auto   q_b_weight_b           = torchTensor2Buffer(weights[5]);
        auto   kv_a_norm_weight_gamma = torchTensor2Buffer(weights[6]);
        auto   kv_a_norm_weight_beta  = torchTensor2Buffer(weights[7]);
        auto   output_weight_b        = torchTensor2Buffer(weights[8]);
        auto   cos_sin_cache_b        = torchTensor2Buffer(weights[9]);
        auto   hidden_buffer          = torchTensor2Buffer(hidden);
        auto   qkv_output             = device_->allocateBuffer(
            {hidden_buffer->type(),
                           {token_num, attn_configs.head_num, attn_configs.nope_head_dim + attn_configs.rope_head_dim}},
            {"output"});
        auto kc_dense_weight        = std::make_shared<DenseWeights>(kc_weight_b);
        auto vc_dense_weight        = std::make_shared<DenseWeights>(vc_t_weight_b);
        auto k_cache_buffer         = torchTensor2Buffer(ckv_cache);
        auto v_cache_buffer         = torchTensor2Buffer(kpe_caches);
        auto attn_layer_weight      = AttentionLayerWeights();
        attn_layer_weight.kc_weight = kc_dense_weight;
        attn_layer_weight.vc_weight = vc_dense_weight;
        attn_layer_weight.q_a_norm_weight =
            std::make_shared<LayerNormWeights>(q_a_norm_weight_gamma, q_a_norm_weight_beta);
        attn_layer_weight.kv_a_norm_weight =
            std::make_shared<LayerNormWeights>(kv_a_norm_weight_gamma, kv_a_norm_weight_beta);
        attn_layer_weight.fusedqkrope_weight  = std::make_shared<DenseWeights>(mla_fusedqkrope_w);
        attn_layer_weight.q_b_weight          = std::make_shared<DenseWeights>(q_b_weight_b);
        attn_layer_weight.output_weight       = std::make_shared<DenseWeights>(output_weight_b);
        attn_layer_weight.rope_cos_sin_cache  = cos_sin_cache_b;
        auto attn_common_inputs               = AttentionCommonInputs();
        attn_common_inputs.context_batch_size = 0;
        attn_common_inputs.decoder_batch_size = hidden.size(0);
        attn_common_inputs.kv_cache           = std::make_optional<KvCacheInfo>(
            {1, kvcache_block_id_host, kv_cache_buffer, nullptr});
        attn_common_inputs.prefill_flash_infer_attn = context_flash_infer_attn;
        attn_common_inputs.decode_flash_infer_attn  = decode_flash_infer_attn;
        attn_common_inputs.input_lengths            = input_lengths_host;
        attn_common_inputs.sequence_lengths         = sequence_lengths_host;
        LayerNormConfig layernorm_config            = LayerNormConfig();
        layernorm_config.eps                        = 1e-6;
        layernorm_config.norm_type                  = NormType::rmsnorm;
        auto mla_params                             = AttentionLayerParams({0,
                                                                            *hidden_buffer,
                                                                            qkv_output,
                                                                            attn_configs,
                                                                            attn_layer_weight,
                                                                            attn_common_inputs,
                                                                            std::nullopt,
                                                                            layernorm_config,
                                                                            QScheme::NoQuantize});
        // auto mla_params = MlaDecoderAttentionParams{
        //     0, *hidden_buffer, qkv_output, attn_common_inputs, attn_layer_weight, attn_configs, QScheme::NoQuantize};
        auto output = device_->mlaAttentionLayer(mla_params);
        return Buffer2torchTensor(*qkv_output, false).contiguous().detach().clone();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        abort();
    }
}
void MlaAttnLayerOp::reset_head_num(int64_t head_num) {
    attn_configs.head_num    = static_cast<size_t>(head_num);
    attn_configs.kv_head_num = static_cast<size_t>(head_num);
}
}  // namespace unittest
static auto MlaAttnLayerOp =
    torch::jit::class_<unittest::MlaAttnLayerOp>("unittest", "MlaAttnLayerOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("forward", &unittest::MlaAttnLayerOp::forward)
        .def("reset_head_num", &unittest::MlaAttnLayerOp::reset_head_num);
