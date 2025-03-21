#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
using namespace fastertransformer;

namespace unittest {

class MlaRotaryKVCacheOp: public torch::jit::CustomClassHolder {
public:
    MlaRotaryKVCacheOp(int64_t mla_type, 
                       int64_t head_num,
                       int64_t nope_head_dim,
                       int64_t rope_head_dim,
                       int64_t v_head_dim,
                       int64_t q_lora_rank,
                       int64_t kv_lora_rank,
                       int64_t hidden_size,
                       double  softmax_extra_scale);
    
    AttentionConfigs                     attn_configs = AttentionConfigs({});
    void init(torch::Tensor sequence_length, torch::Tensor input_length, int64_t page_size, torch::Tensor block_id_map);
    void applyRotaryKVCache(
        torch::Tensor q, torch::Tensor ckv, torch::Tensor k_rope, torch::Tensor ckv_cache, torch::Tensor kpe_cache, torch::Tensor cos_sin_cache);
    DeviceBase* device_;
    FlashInferAttnParamsPtr params_;
    int64_t context_batch_size_;
    int64_t decoder_batch_size_;
};

MlaRotaryKVCacheOp::MlaRotaryKVCacheOp(int64_t mla_type, 
                                       int64_t head_num,
                                       int64_t nope_head_dim,
                                       int64_t rope_head_dim,
                                       int64_t v_head_dim,
                                       int64_t q_lora_rank,
                                       int64_t kv_lora_rank,
                                       int64_t hidden_size,
                                       double  softmax_extra_scale) {
    rtp_llm::initLogger();
    
    auto gpt_params = GptInitParameter();
    gpt_params.mla_ops_type_ = MlaOpsType(mla_type);
    fastertransformer::DeviceFactory::initDevices(gpt_params);
    device_      = fastertransformer::DeviceFactory::getDefaultDevice();
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

void MlaRotaryKVCacheOp::init(torch::Tensor sequence_length, torch::Tensor input_length, int64_t page_size, torch::Tensor block_id_map) {
    attn_configs.tokens_per_block = page_size;
    params_ = FlashInferAttnParams::prepareFlashInferAttnParams(device_,
                                                                attn_configs,
                                                                torchTensor2Buffer(sequence_length),
                                                                torchTensor2Buffer(input_length),
                                                                torchTensor2Buffer(block_id_map),
                                                                DataType::TYPE_FP16);
    FT_CHECK_WITH_INFO(params_ != nullptr, "flashinfer params is nullptr");
    context_batch_size_ = input_length.size(0) - sequence_length.size(0);
    decoder_batch_size_ = sequence_length.size(0);
}

void MlaRotaryKVCacheOp::applyRotaryKVCache(
    torch::Tensor q, torch::Tensor ckv, torch::Tensor k_rope, torch::Tensor ckv_cache, torch::Tensor kpe_cache, torch::Tensor cos_sin_cache) {

    auto attn_layer_weight                = AttentionLayerWeights();
    attn_layer_weight.rope_cos_sin_cache = torchTensor2Buffer(cos_sin_cache);
    auto attn_common_inputs               = AttentionCommonInputs();
    attn_common_inputs.context_batch_size = context_batch_size_;
    attn_common_inputs.decoder_batch_size = decoder_batch_size_;
    attn_common_inputs.flash_infer_attn_params = params_;
    attn_common_inputs.kv_cache =
            std::make_optional<KvCacheInfo>({1, nullptr, torchTensor2Buffer(ckv_cache), torchTensor2Buffer(kpe_cache), nullptr, nullptr});
    attn_common_inputs.flash_infer_attn_params = params_;

    auto q_buf = torchTensor2Buffer(q);
    auto ckv_buf = torchTensor2Buffer(ckv);
    auto k_rope_buf = torchTensor2Buffer(k_rope);

    MlaRotaryWriteKVCacheParams params = {
        *q_buf,
        *ckv_buf,
        *k_rope_buf,
        attn_common_inputs,        
        attn_layer_weight,
        attn_configs,
        QScheme::NoQuantize,
    };
    device_->mlaRotaryWriteKVCache(params);
}
}  // namespace unittest

static auto MlaRotaryKVCacheOp =
    torch::jit::class_<unittest::MlaRotaryKVCacheOp>("unittest", "MlaRotaryKVCacheOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("init", &unittest::MlaRotaryKVCacheOp::init)
        .def("applyRotaryKVCache", &unittest::MlaRotaryKVCacheOp::applyRotaryKVCache);