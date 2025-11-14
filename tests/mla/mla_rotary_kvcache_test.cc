#ifdef USING_ROCM
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#else
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#endif
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

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

    AttentionConfigs attn_configs = AttentionConfigs({});
    void             init(torch::Tensor prefix_length,
                          torch::Tensor sequence_length,
                          torch::Tensor input_length,
                          int64_t       page_size,
                          torch::Tensor block_id_map,
                          torch::Tensor block_id_map_device);
    void             applyRotaryKVCache(torch::Tensor q,
                                        torch::Tensor fused_qkv,
                                        int64_t       kv_offset,
                                        torch::Tensor ckv_cache,
                                        torch::Tensor kpe_cache,
                                        torch::Tensor cos_sin_cache);
    DeviceBase*      device_;
    ParamsPtr        context_params_;
    ParamsPtr        decode_params_;
    int64_t          context_batch_size_;
    int64_t          decoder_batch_size_;
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

    ParallelismConfig parallelism_config;
    ModelConfig model_config;
    model_config.mla_ops_type = MlaOpsType(mla_type);
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

    rtp_llm::DeviceFactory::initDevices(
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
    device_      = rtp_llm::DeviceFactory::getDefaultDevice();

    attn_configs.head_num = static_cast<size_t>(head_num);
    attn_configs.kv_head_num = static_cast<size_t>(head_num);
    attn_configs.size_per_head = static_cast<size_t>(nope_head_dim + rope_head_dim);
    attn_configs.tokens_per_block = 64;
    attn_configs.q_scaling = 1.0f;
    attn_configs.fuse_qkv_add_bias = true;
    attn_configs.use_logn_attn = false;
    attn_configs.is_causal = true;
    attn_configs.use_mla = true;
    attn_configs.q_lora_rank = static_cast<size_t>(q_lora_rank);
    attn_configs.kv_lora_rank = static_cast<size_t>(kv_lora_rank);
    attn_configs.nope_head_dim = static_cast<size_t>(nope_head_dim);
    attn_configs.rope_head_dim = static_cast<size_t>(rope_head_dim);
    attn_configs.v_head_dim = static_cast<size_t>(v_head_dim);
    attn_configs.softmax_extra_scale = static_cast<float>(softmax_extra_scale);
    attn_configs.kv_cache_dtype = KvCacheDataType::BASE;
}

void MlaRotaryKVCacheOp::init(torch::Tensor prefix_length,
                              torch::Tensor sequence_length,
                              torch::Tensor input_length,
                              int64_t       page_size,
                              torch::Tensor block_id_map,
                              torch::Tensor block_id_map_device) {
    attn_configs.tokens_per_block = page_size;
    context_batch_size_           = input_length.size(0) - sequence_length.size(0);
    decoder_batch_size_           = sequence_length.size(0);
#ifdef USING_ROCM
    context_params_ = FlashInferAttnParams::preparePrefillFlashInferAttnParams(device_,
                                                                               attn_configs,
                                                                               torchTensor2Buffer(prefix_length),
                                                                               torchTensor2Buffer(sequence_length),
                                                                               torchTensor2Buffer(input_length),
                                                                               torchTensor2Buffer(block_id_map),
                                                                               DataType::TYPE_FP16);
    decode_params_  = FlashInferAttnParams::prepareDecodeFlashInferAttnParams(device_,
                                                                             attn_configs,
                                                                             torchTensor2Buffer(sequence_length),
                                                                             torchTensor2Buffer(input_length),
                                                                             torchTensor2Buffer(block_id_map),
                                                                             DataType::TYPE_FP16);
#else
    context_params_ = FlashInferAttnParams::prepare(
        device_,
        attn_configs,
        torchTensor2Buffer(prefix_length),
        nullptr,
        torchTensor2Buffer(input_length)->slice(decoder_batch_size_, context_batch_size_, false),
        torchTensor2Buffer(block_id_map)->slice(decoder_batch_size_, context_batch_size_, false),
        torchTensor2Buffer(block_id_map_device)->slice(decoder_batch_size_, context_batch_size_, false),
        DataType::TYPE_FP16);
    decode_params_ =
        FlashInferAttnParams::prepare(device_,
                                      attn_configs,
                                      nullptr,
                                      torchTensor2Buffer(sequence_length)->slice(0, decoder_batch_size_, false),
                                      torchTensor2Buffer(input_length)->slice(0, decoder_batch_size_, false),
                                      torchTensor2Buffer(block_id_map)->slice(0, decoder_batch_size_, false),
                                      torchTensor2Buffer(block_id_map_device)->slice(0, decoder_batch_size_, false),
                                      DataType::TYPE_FP16);
#endif
}

void MlaRotaryKVCacheOp::applyRotaryKVCache(torch::Tensor q,
                                            torch::Tensor fused_qkv,
                                            int64_t       kv_offset,
                                            torch::Tensor ckv_cache,
                                            torch::Tensor kpe_cache,
                                            torch::Tensor cos_sin_cache) {

    auto attn_layer_weight                = AttentionLayerWeights();
    attn_layer_weight.rope_cos_sin_cache  = torchTensor2Buffer(cos_sin_cache);
    auto attn_common_inputs               = AttentionCommonInputs();
    attn_common_inputs.context_batch_size = context_batch_size_;
    attn_common_inputs.decoder_batch_size = decoder_batch_size_;
    attn_common_inputs.kv_cache           = std::make_optional<KvCacheInfo>(
        {1, nullptr, torchTensor2Buffer(ckv_cache), torchTensor2Buffer(kpe_cache), nullptr, nullptr});

    auto q_buf         = torchTensor2Buffer(q);
    auto fused_qkv_buf = torchTensor2Buffer(fused_qkv);

    RTP_LLM_LOG_INFO("before run");
    if (context_params_ != nullptr) {
        RTP_LLM_LOG_INFO("run context");
        auto context_q_buf = q_buf->slice(decoder_batch_size_, q_buf->shape()[0] - decoder_batch_size_);
        auto context_fused_qkv_buf =
            fused_qkv_buf->slice(decoder_batch_size_, fused_qkv_buf->shape()[0] - decoder_batch_size_);

        MlaRotaryWriteKVCacheParams context_params = {
            *context_q_buf,
            nullptr,
            *context_fused_qkv_buf,
            kv_offset,
            context_params_,
            attn_common_inputs,
            attn_layer_weight,
            attn_configs,
            QScheme::NoQuantize,
        };
        device_->mlaRotaryWriteKVCache(context_params);
    }
    if (decode_params_ != nullptr) {
        RTP_LLM_LOG_INFO("run decode");
        auto decode_q_buf         = q_buf->slice(0, decoder_batch_size_);
        auto decode_fused_qkv_buf = fused_qkv_buf->slice(0, decoder_batch_size_);

        MlaRotaryWriteKVCacheParams decode_params = {
            *decode_q_buf,
            nullptr,
            *decode_fused_qkv_buf,
            kv_offset,
            decode_params_,
            attn_common_inputs,
            attn_layer_weight,
            attn_configs,
            QScheme::NoQuantize,
        };
        device_->mlaRotaryWriteKVCache(decode_params);
    }
    RTP_LLM_LOG_INFO("after run");
}

static auto _OP =
    torch::jit::class_<MlaRotaryKVCacheOp>("unittest", "MlaRotaryKVCacheOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("init", &MlaRotaryKVCacheOp::init)
        .def("applyRotaryKVCache", &MlaRotaryKVCacheOp::applyRotaryKVCache);

}

