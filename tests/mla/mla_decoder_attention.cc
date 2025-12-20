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

using Slice          = torch::indexing::Slice;
constexpr auto TNone = torch::indexing::None;

class FlashInferParams: public torch::jit::CustomClassHolder {
public:
    FlashInferParams(torch::Tensor batch_indices,
                     torch::Tensor positions,
                     torch::Tensor kv_last_page_len,
                     torch::Tensor page_indptr,
                     torch::Tensor page_indices) {
        this->batch_indices    = batch_indices;
        this->positions        = positions;
        this->kv_last_page_len = kv_last_page_len;
        this->page_indptr      = page_indptr;
        this->page_indices     = page_indices;
    }

    torch::Tensor get_batch_indices() const {
        return batch_indices;
    }
    torch::Tensor get_positions() const {
        return positions;
    }
    torch::Tensor get_kv_last_page_len() const {
        return kv_last_page_len;
    }
    torch::Tensor get_page_indptr() const {
        return page_indptr;
    }
    torch::Tensor get_page_indices() const {
        return page_indices;
    }

    void set_batch_indices(torch::Tensor value) {
        batch_indices = value;
    }
    void set_positions(torch::Tensor value) {
        positions = value;
    }
    void set_kv_last_page_len(torch::Tensor value) {
        kv_last_page_len = value;
    }
    void set_page_indptr(torch::Tensor value) {
        page_indptr = value;
    }
    void set_page_indices(torch::Tensor value) {
        page_indices = value;
    }

public:
    torch::Tensor batch_indices;
    torch::Tensor positions;
    torch::Tensor kv_last_page_len;
    torch::Tensor page_indptr;
    torch::Tensor page_indices;
};

class MlaDecoderAttnOp: public torch::jit::CustomClassHolder {
public:
    MlaDecoderAttnOp(int64_t mla_ops_type,
                     int64_t head_num,
                     int64_t nope_head_dim,
                     int64_t rope_head_dim,
                     int64_t v_head_dim,
                     int64_t q_lora_rank,
                     int64_t kv_lora_rank,
                     int64_t hidden_size,
                     double  softmax_extra_scale);

    torch::Tensor forward(torch::Tensor q,
                          torch::Tensor fused_qkv,
                          int64_t       kv_offset,
                          torch::Tensor kc_t_weight,
                          torch::Tensor vc_t_weight,
                          torch::Tensor cos_sin_cache,
                          torch::Tensor ckv_cache,
                          torch::Tensor kpe_caches,
                          torch::Tensor sequence_length_t,
                          torch::Tensor kvcache_block_id,
                          int64_t       page_size);

    c10::intrusive_ptr<FlashInferParams> createDecodeFlashInferParams(torch::Tensor sequence_length,
                                                                      torch::Tensor input_length,
                                                                      int64_t       page_size,
                                                                      torch::Tensor block_id_map);
    c10::intrusive_ptr<FlashInferParams> createContextFlashInferParams(torch::Tensor prefix_length,
                                                                       torch::Tensor sequence_length,
                                                                       torch::Tensor input_length,
                                                                       int64_t       page_size,
                                                                       torch::Tensor block_id_map);

    AttentionConfigs attn_configs = AttentionConfigs({});
    DeviceBase*      device_;
};

MlaDecoderAttnOp::MlaDecoderAttnOp(int64_t mla_ops_type,
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
    model_config.mla_ops_type = MlaOpsType(mla_ops_type);
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

c10::intrusive_ptr<FlashInferParams> MlaDecoderAttnOp::createContextFlashInferParams(torch::Tensor prefix_length,
                                                                                     torch::Tensor sequence_length,
                                                                                     torch::Tensor input_length,
                                                                                     int64_t       page_size,
                                                                                     torch::Tensor block_id_map) {
    auto decode_batch_size        = sequence_length.sizes()[0];
    auto context_batch_size       = input_length.sizes()[0] - decode_batch_size;
    attn_configs.tokens_per_block = page_size;
#ifdef USING_ROCM
    auto params                  = FlashInferAttnParams::preparePrefillFlashInferAttnParams(device_,
                                                                           attn_configs,
                                                                           torchTensor2Buffer(prefix_length),
                                                                           torchTensor2Buffer(sequence_length),
                                                                           torchTensor2Buffer(input_length),
                                                                           torchTensor2Buffer(block_id_map),
                                                                           DataType::TYPE_FP16);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.get();
    return c10::make_intrusive<FlashInferParams>(flash_infer_attn_params->batch_indice_t,
                                                 flash_infer_attn_params->positions_t,
                                                 flash_infer_attn_params->paged_kv_last_page_len_1_t,
                                                 flash_infer_attn_params->page_indptr_t,
                                                 flash_infer_attn_params->page_indice_t);
#else
    auto params = FlashInferAttnParams::prepare(
        device_,
        attn_configs,
        torchTensor2Buffer(prefix_length),
        nullptr,
        torchTensor2Buffer(input_length)->slice(decode_batch_size, context_batch_size, false),
        torchTensor2Buffer(block_id_map)->slice(decode_batch_size, context_batch_size, false),
        torchTensor2Buffer(block_id_map.to("cuda"))->slice(decode_batch_size, context_batch_size, false),
        DataType::TYPE_FP16);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.get();
    return c10::make_intrusive<FlashInferParams>(flash_infer_attn_params->batch_indice_d,
                                                 flash_infer_attn_params->positions_d,
                                                 flash_infer_attn_params->paged_kv_last_page_len_d,
                                                 flash_infer_attn_params->page_indptr_d,
                                                 flash_infer_attn_params->page_indice_d);
#endif
}

c10::intrusive_ptr<FlashInferParams> MlaDecoderAttnOp::createDecodeFlashInferParams(torch::Tensor sequence_length,
                                                                                    torch::Tensor input_length,
                                                                                    int64_t       page_size,
                                                                                    torch::Tensor block_id_map) {
    attn_configs.tokens_per_block = page_size;
    auto decode_batch_size        = sequence_length.sizes()[0];
#ifdef USING_ROCM
    auto params                  = FlashInferAttnParams::prepareDecodeFlashInferAttnParams(device_,
                                                                          attn_configs,
                                                                          torchTensor2Buffer(sequence_length),
                                                                          torchTensor2Buffer(input_length),
                                                                          torchTensor2Buffer(block_id_map),
                                                                          DataType::TYPE_FP16);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.get();
    return c10::make_intrusive<FlashInferParams>(flash_infer_attn_params->batch_indice_t,
                                                 flash_infer_attn_params->positions_t,
                                                 flash_infer_attn_params->paged_kv_last_page_len_1_t,
                                                 flash_infer_attn_params->page_indptr_t,
                                                 flash_infer_attn_params->page_indice_t);
#else
    auto params =
        FlashInferAttnParams::prepare(device_,
                                      attn_configs,
                                      nullptr,
                                      torchTensor2Buffer(sequence_length)->slice(0, decode_batch_size, false),
                                      torchTensor2Buffer(input_length)->slice(0, decode_batch_size, false),
                                      torchTensor2Buffer(block_id_map)->slice(0, decode_batch_size, false),
                                      torchTensor2Buffer(block_id_map.to("cuda"))->slice(0, decode_batch_size, false),
                                      DataType::TYPE_FP16);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.get();
    return c10::make_intrusive<FlashInferParams>(flash_infer_attn_params->batch_indice_d,
                                                 flash_infer_attn_params->positions_d,
                                                 flash_infer_attn_params->paged_kv_last_page_len_d,
                                                 flash_infer_attn_params->page_indptr_d,
                                                 flash_infer_attn_params->page_indice_d);
#endif
}

torch::Tensor MlaDecoderAttnOp::forward(torch::Tensor q,
                                        torch::Tensor fused_qkv,
                                        int64_t       kv_offset,
                                        torch::Tensor kc_t_weight,
                                        torch::Tensor vc_t_weight,
                                        torch::Tensor cos_sin_cache,
                                        torch::Tensor ckv_cache,
                                        torch::Tensor kpe_caches,
                                        torch::Tensor sequence_length_t,
                                        torch::Tensor kvcache_block_id,
                                        int64_t       page_size) {
    try {
        attn_configs.tokens_per_block     = page_size;
        BufferPtr sequence_lengths_host   = torchTensor2Buffer(sequence_length_t);
        BufferPtr kvcache_block_id_host   = torchTensor2Buffer(kvcache_block_id);
        BufferPtr kvcache_block_id_device = device_->clone({*kvcache_block_id_host});
#ifdef USING_ROCM
        auto flash_infer_attn_params = FlashInferAttnParams::prepareDecodeFlashInferAttnParams(device_,
                                                                                               attn_configs,
                                                                                               sequence_lengths_host,
                                                                                               sequence_lengths_host,
                                                                                               kvcache_block_id_host,
                                                                                               DataType::TYPE_BF16);
#else
        auto flash_infer_attn_params = FlashInferAttnParams::prepare(device_,
                                                                     attn_configs,
                                                                     nullptr,
                                                                     sequence_lengths_host,
                                                                     sequence_lengths_host,
                                                                     kvcache_block_id_host,
                                                                     kvcache_block_id_device,
                                                                     DataType::TYPE_BF16);
#endif

        size_t token_num     = q.size(0);
        auto   kc_weight_b   = torchTensor2Buffer(kc_t_weight);
        auto   vc_t_weight_b = torchTensor2Buffer(vc_t_weight);

        std::vector<int32_t> cu_seqlens_data(1 + 1, 0);
        cu_seqlens_data[0] = 0;
        cu_seqlens_data[1] = token_num;

        auto q_buffer         = torchTensor2Buffer(q);
        auto fused_qkv_buffer = torchTensor2Buffer(fused_qkv);

        auto qkv_output = device_->allocateBuffer(
            {q_buffer->type(), {token_num, attn_configs.head_num, attn_configs.nope_head_dim}}, {"output"});

        auto kc_dense_weight = std::make_shared<DenseWeights>(kc_weight_b);
        auto vc_dense_weight = std::make_shared<DenseWeights>(vc_t_weight_b);

        auto k_cache_buffer = torchTensor2Buffer(ckv_cache);
        auto v_cache_buffer = torchTensor2Buffer(kpe_caches);

        auto cos_sin_cache_buffer = torchTensor2Buffer(cos_sin_cache);

        auto attn_layer_weight                = AttentionLayerWeights();
        attn_layer_weight.rope_cos_sin_cache  = cos_sin_cache_buffer;
        attn_layer_weight.kc_weight           = kc_dense_weight;
        attn_layer_weight.vc_weight           = vc_dense_weight;
        auto attn_common_inputs               = AttentionCommonInputs();
        attn_common_inputs.context_batch_size = 0;
        attn_common_inputs.decoder_batch_size = q.size(0);
        attn_common_inputs.kv_cache =
            std::make_optional<KvCacheInfo>({1, nullptr, kv_cache_buffer, nullptr});
        attn_common_inputs.decode_flash_infer_attn = flash_infer_attn_params;

        auto mla_params = MlaAttentionModuleParams{0,
                                                   *q_buffer,
                                                   *fused_qkv_buffer,
                                                   kv_offset,
                                                   qkv_output,
                                                   attn_common_inputs,
                                                   attn_layer_weight,
                                                   attn_configs,
                                                   QScheme::NoQuantize};
        device_->mlaAbsorbAttention(mla_params);

        auto output_t = Buffer2torchTensorWithStride(
            *qkv_output, {(int64_t)token_num, (int64_t)attn_configs.head_num, (int64_t)attn_configs.v_head_dim}, 0);
        return output_t.detach().clone();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        throw;
    }
}


static auto FlashInferParamsReg =
    torch::jit::class_<FlashInferParams>("unittest", "FlashInferParams")
        .def(torch::jit::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())
        .def_readwrite("batch_indices", &FlashInferParams::batch_indices)
        .def_readwrite("positions", &FlashInferParams::positions)
        .def_readwrite("kv_last_page_len", &FlashInferParams::kv_last_page_len)
        .def_readwrite("page_indptr", &FlashInferParams::page_indptr)
        .def_readwrite("page_indices", &FlashInferParams::page_indices);

static auto MlaDecoderAttnOpClass =
    torch::jit::class_<MlaDecoderAttnOp>("unittest", "MlaDecoderAttnOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("createDecodeFlashInferParams", &MlaDecoderAttnOp::createDecodeFlashInferParams)
        .def("createContextFlashInferParams", &MlaDecoderAttnOp::createContextFlashInferParams)
        .def("forward", &MlaDecoderAttnOp::forward);

} 

