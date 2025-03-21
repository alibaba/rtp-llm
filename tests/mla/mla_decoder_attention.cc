#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
using namespace fastertransformer;

namespace unittest {

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
                          torch::Tensor kc_t_weight,
                          torch::Tensor vc_t_weight,
                          torch::Tensor ckv_cache,
                          torch::Tensor kpe_caches,
                          torch::Tensor sequence_length_t,
                          torch::Tensor kvcache_block_id,
                          int64_t       page_size);
    c10::intrusive_ptr<FlashInferParams> createFlashInferParams(torch::Tensor sequence_length,
                                                                torch::Tensor input_length,
                                                                int64_t       page_size,
                                                                torch::Tensor block_id_map);
    AttentionConfigs                     attn_configs = AttentionConfigs({});
    DeviceBase*                          device_;
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
    
    auto gpt_params = GptInitParameter();
    gpt_params.mla_ops_type_ = MlaOpsType(mla_ops_type);
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

c10::intrusive_ptr<FlashInferParams> MlaDecoderAttnOp::createFlashInferParams(torch::Tensor sequence_length,
                                                                              torch::Tensor input_length,
                                                                              int64_t       page_size,
                                                                              torch::Tensor block_id_map) {
    attn_configs.tokens_per_block = page_size;
    auto params                   = FlashInferAttnParams::prepareFlashInferAttnParams(device_,
                                                                    attn_configs,
                                                                    torchTensor2Buffer(sequence_length),
                                                                    torchTensor2Buffer(input_length),
                                                                    torchTensor2Buffer(block_id_map),
                                                                    DataType::TYPE_FP16);
    auto flash_infer_attn_params  = (FlashInferAttnParams*)params.get();
    return c10::make_intrusive<FlashInferParams>(flash_infer_attn_params->total_batch_indices_t,
                                                 flash_infer_attn_params->total_positions_t,
                                                 flash_infer_attn_params->total_kv_last_page_len_1_t,
                                                 flash_infer_attn_params->total_page_indptr_t,
                                                 flash_infer_attn_params->total_page_indices_t);
}

torch::Tensor MlaDecoderAttnOp::forward(torch::Tensor q,
                                        torch::Tensor kc_t_weight,
                                        torch::Tensor vc_t_weight,
                                        torch::Tensor ckv_cache,
                                        torch::Tensor kpe_caches,
                                        torch::Tensor sequence_length_t,
                                        torch::Tensor kvcache_block_id,
                                        int64_t       page_size) {
    try {
        attn_configs.tokens_per_block     = page_size;
        BufferPtr sequence_lengths_host   = torchTensor2Buffer(sequence_length_t);
        BufferPtr kvcache_block_id_host   = torchTensor2Buffer(kvcache_block_id);
        auto      flash_infer_attn_params = FlashInferAttnParams::prepareFlashInferAttnParams(device_,
                                                                                         attn_configs,
                                                                                         sequence_lengths_host,
                                                                                         sequence_lengths_host,
                                                                                         kvcache_block_id_host,
                                                                                         DataType::TYPE_BF16);

        size_t token_num     = q.size(0);
        auto   q_b           = torchTensor2Buffer(q);
        auto   kc_weight_b   = torchTensor2Buffer(kc_t_weight);
        auto   vc_t_weight_b = torchTensor2Buffer(vc_t_weight);

        std::vector<int32_t> cu_seqlens_data(1 + 1, 0);
        cu_seqlens_data[0] = 0;
        cu_seqlens_data[1] = token_num;

        auto q_buffer = torchTensor2Buffer(q);

        auto qkv_output = device_->allocateBuffer(
            {q_buffer->type(), {token_num, attn_configs.head_num, attn_configs.nope_head_dim + attn_configs.rope_head_dim}}, {"output"});

        auto kc_dense_weight = std::make_shared<DenseWeights>(kc_weight_b);
        auto vc_dense_weight = std::make_shared<DenseWeights>(vc_t_weight_b);

        auto k_cache_buffer = torchTensor2Buffer(ckv_cache);
        auto v_cache_buffer = torchTensor2Buffer(kpe_caches);

        auto attn_layer_weight                = AttentionLayerWeights();
        attn_layer_weight.kc_weight           = kc_dense_weight;
        attn_layer_weight.vc_weight           = vc_dense_weight;
        auto attn_common_inputs               = AttentionCommonInputs();
        attn_common_inputs.context_batch_size = 0;
        attn_common_inputs.decoder_batch_size = q.size(0);
        attn_common_inputs.kv_cache =
            std::make_optional<KvCacheInfo>({1, nullptr, k_cache_buffer, v_cache_buffer, nullptr, nullptr});
        attn_common_inputs.flash_infer_attn_params = flash_infer_attn_params;

        auto mla_params = MlaDecoderAttentionParams{
            0, *q_buffer, qkv_output, attn_common_inputs, attn_layer_weight, attn_configs, QScheme::NoQuantize};
        device_->mlaDecoderSelfAttention(mla_params);

        auto output_t = Buffer2torchTensorWithStride(*qkv_output, {    (int64_t)token_num, (int64_t)attn_configs.head_num, (int64_t)attn_configs.v_head_dim}, 0);
        return output_t.detach().clone();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        abort();
    }
}

}  // namespace unittest

static auto FlashInferParamsReg =
    torch::jit::class_<unittest::FlashInferParams>("unittest", "FlashInferParams")
        .def(torch::jit::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())
        .def_readwrite("batch_indices", &unittest::FlashInferParams::batch_indices)
        .def_readwrite("positions", &unittest::FlashInferParams::positions)
        .def_readwrite("kv_last_page_len", &unittest::FlashInferParams::kv_last_page_len)
        .def_readwrite("page_indptr", &unittest::FlashInferParams::page_indptr)
        .def_readwrite("page_indices", &unittest::FlashInferParams::page_indices);

static auto MlaDecoderAttnOp =
    torch::jit::class_<unittest::MlaDecoderAttnOp>("unittest", "MlaDecoderAttnOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("createFlashInferParams", &unittest::MlaDecoderAttnOp::createFlashInferParams)
        .def("forward", &unittest::MlaDecoderAttnOp::forward);