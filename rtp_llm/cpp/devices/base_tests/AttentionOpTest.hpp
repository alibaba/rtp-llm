#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"

#ifdef USING_ROCM
#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#endif

#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#endif

using namespace std;
using namespace rtp_llm;

torch::Tensor rotate(const torch::Tensor& x) {
    torch::Tensor x1 = x.slice(-1, 0, x.size(-1) / 2);
    torch::Tensor x2 = x.slice(-1, x.size(-1) / 2, x.size(-1));
    return torch::cat({-x2, x1}, -1);
}

std::tuple<torch::Tensor, torch::Tensor>
apply_rotary_emb(const torch::Tensor& q, const torch::Tensor& k, int cache_len, int rope_theta, int rope_dim) {
    auto inv_freq =
        1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto          t     = torch::arange(cache_len, torch::kInt64).to(torch::kFloat32);
    auto          freqs = torch::outer(t, inv_freq);
    torch::Tensor emb   = torch::cat({freqs, freqs}, -1);
    torch::Tensor cos   = emb.cos().to(torch::kFloat32);
    torch::Tensor sin   = emb.sin().to(torch::kFloat32);

    torch::Tensor index    = torch::tensor(cache_len - 1);
    auto          cos_pos  = cos.index_select(0, index).unsqueeze(0).unsqueeze(1);
    auto          sin_pos  = sin.index_select(0, index).unsqueeze(0).unsqueeze(1);
    auto          q_rotate = rotate(q);
    auto          k_rotate = rotate(k);
    auto          q_emb    = (q * cos_pos) + (q_rotate * sin_pos);
    auto          k_emb    = (k * cos_pos) + (k_rotate * sin_pos);

    return std::make_tuple(q_emb, k_emb);
}

struct AttentionImpl: torch::nn::Module {
    AttentionImpl() {
        // register_module() is needed if we want to use the parameters() method later on
        // register_module("rope", rope);
    }

    std::vector<torch::Tensor> forward(torch::Tensor&               query_states,
                                       torch::Tensor&               key_states,
                                       torch::Tensor&               value_states,
                                       std::optional<torch::Tensor> attention_mask = std::nullopt,
                                       std::optional<torch::Tensor> k_cache        = std::nullopt,
                                       std::optional<torch::Tensor> v_cache        = std::nullopt,
                                       bool                         use_rope       = false,
                                       int                          rope_theta     = 1000000,
                                       int                          rope_dim       = 128) {
        auto batch_size  = query_states.size(0);
        auto seq_len     = query_states.size(1);
        auto head_num    = query_states.size(2);
        auto head_kv_num = key_states.size(2);
        auto head_dim    = query_states.size(3);

        auto q = query_states.transpose(1, 2);
        auto k = key_states.transpose(1, 2);
        auto v = value_states.transpose(1, 2);

        if (use_rope) {
            int cache_len  = k_cache.has_value() ? ((*k_cache).size(2) + k.size(2)) : k.size(2);
            std::tie(q, k) = apply_rotary_emb(q, k, cache_len, rope_theta, rope_dim);
        }

        if (k_cache.has_value() && v_cache.has_value()) {
            k = torch::cat({*k_cache, k}, 2);
            v = torch::cat({*v_cache, v}, 2);
        }

        auto kv_seq_len = k.size(2);
        if (head_num > head_kv_num) {
            k = k.reshape({batch_size, head_kv_num, 1, kv_seq_len, head_dim})
                    .expand({-1, -1, head_num / head_kv_num, -1, -1});
            v = v.reshape({batch_size, head_kv_num, 1, kv_seq_len, head_dim})
                    .expand({-1, -1, head_num / head_kv_num, -1, -1});
            k = k.reshape({batch_size, head_num, kv_seq_len, head_dim});
            v = v.reshape({batch_size, head_num, kv_seq_len, head_dim});
        }
        auto attn_weights = torch::matmul(q, k.transpose(2, 3));
        if (attention_mask.has_value()) {
            attention_mask = attention_mask->view({batch_size, 1, seq_len, kv_seq_len});
        } else {
            attention_mask = torch::zeros({batch_size, 1, seq_len, kv_seq_len});
        }
        auto scores = torch::softmax((attn_weights / sqrtf(head_dim * 1.0f) + *attention_mask), -1);
#ifdef USING_ROCM
        auto output = torch::matmul(scores.to(torch::kFloat32), v.to(torch::kFloat32));
#else
        auto output = torch::matmul(scores, v);
#endif
        auto transpose_output = output.transpose(1, 2);
        return {q, k, v, attn_weights, scores, output, transpose_output};
    }
};
TORCH_MODULE(Attention);

class AttentionOpTest: public DeviceTestBase {
public:
    void contextAttentionOpTest(size_t        batch_size,
                                size_t        seq_len,
                                size_t        num_heads,
                                size_t        num_key_value_heads,
                                size_t        head_dim,
                                const QScheme qscheme = QScheme::NoQuantize);

    void selfAttentionOpTest(size_t batch_size,
                             size_t seq_len,
                             size_t kv_seq_len,
                             size_t num_heads,
                             size_t num_key_value_heads,
                             size_t head_dim);
#ifdef USING_ROCM
    void aiterPageAttentionOpTest(size_t batch_size,
                                  size_t seq_len,
                                  size_t kv_seq_len,
                                  size_t num_heads,
                                  size_t num_key_value_heads,
                                  size_t head_dim);
#endif

#ifdef USING_CUDA12
    void xqaAttentionOpTest(size_t batch_size,
                            size_t seq_len,
                            size_t kv_seq_len,
                            size_t num_heads,
                            size_t num_key_value_heads,
                            size_t head_dim,
                            size_t tokens_per_block,
                            bool   is_kv_cache_fp8);

    void flashinferPrefillOpTest(size_t        batch_size,
                                 size_t        seq_len,
                                 size_t        kv_seq_len,
                                 size_t        num_heads,
                                 size_t        num_key_value_heads,
                                 size_t        head_dim,
                                 const QScheme qscheme = QScheme::NoQuantize);

    void xqaPrefillOpTest(size_t        batch_size,
                          size_t        seq_len,
                          size_t        kv_seq_len,
                          size_t        num_heads,
                          size_t        num_key_value_heads,
                          size_t        head_dim,
                          const QScheme qscheme = QScheme::NoQuantize);
#endif
};

void AttentionOpTest::contextAttentionOpTest(size_t        batch_size,
                                             size_t        seq_len,
                                             size_t        num_heads,
                                             size_t        num_key_value_heads,
                                             size_t        head_dim,
                                             const QScheme qscheme) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;
#ifdef USING_ROCM
    auto tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
#else
    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
#endif
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto qkv_states_host = torch::cat({query_states_host, key_states_host, value_states_host}, 2);
    auto scale_host      = torch::ones({1});

    qkv_states_host =
        qkv_states_host.view({(int)(batch_size * seq_len), (int)(num_heads + 2 * num_key_value_heads), (int)head_dim});

    const auto       input_lengths    = createBuffer<int32_t>({batch_size}, std::vector<int32_t>(batch_size, seq_len));
    const auto       sequence_lengths = createBuffer<int32_t>({0}, {});
    std::vector<int> cu_seqlens(batch_size + 1);
    std::vector<int> cu_seqlens_without_prefix(batch_size + 1);
    for (int i = 0; i < batch_size + 1; i++) {
        cu_seqlens[i]                = seq_len * i;
        cu_seqlens_without_prefix[i] = seq_len * i;
    }
    auto token_num       = batch_size * seq_len;
    auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);
    auto cu_seqlens_without_prefix_host =
        torch::from_blob((void*)cu_seqlens_without_prefix.data(), {(int)batch_size + 1}, int_tensor_options);
    auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    auto position_ids_host   = torch::arange((int)token_num, int_tensor_options);
    auto bias_host           = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);
    auto attention_mask_host = torch::zeros({(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);

    auto qkv_input_device = createDeviceBuffer<half>(qkv_states_host);

    auto bias_device                      = createDeviceBuffer<half>(bias_host);
    auto position_ids_device              = createDeviceBuffer<int>(position_ids_host);
    auto padding_offset_device            = createDeviceBuffer<int>(padding_offset_host);
    auto cu_seqlens_device                = createDeviceBuffer<int>(cu_seqlens_host);
    auto cu_seqlens_without_prefix_device = createDeviceBuffer<int>(cu_seqlens_without_prefix_host);
    auto attention_mask_device            = createDeviceBuffer<half>(attention_mask_host);
    auto scale_device                     = createDeviceBuffer<float>(scale_host);
#ifdef USING_ROCM
    auto rope_config = RopeConfig({RopeStyle::Base, (int)head_dim, 10000, 1, 2048, 1, 1});

    size_t tokensPerBlock = 16;
    int    block_num      = batch_size * ((seq_len + tokensPerBlock - 1) / tokensPerBlock + 1);
    auto   cache_conf     = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokensPerBlock, DataType::TYPE_BF16);
    auto kv_cache_block_id = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {batch_size, block_num / batch_size}, rtp_llm::AllocationType::HOST});

    cache_manager_ = std::make_shared<rtp_llm::KVCacheManager>(cache_conf, device_);
    ASSERT_TRUE(cache_manager_->init());
    auto kv_cache_buffer      = cache_manager_->kvCacheBuffer();
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);
    auto common_inputs        = AttentionCommonInputs({input_lengths, sequence_lengths});
    common_inputs.kv_cache    = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
#else
    auto rope_config   = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});
    auto common_inputs = AttentionCommonInputs({input_lengths, sequence_lengths});
#endif
    common_inputs.cu_seqlens                = move(cu_seqlens_device);
    common_inputs.cu_seqlens_without_prefix = move(cu_seqlens_without_prefix_device);
    common_inputs.cu_kv_seqlens             = common_inputs.cu_seqlens;
    common_inputs.padding_offset            = move(padding_offset_device);
    common_inputs.position_ids              = position_ids_device;
    common_inputs.attention_mask            = attention_mask_device;
    common_inputs.context_batch_size        = batch_size;
    common_inputs.context_max_seq_len       = seq_len;
    common_inputs.decoder_batch_size        = 0;
    common_inputs.decoder_max_seq_len       = 0;
    common_inputs.max_prefix_length         = 0;

    auto buffer_nullptr   = BufferPtr(nullptr);
    auto attention_weight = AttentionLayerWeights();
#ifdef USING_ROCM
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr));
#else
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));
#endif

    attention_weight.static_scale_reciprocal_weight = make_shared<const DenseWeights>(DenseWeights(scale_device));

    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;

    auto output_data_type = qscheme == QScheme::Qfp8PerTensor ? DataType::TYPE_FP8_E4M3 : qkv_input_device->type();
    auto qkv_output       = device_->allocateBuffer({output_data_type, {batch_size, seq_len, num_heads, head_dim}});
#ifdef USING_ROCM
    device_->initParamsRef().use_asm_pa  = true;
    device_->initParamsRef().max_seq_len = 150000;
    device_->contextAttention({0,
                               *qkv_input_device,
                               *qkv_output,
                               common_inputs,
                               attention_weight,
                               attention_config,
                               qscheme,
                               DataType::TYPE_INVALID});
    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host,
                                         std::nullopt,
                                         std::nullopt,
                                         true,
                                         rope_config.base,
                                         rope_config.dim);
#else
    device_->contextAttention(
        {0, *qkv_input_device, *qkv_output, common_inputs, attention_weight, attention_config, qscheme});
    auto result_ref = attention->forward(query_states_host, key_states_host, value_states_host, attention_mask_host);
#endif
    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()), 1e-2, 1e-2);
}

void AttentionOpTest::selfAttentionOpTest(size_t batch_size,
                                          size_t seq_len,
                                          size_t kv_seq_len,
                                          size_t num_heads,
                                          size_t num_key_value_heads,
                                          size_t head_dim) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;
#ifdef USING_ROCM
    auto tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
#else
    auto tensor_options      = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto half_tensor_options = torch::TensorOptions(torch::kHalf).device(torch::Device(torch::kCPU));
#endif
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto qkv_states_host =
        torch::cat({query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim}),
                    key_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim}),
                    value_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim})},
                   1);

    std::vector<int> sequence_lengths(batch_size);
    std::vector<int> input_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        sequence_lengths[i] = kv_seq_len + seq_len - 1;
        input_lengths[i]    = kv_seq_len;
    }
    size_t step = *std::max_element(sequence_lengths.begin(), sequence_lengths.end());
    step        = step + 1;
    auto sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);

#ifdef USING_ROCM
    size_t tokensPerBlock = 16;
#else
    size_t tokensPerBlock = 8;
#endif

    size_t padding_kv_seq_len = ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) * tokensPerBlock;
    padding_kv_seq_len        = (kv_seq_len == 0) ? 2 * tokensPerBlock : padding_kv_seq_len;
    auto kvcache_pad =
        torch::zeros({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
#ifdef USING_ROCM
                     tensor_options
#else
                     half_tensor_options
#endif
        );

    auto k_cache_host =
        kvcache_pad
            .index({0, torch::indexing::Slice(), 0, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()})
            .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
            .transpose(1, 2)
            .contiguous()
            .clone();

    auto v_cache_host =
        kvcache_pad
            .index({0, torch::indexing::Slice(), 1, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()})
            .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
            .transpose(1, 2)
            .contiguous()
            .clone();

    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto bias_host = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);

    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_host);
    auto bias_device             = createDeviceBuffer<half>(bias_host);
    auto qkv_states_device       = createDeviceBuffer<half>(qkv_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);
#ifdef USING_ROCM
    auto rope_config = RopeConfig({RopeStyle::Base, (int)head_dim, 10000, 1, 2048, 1, 1});
#else
    auto rope_config = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});
#endif

// cache manager need one block for preserve and every seq need one block for preserve.
#ifdef USING_ROCM
    auto block_num = batch_size * ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) + 1;
#else
    auto block_num = 2 * batch_size * ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) + 1;
#endif

    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokensPerBlock, DataType::TYPE_FP16);
    cache_manager_            = nullptr;
    auto kv_cache_block_id    = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    auto kv_cache_buffer      = cache_manager_->kvCacheBuffer();
    auto common_inputs        = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);

    common_inputs.kv_cache = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size  = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size  = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;
    common_inputs.max_prefix_length   = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    auto token_num = batch_size * seq_len;

    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;
    attention_config.tokens_per_block = tokensPerBlock;

#ifdef USING_CUDA12
    CudaDevice* device            = dynamic_cast<CudaDevice*>(device_);
    common_inputs.decode_trt_attn = device->prepareTrtAttn(
        attention_config, kv_cache_buffer.k_blocks, common_inputs.kv_cache->kv_cache_block_id, batch_size);
#endif

#ifdef USING_ROCM
    ROCmDevice* device = dynamic_cast<ROCmDevice*>(device_);
    common_inputs.decode_aiter_attn =
        AiterAttnParams::prepareDecodeAiterAttnParams(device, torchTensor2Buffer(sequence_lengths_host), attention_config, 0, common_inputs.kv_cache->kv_cache_block_id);
#endif

    auto qkv_output = device_->allocateBuffer({qkv_states_device->type(), {token_num, num_heads, head_dim}});
#ifdef USING_ROCM
    device_->initParamsRef().use_asm_pa  = true;
    device_->initParamsRef().max_seq_len = 150000;
    device_->decoderSelfAttention({0,
                                   *qkv_states_device,
                                   *qkv_output,
                                   common_inputs,
                                   attention_weight,
                                   attention_config,
                                   QScheme::NoQuantize,
                                   DataType::TYPE_INVALID});
    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host,
                                         k_cache_host,
                                         v_cache_host,
                                         true,
                                         rope_config.base,
                                         rope_config.dim);
#else
    device_->decoderSelfAttention(
        {0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config});
    auto result_ref = attention->forward(
        query_states_host, key_states_host, value_states_host, attention_mask_host, k_cache_host, v_cache_host);
#endif
    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 1e-2);
}

#ifdef USING_ROCM
void AttentionOpTest::aiterPageAttentionOpTest(size_t batch_size,
                                               size_t seq_len,
                                               size_t kv_seq_len,
                                               size_t num_heads,
                                               size_t num_key_value_heads,
                                               size_t head_dim) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;
    ROCmDevice*        device              = dynamic_cast<ROCmDevice*>(device_);
    auto               tensor_options      = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto               bf16_tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto               int_tensor_options  = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    // auto fp8_tensor_options = torch::TensorOptions(torch::kFloat8_e4m3fnuz).device(torch::Device(torch::kCPU));
    auto uint8_tensor_options = torch::TensorOptions(torch::kUInt8).device(torch::Device(torch::kCPU));
    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, bf16_tensor_options);

    auto qkv_states_host = query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim});
    std::vector<int> sequence_lengths(batch_size);
    std::vector<int> input_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        sequence_lengths[i] = kv_seq_len + seq_len - 1;
        input_lengths[i]    = kv_seq_len;
    }
    size_t step = *std::max_element(sequence_lengths.begin(), sequence_lengths.end());
    step        = step + 1;
    auto sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto   input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);
    size_t tokens_per_block   = 16;
    size_t padding_kv_seq_len = ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) * tokens_per_block;
    padding_kv_seq_len        = (kv_seq_len == 0) ? 2 * tokens_per_block : padding_kv_seq_len;
    auto kvcache_pad =
        torch::rand({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
                    bf16_tensor_options);
    auto kvcache_pad_fp8 =
        torch::zeros({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
                     uint8_tensor_options)
            .view(torch::kFloat8_e4m3fnuz);
    auto k_cache_host =
        kvcache_pad
            .index({0, torch::indexing::Slice(), 0, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()})
            .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
            .contiguous()
            .clone();
    auto v_cache_host =
        kvcache_pad
            .index({0, torch::indexing::Slice(), 1, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()})
            .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
            .contiguous()
            .clone();

    int  H_x       = ((int)head_dim) / 8;
    int  num_block = batch_size * kv_seq_len / tokens_per_block;
    auto k_cache_host_view =
        k_cache_host.reshape({num_block, (int)tokens_per_block, (int)num_key_value_heads, H_x, 8})
            .permute({0, 2, 3, 1, 4})
            .reshape({1, (int)batch_size, 1, (int)kv_seq_len, (int)num_key_value_heads * (int)head_dim});
    auto v_cache_host_view =
        v_cache_host.reshape({num_block, (int)tokens_per_block, (int)num_key_value_heads, (int)head_dim})
            .permute({0, 2, 3, 1})
            .reshape({1, (int)batch_size, 1, (int)kv_seq_len, (int)num_key_value_heads * (int)head_dim});
    kvcache_pad = torch::cat({k_cache_host_view, v_cache_host_view}, 2);
    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);
    auto bias_host = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);
    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_host);
    auto bias_device             = createDeviceBuffer<__nv_bfloat16>(bias_host);
    auto qkv_states_device       = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);
    // auto rope_config             = RopeConfig({RopeStyle::Base, (int)head_dim, 1000000, 1., 0., 0., 40960});
    auto rope_config = RopeConfig({RopeStyle::Base, (int)head_dim, 10000, 1, 2048, 1, 1});
    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_BF16);
    cache_manager_         = nullptr;
    auto kv_cache_block_id = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    // cache, kv_cache_block_id = [batch_size, xxx]
    auto kv_cache_buffer      = cache_manager_->kvCacheBuffer();
    auto common_inputs        = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);
    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;
    attention_config.tokens_per_block = tokens_per_block;
    common_inputs.kv_cache    = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size  = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size  = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;
    common_inputs.max_prefix_length   = 0;
    common_inputs.decode_aiter_attn =
        AiterAttnParams::prepareDecodeAiterAttnParams(device, torchTensor2Buffer(sequence_lengths_host), attention_config, 0, common_inputs.kv_cache->kv_cache_block_id);
    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));
    auto token_num              = batch_size * seq_len;
    auto qkv_output = device->allocateBuffer({qkv_states_device->type(), {token_num, num_heads, head_dim}});
    AttentionModuleParams params = {
        0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config};
    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
    auto       kv_cache_page_List =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                               {"kv_cache_page_List"});
    KVBlockArray kv_block_array = device->getKVBlockArray(params, *kv_cache_page_List, batch_size, false);

    runAiterAsmPA(params, device, *qkv_states_device);

    device->syncAndCheck();
    auto q_host_fp32 = query_states_host.to(tensor_options);
    auto k_host_fp32 = k_cache_host.to(tensor_options);
    auto v_host_fp32 = v_cache_host.to(tensor_options);
    auto result_ref  = attention->forward(q_host_fp32, k_host_fp32, v_host_fp32);

    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 5e-2);
}
#endif

#ifdef USING_CUDA12

torch::Tensor rand_n(at::IntArrayRef size, at::TensorOptions options = {}) {
    auto rand_int = torch::randint(-100, 100, size, options);
    return rand_int.to(torch::kBFloat16) / 100;
}

void AttentionOpTest::xqaAttentionOpTest(size_t batch_size,
                                         size_t seq_len,
                                         size_t kv_seq_len,
                                         size_t num_heads,
                                         size_t num_key_value_heads,
                                         size_t head_dim,
                                         size_t tokens_per_block,
                                         bool   is_kv_cache_fp8) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    CudaDevice* device = dynamic_cast<CudaDevice*>(device_);

    // auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto bf16_tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto int_tensor_options  = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    // auto fp8_tensor_options = torch::TensorOptions(torch::kFloat8_e4m3fn).device(torch::Device(torch::kCPU));

    std::vector<int> sequence_lengths(batch_size);
    std::vector<int> input_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        // actual input is qkv
        sequence_lengths[i] = kv_seq_len;
        input_lengths[i]    = kv_seq_len;
    }

    auto sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);

    size_t padding_kv_seq_len =
        ((kv_seq_len + seq_len + tokens_per_block - 1) / tokens_per_block + 1) * tokens_per_block;
    padding_kv_seq_len = (kv_seq_len == 0) ? 2 * tokens_per_block : padding_kv_seq_len;
    auto k_cache_host = torch::rand({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim},
                                    bf16_tensor_options);
    auto v_cache_host = torch::rand({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim},
                                    bf16_tensor_options);

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, bf16_tensor_options);
    auto key_states_host = k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                               .contiguous();
    auto value_states_host = v_cache_host
                                 .index({torch::indexing::Slice(),
                                         torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice()})
                                 .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                                 .contiguous();

    auto kvcache_pad =
        torch::cat({k_cache_host, v_cache_host}, 1)
            .reshape({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim});
    auto kvcache_pad_fp8 = kvcache_pad.to(torch::kFloat8_e4m3fn);

    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, bf16_tensor_options);
    auto bias_host = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, bf16_tensor_options);

    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_host);
    auto bias_device             = createDeviceBuffer<__nv_bfloat16>(bias_host);
    auto query_states_device     = createDeviceBuffer<__nv_bfloat16>(query_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);

    int  rope_dim                = static_cast<int>(head_dim);
    int  rope_theta              = 1000000;
    int  max_position_embeddings = 10240;
    auto rope_config = RopeConfig({RopeStyle::Base, rope_dim, rope_theta, 1., 0., 0., max_position_embeddings});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((kv_seq_len + seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = is_kv_cache_fp8 ? makeMhaCacheConfig(1,
                                                           (uint)block_num,
                                                           (uint)num_key_value_heads,
                                                           (uint)head_dim,
                                                           (uint)tokens_per_block,
                                                           DataType::TYPE_FP8_E4M3) :
                                        makeMhaCacheConfig(1,
                                                           (uint)block_num,
                                                           (uint)num_key_value_heads,
                                                           (uint)head_dim,
                                                           (uint)tokens_per_block,
                                                           DataType::TYPE_BF16);
    cache_manager_  = nullptr;
    auto kv_cache_block_id    = is_kv_cache_fp8 ? allocateKVBlocks(cache_conf, input_lengths, kvcache_pad_fp8, false) :
                                                  allocateKVBlocks(cache_conf, input_lengths, kvcache_pad, false);
    auto kv_cache_buffer      = cache_manager_->kvCacheBuffer();
    auto common_inputs        = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);
    common_inputs.kv_cache    = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size  = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size  = batch_size;
    common_inputs.decoder_max_seq_len = kv_seq_len + seq_len;
    common_inputs.max_prefix_length   = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    auto token_num = batch_size * seq_len;

    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;
    attention_config.tokens_per_block = tokens_per_block;

    auto qkv_output = device->allocateBuffer({query_states_device->type(), {token_num, num_heads, head_dim}});

    AttentionModuleParams params = {
        0, *query_states_device, *qkv_output, common_inputs, attention_weight, attention_config};

    auto attn = device->prepareTrtAttn(
        attention_config, kv_cache_buffer.k_blocks, params.common.kv_cache->kv_cache_block_id, batch_size);
    auto trt_attn = reinterpret_cast<TRTAttn*>(attn.get());
    TRTAttn::setKvCache(trt_attn->kv_block_array, *common_inputs.kv_cache);

    runXqa(params.input.data(),
           true,
           params.output.data(),
           num_heads,
           num_key_value_heads,
           head_dim,
           batch_size,
           static_cast<size_t>(trt_attn->kv_block_array.mMaxBlocksPerSeq),
           kv_seq_len + seq_len,
           tokens_per_block,
           trt_attn->kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>(const_cast<KVCacheIndex*>(trt_attn->kv_block_array.data)),
           is_kv_cache_fp8,
           reinterpret_cast<uint32_t*>(params.common.sequence_lengths->data()));

    device->syncAndCheck();

    auto result_ref =
        attention->forward(query_states_host,
                           key_states_host,
                           value_states_host,
                           attention_mask_host,
                           k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone(),
                           v_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone());

    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 5e-2);
}

void AttentionOpTest::flashinferPrefillOpTest(size_t        batch_size,
                                              size_t        seq_len,
                                              size_t        kv_seq_len,
                                              size_t        num_heads,
                                              size_t        num_key_value_heads,
                                              size_t        head_dim,
                                              const QScheme qscheme) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;
    CudaDevice*        device             = dynamic_cast<CudaDevice*>(device_);
    auto               tensor_options     = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto               int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    torch::manual_seed(114514);
    auto query_states_host = rand_n({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);
    std::vector<int> input_lengths(batch_size);
    std::vector<int> prefix_lengths(batch_size);
    std::vector<int> kv_seq_lengths(batch_size);
    std::vector<int> sequence_lengths;
    std::vector<int> cu_seqlens(batch_size + 1, 0);
    std::vector<int> cu_seqlens_without_prefix(batch_size + 1, 0);
    for (int i = 0; i < batch_size; i++) {
        input_lengths[i]                 = seq_len;
        prefix_lengths[i]                = kv_seq_len;
        kv_seq_lengths[i]                = kv_seq_len + seq_len;
        cu_seqlens[i + 1]                = seq_len * (i + 1);
        cu_seqlens_without_prefix[i + 1] = seq_len * (i + 1);
    }
    std::vector<int> positions(batch_size, kv_seq_len);

    auto input_lengths_host  = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto prefix_lengths_host = torch::from_blob((void*)prefix_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto cu_seqlens_host     = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);
    auto cu_seqlens_without_prefix_host =
        torch::from_blob((void*)cu_seqlens_without_prefix.data(), {(int)batch_size + 1}, int_tensor_options);
    auto   sequence_lengths_host = torch::from_blob((void*)sequence_lengths.data(), {(int)0}, int_tensor_options);
    size_t tokens_per_block      = 64;
    size_t padding_kv_seq_len = ((kv_seq_len + seq_len + tokens_per_block - 1) / tokens_per_block) * tokens_per_block;
    padding_kv_seq_len        = (kv_seq_len == 0) ? 1 * tokens_per_block : padding_kv_seq_len;
    torch::manual_seed(1145);
    auto k_cache_host =
        rand_n({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);
    torch::manual_seed(11);
    auto v_cache_host =
        rand_n({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);
    torch::manual_seed(666666);
    auto key_states_host = k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                               .contiguous();
    auto value_states_host = v_cache_host
                                 .index({torch::indexing::Slice(),
                                         torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice()})
                                 .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                                 .contiguous();
    auto qkv_states_host =
        torch::cat({query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim}),
                    key_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim}),
                    value_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim})},
                   1);
    auto kvcache_pad =
        torch::cat({k_cache_host, v_cache_host}, 1)
            .reshape({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim})
            .to(torch::kBFloat16);
    auto kvcache_pad_fp8 = kvcache_pad.to(torch::kFloat8_e4m3fn);

    auto token_num           = batch_size * seq_len;
    auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    auto position_ids_host   = torch::from_blob((void*)positions.data(), {(int)batch_size}, int_tensor_options);
    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto attention_mask_device            = createDeviceBuffer<float>(attention_mask_host);
    auto qkv_states_device                = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto query_states_device              = createDeviceBuffer<__nv_bfloat16>(query_states_host);
    auto sequence_lengths_device          = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device             = createDeviceBuffer<int>(input_lengths_host);
    auto cu_seqlens_device                = createDeviceBuffer<int>(cu_seqlens_host);
    auto cu_seqlens_without_prefix_device = createDeviceBuffer<int>(cu_seqlens_without_prefix_host);
    auto padding_offset_device            = createDeviceBuffer<int>(padding_offset_host);
    auto position_ids_device              = createDeviceBuffer<int>(position_ids_host);
    auto rope_config                      = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});
    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_BF16);
    cache_manager_         = nullptr;
    auto kv_cache_block_id = allocateKVBlocks(cache_conf, kv_seq_lengths, kvcache_pad, false);
    auto kv_cache_buffer   = cache_manager_->kvCacheBuffer();
    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;
    attention_config.tokens_per_block = tokens_per_block;
    attention_config.kv_cache_dtype = KvCacheDataType::BASE;
    attention_config.skip_append_kv_cache   = true;
    BufferPtr        prefix_lengths_buf     = tensorToBuffer(prefix_lengths_host, AllocationType::HOST);
    BufferPtr        sequence_lengths_buf   = tensorToBuffer(sequence_lengths_host, AllocationType::HOST);
    BufferPtr        input_lengths_buf      = tensorToBuffer(input_lengths_host, AllocationType::HOST);
    BufferPtr        kv_cache_block_id_d    = device->clone({*kv_cache_block_id, AllocationType::DEVICE});
    DevicePrepOutput prep_output            = device->prepareModelRun({attention_config,
                                                                       prefix_lengths_buf,
                                                                       sequence_lengths_buf,
                                                                       input_lengths_buf,
                                                                       kv_cache_block_id,
                                                                       kv_cache_block_id_d,
                                                                       kv_cache_buffer.k_blocks,
                                                                       DataType::TYPE_BF16,
                                                                       batch_size,
                                                                       0,
                                                                       true,
                                                                       false});
    const auto       const_sequence_lengths = createBuffer<int32_t>({0}, {});
    auto             common_inputs          = AttentionCommonInputs({input_lengths_device, const_sequence_lengths});
    auto             layer_k_cache_buffer   = kv_cache_buffer.k_blocks->index(0);
    auto             layer_v_cache_buffer   = kv_cache_buffer.v_blocks->index(0);
    common_inputs.kv_cache                  = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.cu_seqlens                = cu_seqlens_device;
    common_inputs.cu_seqlens_without_prefix = cu_seqlens_without_prefix_device;
    common_inputs.padding_offset            = padding_offset_device;
    common_inputs.position_ids              = position_ids_device;
    common_inputs.attention_mask            = attention_mask_device;
    common_inputs.context_batch_size        = batch_size;
    common_inputs.context_max_seq_len       = seq_len;
    common_inputs.decoder_batch_size        = 0;
    common_inputs.decoder_max_seq_len       = 0;
    common_inputs.max_prefix_length         = 0;
    common_inputs.decode_flash_infer_attn.swap(prep_output.decode_flash_infer_attn);
    common_inputs.prefill_flash_infer_attn.swap(prep_output.prefill_flash_infer_attn);
    common_inputs.prefill_trt_attn.swap(prep_output.prefill_trt_attn);
    common_inputs.decode_trt_attn.swap(prep_output.decode_trt_attn);
    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr));
    auto qkv_output = device->allocateBuffer({query_states_device->type(), {token_num, num_heads, head_dim}});
    device->contextAttention(
        {0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config, qscheme});
    auto result_ref =
        attention->forward(query_states_host,
                           key_states_host,
                           value_states_host,
                           attention_mask_host,
                           k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone(),
                           v_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone());
    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 1e-2);
}

void AttentionOpTest::xqaPrefillOpTest(size_t        batch_size,
                                       size_t        seq_len,
                                       size_t        kv_seq_len,
                                       size_t        num_heads,
                                       size_t        num_key_value_heads,
                                       size_t        head_dim,
                                       const QScheme qscheme) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;
    CudaDevice*        device             = dynamic_cast<CudaDevice*>(device_);
    auto               tensor_options     = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto               int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    torch::manual_seed(114514);
    auto query_states_host = rand_n({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);
    std::vector<int> input_lengths(batch_size);
    std::vector<int> prefix_lengths(batch_size);
    std::vector<int> kv_seq_lengths(batch_size);
    std::vector<int> sequence_lengths;
    std::vector<int> cu_seqlens(batch_size + 1, 0);
    for (int i = 0; i < batch_size; i++) {
        input_lengths[i]  = seq_len;
        prefix_lengths[i] = kv_seq_len;
        kv_seq_lengths[i] = kv_seq_len + seq_len;
        cu_seqlens[i + 1] = seq_len * (i + 1);
    }
    auto   input_lengths_host  = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto   prefix_lengths_host = torch::from_blob((void*)prefix_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto   cu_seqlens_host     = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);
    auto   sequence_lengths_host = torch::from_blob((void*)sequence_lengths.data(), {(int)0}, int_tensor_options);
    size_t tokens_per_block      = 64;
    size_t padding_kv_seq_len = ((kv_seq_len + seq_len + tokens_per_block - 1) / tokens_per_block) * tokens_per_block;
    padding_kv_seq_len        = (kv_seq_len == 0) ? 1 * tokens_per_block : padding_kv_seq_len;
    torch::manual_seed(1145);
    auto k_cache_host =
        rand_n({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);
    torch::manual_seed(11);
    auto v_cache_host =
        rand_n({(int)batch_size, (int)padding_kv_seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);
    torch::manual_seed(666666);
    auto key_states_host = k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                               .contiguous();
    auto value_states_host = v_cache_host
                                 .index({torch::indexing::Slice(),
                                         torch::indexing::Slice(kv_seq_len, kv_seq_len + seq_len),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice()})
                                 .reshape({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim})
                                 .contiguous();
    auto qkv_states_host =
        torch::cat({query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim}),
                    key_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim}),
                    value_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim})},
                   1);
    auto kvcache_pad =
        torch::cat({k_cache_host, v_cache_host}, 1)
            .reshape({1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim})
            .to(torch::kBFloat16);
    auto kvcache_pad_fp8 = kvcache_pad.to(torch::kFloat8_e4m3fn);
    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);
    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_host);
    auto qkv_states_device       = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto query_states_device     = createDeviceBuffer<__nv_bfloat16>(query_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);
    auto rope_config             = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});
    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_FP8_E4M3);
    cache_manager_         = nullptr;
    auto kv_cache_block_id = allocateKVBlocks(cache_conf, kv_seq_lengths, kvcache_pad_fp8, false);
    auto kv_cache_buffer   = cache_manager_->kvCacheBuffer();
    AttentionConfigs attention_config;
    attention_config.head_num = num_heads;
    attention_config.kv_head_num = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config = rope_config;
    attention_config.tokens_per_block = tokens_per_block;
    attention_config.kv_cache_dtype = KvCacheDataType::FP8;
    attention_config.skip_append_kv_cache   = true;
    BufferPtr        prefix_lengths_buf     = tensorToBuffer(prefix_lengths_host, AllocationType::HOST);
    BufferPtr        sequence_lengths_buf   = tensorToBuffer(sequence_lengths_host, AllocationType::HOST);
    BufferPtr        input_lengths_buf      = tensorToBuffer(input_lengths_host, AllocationType::HOST);
    BufferPtr        kv_cache_block_id_d    = device->clone({*kv_cache_block_id, AllocationType::DEVICE});
    DevicePrepOutput prep_output            = device->prepareModelRun({attention_config,
                                                                       prefix_lengths_buf,
                                                                       sequence_lengths_buf,
                                                                       input_lengths_buf,
                                                                       kv_cache_block_id,
                                                                       kv_cache_block_id_d,
                                                                       kv_cache_buffer.k_blocks,
                                                                       DataType::TYPE_FP8_E4M3,
                                                                       batch_size,
                                                                       0,
                                                                       true,
                                                                       false});
    const auto       const_sequence_lengths = createBuffer<int32_t>({0}, {});
    auto             common_inputs          = AttentionCommonInputs({input_lengths_device, const_sequence_lengths});
    auto             layer_k_cache_buffer   = kv_cache_buffer.k_blocks->index(0);
    auto             layer_v_cache_buffer   = kv_cache_buffer.v_blocks->index(0);
    common_inputs.kv_cache                  = KvCacheInfo(
        {(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size  = batch_size;
    common_inputs.context_max_seq_len = seq_len;
    common_inputs.decoder_batch_size  = 0;
    common_inputs.decoder_max_seq_len = 0;
    common_inputs.max_prefix_length   = kv_seq_len;
    common_inputs.decode_flash_infer_attn.swap(prep_output.decode_flash_infer_attn);
    common_inputs.prefill_flash_infer_attn.swap(prep_output.prefill_flash_infer_attn);
    common_inputs.prefill_trt_attn.swap(prep_output.prefill_trt_attn);
    common_inputs.decode_trt_attn.swap(prep_output.decode_trt_attn);
    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr));
    auto token_num              = batch_size * seq_len;
    auto qkv_output = device->allocateBuffer({query_states_device->type(), {token_num, num_heads, head_dim}});
    device->contextAttention(
        {0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config, qscheme});
    auto result_ref =
        attention->forward(query_states_host,
                           key_states_host,
                           value_states_host,
                           attention_mask_host,
                           k_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone(),
                           v_cache_host
                               .index({torch::indexing::Slice(),
                                       torch::indexing::Slice(0, kv_seq_len),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice()})
                               .reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim})
                               .transpose(1, 2)
                               .contiguous()
                               .clone());
    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 1e-2);
}
#endif
