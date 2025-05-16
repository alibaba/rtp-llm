#pragma once

#include <torch/torch.h>
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/devices/testing/TestBase.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"

#ifdef USING_CUDA12
#include "maga_transformer/cpp/devices/cuda_impl/CudaXqa.h"
#endif

using namespace std;
using namespace rtp_llm;

torch::Tensor rotate(const torch::Tensor& x) {
    torch::Tensor x1 = x.slice(-1, 0, x.size(-1) / 2);
    torch::Tensor x2 = x.slice(-1, x.size(-1) / 2, x.size(-1));
    return torch::cat({-x2, x1}, -1);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb(const torch::Tensor& q,
                                                          const torch::Tensor& k,
                                                          int cache_len,
                                                          int rope_theta,
                                                          int rope_dim) {
    auto inv_freq = 1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(cache_len, torch::kInt64).to(torch::kFloat32);
    auto freqs = torch::outer(t, inv_freq);
    torch::Tensor emb = torch::cat({freqs, freqs}, -1);
    torch::Tensor cos = emb.cos().to(torch::kFloat32);
    torch::Tensor sin = emb.sin().to(torch::kFloat32);

    torch::Tensor index = torch::tensor(cache_len - 1);
    auto cos_pos = cos.index_select(0, index).unsqueeze(0).unsqueeze(1);
    auto sin_pos = sin.index_select(0, index).unsqueeze(0).unsqueeze(1);
    auto q_rotate = rotate(q);
    auto k_rotate = rotate(k);
    auto q_emb = (q * cos_pos) + (q_rotate * sin_pos);
    auto k_emb = (k * cos_pos) + (k_rotate * sin_pos);

    return std::make_tuple(q_emb, k_emb);
}

struct AttentionImpl : torch::nn::Module {
    AttentionImpl() {
        // register_module() is needed if we want to use the parameters() method later on
        // register_module("rope", rope);
    }

    std::vector<torch::Tensor> forward(torch::Tensor& query_states,
                                       torch::Tensor& key_states,
                                       torch::Tensor& value_states,
                                       std::optional<torch::Tensor> attention_mask = std::nullopt,
                                       std::optional<torch::Tensor> k_cache = std::nullopt,
                                       std::optional<torch::Tensor> v_cache = std::nullopt,
                                       bool use_rope = false,
                                       int rope_theta = 1000000,
                                       int rope_dim = 128) {
        auto batch_size = query_states.size(0);
        auto seq_len = query_states.size(1);
        auto head_num = query_states.size(2);
        auto head_kv_num = key_states.size(2);
        auto head_dim = query_states.size(3);

        auto q = query_states.transpose(1, 2);
        auto k = key_states.transpose(1, 2);
        auto v = value_states.transpose(1, 2);

        if (use_rope) {
            int cache_len = k_cache.has_value() ? ((*k_cache).size(2) + k.size(2)) : k.size(2);
            std::tie(q, k) = apply_rotary_emb(q, k, cache_len, rope_theta, rope_dim);
        }

        if (k_cache.has_value() && v_cache.has_value()) {
            k = torch::cat({*k_cache, k}, 2);
            v = torch::cat({*v_cache, v}, 2);
        }

        auto kv_seq_len = k.size(2);
        if (head_num > head_kv_num) {
            k = k.reshape({batch_size, head_kv_num, 1, kv_seq_len, head_dim}).expand({-1, -1, head_num / head_kv_num, -1, -1});
            v = v.reshape({batch_size, head_kv_num, 1, kv_seq_len, head_dim}).expand({-1, -1, head_num / head_kv_num, -1, -1});
            k = k.reshape({batch_size, head_num, kv_seq_len, head_dim});
            v = v.reshape({batch_size, head_num, kv_seq_len, head_dim});
        }
        auto attn_weights = torch::matmul(q, k.transpose(2, 3));
        if (attention_mask.has_value()) {
            attention_mask = attention_mask->view({batch_size, 1, seq_len, kv_seq_len});
        } else {
            attention_mask = torch::zeros({batch_size, 1, seq_len, kv_seq_len});
        }
        auto scores  = torch::softmax(
                (attn_weights / sqrtf(head_dim * 1.0f) + *attention_mask), -1);

        auto output = torch::matmul(scores, v);
        auto transpose_output = output.transpose(1, 2);
        return {q, k, v, attn_weights, scores, output, transpose_output};
    }
};
TORCH_MODULE(Attention);

class AttentionOpTest: public DeviceTestBase {
public:


    void contextAttentionOpTest(size_t batch_size,
                                size_t seq_len,
                                size_t num_heads,
                                size_t num_key_value_heads,
                                size_t head_dim,
				                const QScheme qscheme = QScheme::NoQuantize);

    void selfAttentionOpTest(size_t batch_size,
                             size_t seq_len,
                             size_t kv_seq_len,
                             size_t num_heads,
                             size_t num_key_value_heads,
                             size_t head_dim);

#ifdef USING_CUDA12
    void xqaAttentionOpTest(size_t batch_size,
                            size_t seq_len,
                            size_t kv_seq_len,
                            size_t num_heads,
                            size_t num_key_value_heads,
                            size_t head_dim);
#endif
};


void AttentionOpTest::contextAttentionOpTest(size_t batch_size,
                                             size_t seq_len,
                                             size_t num_heads,
                                             size_t num_key_value_heads,
                                             size_t head_dim,
					                         const QScheme qscheme) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto qkv_states_host = torch::cat(
        {query_states_host, key_states_host, value_states_host}, 2);
    auto scale_host = torch::ones({1});

    qkv_states_host = qkv_states_host.view({(int)(batch_size * seq_len),
                                            (int)(num_heads + 2 * num_key_value_heads),
                                            (int)head_dim});

    const auto input_lengths = createBuffer<int32_t>(
        {batch_size}, std::vector<int32_t>(batch_size, seq_len));
    const auto sequence_lengths = createBuffer<int32_t>({0}, {});
    std::vector<int> cu_seqlens(batch_size + 1);
    for (int i = 0; i < batch_size + 1; i++) {
        cu_seqlens[i] = seq_len * i;
    }
    auto token_num = batch_size * seq_len;
    auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);
    auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    auto position_ids_host = torch::arange((int)token_num, int_tensor_options);
    auto bias_host = torch::zeros(
        {(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);
    auto attention_mask_host = torch::zeros(
        {(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);

    auto qkv_input_device = createDeviceBuffer<half>(qkv_states_host);

    auto bias_device            = createDeviceBuffer<half>(bias_host);
    auto position_ids_device    = createDeviceBuffer<int>(position_ids_host);
    auto padding_offset_device  = createDeviceBuffer<int>(padding_offset_host);
    auto cu_seqlens_device      = createDeviceBuffer<int>(cu_seqlens_host);
    auto attention_mask_device  = createDeviceBuffer<half>(attention_mask_host);
    auto scale_device           = createDeviceBuffer<float>(scale_host);
    auto rope_config            = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});

    auto common_inputs      = AttentionCommonInputs({input_lengths, sequence_lengths});
    common_inputs.cu_seqlens = move(cu_seqlens_device);
    common_inputs.cu_kv_seqlens = common_inputs.cu_seqlens;
    common_inputs.padding_offset = move(padding_offset_device);
    common_inputs.position_ids = position_ids_device;
    common_inputs.attention_mask = attention_mask_device;
    common_inputs.context_batch_size = batch_size;
    common_inputs.context_max_seq_len = seq_len;
    common_inputs.decoder_batch_size = 0;
    common_inputs.decoder_max_seq_len = 0;
    common_inputs.max_prefix_length = 0;

    auto buffer_nullptr = BufferPtr(nullptr);
    auto attention_weight   = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    attention_weight.static_scale_reciprocal_weight = make_shared<const DenseWeights>(DenseWeights(scale_device));

    auto attention_config   = AttentionConfigs({num_heads,
                                                num_key_value_heads,
                                                head_dim,
                                                num_heads * head_dim,
                                                rope_config});

    auto output_data_type = qscheme == QScheme::Qfp8PerTensor ? DataType::TYPE_FP8_E4M3 : qkv_input_device->type();
    auto qkv_output = device_->allocateBuffer(
        {output_data_type, {batch_size, seq_len, num_heads, head_dim}}
    );
    device_->contextAttention({0, *qkv_input_device,
                               *qkv_output,
                                common_inputs,
                                attention_weight,
                                attention_config,
                                qscheme});

    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host);

    auto result  = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()));
}

void AttentionOpTest::selfAttentionOpTest(size_t batch_size,
                                          size_t seq_len,
                                          size_t kv_seq_len,
                                          size_t num_heads,
                                          size_t num_key_value_heads,
                                          size_t head_dim) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto half_tensor_options = torch::TensorOptions(torch::kHalf).device(torch::Device(torch::kCPU));
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);


    auto qkv_states_host = torch::cat(
        {query_states_host.view({(int)batch_size * (int)seq_len , (int)num_heads, (int)head_dim}),
         key_states_host.view({(int)batch_size * (int)seq_len , (int)num_key_value_heads, (int)head_dim}),
         value_states_host.view({(int)batch_size * (int)seq_len , (int)num_key_value_heads, (int)head_dim})}, 1);


    std::vector<int> sequence_lengths(batch_size);
    std::vector<int> input_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        sequence_lengths[i] = kv_seq_len + seq_len - 1;
        input_lengths[i]    = kv_seq_len;
    }
    size_t step = *std::max_element(sequence_lengths.begin(), sequence_lengths.end());
    step = step + 1;
    auto sequence_lengths_host = torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);

    size_t tokensPerBlock = 8;

    size_t padding_kv_seq_len = ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) * tokensPerBlock;
    padding_kv_seq_len = (kv_seq_len == 0) ? 2 * tokensPerBlock : padding_kv_seq_len;
    auto kvcache_pad = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
        half_tensor_options);

    auto k_cache_host = kvcache_pad.index(
        {0, torch::indexing::Slice(), 0, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()}
    ).reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim}).transpose(1, 2).contiguous().clone();

    auto v_cache_host = kvcache_pad.index(
        {0, torch::indexing::Slice(), 1, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()}
    ).reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim}).transpose(1, 2).contiguous().clone();

    auto attention_mask_host = torch::zeros(
        {(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto bias_host = torch::zeros(
        {(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);


    auto attention_mask_device      = createDeviceBuffer<float>(attention_mask_host);
    auto bias_device                = createDeviceBuffer<half>(bias_host);
    auto qkv_states_device          = createDeviceBuffer<half>(qkv_states_host);
    auto sequence_lengths_device    = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device       = createDeviceBuffer<int>(input_lengths_host);

    auto rope_config = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num = 2 * batch_size * ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) + 1;
    rtp_llm::CacheConfig cache_conf(rtp_llm::KVCacheParam({1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokensPerBlock, DataType::TYPE_FP16}));
    cache_manager_ = nullptr;
    auto kv_cache_block_id = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    auto kv_cache_buffer = cache_manager_->kvCacheBuffer();
    auto common_inputs = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);
    common_inputs.kv_cache = KvCacheInfo({(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;
    common_inputs.max_prefix_length = 0;

    auto buffer_nullptr = BufferPtr(nullptr);
    auto attention_weight = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    auto token_num = batch_size * seq_len;

    auto attention_config   = AttentionConfigs({num_heads,
                                                num_key_value_heads,
                                                head_dim,
                                                num_heads * head_dim,
                                                rope_config,
                                                tokensPerBlock});

    auto qkv_output = device_->allocateBuffer(
        {qkv_states_device->type(), {token_num, num_heads, head_dim}}
    );
    device_->decoderSelfAttention({0, *qkv_states_device,
                                    *qkv_output,
                                    common_inputs,
                                    attention_weight,
                                    attention_config});

    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host,
                                         k_cache_host,
                                         v_cache_host);

    auto result  = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 1e-2);
}

#ifdef USING_CUDA12
void AttentionOpTest::xqaAttentionOpTest(size_t batch_size,
                                         size_t seq_len,
                                         size_t kv_seq_len,
                                         size_t num_heads,
                                         size_t num_key_value_heads,
                                         size_t head_dim) {
    if (!USE_INPUT_KV || !ROPE_STYLE || ROPE_STYLE != 1) {
        throw std::runtime_error("need to enable USE_INPUT_KV and set ROPE_STYLE to 1 in file 3rdparty/xqa/defines.h");
    }
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    CudaDevice* device = dynamic_cast<CudaDevice*>(device_);

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto bf16_tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    auto fp8_tensor_options = torch::TensorOptions(torch::kFloat8_e4m3fn).device(torch::Device(torch::kCPU));

    auto query_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host = torch::rand(
        {(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto qkv_states_host = torch::cat(
        {query_states_host.view({(int)batch_size * (int)seq_len , (int)num_heads, (int)head_dim}),
         key_states_host.view({(int)batch_size * (int)seq_len , (int)num_key_value_heads, (int)head_dim}),
         value_states_host.view({(int)batch_size * (int)seq_len , (int)num_key_value_heads, (int)head_dim})}, 1);

    std::vector<int> sequence_lengths(batch_size);
    std::vector<int> input_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        sequence_lengths[i] = kv_seq_len + seq_len - 1;
        input_lengths[i]    = kv_seq_len;
    }
    size_t step = *std::max_element(sequence_lengths.begin(), sequence_lengths.end());
    step = step + 1;
    auto sequence_lengths_host = torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);

    size_t tokens_per_block = TOKENS_PER_PAGE;

    size_t padding_kv_seq_len = ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) * tokens_per_block;
    padding_kv_seq_len = (kv_seq_len == 0) ? 2 * tokens_per_block : padding_kv_seq_len;
    auto kvcache_pad = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
        bf16_tensor_options);

    auto kvcache_pad_fp8 = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
        fp8_tensor_options);

    auto k_cache_host = kvcache_pad.index(
        {0, torch::indexing::Slice(), 0, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()}
    ).reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim}).transpose(1, 2).contiguous().clone();

    auto v_cache_host = kvcache_pad.index(
        {0, torch::indexing::Slice(), 1, torch::indexing::Slice(0, kv_seq_len), torch::indexing::Slice()}
    ).reshape({(int)batch_size, (int)kv_seq_len, (int)num_key_value_heads, (int)head_dim}).transpose(1, 2).contiguous().clone();

    auto attention_mask_host = torch::zeros(
        {(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto bias_host = torch::zeros(
        {(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);

    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_host);
    auto bias_device             = createDeviceBuffer<__nv_bfloat16>(bias_host);
    auto qkv_states_device       = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);

    auto rope_config = RopeConfig({RopeStyle::Base, (int)head_dim, 1000000, 1., 0., 0., 40960});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num = 2 * batch_size * ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    rtp_llm::CacheConfig cache_conf(
        rtp_llm::KVCacheParam({1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_FP8_E4M3})
    );
    cache_manager_ = nullptr;
    auto kv_cache_block_id = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad_fp8);
    auto kv_cache_buffer = cache_manager_->kvCacheBuffer();
    auto common_inputs = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_k_cache_buffer = kv_cache_buffer.k_blocks->index(0);
    auto layer_v_cache_buffer = kv_cache_buffer.v_blocks->index(0);
    common_inputs.kv_cache = KvCacheInfo({(int)kv_cache_buffer.k_blocks->shape()[0], kv_cache_block_id, layer_k_cache_buffer, layer_v_cache_buffer});
    common_inputs.context_batch_size = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;
    common_inputs.max_prefix_length = 0;

    auto buffer_nullptr = BufferPtr(nullptr);
    auto attention_weight = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    auto token_num = batch_size * seq_len;

    auto attention_config = AttentionConfigs({num_heads,
                                              num_key_value_heads,
                                              head_dim,
                                              num_heads * head_dim,
                                              rope_config,
                                              tokens_per_block});

    auto qkv_output = device->allocateBuffer(
        {qkv_states_device->type(), {token_num, num_heads, head_dim}}
    );

    AttentionModuleParams params = {0, 
                                    *qkv_states_device,
                                    *qkv_output,
                                    common_inputs,
                                    attention_weight,
                                    attention_config};

    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
    auto kv_cache_page_List = device->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE}, {"kv_cache_page_List"}
    );
    KVBlockArray kv_block_array = device->getKVBlockArray(params, *kv_cache_page_List, batch_size, false);

    runXqa(params.input.data(),
           params.output.data(),
           num_heads,
           num_key_value_heads,
           batch_size,
           kv_seq_len,
           tokens_per_block,
           kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>(const_cast<KVCacheIndex*>(kv_block_array.data)),
           reinterpret_cast<uint32_t*>(params.common.sequence_lengths->data()),
           device);

    device->syncAndCheck();

    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host,
                                         k_cache_host,
                                         v_cache_host,
                                         true);

    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 5e-2);
}
#endif
