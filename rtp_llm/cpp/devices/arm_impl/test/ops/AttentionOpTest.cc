#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <torch/torch.h>

#include <chrono>
#include <omp.h>

using namespace std;
using namespace rtp_llm;

class ArmAttentionOpTest: public DeviceTestBase {
public:
    void contextAttentionOpTest(
        size_t batch_size, size_t seq_len, size_t num_heads, size_t num_key_value_heads, size_t head_dim);

    void selfAttentionOpTest(size_t batch_size,
                             size_t seq_len,
                             size_t kv_seq_len,
                             size_t num_heads,
                             size_t num_key_value_heads,
                             size_t head_dim);
};

struct AttentionImpl: torch::nn::Module {
    AttentionImpl() {}

    std::vector<torch::Tensor> forward(torch::Tensor&               query_states,
                                       torch::Tensor&               key_states,
                                       torch::Tensor&               value_states,
                                       std::optional<torch::Tensor> attention_mask = std::nullopt,
                                       std::optional<torch::Tensor> k_cache        = std::nullopt,
                                       std::optional<torch::Tensor> v_cache        = std::nullopt) {

        auto batch_size = query_states.size(0);
        auto seq_len    = query_states.size(1);
        auto head_dim   = query_states.size(3);

        auto q = query_states.transpose(1, 2);
        auto k = key_states.transpose(1, 2);
        auto v = value_states.transpose(1, 2);

        if (k_cache.has_value() && v_cache.has_value()) {
            k = torch::cat({*k_cache, k}, 2);
            v = torch::cat({*v_cache, v}, 2);
        }

        auto kv_seq_len = k.size(2);

        auto attn_weights = torch::matmul(q, k.transpose(2, 3));
        if (attention_mask.has_value()) {
            attention_mask = attention_mask->view({batch_size, 1, seq_len, kv_seq_len});
        } else {
            attention_mask = torch::zeros({batch_size, 1, seq_len, kv_seq_len});
        }

        auto scores = torch::softmax((attn_weights / sqrtf(head_dim * 1.0f) + *attention_mask), -1);

        auto output = torch::matmul(scores, v);

        auto transpose_output = output.transpose(1, 2);

        return {q, k, v, attn_weights, scores, output, transpose_output};
    }
};
TORCH_MODULE(Attention);

void ArmAttentionOpTest::contextAttentionOpTest(
    size_t batch_size, size_t seq_len, size_t num_heads, size_t num_key_value_heads, size_t head_dim) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto tensor_options     = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, tensor_options);

    auto key_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto value_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    auto qkv_states_host = torch::cat({query_states_host, key_states_host, value_states_host}, 2);

    qkv_states_host =
        qkv_states_host.view({(int)(batch_size * seq_len), (int)(num_heads + 2 * num_key_value_heads), (int)head_dim});

    const auto input_lengths =
        createBuffer<int32_t>({batch_size}, std::vector<int32_t>(batch_size, seq_len), AllocationType::HOST);
    const auto       sequence_lengths = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    std::vector<int> cu_seqlens(batch_size);
    for (int i = 0; i < batch_size; i++) {
        cu_seqlens[i] = seq_len * (i + 1);
    }
    auto token_num           = batch_size * seq_len;
    auto cu_seqlens_host     = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size}, int_tensor_options);
    auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    auto position_ids_host   = torch::arange((int)token_num, int_tensor_options);
    auto bias_host           = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);
    /* Torch and device use different mask notations */
    auto attention_mask_host = torch::zeros({(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);
    auto attention_mask_     = torch::ones({(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);

    auto qkv_input_device = createDeviceBuffer<float>(qkv_states_host);

    auto bias_device           = createDeviceBuffer<float>(bias_host);
    auto position_ids_device   = createDeviceBuffer<int>(position_ids_host);
    auto padding_offset_device = createDeviceBuffer<int>(padding_offset_host);
    auto cu_seqlens_device     = createDeviceBuffer<int>(cu_seqlens_host);
    auto attention_mask_device = createDeviceBuffer<float>(attention_mask_);
    auto rope_config           = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});

    AttentionCommonInputs common_inputs({input_lengths, sequence_lengths});
    common_inputs.cu_seqlens          = move(cu_seqlens_device);
    common_inputs.padding_offset      = move(padding_offset_device);
    common_inputs.position_ids        = position_ids_device;
    common_inputs.attention_mask      = attention_mask_device;
    common_inputs.context_batch_size  = batch_size;
    common_inputs.context_max_seq_len = seq_len;
    common_inputs.decoder_batch_size  = 0;
    common_inputs.decoder_max_seq_len = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    AttentionConfigs attention_config;
    attention_config.head_num      = num_heads;
    attention_config.kv_head_num   = num_key_value_heads;
    attention_config.size_per_head = head_dim;
    attention_config.rope_config   = rope_config;

    auto qkv_output = device_->allocateBuffer({qkv_input_device->type(), {batch_size, seq_len, num_heads, head_dim}});
    auto result_ref = attention->forward(query_states_host, key_states_host, value_states_host, attention_mask_host);

    device_->contextAttention({0, *qkv_input_device, *qkv_output, common_inputs, attention_weight, attention_config});

    auto result = bufferToTensor(*qkv_output);
    // assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()));
    assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()), 1e-2, 1e-2);
}

void ArmAttentionOpTest::selfAttentionOpTest(size_t batch_size,
                                             size_t seq_len,
                                             size_t kv_seq_len,
                                             size_t num_heads,
                                             size_t num_key_value_heads,
                                             size_t head_dim) {
    Attention attention = Attention();
    attention.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = attention.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto tensor_options     = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
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
    auto   sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);

    size_t tokensPerBlock = 8;

    size_t padding_kv_seq_len = ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) * tokensPerBlock;
    padding_kv_seq_len        = (kv_seq_len == 0) ? 2 * tokensPerBlock : padding_kv_seq_len;
    auto kvcache_pad          = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim}, tensor_options);

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

    /* Torch and device use different mask notations */
    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);
    auto attention_mask_ = torch::ones({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto bias_host = torch::zeros({(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);

    auto attention_mask_device   = createDeviceBuffer<float>(attention_mask_);
    auto bias_device             = createDeviceBuffer<float>(bias_host);
    auto qkv_states_device       = createDeviceBuffer<float>(qkv_states_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);

    auto rope_config = RopeConfig({RopeStyle::No, (int)head_dim, 10000, 1, 2048, 1, 1});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num             = 2 * batch_size * ((kv_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1) + 1;
    auto cache_conf            = makeMhaCacheConfig(1,
                                         static_cast<uint>(block_num),
                                         static_cast<uint>(num_heads),
                                         static_cast<uint>(head_dim),
                                         static_cast<uint>(tokensPerBlock),
                                         rtp_llm::TYPE_FP32);
    cache_manager_             = nullptr;
    auto kv_cache_block_id     = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    auto kv_cache_buffer       = cache_manager_->allLayerCacheBase();
    auto layer_kv_cache_buffer = kv_cache_buffer.layers_to_kv_buffer_ptrs[0];
    auto common_inputs         = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    common_inputs.kv_cache     = KvCacheInfo{
        (int)cache_manager_->cacheConfig().layer_num, kv_cache_block_id, {}, layer_kv_cache_buffer, nullptr};
    common_inputs.context_batch_size  = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size  = batch_size;
    common_inputs.decoder_max_seq_len = step;
    common_inputs.max_prefix_length   = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, bias_device));

    AttentionConfigs attention_config;
    attention_config.head_num         = num_heads;
    attention_config.kv_head_num      = num_key_value_heads;
    attention_config.size_per_head    = head_dim;
    attention_config.rope_config      = rope_config;
    attention_config.tokens_per_block = tokensPerBlock;

    auto qkv_output = device_->allocateBuffer({qkv_states_device->type(), {batch_size, seq_len, num_heads, head_dim}});
    auto result_ref = attention->forward(
        query_states_host, key_states_host, value_states_host, attention_mask_host, k_cache_host, v_cache_host);

    device_->decoderSelfAttention(
        {0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config});

    auto result = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6].to(result.dtype()), result, 1e-2, 1e-2);
}

TEST_F(ArmAttentionOpTest, ContextAttentionOpTest) {
    std::vector<size_t> batch = {1, 2};
    std::vector<size_t> seq   = {1, 2};

    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 16;
            size_t num_key_value_heads = num_heads;
            size_t head_dim            = 128;

            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(ArmAttentionOpTest, SelfAttentionOpTest) {
    std::vector<size_t> batch  = {1, 2};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {0, 1, 2};

    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 16;
                size_t num_key_value_heads = num_heads;
                size_t head_dim            = 128;

                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}
