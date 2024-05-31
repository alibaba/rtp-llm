#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#define private public
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include "maga_transformer/cpp/cache/CacheConfig.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaAttentionOpTest: public DeviceTestBase {
public:

    void contextAttentionOpTest(size_t batch_size,
                                size_t seq_len,
                                size_t num_heads,
                                size_t num_key_value_heads,
                                size_t head_dim);

    void selfAttentionOpTest(size_t batch_size,
                             size_t seq_len,
                             size_t kv_seq_len,
                             size_t num_heads,
                             size_t num_key_value_heads,
                             size_t head_dim);
};

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
                                       std::optional<torch::Tensor> v_cache = std::nullopt) {

        auto batch_size = query_states.size(0);
        auto seq_len = query_states.size(1);
        auto head_num = query_states.size(2);
        auto head_kv_num = key_states.size(2);
        auto head_dim = query_states.size(3);

        auto q = query_states.transpose(1, 2);
        auto k = key_states.transpose(1, 2);
        auto v = value_states.transpose(1, 2);

        if (k_cache.has_value() && v_cache.has_value()) {
            k = torch::cat({k, *k_cache}, 2);
            v = torch::cat({v, *v_cache}, 2);
        }

        auto kv_seq_len = k.size(2);

        // auto position_ids = torch::arange(seq_len);
        // q = rope->forward(q, position_ids);
        // k = rope->forward(k, position_ids);
        // std::cout << "k shape is " << k.sizes() << "\n";
        // std::cout << "q shape is " << q.sizes() << "\n";

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

    // RotaryEmbedding rope;
};
TORCH_MODULE(Attention);

void CudaAttentionOpTest::contextAttentionOpTest(size_t batch_size,
                                                 size_t seq_len,
                                                 size_t num_heads,
                                                 size_t num_key_value_heads,
                                                 size_t head_dim) {
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
    auto rope_config            = RopeConfig({RopeType::NOROPE, head_dim, 10000, 1, 2048, 1, 1});

    auto common_inputs      = AttentionCommonInputs({*input_lengths, *sequence_lengths});
    common_inputs.cu_seqlens = move(cu_seqlens_device);
    common_inputs.padding_offset = move(padding_offset_device);
    common_inputs.position_ids = position_ids_device;
    common_inputs.attention_mask = attention_mask_device;
    common_inputs.context_batch_size = batch_size;
    common_inputs.context_max_seq_len = seq_len;
    common_inputs.decoder_batch_size = 0;
    common_inputs.decoder_max_seq_len = 0;

    auto buffer_nullptr = BufferPtr(nullptr);
    auto attention_weight   = AttentionLayerWeights(std::make_unique<const DenseWeights>(
                                                    DenseWeights(buffer_nullptr, bias_device)));

    auto attention_config   = AttentionConfigs({num_heads,
                                                num_key_value_heads,
                                                head_dim,
                                                rope_config});

    auto qkv_output = device_->allocateBuffer(
        {qkv_input_device->type(), {batch_size, seq_len, num_heads, head_dim}}
    );
    device_->contextAttention({*qkv_input_device,
                               *qkv_output,
                                common_inputs,
                                attention_weight,
                                attention_config});

    auto result_ref = attention->forward(query_states_host,
                                         key_states_host,
                                         value_states_host,
                                         attention_mask_host);

    auto result  = bufferToTensor(*qkv_output);
    assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()));
}

void CudaAttentionOpTest::selfAttentionOpTest(size_t batch_size,
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
    
    size_t padding_kv_seq_len = ((kv_seq_len + tokensPerBlock) / tokensPerBlock) * tokensPerBlock;
    auto kvcache_pad = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_kv_seq_len, (int)num_key_value_heads * (int)head_dim},
        tensor_options);

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

    auto rope_config = RopeConfig({RopeType::NOROPE, head_dim, 10000, 1, 2048, 1, 1});
    
    rtp_llm::CacheConfig cache_conf(1, 4096, num_heads, head_dim, tokensPerBlock, DataType::TYPE_FP16);
    auto kv_cache = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);

    auto common_inputs = AttentionCommonInputs(*input_lengths_device, *sequence_lengths_device);
    common_inputs.kv_cache_blocks = *kv_cache;
    common_inputs.context_batch_size = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;

    auto buffer_nullptr = BufferPtr(nullptr);
    auto attention_weight = AttentionLayerWeights(std::make_unique<const DenseWeights>(
                                                    DenseWeights(buffer_nullptr, bias_device)));


    auto token_num = batch_size * seq_len;

    auto attention_config   = AttentionConfigs({num_heads,
                                                num_key_value_heads,
                                                head_dim,
                                                rope_config,
                                                tokensPerBlock});

    auto qkv_output = device_->allocateBuffer(
        {qkv_states_device->type(), {token_num, num_heads, head_dim}}
    );
    device_->decoderSelfAttention({*qkv_states_device,
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


TEST_F(CudaAttentionOpTest, SelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    std::vector<size_t> batch = {2, 4, 8};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_heads = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim = 64;
                selfAttentionOpTest(batch_size,
                                    seq_len,
                                    kv_seq_len,
                                    num_heads,
                                    num_key_value_heads,
                                    head_dim);
            }
        }
    }
}

TEST_F(CudaAttentionOpTest, ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_openSource_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            size_t dim = head_dim;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(CudaAttentionOpTest, OpenSourceFMHAContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_openSource_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            size_t dim = head_dim;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}


TEST_F(CudaAttentionOpTest, TrtV2ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "ON", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_openSource_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            size_t dim = head_dim;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(CudaAttentionOpTest, TrtV1ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_openSource_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            size_t dim = head_dim;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

