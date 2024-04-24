#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

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
        {batch_size}, std::vector<int32_t>(batch_size, seq_len), AllocationType::HOST);
    const auto sequence_lengths = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    std::vector<int> cu_seqlens(batch_size);
    for (int i = 0; i < batch_size; i++) {
        cu_seqlens[i] = seq_len * (i + 1);
    }
    auto token_num = batch_size * seq_len;
    auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size}, int_tensor_options);
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
    auto attention_mask_device  = createDeviceBuffer<float>(attention_mask_host);
    auto rope_config            = RopeConfig({RopeType::NOROPE, head_dim, 10000, 1, 2048, 1, 1});

    auto common_inputs      = AttentionCommonInputs({*input_lengths, *sequence_lengths});
    common_inputs.cu_seqlens = move(cu_seqlens_device);
    common_inputs.padding_offset = move(padding_offset_device);
    common_inputs.position_ids = *position_ids_device;
    common_inputs.attention_mask = *attention_mask_device;
    common_inputs.context_batch_size = batch_size;
    common_inputs.context_max_seq_len = seq_len;
    common_inputs.decoder_batch_size = 0;
    common_inputs.decoder_max_seq_len = 0;

    auto buffer_nullptr = unique_ptr<Buffer>(nullptr);
    auto attention_weight   = AttentionLayerWeights(std::make_unique<const DenseWeights>(
                                                    DenseWeights(buffer_nullptr, bias_device)));

    auto attention_config   = AttentionConfigs({num_heads,
                                                num_key_value_heads,
                                                head_dim,
                                                rope_config});

    auto qkv_output = device_->allocateBuffer(
        {qkv_input_device->type(), {token_num, num_heads + 2 * num_key_value_heads, head_dim}}
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
    // assertTensorClose(result_ref[6], result.to(result_ref[6].dtype()));
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

    auto k_cache_host = torch::zeros(
        {(int)batch_size, (int)num_key_value_heads, (int)kv_seq_len, (int)head_dim}, tensor_options);

    auto v_cache_host = torch::zeros(
        {(int)batch_size, (int)num_key_value_heads, (int)kv_seq_len, (int)head_dim}, tensor_options);

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

    size_t tokensPerBlock = 4;
    size_t maxBlocksPerSeq = ((kv_seq_len + seq_len + 1) % tokensPerBlock == 0) ?
                             ((kv_seq_len + seq_len + 1) / tokensPerBlock) :
                             ((kv_seq_len + seq_len + 1) / tokensPerBlock) + 1;

    // k, v tensor shape is [batch_size, head_kv_size, kv_seq_len, head_dim].
    // split tensor to small tensor which shape is [head_size, tokensPerBlock, head_dim].
    // and the tensor map is [block_size, 2, block_num]

    EXPECT_GE(maxBlocksPerSeq * tokensPerBlock, kv_seq_len + seq_len);
    EXPECT_EQ(kv_seq_len % tokensPerBlock, 0);
    auto k_tensor = k_cache_host.view({(int)batch_size,
                                    (int)num_key_value_heads,
                                    (int)(kv_seq_len / tokensPerBlock),
                                    (int)tokensPerBlock,
                                    (int)head_dim});
    k_tensor = k_tensor.transpose(1, 2);

    auto v_tensor = v_cache_host.view({(int)batch_size,
                                    (int)num_key_value_heads,
                                    (int)(kv_seq_len / tokensPerBlock),
                                    (int)tokensPerBlock,
                                    (int)head_dim});
    v_tensor = v_tensor.transpose(1, 2);

    std::vector<void*> block_pointers(batch_size * 2 * maxBlocksPerSeq, nullptr);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < maxBlocksPerSeq; j++) {
            if (j < (int)(kv_seq_len / tokensPerBlock)) {
                auto k_tmp = k_tensor.index({i, j, "..."});
                auto v_tmp = v_tensor.index({i, j, "..."});
                auto k_buffer = createDeviceBuffer<half>(k_tmp);
                auto v_buffer = createDeviceBuffer<half>(v_tmp);
                block_pointers[i * maxBlocksPerSeq * 2 + j] = k_buffer->data();
                block_pointers[i * maxBlocksPerSeq * 2 + maxBlocksPerSeq + j] = v_buffer->data();
            } else {
                auto k_tmp = torch::zeros({1, 1, (int)num_key_value_heads, (int)tokensPerBlock, (int)head_dim});
                auto v_tmp = torch::zeros({1, 1, (int)num_key_value_heads, (int)tokensPerBlock, (int)head_dim});
                auto k_buffer = createDeviceBuffer<half>(k_tmp);
                auto v_buffer = createDeviceBuffer<half>(v_tmp);
                block_pointers[i * maxBlocksPerSeq * 2 + j] = k_buffer->data();
                block_pointers[i * maxBlocksPerSeq * 2 + maxBlocksPerSeq + j] = v_buffer->data();
            }
        }
    }
    for (auto ptr : block_pointers) {
        EXPECT_NE(ptr, nullptr);
    }


    auto kv_cache = device_->allocateBuffer(
        {DataType::TYPE_UINT64, {(size_t)batch_size, maxBlocksPerSeq}, AllocationType::HOST}, {});

    std::memcpy(kv_cache->data(), block_pointers.data(), block_pointers.size() * sizeof(void*));

    auto common_inputs = AttentionCommonInputs(*input_lengths_device, *sequence_lengths_device);
    common_inputs.kv_cache_blocks = *kv_cache;
    common_inputs.context_batch_size = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size = batch_size;
    common_inputs.decoder_max_seq_len = step - 1;

    auto buffer_nullptr = unique_ptr<Buffer>(nullptr);
    auto attention_weight   = AttentionLayerWeights(std::make_unique<const DenseWeights>(
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
    std::vector<size_t> batch = {1};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {16};
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
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 100};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 32;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 32;
            size_t dim = head_dim;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

