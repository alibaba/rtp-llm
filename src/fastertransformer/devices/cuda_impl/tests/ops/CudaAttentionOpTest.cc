#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaAttentionOpTest: public DeviceTestBase<DeviceType::Cuda> {
public:

};

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(size_t dim, size_t max_position_embeddings=2048, float base=10000) :
        dim(dim), max_position_embeddings(max_position_embeddings), base(base) {}

    torch::Tensor rotate_half(torch::Tensor x) {
        // Rotates half the hidden dims of the input.
        int half_size = x.sizes()[3] / 2;
        auto x1 = x.index({"...", at::indexing::Slice(0, half_size)});
        auto x2 = x.index({"...", at::indexing::Slice(half_size-1, -1)});
        auto result = torch::cat({-x2, x1}, -1);
        return result;
    }

    torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor position_ids) {
        int seq_len = position_ids.sizes()[0];
        auto inv_freq = 1.0 / torch::pow(base, (torch::arange(0, (int)dim, 2).to(torch::kFloat) / (int)dim));
        auto t = torch::arange(seq_len);
        auto freqs = torch::outer(t, inv_freq);
        auto emb = torch::cat({freqs, freqs}, -1);
        auto cos = emb.cos();
        auto sin = emb.sin();
        auto embed = (hidden_states * cos) + (rotate_half(hidden_states) * sin);
        return embed;
    }

    size_t dim;
    size_t max_position_embeddings;
    float base;

};

TORCH_MODULE(RotaryEmbedding);

struct AttentionImpl : torch::nn::Module {
    AttentionImpl(int batch_size,
                  int head_num,
                  int head_kv_num,
                  int head_dim,
                  int seq_len,
                  int dim) :
        rope(dim), head_kv_num(head_kv_num),
        batch_size(batch_size), seq_len(seq_len), head_num(head_num), head_dim(head_dim)
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("rope", rope);
    }


    std::vector<torch::Tensor> forward(torch::Tensor& query_states,
                                       torch::Tensor& key_states,
                                       torch::Tensor& value_states,
                                       torch::Tensor& attention_mask) {

        // std::cout << "batch_size is " << batch_size << std::endl;
        // std::cout << "seq_len is " << seq_len << std::endl;
        // std::cout << "head_num is " << head_num << std::endl;
        // std::cout << "head_kv_num is " << head_kv_num << std::endl;
        // std::cout << "head_dim is " << head_dim << std::endl;
        auto q = query_states.view({batch_size, seq_len, head_num, head_dim}).transpose(1, 2);
        auto k = key_states.view({batch_size, seq_len, head_kv_num, head_dim}).transpose(1, 2);
        auto v = value_states.view({batch_size, seq_len, head_kv_num, head_dim}).transpose(1, 2);

        auto position_ids = torch::arange(seq_len);
        // q = rope->forward(q, position_ids);
        // k = rope->forward(k, position_ids);
        // std::cout << "k shape is " << k.sizes() << "\n";
        // std::cout << "q shape is " << q.sizes() << "\n";

        auto attn_weights = torch::matmul(q, k.transpose(2, 3));
        attention_mask = attention_mask.view({batch_size, 1, seq_len, seq_len});
        auto scores  = torch::softmax(
            (attn_weights / sqrtf(head_dim * 1.0f) + attention_mask), -1);

        auto output = torch::matmul(scores, v);
        auto transpose_output = output.transpose(1, 2);
        return {q, k, v, attn_weights, scores, output, transpose_output};
    }

    RotaryEmbedding rope;
    int batch_size;
    int seq_len;
    int head_num;
    int head_dim;
    int head_kv_num;
};
TORCH_MODULE(Attention);


TEST_F(CudaAttentionOpTest, ContextAttentionOpTest) {
    // size_t batch_size = 64;
    // size_t seq_len = 100;
    // size_t num_heads = 32;
    // size_t num_key_value_heads = num_heads;
    // size_t head_dim = 32;
    // size_t dim = head_dim;
    // size_t token_num = batch_size * seq_len;

    // Attention attention(batch_size,
    //                     num_heads,
    //                     num_key_value_heads,
    //                     head_dim,
    //                     seq_len,
    //                     head_dim);
    // attention.ptr()->to(torch::Device(torch::kCPU));
    // auto state_dict = attention.ptr()->named_parameters();
    // torch::NoGradGuard no_grad;

    // auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    // auto int_tensor_options = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    // auto query_states_host = torch::rand(
    //     {(int)token_num, (int)num_heads, (int)head_dim}, tensor_options);

    // auto key_states_host = torch::rand(
    //     {(int)token_num, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    // auto value_states_host = torch::rand(
    //     {(int)token_num, (int)num_key_value_heads, (int)head_dim}, tensor_options);

    // auto qkv_states_host = torch::cat(
    //     {query_states_host, key_states_host, value_states_host}, 1);

    // auto q_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_heads, (int)seq_len, (int)head_dim},
    //     tensor_options
    // );
    // auto k_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_key_value_heads, (int)seq_len, (int)head_dim},
    //     tensor_options
    // );
    // auto v_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_key_value_heads, (int)seq_len, (int)head_dim},
    //     tensor_options
    // );

    // std::vector<int> cu_seqlens(batch_size);
    // for (int i = 0; i < batch_size; i++) {
    //     cu_seqlens[i] = seq_len * (i + 1);
    // }

    // auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size}, int_tensor_options);
    // auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    // auto position_ids_host = torch::arange((int)token_num, int_tensor_options);
    // auto bias_host = torch::zeros(
    //     {(int)((num_heads + 2 * num_key_value_heads) * head_dim)}, tensor_options);
    // auto attention_mask_host = torch::zeros(
    //     {(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);

    // auto qkv_input_device = CreateDeviceBuffer<half>(qkv_states_host);
    // auto q_output_device = CreateDeviceBuffer<half>(q_output_host);
    // auto k_output_device = CreateDeviceBuffer<half>(k_output_host);
    // auto v_output_device = CreateDeviceBuffer<half>(v_output_host);
    // auto bias_device = CreateDeviceBuffer<half>(bias_host);
    // auto position_ids_device = CreateDeviceBuffer<int>(position_ids_host);
    // auto padding_offset_device = CreateDeviceBuffer<int>(padding_offset_host);
    // auto cu_seqlens_device = CreateDeviceBuffer<int>(cu_seqlens_host);
    // auto attention_mask_device = CreateDeviceBuffer<float>(attention_mask_host);
    // auto rope_config = RopeConfig({RopeType::NOROPE, dim, 10000, 1, 2048, 1, 1});
    // // tmp
    // auto qk_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_heads, (int)seq_len, (int)seq_len},
    //     tensor_options);
    // auto qk_output_device = CreateDeviceBuffer<float>(qk_output_host);

    // auto qk_softmax_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_heads, (int)seq_len, (int)seq_len},
    //     tensor_options);
    // auto qk_softmax_output_device = CreateDeviceBuffer<half>(qk_softmax_output_host);

    // auto qkv_output_host = torch::zeros(
    //     {(int)batch_size, (int)num_heads, (int)seq_len, (int)head_dim},
    //     tensor_options);
    // auto qkv_output_device = CreateDeviceBuffer<half>(qkv_output_host);

    // auto qkv_transpose_output_host = torch::zeros(
    //     {(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim},
    //     tensor_options);
    // auto qkv_transpose_output_device = CreateDeviceBuffer<half>(qkv_transpose_output_host);


    // device_->contextAttention({*qkv_input_device,
    //                            *q_output_device,
    //                            *k_output_device,
    //                            *v_output_device,
    //                            *bias_device,
    //                            *position_ids_device,
    //                            *padding_offset_device,
    //                            *cu_seqlens_device,
    //                            *attention_mask_device,
    //                            rope_config,
    //                            *qk_output_device,
    //                            *qk_softmax_output_device,
    //                            *qkv_output_device,
    //                            *qkv_transpose_output_device});

    // auto result = attention->forward(query_states_host,
    //                                  key_states_host,
    //                                  value_states_host,
    //                                  attention_mask_host);

    // auto q  = CreateTensor(*q_output_device);
    // auto k  = CreateTensor(*k_output_device);
    // auto v  = CreateTensor(*v_output_device);
    // auto qk  = CreateTensor(*qk_output_device);
    // auto qk_softmax  = CreateTensor(*qk_softmax_output_device);
    // auto qkv_out  = CreateTensor(*qkv_output_device);
    // auto qkv_transpose_out  = CreateTensor(*qkv_transpose_output_device);

    // // std::cout << "qkv_out is " << qk << std::endl;
    // // std::cout << "result[5] is " << result[5] << std::endl;
    // ASSERT_TRUE(torch::allclose(q, result[0], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(k, result[1], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(v, result[2], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(qk, result[3], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(qk_softmax, result[4], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(qkv_out, result[5], rtol_, atol_));
    // ASSERT_TRUE(torch::allclose(qkv_transpose_out, result[6], rtol_, atol_));


}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
