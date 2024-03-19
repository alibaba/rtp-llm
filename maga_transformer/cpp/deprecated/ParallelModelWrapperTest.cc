
#include "gtest/gtest.h"
#include <memory>

#define private public
#include "maga_transformer/cpp/common/cuda_resources.h"
#include "maga_transformer/cpp/components/ParallelModelWrapper.h"
#include "maga_transformer/cpp/test/utils/test_base.h"
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/core/Tensor.h"

#include <chrono>
#include <thread>

using namespace std;
// using namespace torch;
using namespace rtp_llm;

namespace rtp_llm {

class ModelTest: public TestBase {
protected:
    void SetUp() override {}

    void TearDown() override {}

protected:
};

template<typename T>
void write_tensor(torch::Tensor& t, const vector<T> values)
{
    T* base = t.data_ptr<T>();
    for (auto i = 0; i < values.size(); ++i) {
        *(base + i) = values[i];
    }
}

TEST_F(ModelTest, testSample)
{
    GptInitParameter params(2, 64, 2, 20, 20);
    params.head_num_kv_ = 2;

    const size_t inter_size    = 512;
    params.inter_size_         = inter_size;
    params.inter_padding_size_ = inter_size;

    typedef half         T;
    const at::ScalarType scalar_type = at::ScalarType::Half;
    const DataType       data_type   = getTensorType<T>();
    ft::IAllocator*      allocator_;
    auto&                cuda_resources = CudaResourcesSingleton::getInstance();
    allocator_                          = cuda_resources.cuda_allocator.get();

    const size_t hidden_units = 128;
    char*        data         = nullptr;
    data                      = reinterpret_cast<char*>(allocator_->reMalloc(data, 100, true));

    ft::Tensor word_embeddings(MEMORY_GPU, data_type, {(size_t)20, hidden_units}, allocator_, true);
    ft::Tensor pre_layernorm_weights(MEMORY_GPU, data_type, {hidden_units}, allocator_, true);
    ft::Tensor post_layernorm_weights(MEMORY_GPU, data_type, {hidden_units}, allocator_, true);
    ft::Tensor qkv_weights(MEMORY_GPU, data_type, {hidden_units, 3, hidden_units}, allocator_, true);
    ft::Tensor attention_layernorm(MEMORY_GPU, data_type, {hidden_units}, allocator_, true);
    ft::Tensor attention_output_weight(MEMORY_GPU, data_type, {hidden_units, hidden_units}, allocator_, true);
    ft::Tensor ffn_weight(MEMORY_GPU, data_type, {hidden_units, inter_size}, allocator_, true);
    ft::Tensor ffn_output_weight(MEMORY_GPU, data_type, {inter_size, hidden_units}, allocator_, true);
    ft::Tensor ffn_layer_norm(MEMORY_GPU, data_type, {inter_size}, allocator_, true);

    std::unordered_map<std::string, ft::Tensor>              global_weights = {{W::embedding, word_embeddings},
                                                                               {W::wpe, word_embeddings},
                                                                               {W::lm_head, word_embeddings},
                                                                               {W::pre_decoder_ln_gamma, pre_layernorm_weights},
                                                                               {W::pre_decoder_ln_beta, pre_layernorm_weights},
                                                                               {W::final_ln_gamma, post_layernorm_weights},
                                                                               {W::final_ln_beta, post_layernorm_weights}};
    std::vector<std::unordered_map<std::string, ft::Tensor>> layer_int8_weights, layer_int8_scales;
    std::vector<std::unordered_map<std::string, ft::Tensor>> layer_weights;

    for (int i = 0; i < params.num_layers_; ++i) {
        std::unordered_map<std::string, ft::Tensor> weight = {{W::pre_ln_gamma, pre_layernorm_weights},
                                                              {W::pre_ln_beta, pre_layernorm_weights},
                                                              {W::pre_attn_ln_gamma, pre_layernorm_weights},
                                                              {W::pre_attn_ln_beta, pre_layernorm_weights},
                                                              {W::attn_qkv_w, qkv_weights},
                                                              {W::attn_qkv_b, qkv_weights},
                                                              {W::attn_ln_gamma, attention_layernorm},
                                                              {W::attn_ln_beta, attention_layernorm},
                                                              {W::attn_o_w, attention_output_weight},
                                                              {W::attn_o_b, attention_output_weight},
                                                              {W::post_ln_gamma, post_layernorm_weights},
                                                              {W::post_ln_beta, post_layernorm_weights},
                                                              {W::ffn_w1, ffn_weight},
                                                              {W::ffn_b1, ffn_weight},
                                                              {W::ffn_w3, ffn_weight},
                                                              {W::ffn_b3, ffn_weight},
                                                              {W::ffn_w2, ffn_output_weight},
                                                              {W::ffn_b2, ffn_output_weight},
                                                              {W::ffn_ln_gamma, ffn_layer_norm},
                                                              {W::ffn_ln_beta, ffn_layer_norm}};
        layer_weights.push_back(weight);
    }
    ParallelModelWrapper model(params, 1, "", 0, global_weights, layer_weights, layer_int8_weights, layer_int8_scales);
    ModelRequest         model_request;
    model_request.generate_batch_size = 3;
    model_request.context_batch_size  = 2;
    model_request.combo_tokens =
        torch::zeros({6}, torch::dtype(at::ScalarType::Int).device(torch::kCPU).requires_grad(false));
    model_request.input_lengths =
        torch::zeros({5}, torch::dtype(at::ScalarType::Int).device(torch::kCPU).requires_grad(false));
    model_request.sequence_lengths =
        torch::zeros({5}, torch::dtype(at::ScalarType::Int).device(torch::kCPU).requires_grad(false));
    model_request.prefix_lengths =
        torch::zeros({5}, torch::dtype(at::ScalarType::Int).device(torch::kCPU).requires_grad(false));
    model_request.count_length =
        torch::zeros({1}, torch::dtype(at::ScalarType::Bool).device(torch::kCPU).requires_grad(false));
    model_request.kv_cache_blocks =
        torch::zeros({2, 5, 2, 2}, torch::dtype(at::ScalarType::Long).device(torch::kCPU).requires_grad(false));
    int* aaaaa = model_request.combo_tokens.data_ptr<int>();
    write_tensor(model_request.combo_tokens, vector<int>{1, 2, 3, 4, 5, 6});
    write_tensor(model_request.input_lengths, vector<int>{5, 4, 3, 2, 1});
    write_tensor(model_request.sequence_lengths, vector<int>{5, 4, 3, 2, 1});
    vector<ft::Tensor> kv_cache;
    vector<int64_t>    block_pointer;
    const size_t       max_blocks_per_batch = 2;
    for (int i = 0; i < params.num_layers_; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int z = 0; z < 2; ++z) {
                for (int k = 0; k < max_blocks_per_batch; ++k) {
                    ft::Tensor cache(MEMORY_GPU, data_type, {(size_t)8, hidden_units}, allocator_, true);
                    kv_cache.push_back(cache);
                    block_pointer.push_back(int64_t(cache.getPtr<void>()));
                }
            }
        }
    }
    write_tensor(model_request.kv_cache_blocks, block_pointer);

    model.forward(model_request);
}

}  // namespace rtp_llm
