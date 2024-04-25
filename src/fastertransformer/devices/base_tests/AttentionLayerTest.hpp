#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/torch_impl/GptModel.hpp"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/utils/KvCacheUtils.h"

#define private public
#include "maga_transformer/cpp/models/GptModel.h"

using namespace std;
using namespace rtp_llm;
using namespace fastertransformer;

template <typename TestT>
class AttentionLayerTest : public DeviceTestBase {
public:
    using TestType = TestT;
    void SetUp() override {
        DeviceTestBase::SetUp();
        GptModelDescription description;
        Weights weights;
        model_.reset(new GptModel({device_, weights, description}));
        GptModel model({device_, weights, description});
    }

    void testAttentionLayer (const AttentionConfigs& attention_conf,
                             const std::vector<int32_t>& input_lengths,
                             const std::vector<int32_t>& sequence_lengths);

    BufferPtr allocateKVBlocks(const AttentionConfigs& attention_conf,
                                const std::vector<int32_t>& input_lengths,
                                const std::vector<int32_t>& sequence_lengths);

    AttentionLayerWeights getAttentionWeights(const GptAttention& gpt_attention);

protected:
    CacheManagerPtr cache_manager_;
    std::shared_ptr<GptModel> model_;

};

torch::Tensor fakeAttentionInputs(const int64_t hidden_size, const int64_t token_num) {
    torch::manual_seed(1234);
    return torch::rand({token_num, hidden_size});
}

template <typename T>
BufferPtr AttentionLayerTest<T>::allocateKVBlocks(const AttentionConfigs& attention_conf,
                                                const std::vector<int32_t>& input_lengths,
                                                const std::vector<int32_t>& sequence_lengths) {
    const auto max_seq_len = *std::max_element(input_lengths.begin(), input_lengths.end());
    const auto batch_layer_kv_block_num =
        ((max_seq_len / attention_conf.tokens_per_block) + 2) * input_lengths.size();
    const auto batch_size = input_lengths.size();
    const auto num_layers = 1;

    auto kv_blocks_buf = device_->allocateBuffer({
        DataType::TYPE_INT64, {num_layers, 2, batch_size, batch_layer_kv_block_num}, AllocationType::HOST
    });
    BatchKVCacheBlockAddr batch_kv_cache;

    for (auto i = 0; i < batch_size; i++) {
        auto [success, kv_cache] = cache_manager_->malloc(batch_layer_kv_block_num);
        EXPECT_TRUE(success);
        batch_kv_cache.pushBack(kv_cache);
    }
    for (auto i = 0; i < batch_size; i++) {
        memcpyKvCache(
            kv_blocks_buf->data<uint64_t>(),
            batch_kv_cache.k_ptr[i],
            batch_kv_cache.v_ptr[i],
            1,
            kv_blocks_buf->shape().back(),
            batch_size,
            i
        );
    }

    return move(kv_blocks_buf);
}

template <typename T>
AttentionLayerWeights AttentionLayerTest<T>::getAttentionWeights(const GptAttention& gpt_attention) {
    AttentionLayerWeights attention_weights;
    auto qkv_tensor = torch::concat({
        gpt_attention->q_proj->weight.transpose(0, 1),
        gpt_attention->k_proj->weight.transpose(0, 1),
        gpt_attention->v_proj->weight.transpose(0, 1)
    }, 1).to(dataTypeToTorchType(getTensorType<TestType>()));
    auto qkv_buf = tensorToBuffer(qkv_tensor);
    attention_weights.qkv_weight.reset(new DenseWeights(qkv_buf));
    auto o_buf = tensorToBuffer(gpt_attention->o_proj->weight.transpose(0, 1).contiguous().to(
        dataTypeToTorchType(getTensorType<TestType>())));
    attention_weights.output_weight.reset(new DenseWeights(o_buf));
    return move(attention_weights);
}

template <typename T>
void AttentionLayerTest<T>::testAttentionLayer(const AttentionConfigs& attention_conf,
                                            const std::vector<int32_t>& input_lengths,
                                            const std::vector<int32_t>& sequence_lengths)
{
    // 1. prepare inputs
    const auto context_token_num = std::accumulate(
        input_lengths.begin() + sequence_lengths.size(), input_lengths.end(), 0);
    const auto total_token_num = context_token_num + sequence_lengths.size();

    // NOTE: this relationship does not hold for all models, e.g. gemma
    const auto hidden_size = attention_conf.hidden_size;
    const auto input_tensor = fakeAttentionInputs(hidden_size, total_token_num);
    const auto context_lengths = std::vector<int32_t>(
        input_lengths.begin() + sequence_lengths.size(), input_lengths.end());
    const auto mask_tensor = create_context_mask(context_lengths)
        .to(dataTypeToTorchType(getTensorType<TestType>()));
    std::cout << "mask: " << mask_tensor << std::endl;
    const auto input_buffer = tensorToBuffer(
        input_tensor.to(dataTypeToTorchType(getTensorType<TestType>())));

    GptModelInputs model_inputs;
    model_inputs.combo_tokens = tensorToBuffer(input_tensor);
    model_inputs.input_lengths = vector2Buffer(input_lengths);
    model_inputs.sequence_lengths = vector2Buffer(sequence_lengths);
    const auto mask_buf = tensorToBuffer(mask_tensor);
    model_inputs.attention_mask = *mask_buf;
    model_inputs.kv_cache_blocks = allocateKVBlocks(attention_conf, input_lengths, sequence_lengths);
    auto common_inputs = model_->prepareAttentionInputs(model_inputs);
    auto layer_cache_blocks = (*model_inputs.kv_cache_blocks)[0];
    common_inputs.kv_cache_blocks = layer_cache_blocks;
    printBufferData(*model_inputs.kv_cache_blocks, "kv_cache_blocks");

    // 2. compute reference implementation result
    GptAttention gpt_attention(attention_conf);
    torch::manual_seed(1234);
    torch::nn::init::normal_(gpt_attention->q_proj->weight);
    torch::nn::init::normal_(gpt_attention->k_proj->weight);
    torch::nn::init::normal_(gpt_attention->v_proj->weight);
    torch::nn::init::normal_(gpt_attention->o_proj->weight);
    torch::nn::init::zeros_(gpt_attention->q_proj->bias);
    torch::nn::init::zeros_(gpt_attention->k_proj->bias);
    torch::nn::init::zeros_(gpt_attention->v_proj->bias);
    torch::nn::init::zeros_(gpt_attention->o_proj->bias);

    auto position_ids = create_position_ids(input_lengths);
    auto torch_output = gpt_attention->forward(input_tensor.unsqueeze(0), mask_tensor, position_ids)
                                     .reshape({-1, (int64_t)hidden_size});
    std::cout << "torch output: " << torch_output.sizes() << std::endl;

    // 3. compute kernel result and compare
    auto attention_weights = getAttentionWeights(gpt_attention);
    AttentionLayerParams params {
        *input_buffer,
        attention_conf,
        attention_weights,
        common_inputs
    };
    auto attn_output = device_->attentionLayer(params);
    auto output_tensor = bufferToTensor(*attn_output.hidden_states);
    assertTensorClose(output_tensor, torch_output, 1e-3, 1);
}

class AttentionLayerTestFp16 : public AttentionLayerTest<half> {};

TEST_F(AttentionLayerTestFp16, testSimpleContextAttention) {
    AttentionConfigs attention_conf;
    attention_conf.head_num = 4;
    attention_conf.kv_head_num = 4;
    attention_conf.size_per_head = 8;
    attention_conf.tokens_per_block = 4;
    attention_conf.hidden_size = 32;

    attention_conf.rope_config.embedding_style = RopeType::Base;
    attention_conf.rope_config.embedding_dim = attention_conf.size_per_head;
    attention_conf.rope_config.embedding_base = 1000000;

    const size_t layer_num = 2;
    const size_t block_num = 1024;
    CacheConfig cache_conf(
        layer_num, block_num, attention_conf.kv_head_num, attention_conf.size_per_head,
        attention_conf.tokens_per_block, getTensorType<TestType>());
    cache_manager_ = std::make_shared<CacheManager>(cache_conf, device_);
    testAttentionLayer(attention_conf, {5}, {});

    attention_conf.head_num = 16;
    attention_conf.kv_head_num = 16;
    attention_conf.size_per_head = 64;
    attention_conf.hidden_size = 1024;
    attention_conf.rope_config.embedding_dim = attention_conf.size_per_head;
    testAttentionLayer(attention_conf, {3}, {});
}

