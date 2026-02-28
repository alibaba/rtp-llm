#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

#include "rtp_llm/cpp/test/ModelTestUtil.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/torch_impl/GptModel.hpp"

using namespace std;
using namespace rtp_llm;

class GptModelTest: public DeviceTestBase {};

TEST_F(GptModelTest, testSimple) {
    const auto path    = test_data_path_ + "../../test/model_test/fake_test/testdata/qwen_0.5b";
    auto       weights = loadWeightsFromDir(path);
    RTP_LLM_CHECK(weights->lm_head->kernel != nullptr);
    RTP_LLM_CHECK(weights->embedding != nullptr);
    RTP_LLM_CHECK(weights->layers.size() == 4);

    GptModelDescription description;
    description.ffn_conf.activation_type = ActivationType::Swiglu;
    description.norm_type                = NormType::rmsnorm;
    auto& attention_conf                 = description.attention_conf;
    attention_conf.head_num              = 16;
    attention_conf.kv_head_num           = 16;
    attention_conf.size_per_head         = 64;
    attention_conf.tokens_per_block      = 16;
    attention_conf.rope_config.style     = RopeStyle::Base;
    attention_conf.rope_config.dim       = 64;
    attention_conf.rope_config.base      = 1000000;
    attention_conf.is_causal             = true;

    const auto  cache_block_num = 128;
    CacheConfig cache_config(KVCacheParam({static_cast<uint>(weights->layers.size()),
                                           cache_block_num,
                                           static_cast<uint>(attention_conf.kv_head_num),
                                           static_cast<uint>(attention_conf.size_per_head),
                                           static_cast<uint>(attention_conf.tokens_per_block),
                                           DataType::TYPE_FP16}));

    const std::vector<int32_t> input_lengths_vec = {3};

    auto       combo_tokens      = createBuffer<int32_t>({3}, {13048, 11, 220}, AllocationType::HOST);
    auto       input_lengths     = createBuffer<int32_t>({1}, input_lengths_vec, AllocationType::HOST);
    auto       sequence_lengths  = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    auto       kv_cache          = torch::empty(0);
    auto       kv_cache_block_id = allocateKVBlocks(cache_config, input_lengths_vec, kv_cache);
    const auto mask_tensor       = create_context_mask(input_lengths_vec).to(torch::kFloat16);
    const auto mask_buf          = tensorToBuffer(mask_tensor);
    auto       model = createGptModel({device_, *weights, description, cache_manager_->allLayerCacheBase()});

    GptModelInputs inputs;
    inputs.combo_tokens     = std::move(combo_tokens);
    inputs.input_lengths    = std::move(input_lengths);
    inputs.sequence_lengths = std::move(sequence_lengths);

    inputs.prefix_lengths    = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    inputs.lm_output_indexes = createBuffer<int32_t>({1}, {2}, AllocationType::HOST);
    inputs.attention_mask    = mask_buf;
    inputs.kv_cache_block_id = kv_cache_block_id;
    device_->syncAndCheck();

    // temporarily disable test for cpu device
    // enable this back when device is ready.
    if (DeviceFactory::getDefaultDevice()->getDeviceProperties().type == DeviceType::Cpu) {
        try {
            auto outputs = model->forward(inputs);
            printBufferData(*outputs.logits, "logits");
            auto output_tensor = bufferToTensor(*outputs.logits);
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        return;
    }

    auto outputs = model->forward(inputs);
    device_->syncAndCheck();
    printBufferData(*outputs.logits, "logits");
    auto output_tensor = bufferToTensor(*outputs.logits);

    // expected to output token 151645
    assertTensorClose(output_tensor.index({torch::indexing::Slice(), 151645}),
                      bufferToTensor(*createBuffer<float>({1}, {15.7891}, AllocationType::HOST)),
                      1e-1,
                      5e-2);
    assertTensorClose(output_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}),
                      bufferToTensor(*createBuffer<float>({1, 3}, {7.1562, -9.3672, -0.8486}, AllocationType::HOST)),
                      0.2,
                      0.1);

    inputs.combo_tokens      = createBuffer<int32_t>({1}, {151645}, AllocationType::HOST);
    inputs.input_lengths     = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.sequence_lengths  = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.prefix_lengths    = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    inputs.lm_output_indexes = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    device_->syncAndCheck();
    outputs = model->forward(inputs);
    device_->syncAndCheck();
    output_tensor = bufferToTensor(*outputs.logits);

    // expected output token 198
    assertTensorClose(output_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(190, 200)}),
                      bufferToTensor(*createBuffer<float>(
                          {10},
                          {1.1670, -0.6973, -0.6919, -1.7705, -1.4453, -0.9766, -0.1351, 2.6152, 28.2500, -0.1479},
                          AllocationType::HOST)),
                      1e-1,
                      2e-2);

    // combining two previous queries.
    inputs.combo_tokens      = createBuffer<int32_t>({4}, {151645, 13048, 11, 220}, AllocationType::HOST);
    inputs.input_lengths     = createBuffer<int32_t>({2}, {3, 3}, AllocationType::HOST);
    inputs.sequence_lengths  = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.prefix_lengths    = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    inputs.lm_output_indexes = createBuffer<int32_t>({2}, {0, 3}, AllocationType::HOST);

    inputs.kv_cache_block_id = allocateKVBlocks(cache_config, {3, 3}, kv_cache);
    device_->copy({inputs.kv_cache_block_id->view(0, 1), *kv_cache_block_id});

    device_->syncAndCheck();
    outputs = model->forward(inputs);
    device_->syncAndCheck();
    output_tensor = bufferToTensor(*outputs.logits);
    assertTensorClose(output_tensor.index({torch::indexing::Slice(1, 2), 151645}),
                      bufferToTensor(*createBuffer<float>({1}, {15.7891}, AllocationType::HOST)),
                      1e-1,
                      5e-2);
    assertTensorClose(output_tensor.index({torch::indexing::Slice(1, 2), torch::indexing::Slice(0, 3)}),
                      bufferToTensor(*createBuffer<float>({1, 3}, {7.1562, -9.3672, -0.8486}, AllocationType::HOST)),
                      0.2,
                      0.1);
    assertTensorClose(output_tensor.index({torch::indexing::Slice(0, 1), torch::indexing::Slice(190, 200)}),
                      bufferToTensor(*createBuffer<float>(
                          {10},
                          {1.1670, -0.6973, -0.6919, -1.7705, -1.4453, -0.9766, -0.1351, 2.6152, 28.2500, -0.1479},
                          AllocationType::HOST)),
                      1e-1,
                      2e-2);
}

TEST_F(GptModelTest, testAttentionInputs) {
    auto                dtype = rtp_llm::DataType::TYPE_FP16;
    GptModelDescription description;
    auto&               attention_conf = description.attention_conf;
    attention_conf.head_num            = 16;
    attention_conf.kv_head_num         = 16;
    attention_conf.size_per_head       = 64;
    attention_conf.tokens_per_block    = 16;

    Weights        weights;
    auto           model = createGptModel({device_, weights, description});
    GptModelInputs inputs;
    inputs.input_lengths    = createBuffer<int32_t>({4}, {3, 5, 2, 7}, AllocationType::HOST);
    inputs.prefix_lengths   = createBuffer<int32_t>({4}, {0, 0, 0, 0}, AllocationType::HOST);
    inputs.sequence_lengths = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    inputs.combo_tokens     = createBuffer<int32_t>({17}, std::vector<int32_t>(17, 0), AllocationType::HOST);
    AttentionCommonInputs attention_inputs({inputs.input_lengths, inputs.sequence_lengths});

    {
        device_->syncAndCheck();
        attention_inputs = model->prepareAttentionInputs(inputs, dtype, nullptr);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 3, 8, 10, 17});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset,
                                        {0, 0, 0, 4, 4, 4, 4, 4, 6, 6, 11, 11, 11, 11, 11, 11, 11});
        ASSERT_EQ(attention_inputs.context_batch_size, 4);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 0);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 0);
    }

    inputs.sequence_lengths = createBuffer<int32_t>({3}, {4, 19, 23}, AllocationType::HOST);
    inputs.prefix_lengths   = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    inputs.combo_tokens     = createBuffer<int32_t>({10}, std::vector<int32_t>(10, 0), AllocationType::HOST);
    {
        device_->syncAndCheck();
        attention_inputs = model->prepareAttentionInputs(inputs, dtype, nullptr);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 7});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        ASSERT_EQ(attention_inputs.context_batch_size, 1);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 3);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 23);
    }

    inputs.sequence_lengths = createBuffer<int32_t>({2}, {4, 6}, AllocationType::HOST);
    inputs.prefix_lengths   = createBuffer<int32_t>({2}, {0, 0}, AllocationType::HOST);
    inputs.combo_tokens     = createBuffer<int32_t>({11}, std::vector<int32_t>(11, 0), AllocationType::HOST);
    {
        device_->syncAndCheck();
        attention_inputs = model->prepareAttentionInputs(inputs, dtype, nullptr);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 2, 9});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset, {0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0});
        ASSERT_EQ(attention_inputs.context_batch_size, 2);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 2);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 6);
    }
}
