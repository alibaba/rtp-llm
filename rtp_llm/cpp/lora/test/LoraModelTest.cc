
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/LoraWeights.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class LoraModelTest: public DeviceTestBase {
protected:
};

TEST_F(LoraModelTest, testLoraModelImplConstruct) {
    auto lora_model_impl = rtp_llm::lora::LoraModelImpl();
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::attn_qkv_w), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::attn_o_w), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w1), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w3), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w2), nullptr);

    auto lora_a            = torch::rand({10, 8});
    auto lora_b            = torch::rand({8, 10});
    auto lora_a_buffer_ptr = tensorToBuffer(lora_a);
    auto lora_b_buffer_ptr = tensorToBuffer(lora_b);

    lora_model_impl.setLoraWeigths(rtp_llm::W::attn_qkv_w, lora_a_buffer_ptr, lora_b_buffer_ptr);
    auto lora_weights = lora_model_impl.getLoraWeights(rtp_llm::W::attn_qkv_w);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(lora_weights->lora_a_)), lora_a);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(lora_weights->lora_b_)), lora_b);

    EXPECT_NE(lora_model_impl.getLoraWeights(rtp_llm::W::attn_qkv_w), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::attn_o_w), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w1), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w3), nullptr);
    EXPECT_EQ(lora_model_impl.getLoraWeights(rtp_llm::W::ffn_w2), nullptr);
}

TEST_F(LoraModelTest, testLoraModelErrorLayernumConstruct) {
    rtp_llm::lora::loraLayerWeightsMap lora_a_map(1);
    rtp_llm::lora::loraLayerWeightsMap lora_b_map(2);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraModel(lora_a_map, lora_b_map));
}

TEST_F(LoraModelTest, testLoraModelConstructCase0) {
    rtp_llm::lora::loraLayerWeightsMap lora_a_map(1);
    rtp_llm::lora::loraLayerWeightsMap lora_b_map(1);
    auto                               lora_model = rtp_llm::lora::LoraModel(lora_a_map, lora_b_map);

    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::attn_qkv_w), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::attn_o_w), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w1), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w2), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w3), nullptr);

    EXPECT_ANY_THROW(lora_model.getLoraWeights(1, rtp_llm::W::attn_o_w));
}

TEST_F(LoraModelTest, testLoraModelConstructCase1) {
    rtp_llm::lora::loraLayerWeightsMap lora_a_map(1);
    rtp_llm::lora::loraLayerWeightsMap lora_b_map(1);

    auto lora_a            = torch::rand({10, 8});
    auto lora_b            = torch::rand({8, 10});
    auto lora_a_buffer_ptr = tensorToBuffer(lora_a);
    auto lora_b_buffer_ptr = tensorToBuffer(lora_b);

    lora_a_map[0][rtp_llm::W::attn_qkv_w] = lora_a_buffer_ptr;
    lora_b_map[0][rtp_llm::W::attn_qkv_w] = lora_b_buffer_ptr;

    auto lora_model = rtp_llm::lora::LoraModel(lora_a_map, lora_b_map);

    EXPECT_NE(lora_model.getLoraWeights(0, rtp_llm::W::attn_qkv_w), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::attn_o_w), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w1), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w2), nullptr);
    EXPECT_EQ(lora_model.getLoraWeights(0, rtp_llm::W::ffn_w3), nullptr);

    auto lora_weights = lora_model.getLoraWeights(0, rtp_llm::W::attn_qkv_w);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(lora_weights->lora_a_)), lora_a);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(lora_weights->lora_b_)), lora_b);
}

}  // namespace rtp_llm
