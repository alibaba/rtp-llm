
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/LoraWeights.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class LoraWeightsTest: public DeviceTestBase {
protected:
};

TEST_F(LoraWeightsTest, testLoraWeightsConstruct) {
    auto lora_a = torch::rand({10, 8});
    auto lora_b = torch::rand({8, 10});
    // nullptr, nullptr
    BufferPtr lora_a_buffer_ptr = nullptr;
    BufferPtr lora_b_buffer_ptr = nullptr;
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
    // lora_a, nullptr
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = nullptr;
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
    // nullptr, lora_b
    lora_a_buffer_ptr = nullptr;
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
    // lora_a, lora_b
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_NO_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
    auto result = rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(result.lora_a_)), lora_a);
    torch::equal(bufferToTensor(*std::const_pointer_cast<Buffer>(result.lora_b_)), lora_b);

    // dim same test
    lora_a            = torch::rand({10, 8, 10});
    lora_b            = torch::rand({8, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    // dim >= 2 test
    // error test
    lora_a            = torch::rand({10});
    lora_b            = torch::rand({10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
    // dim = 3 correct test
    lora_a            = torch::rand({2, 10, 8});
    lora_b            = torch::rand({2, 8, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_NO_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    // same rank test
    lora_a            = torch::rand({10, 8});
    lora_b            = torch::rand({7, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    lora_a            = torch::rand({2, 8, 8});
    lora_b            = torch::rand({2, 7, 7});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    // rank size test

    lora_a            = torch::rand({10, 11});
    lora_b            = torch::rand({11, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    lora_a            = torch::rand({10, 10, 11});
    lora_b            = torch::rand({10, 11, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    // same dtype test
    lora_a            = torch::rand({10, 8}).to(torch::kHalf);
    lora_b            = torch::rand({8, 10}).to(torch::kFloat32);
    lora_a_buffer_ptr = tensorToBuffer(lora_a);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));

    // same memory test
    lora_a            = torch::rand({10, 8});
    lora_b            = torch::rand({8, 10});
    lora_a_buffer_ptr = tensorToBuffer(lora_a, rtp_llm::AllocationType::HOST);
    lora_b_buffer_ptr = tensorToBuffer(lora_b);
    EXPECT_ANY_THROW(rtp_llm::lora::LoraWeights(lora_a_buffer_ptr, lora_b_buffer_ptr));
}

}  // namespace rtp_llm
