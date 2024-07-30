
#include "gtest/gtest.h"

#include "maga_transformer/cpp/lora/LoraManager.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class LoraManagerTest: public DeviceTestBase {
protected:

};

TEST_F(LoraManagerTest, testSimple) {
    auto lora_manager = lora::LoraManager();
    ft::lora::loraLayerWeightsMap lora_a_map(1);
    ft::lora::loraLayerWeightsMap lora_b_map(1);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);

    lora_manager.addLora(0, lora_a_map, lora_b_map);
    EXPECT_NE(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), true);

    lora_manager.removeLora(0);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);
}

TEST_F(LoraManagerTest, testPressure) {
    auto lora_manager = lora::LoraManager();
    ft::lora::loraLayerWeightsMap lora_a_map(1);
    ft::lora::loraLayerWeightsMap lora_b_map(1);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);

    for (int i = 0; i < 1000; i++) {
        lora_manager.addLora(i, lora_a_map, lora_b_map);
        EXPECT_NE(lora_manager.getLora(i), nullptr);
        EXPECT_EQ(lora_manager.hasLora(i), true);
    }

    for (int i = 0; i < 1000; i++) {
        lora_manager.removeLora(i);
        EXPECT_EQ(lora_manager.getLora(i), nullptr);
        EXPECT_EQ(lora_manager.hasLora(i), false);
    }
}

TEST_F(LoraManagerTest, testRemoveSimple) {
    auto lora_manager = lora::LoraManager();
    ft::lora::loraLayerWeightsMap lora_a_map(1);
    ft::lora::loraLayerWeightsMap lora_b_map(1);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);

    lora_manager.addLora(0, lora_a_map, lora_b_map);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    {
        auto lora_resource = lora_manager.getLora(0);

        EXPECT_NE(lora_resource, nullptr);
        EXPECT_EQ(lora_resource.use_count(), 2);
    }
    auto lora_resource = lora_manager.getLora(0);

    EXPECT_NE(lora_resource, nullptr);
    EXPECT_EQ(lora_resource.use_count(), 2);
    std::this_thread::sleep_for(2000ms);
    lora_resource = nullptr;


    auto removeLoraFunc = [&lora_manager](int64_t lora_id) {
        lora_manager.removeLora(lora_id);
    };
    std::thread removeLora(removeLoraFunc, 0);
    removeLora.join();

    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);
}

TEST_F(LoraManagerTest, testTimeoutSimple) {
    auto lora_manager = lora::LoraManager(10);
    ft::lora::loraLayerWeightsMap lora_a_map(1);
    ft::lora::loraLayerWeightsMap lora_b_map(1);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);

    lora_manager.addLora(0, lora_a_map, lora_b_map);
    auto lora_resource = lora_manager.getLora(0);
    EXPECT_NE(lora_resource, nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    EXPECT_EQ(lora_resource.use_count(), 2);

    EXPECT_ANY_THROW(lora_manager.removeLora(0));

    EXPECT_NE(lora_resource, nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    EXPECT_EQ(lora_resource.use_count(), 2);
}




}  // namespace rtp_llm
