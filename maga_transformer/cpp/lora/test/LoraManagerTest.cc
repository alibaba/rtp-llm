
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
    auto removeLoraFunc = [&lora_manager](int64_t lora_id) {
        lora_manager.removeLora(lora_id);
    };
    std::thread removeLora(removeLoraFunc, 0);
    std::this_thread::sleep_for(2000ms);
    EXPECT_NE(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    lora_resource = nullptr;
    lora_manager.releaseSignal();
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

TEST_F(LoraManagerTest, testisLoraAliveSimple) {
    auto lora_manager = lora::LoraManager();
    ft::lora::loraLayerWeightsMap lora_a_map(1);
    ft::lora::loraLayerWeightsMap lora_b_map(1);
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);

    lora_manager.addLora(0, lora_a_map, lora_b_map);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    auto lora_resource = lora_manager.getLora(0);

    EXPECT_NE(lora_resource, nullptr);
    EXPECT_EQ(lora_resource.use_count(), 2);
    EXPECT_EQ(lora_manager.isLoraAlive(0), true);

    // start remove lora
    auto removeLoraFunc = [&lora_manager](int64_t lora_id) {
        lora_manager.removeLora(lora_id);
    };
    std::thread removeLora(removeLoraFunc, 0);
    // when lora resource ref cout not release and removing lora
    // we can still get lora resource from getLora, But isLoraAlive will return false.
    std::this_thread::sleep_for(2000ms);
    EXPECT_EQ(lora_manager.hasLora(0), true);
    EXPECT_NE(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.isLoraAlive(0), false);


    // after lora resource release, we can not get lora.
    // hasLora will return nullptr and isLoraAlive will return false.
    lora_resource = nullptr;
    lora_manager.releaseSignal();
    removeLora.join();
    EXPECT_EQ(lora_manager.getLora(0), nullptr);
    EXPECT_EQ(lora_manager.hasLora(0), false);
    EXPECT_EQ(lora_manager.isLoraAlive(0), false);
}


TEST_F(LoraManagerTest, testMultiAddWithMultiRemove) {
    auto lora_manager = lora::LoraManager();

    auto addLoraFunc = [&](size_t lora_num) {
        ft::lora::loraLayerWeightsMap lora_a_map(1);
        ft::lora::loraLayerWeightsMap lora_b_map(1);
        for (int i = 0; i < lora_num; i ++) {
            lora_manager.addLora(i, lora_a_map, lora_b_map);
        }
    };

    auto removeLoraFunc = [&](size_t lora_num) {
        for (int i = 0; i < lora_num; i ++) {
            while(true) {
                if (lora_manager.hasLora(i) && lora_manager.isLoraAlive(i)) {
                    break;
                }
            }
            lora_manager.removeLora(i);
        }
    };

    std::thread addLora(addLoraFunc, 1000);
    std::thread removeLora(removeLoraFunc, 1000);

    addLora.join();
    removeLora.join();

    for (int i = 0; i < 1000; i++) {
        EXPECT_EQ(lora_manager.hasLora(0), false);
        EXPECT_EQ(lora_manager.getLora(0), nullptr);
        EXPECT_EQ(lora_manager.isLoraAlive(0), false);
    }


}




}  // namespace rtp_llm
