#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class RequestBlockBufferStoreTest: public CacheStoreTestBase {};

TEST_F(RequestBlockBufferStoreTest, testBlocksOps) {
    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");
    auto block1        = block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true);
    auto block2        = block_buffer_util_->makeBlockBuffer("b2", 1024, '1', false);
    request_block->addBlock(block1);
    request_block->addBlock(block2);

    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_, rtp_llm::DeviceFactory::getDefaultDevice());
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    store->setRequestBlockBuffer(request_block);
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    auto verify_block1 = store->getBlockBuffer("request-1", "b1");
    ASSERT_TRUE(block1 != nullptr);
    ASSERT_NE(block1, verify_block1);
    ASSERT_EQ(verify_block1->key, block1->key);
    ASSERT_NE(verify_block1->addr, block1->addr);
    ASSERT_FALSE(verify_block1->gpu_mem);
    ASSERT_EQ(verify_block1->len, block1->len);
    ASSERT_EQ(verify_block1->adopted, block1->adopted);
    ASSERT_EQ('0', ((char*)verify_block1->addr.get())[0]);

    auto verify_block2 = store->getBlockBuffer("request-1", "b2");
    ASSERT_EQ(verify_block2, block2);

    auto verify_block3 = store->getBlockBuffer("request-2", "b1");
    ASSERT_TRUE(verify_block3 == nullptr);

    auto verify_block4 = store->getBlockBuffer("request-1", "b3");
    ASSERT_TRUE(verify_block4 == nullptr);

    store->delRequestBlockBuffer("request-1");
    verify_block1 = store->getBlockBuffer("request-1", "b1");
    ASSERT_TRUE(verify_block1 == nullptr);
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    store->delRequestBlockBuffer("request-2");
}

TEST_F(RequestBlockBufferStoreTest, testWatchFunc_SetBeforeBlocks) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_, rtp_llm::DeviceFactory::getDefaultDevice());
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");
    auto block1        = block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true);
    auto block2        = block_buffer_util_->makeBlockBuffer("b2", 1024, '1', false);
    request_block->addBlock(block1);
    request_block->addBlock(block2);

    bool                          callback_flag        = false;
    bool                          failed_callback_flag = false;
    RequestBlockBuffer::WatchFunc watch_func           = [&failed_callback_flag, &callback_flag, block1, block2](
                                                   bool                                            success,
                                                   const std::vector<std::shared_ptr<BlockBuffer>> blocks) {
        if (success) {
            callback_flag = true;
            EXPECT_EQ(2, blocks.size());
            for (auto& block : blocks) {
                EXPECT_TRUE(block->key == "b1" || block->key == "b2");
            }
        } else {
            failed_callback_flag = true;
        }
    };

    // empty block, not trigger callback
    store->setRequestBlockBufferWatchFunc("request-1", std::move(watch_func));
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());
    ASSERT_FALSE(callback_flag);
    ASSERT_FALSE(failed_callback_flag);

    // set blocks, trigger callback
    store->setRequestBlockBuffer(request_block);
    ASSERT_TRUE(callback_flag);
    ASSERT_FALSE(failed_callback_flag);

    // del request block
    store->delRequestBlockBuffer("request-1");
    ASSERT_TRUE(failed_callback_flag);
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());
}

TEST_F(RequestBlockBufferStoreTest, testWatchFunc_SetAfterBlocks) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_, rtp_llm::DeviceFactory::getDefaultDevice());
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");
    auto block1        = block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true);
    auto block2        = block_buffer_util_->makeBlockBuffer("b2", 1024, '1', false);
    request_block->addBlock(block1);
    request_block->addBlock(block2);
    store->setRequestBlockBuffer(request_block);

    bool                          callback_flag        = false;
    bool                          failed_callback_flag = false;
    RequestBlockBuffer::WatchFunc watch_func           = [&failed_callback_flag, &callback_flag, block1, block2](
                                                   bool                                            success,
                                                   const std::vector<std::shared_ptr<BlockBuffer>> blocks) {
        if (success) {
            callback_flag = true;
            EXPECT_EQ(2, blocks.size());
            for (auto& block : blocks) {
                EXPECT_TRUE(block->key == "b1" || block->key == "b2");
            }
        } else {
            failed_callback_flag = true;
        }
    };

    // set blocks, trigger callback
    store->setRequestBlockBufferWatchFunc("request-1", std::move(watch_func));
    ASSERT_TRUE(callback_flag);
    ASSERT_FALSE(failed_callback_flag);
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());

    // del request block
    store->delRequestBlockBuffer("request-1");
    ASSERT_TRUE(failed_callback_flag);
    ASSERT_FALSE(store->debugInfoOnRequest("request-1").empty());
}

TEST_F(RequestBlockBufferStoreTest, testAfterDelRequestBlockBuffer) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_, rtp_llm::DeviceFactory::getDefaultDevice());
    store->delRequestBlockBuffer("request-1");

    ASSERT_TRUE(store->getBlockBuffer("request-1", "b1") == nullptr);

    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");
    auto block1        = block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true);
    auto block2        = block_buffer_util_->makeBlockBuffer("b2", 1024, '1', false);
    request_block->addBlock(block1);
    request_block->addBlock(block2);
    ASSERT_TRUE(store->setRequestBlockBuffer(request_block));

    store->delRequestBlockBuffer("request-1");
    ASSERT_FALSE(store->setRequestBlockBuffer(request_block));
    ASSERT_FALSE(store->setRequestBlockBufferWatchFunc(
        "request-1",
        [](bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) { EXPECT_FALSE(success); }));
}

}  // namespace rtp_llm
