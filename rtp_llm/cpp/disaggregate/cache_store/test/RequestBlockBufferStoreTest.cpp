#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace rtp_llm {

class RequestBlockBufferStoreTest: public CacheStoreTestBase {};

TEST_F(RequestBlockBufferStoreTest, testBlocksOps) {
    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");
    auto block1        = block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true);
    auto block2        = block_buffer_util_->makeBlockBuffer("b2", 1024, '1', false);
    request_block->addBlock(block1);
    request_block->addBlock(block2);

    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
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

TEST_F(RequestBlockBufferStoreTest, testPinnedCopyPayloadAndLifetime) {
    auto                                      store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    std::vector<std::vector<uint8_t>>         payloads;
    std::vector<std::shared_ptr<BlockBuffer>> retained;
    {
        auto request = std::make_shared<RequestBlockBuffer>("pinned-copy");
        for (const uint32_t size : {17U, 4097U, 65537U, 1048583U}) {
            auto& payload = payloads.emplace_back(size);
            for (uint32_t i = 0; i < size; ++i) {
                payload[i] = static_cast<uint8_t>((i + payloads.size()) % 251);
            }
            auto tensor = torch::from_blob(payload.data(), {(int64_t)size}, torch::kUInt8).to(torch::kCUDA);
            auto addr   = std::shared_ptr<void>(tensor.data_ptr(), [tensor](void*) {});
            request->addBlock(std::make_shared<BlockBuffer>(std::to_string(size), addr, size, true, true));
        }
        ASSERT_TRUE(store->setRequestBlockBuffer(request));
        runtimeSyncAndCheck();
        for (const auto& payload : payloads) {
            auto block = store->getBlockBuffer("pinned-copy", std::to_string(payload.size()));
            ASSERT_NE(block, nullptr);
            ASSERT_FALSE(block->gpu_mem);
            ASSERT_TRUE(block->adopted);
            ASSERT_EQ(block->len, payload.size());
            ASSERT_TRUE(torch::from_blob(block->addr.get(), {(int64_t)block->len}, torch::kUInt8).is_pinned());
            retained.push_back(block);
        }
    }
    store->delRequestBlockBuffer("pinned-copy");
    store.reset();

    // Readers retain ownership after both the source tensors and store are gone.
    for (size_t i = 0; i < retained.size(); ++i) {
        auto churn =
            torch::empty({(int64_t)retained[i]->len}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
        churn.fill_(255);
        auto begin = static_cast<const uint8_t*>(retained[i]->addr.get());
        EXPECT_EQ(payloads[i], std::vector<uint8_t>(begin, begin + retained[i]->len));
    }
    std::weak_ptr<void> last_owner = retained.front()->addr;
    retained.clear();
    EXPECT_TRUE(last_owner.expired());
}

TEST_F(RequestBlockBufferStoreTest, testExpiryWithContinuousAndRepeatedCompletions) {
    auto store  = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto old    = std::make_shared<RequestBlockBuffer>("old");
    auto recent = std::make_shared<RequestBlockBuffer>("recent");
    ASSERT_TRUE(store->setRequestBlockBuffer(old));
    ASSERT_TRUE(store->setRequestBlockBuffer(recent));
    store->delRequestBlockBuffer("old");
    store->delRequestBlockBuffer("recent");
    constexpr int64_t retention_us = 1000 * 60 * 60;
    const auto        start        = currentTimeUs();
    auto              previous     = start;
    int64_t           max_gap      = 0;
    int               completed    = 0;
    while (currentTimeUs() - start <= retention_us + 100000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto id = "current-" + std::to_string(completed++);
        ASSERT_TRUE(store->setRequestBlockBuffer(std::make_shared<RequestBlockBuffer>(id)));
        store->delRequestBlockBuffer(id);
        store->delRequestBlockBuffer("recent");
        const auto now = currentTimeUs();
        max_gap        = std::max(max_gap, now - previous);
        previous       = now;
    }
    ASSERT_LT(max_gap, retention_us);
    EXPECT_TRUE(store->setRequestBlockBuffer(old));
    EXPECT_FALSE(store->setRequestBlockBuffer(recent));
}

TEST_F(RequestBlockBufferStoreTest, testMissingCompletionsDoNotAccumulate) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    for (int i = 0; i < 1000; ++i) {
        store->delRequestBlockBuffer("missing-" + std::to_string(i));
    }
    EXPECT_TRUE(store->expired_request_caches_.empty());
    EXPECT_TRUE(store->request_cache_map_.empty());
}

TEST_F(RequestBlockBufferStoreTest, testStaleExpiryPreservesLiveRequest) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    ASSERT_TRUE(store->setRequestBlockBuffer(std::make_shared<RequestBlockBuffer>("live")));
    auto retained = store->getOrInsertRequestBlockBuffer("live");
    store->expired_request_caches_.push_back({"live", currentTimeUs() - 1000 * 60 * 60 - 1});
    store->delRequestBlockBuffer("missing");
    EXPECT_EQ(store->getOrInsertRequestBlockBuffer("live"), retained);
}

TEST_F(RequestBlockBufferStoreTest, testStopClearsExpiryMetadata) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    ASSERT_TRUE(store->setRequestBlockBuffer(std::make_shared<RequestBlockBuffer>("completed")));
    store->delRequestBlockBuffer("completed");
    ASSERT_FALSE(store->expired_request_caches_.empty());
    store->stop();
    EXPECT_TRUE(store->request_cache_map_.empty());
    EXPECT_TRUE(store->expired_request_caches_.empty());
}

TEST_F(RequestBlockBufferStoreTest, testWatchFunc_SetBeforeBlocks) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
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
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
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
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
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
