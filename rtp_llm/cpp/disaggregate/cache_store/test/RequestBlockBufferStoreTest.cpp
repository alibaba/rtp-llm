#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <chrono>

namespace rtp_llm {

class RequestBlockBufferStoreTest: public CacheStoreTestBase {};

namespace {
class RdmaMockMemoryUtil: public MockMemoryUtil {
public:
    explicit RdmaMockMemoryUtil(std::shared_ptr<MemoryUtil> impl): MockMemoryUtil(std::move(impl)) {}
    bool isRdmaMode() override {
        return true;
    }
};
constexpr int64_t kTombstoneRetentionUs =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::hours(1)).count();
}  // namespace

TEST_F(RequestBlockBufferStoreTest, TypedPublicationDistinguishesActiveAndEndedRequests) {
    auto store   = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto request = std::make_shared<RequestBlockBuffer>("typed");
    EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::Stored);
    store->delRequestBlockBuffer("typed");
    EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::RequestEnded);
    EXPECT_FALSE(store->setRequestBlockBuffer(request));

    // Generic delete preserves the legacy unknown-ID contract; definitive
    // finish-before-first-store uses the explicit retain_tombstone overload.
    store->delRequestBlockBuffer("not-registered");
    auto unknown = std::make_shared<RequestBlockBuffer>("not-registered");
    EXPECT_EQ(store->setRequestBlockBufferResult(unknown), RequestBlockBufferStore::StoreResult::Stored);
}

TEST_F(RequestBlockBufferStoreTest, TypedPublicationRejectsHeldBufferEndedDuringValidation) {
    // Use host memory and a mock MR check to end the request after lookup,
    // before the final append. No GPU allocation or copy is needed.
    auto memory  = std::make_shared<RdmaMockMemoryUtil>(memory_util_);
    auto store   = std::make_shared<RequestBlockBufferStore>(memory);
    auto request = std::make_shared<RequestBlockBuffer>("validation-race");
    auto data    = std::make_shared<char>('x');
    request->addBlock(std::make_shared<BlockBuffer>("host", data, 1, false, false));
    EXPECT_CALL(*memory, isMemoryMr(::testing::_, 1, false, false))
        .WillOnce(::testing::Invoke([&](void*, uint64_t, bool, bool) {
            store->delRequestBlockBuffer("validation-race");
            return true;
        }));
    EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::RequestEnded);
    EXPECT_EQ(store->getBlockBuffer("validation-race", "host"), nullptr);
}

TEST_F(RequestBlockBufferStoreTest, ExplicitTerminalEndBeforeFirstPublicationRetainsTombstone) {
    auto store   = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto request = std::make_shared<RequestBlockBuffer>("end-before-store");
    store->delRequestBlockBuffer("end-before-store", true);
    store->delRequestBlockBuffer("end-before-store", true);
    EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::RequestEnded);
    EXPECT_FALSE(store->setRequestBlockBuffer(request));
    bool watch_called = false;
    EXPECT_FALSE(
        store->setRequestBlockBufferWatchFunc("end-before-store", [&](bool, const auto&) { watch_called = true; }));
    EXPECT_FALSE(watch_called);
    EXPECT_EQ(store->getBlockBuffer("end-before-store", "missing"), nullptr);

    // The default generic delete deliberately permits future first publication.
    store->delRequestBlockBuffer("legacy-unknown");
    auto legacy = std::make_shared<RequestBlockBuffer>("legacy-unknown");
    EXPECT_EQ(store->setRequestBlockBufferResult(legacy), RequestBlockBufferStore::StoreResult::Stored);
}

TEST_F(RequestBlockBufferStoreTest, TombstoneRepeatDoesNotExtendRetentionOrGrowQueue) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    store->delRequestBlockBuffer("repeat", true);
    ASSERT_EQ(store->expired_request_caches_.size(), 1);
    const auto first_end = store->expired_request_caches_.front().second;
    for (int i = 0; i < 20; ++i) {
        store->delRequestBlockBuffer("repeat", true);
    }
    EXPECT_EQ(store->expired_request_caches_.size(), 1);
    EXPECT_EQ(store->expired_request_caches_.front().second, first_end);
    auto active = std::make_shared<RequestBlockBuffer>("active-end");
    ASSERT_EQ(store->setRequestBlockBufferResult(active), RequestBlockBufferStore::StoreResult::Stored);
    store->delRequestBlockBuffer("active-end");
    EXPECT_EQ(store->expired_request_caches_.size(), 2);
    store->delRequestBlockBuffer("active-end");
    EXPECT_EQ(store->expired_request_caches_.size(), 2);
}

TEST_F(RequestBlockBufferStoreTest, TombstoneRemainsAfterTenSecondsForLatePublication) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    store->delRequestBlockBuffer("long-prefill", true);
    store->expired_request_caches_.front().second = currentTimeUs() - 10 * 1000 * 1000;
    store->delRequestBlockBuffer("cleanup-trigger");
    ASSERT_EQ(store->expired_request_caches_.size(), 1);
    auto late = std::make_shared<RequestBlockBuffer>("long-prefill");
    EXPECT_EQ(store->setRequestBlockBufferResult(late), RequestBlockBufferStore::StoreResult::RequestEnded);
}

TEST_F(RequestBlockBufferStoreTest, TombstoneCleanupExpiresOldestFirstAndPreservesRecentEntry) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    store->delRequestBlockBuffer("old-1", true);
    store->delRequestBlockBuffer("old-2", true);
    store->delRequestBlockBuffer("recent", true);
    const auto now                           = currentTimeUs();
    store->expired_request_caches_[0].second = now - kTombstoneRetentionUs - 2 * 1000 * 1000;
    store->expired_request_caches_[1].second = now - kTombstoneRetentionUs - 1000 * 1000;
    store->delRequestBlockBuffer("cleanup-trigger");
    ASSERT_EQ(store->expired_request_caches_.size(), 1);
    EXPECT_EQ(store->expired_request_caches_.front().first, "recent");
    EXPECT_EQ(store->request_cache_map_.count("old-1"), 0);
    EXPECT_EQ(store->request_cache_map_.count("old-2"), 0);
    EXPECT_EQ(store->request_cache_map_.count("recent"), 1);
    auto late = std::make_shared<RequestBlockBuffer>("recent");
    EXPECT_EQ(store->setRequestBlockBufferResult(late), RequestBlockBufferStore::StoreResult::RequestEnded);
}

TEST_F(RequestBlockBufferStoreTest, LegacyUnknownDeleteAndStaleExpiryCannotEraseActivePublication) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    store->delRequestBlockBuffer("legacy");
    EXPECT_TRUE(store->expired_request_caches_.empty());
    auto active = std::make_shared<RequestBlockBuffer>("legacy");
    ASSERT_EQ(store->setRequestBlockBufferResult(active), RequestBlockBufferStore::StoreResult::Stored);
    // Model a stale record left by the old unknown-ID delete implementation.
    store->expired_request_caches_.emplace_back("legacy", currentTimeUs() - kTombstoneRetentionUs - 1000 * 1000);
    store->delRequestBlockBuffer("cleanup-trigger");
    EXPECT_TRUE(store->expired_request_caches_.empty());
    EXPECT_EQ(store->request_cache_map_.count("legacy"), 1);
    EXPECT_NE(store->request_cache_map_.at("legacy"), nullptr);
    EXPECT_EQ(store->setRequestBlockBufferResult(active), RequestBlockBufferStore::StoreResult::Stored);
}

TEST_F(RequestBlockBufferStoreTest, ExplicitTerminalEndRecreatesExpiredTombstoneAfterCleanup) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    store->delRequestBlockBuffer("expired-end", true);
    const auto old_end                            = currentTimeUs() - kTombstoneRetentionUs - 1000 * 1000;
    store->expired_request_caches_.front().second = old_end;
    store->delRequestBlockBuffer("expired-end", true);
    ASSERT_EQ(store->expired_request_caches_.size(), 1);
    EXPECT_GT(store->expired_request_caches_.front().second, old_end);
    auto late = std::make_shared<RequestBlockBuffer>("expired-end");
    EXPECT_EQ(store->setRequestBlockBufferResult(late), RequestBlockBufferStore::StoreResult::RequestEnded);
}

TEST_F(RequestBlockBufferStoreTest, TypedPublicationDoesNotHideMrConversionFailure) {
    for (const bool end_during_validation : {false, true}) {
        auto memory  = std::make_shared<RdmaMockMemoryUtil>(memory_util_);
        auto store   = std::make_shared<RequestBlockBufferStore>(memory);
        auto request = std::make_shared<RequestBlockBuffer>("mr-failure");
        auto data    = std::make_shared<char>('x');
        request->addBlock(std::make_shared<BlockBuffer>("host", data, 1, false, false));
        EXPECT_CALL(*memory, isMemoryMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillOnce(::testing::Invoke([&](void*, uint64_t, bool, bool) {
                if (end_during_validation) {
                    store->delRequestBlockBuffer("mr-failure");
                }
                return false;
            }))
            .WillRepeatedly(::testing::Return(false));
        EXPECT_CALL(*memory, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .Times(::testing::AnyNumber())
            .WillRepeatedly(::testing::Return(false));
        // Without initialized runtime conversion already fails; with runtime
        // it reaches the failing MR registration. A concurrent end cannot hide it.
        EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::Failed);
        store->delRequestBlockBuffer("mr-failure");
        EXPECT_EQ(store->setRequestBlockBufferResult(request), RequestBlockBufferStore::StoreResult::RequestEnded);
    }
}

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

TEST_F(RequestBlockBufferStoreTest, testPayloadAndMetadataReadyAtPublication) {
    auto store       = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto request     = std::make_shared<RequestBlockBuffer>("tcp-batch");
    auto passthrough = block_buffer_util_->makeBlockBuffer("cpu", 17, 42, false);
    request->addBlock(passthrough);
    for (size_t i = 0; i < 257; ++i) {
        auto block             = block_buffer_util_->makeBlockBuffer(std::to_string(i), 1024 + i, char(i % 101), true);
        block->partition_count = 4;
        block->partition_id    = 2;
        block->partition_kv_halves = true;
        request->addBlock(block);
    }
    runtimeSyncAndCheck();  // Match runStoreTask's producer-event completion.
    int  successful_callbacks = 0;
    auto verify               = [&](bool success, const std::vector<std::shared_ptr<BlockBuffer>> blocks) {
        if (!success) {
            return;  // Existing request-close notification is separate.
        }
        ++successful_callbacks;
        ASSERT_EQ(blocks.size(), 258);
        for (const auto& block : blocks) {
            if (block->key == "cpu") {
                EXPECT_EQ(block, passthrough);
                continue;
            }
            const size_t index = std::stoul(block->key);
            EXPECT_FALSE(block->gpu_mem);
            EXPECT_EQ(block->len, 1024 + index);
            EXPECT_EQ(block->partition_count, 4);
            EXPECT_EQ(block->partition_id, 2);
            EXPECT_TRUE(block->partition_kv_halves);
            const auto* bytes = static_cast<const unsigned char*>(block->addr.get());
            for (size_t j = 0; j < block->len; ++j) {
                ASSERT_EQ(bytes[j], index % 101);
            }
        }
    };
    ASSERT_TRUE(store->setRequestBlockBufferWatchFunc("tcp-batch", std::move(verify)));
    EXPECT_EQ(successful_callbacks, 0);
    ASSERT_TRUE(store->setRequestBlockBuffer(request));
    EXPECT_EQ(successful_callbacks, 1);
    EXPECT_EQ(store->getBlockBuffer("tcp-batch", "cpu"), passthrough);
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
