#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <unistd.h>

#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class RequestBlockBufferStoreTest: public CacheStoreTestBase {};

namespace {

class SplitMemoryUtil: public NoRdmaMemoryUtilImpl {
public:
    static constexpr uint64_t min_region_bytes = 1024;
    static constexpr uint64_t max_region_bytes = 8192;

    bool isRdmaMode() override {
        return true;
    }

    uint64_t minRegistrationSize() const override {
        return min_region_bytes;
    }

    bool regUserMr(void* ptr, uint64_t bytes, bool gpu, uint64_t alignment) override {
        if (alignment == 0) {
            alignment = min_region_bytes;
        }
        const auto chunk_bytes = max_region_bytes - max_region_bytes % alignment;
        if (gpu || alignment < min_region_bytes || chunk_bytes == 0 || bytes % alignment != 0) {
            return false;
        }

        const auto base = reinterpret_cast<uintptr_t>(ptr);
        for (uint64_t offset = 0; offset < bytes; offset += chunk_bytes) {
            regions_.push_back({base, base + offset, base + std::min(offset + chunk_bytes, bytes)});
        }
        return true;
    }

    bool deregUserMr(void* ptr, bool gpu) override {
        EXPECT_FALSE(gpu);
        const auto base = reinterpret_cast<uintptr_t>(ptr);
        const auto it   = std::remove_if(
            regions_.begin(), regions_.end(), [base](const Region& region) { return region.base == base; });
        EXPECT_NE(it, regions_.end());
        regions_.erase(it, regions_.end());
        return true;
    }

    bool isMemoryMr(void* ptr, uint64_t bytes, bool gpu, bool adopted) override {
        const auto address = reinterpret_cast<uintptr_t>(ptr);
        return !gpu && adopted && std::any_of(regions_.begin(), regions_.end(), [address, bytes](const Region& region) {
            return region.begin <= address && address + bytes <= region.end;
        });
    }

    size_t regionCount() const {
        return regions_.size();
    }

private:
    struct Region {
        uintptr_t base;
        uintptr_t begin;
        uintptr_t end;
    };
    std::vector<Region> regions_;
};

void addHostBlock(RequestBlockBuffer& request, const std::string& key, uint32_t bytes, char fill) {
    auto payload = std::make_shared<std::vector<char>>(bytes, fill);
    request.addBlock(key, std::shared_ptr<void>(payload, payload->data()), bytes, false, true);
}

}  // namespace

TEST_F(RequestBlockBufferStoreTest, testRdmaStagingRegistersOddSizedAndSmallBlocks) {
    auto                    memory = std::make_shared<SplitMemoryUtil>();
    RequestBlockBufferStore store(memory);
    for (uint32_t bytes : {4224u, 128u}) {
        SCOPED_TRACE(bytes);
        auto request = std::make_shared<RequestBlockBuffer>(std::to_string(bytes));
        addHostBlock(*request, "block", bytes, 'a');

        ASSERT_TRUE(store.setRequestBlockBuffer(request));
        auto block = store.getBlockBuffer(request->getRequestId(), "block");
        ASSERT_NE(block, nullptr);
        EXPECT_EQ(block->len, bytes);
        EXPECT_TRUE(memory->isMemoryMr(block->addr.get(), block->len, block->gpu_mem, block->adopted));
        EXPECT_EQ(std::memcmp(block->addr.get(), request->getBlock("block")->addr.get(), bytes), 0);
    }
}

TEST_F(RequestBlockBufferStoreTest, testRdmaStagingKeepsBlocksInsideRegionsUntilReadersReleaseThem) {
    auto                    memory = std::make_shared<SplitMemoryUtil>();
    RequestBlockBufferStore store(memory);
    auto                    request = std::make_shared<RequestBlockBuffer>("split-mr");
    // Four 4,352-byte slots and four 256-byte slots total 18 KiB on the
    // old layout. Registration succeeds, but the 8 KiB split cuts a payload.
    constexpr int block_count = 8;
    for (int i = 0; i < block_count; ++i) {
        addHostBlock(*request, std::to_string(i), i % 2 == 0 ? 128 : 4224, 'a' + i);
    }

    ASSERT_TRUE(store.setRequestBlockBuffer(request));
    std::shared_ptr<BlockBuffer> reader;
    for (int i = 0; i < block_count; ++i) {
        const auto key    = std::to_string(i);
        const auto source = request->getBlock(key);
        reader            = store.getBlockBuffer(request->getRequestId(), key);
        ASSERT_NE(reader, nullptr);
        ASSERT_EQ(reader->len, source->len);
        EXPECT_TRUE(memory->isMemoryMr(reader->addr.get(), reader->len, reader->gpu_mem, reader->adopted)) << key;
        EXPECT_EQ(std::memcmp(reader->addr.get(), source->addr.get(), source->len), 0) << key;
    }

    store.delRequestBlockBuffer(request->getRequestId());
    EXPECT_TRUE(memory->isMemoryMr(reader->addr.get(), reader->len, reader->gpu_mem, reader->adopted));
    reader.reset();
    EXPECT_EQ(memory->regionCount(), 0);
}

TEST_F(RequestBlockBufferStoreTest, testRdmaStagingReleasesRegionsIfLaterRegistrationFails) {
    auto                    memory = std::make_shared<SplitMemoryUtil>();
    RequestBlockBufferStore store(memory);
    auto                    request = std::make_shared<RequestBlockBuffer>("registration-failure");
    addHostBlock(*request, "small", 128, 'a');
    addHostBlock(*request, "large", SplitMemoryUtil::max_region_bytes + 1, 'b');

    EXPECT_FALSE(store.setRequestBlockBuffer(request));
    EXPECT_EQ(store.getBlockBuffer(request->getRequestId(), "small"), nullptr);
    EXPECT_EQ(store.getBlockBuffer(request->getRequestId(), "large"), nullptr);
    EXPECT_EQ(memory->regionCount(), 0);

    addHostBlock(*request, "large", 4224, 'c');
    ASSERT_TRUE(store.setRequestBlockBuffer(request));
    EXPECT_NE(store.getBlockBuffer(request->getRequestId(), "small"), nullptr);
    EXPECT_NE(store.getBlockBuffer(request->getRequestId(), "large"), nullptr);
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

TEST_F(RequestBlockBufferStoreTest, testBatchedStagingKeepsPerBlockContent) {
    auto store         = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto request_block = std::make_shared<RequestBlockBuffer>("request-1");

    constexpr int      block_count = 32;
    constexpr uint32_t block_len   = 1000;  // unaligned on purpose
    for (int i = 0; i < block_count; i++) {
        request_block->addBlock(
            block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), block_len, static_cast<char>('a' + i), true));
    }
    ASSERT_TRUE(store->setRequestBlockBuffer(request_block));

    std::vector<std::pair<char*, char*>> ranges;
    for (int i = 0; i < block_count; i++) {
        auto block = store->getBlockBuffer("request-1", "b" + std::to_string(i));
        ASSERT_TRUE(block != nullptr);
        ASSERT_TRUE(block_buffer_util_->verifyBlock(
            block, "b" + std::to_string(i), block_len, false, static_cast<char>('a' + i)));
        auto* begin = static_cast<char*>(block->addr.get());
        ranges.push_back({begin, begin + block_len});
    }

    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); i++) {
        ASSERT_LE(ranges[i - 1].second, ranges[i].first);
    }
}

TEST_F(RequestBlockBufferStoreTest, testExpiredRequestCacheReclaimed) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    // Shrink the tombstone TTL so reclamation is observable without sleeping
    // for the production default (one hour).
    constexpr int64_t kTestTtlUs = 1000 * 50;  // 50ms
    store->setExpiredRequestCacheTtlUsForTest(kTestTtlUs);

    auto make_request = [this](const std::string& requestid) {
        auto request_block = std::make_shared<RequestBlockBuffer>(requestid);
        request_block->addBlock(block_buffer_util_->makeBlockBuffer("b1", 1024, '0', true));
        return request_block;
    };

    ASSERT_TRUE(store->setRequestBlockBuffer(make_request("request-1")));
    store->delRequestBlockBuffer("request-1");
    ASSERT_FALSE(store->setRequestBlockBuffer(make_request("request-1")));

    // A newer tombstone must not shield the older one from cleanup.
    usleep(kTestTtlUs + 1000);
    store->delRequestBlockBuffer("request-2");

    ASSERT_TRUE(store->setRequestBlockBuffer(make_request("request-1")));
    ASSERT_TRUE(store->getBlockBuffer("request-1", "b1") != nullptr);
}

}  // namespace rtp_llm
