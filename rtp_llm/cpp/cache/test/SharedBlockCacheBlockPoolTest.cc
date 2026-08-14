#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"

namespace rtp_llm::test {
namespace {

class ReferenceObservingPublisher final: public KVCacheEventPublisher {
public:
    explicit ReferenceObservingPublisher(std::shared_ptr<BlockPool> block_pool): block_pool_(std::move(block_pool)) {}

    bool start() noexcept override {
        enabled_ = true;
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        observed_cache_refs_ = block_pool_->blockCacheRefBlocksNum();
        observed_event_      = event;
        return PublishResult::ACCEPTED;
    }

    void stop() noexcept override {
        enabled_ = false;
    }

    PublisherStatus status() const noexcept override {
        PublisherStatus result;
        result.state = enabled_ ? PublisherState::READY : PublisherState::STOPPED;
        return result;
    }

    bool enabled() const noexcept override {
        return enabled_;
    }

    size_t observedCacheRefs() const noexcept {
        return observed_cache_refs_;
    }

    const std::optional<KVCacheEvent>& observedEvent() const noexcept {
        return observed_event_;
    }

private:
    std::shared_ptr<BlockPool>  block_pool_;
    std::optional<KVCacheEvent> observed_event_;
    size_t                      observed_cache_refs_{0};
    bool                        enabled_{false};
};

BlockPoolConfig makeHostBlockPoolConfig() {
    BlockPoolConfig config;
    config.pool_name        = "shared-cache-capacity-test";
    config.block_num        = 3;
    config.total_size_bytes = 12;

    MemoryLayoutConfig layout;
    layout.layer_num                = 1;
    layout.block_num                = config.block_num;
    layout.dtype                    = DataType::TYPE_FP16;
    layout.kv_block_pool_size_bytes = config.total_size_bytes;
    layout.total_size_bytes         = config.total_size_bytes;
    layout.kv_block_stride_bytes    = 4;
    layout.k_block_stride_bytes     = 2;
    layout.v_block_stride_bytes     = 2;
    layout.local_head_num_kv        = 1;
    layout.seq_size_per_block       = 1;
    config.memory_layouts           = {layout};
    return config;
}

}  // namespace

TEST(SharedBlockCacheBlockPoolTest, CapacityRejectionPreservesExistingReference) {
    auto block_pool = std::make_shared<BlockPool>(makeHostBlockPoolConfig(), AllocationType::HOST);
    ASSERT_TRUE(block_pool->init());

    const auto total_free = block_pool->freeBlocksNum();
    auto       blocks     = block_pool->malloc(2);
    ASSERT_EQ(2u, blocks.size());

    SharedBlockCache cache(/*max_capacity=*/1);
    cache.init(/*group_num=*/1, {block_pool});
    cache.put(/*cache_key=*/1, {blocks[0]}, /*is_resident=*/false);
    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(1u, block_pool->blockCacheRefBlocksNum());
    EXPECT_EQ(total_free - 2, block_pool->freeBlocksNum());

    cache.put(/*cache_key=*/2, {blocks[1]}, /*is_resident=*/false);
    block_pool->requestFree(blocks[1]);

    // Capacity rejection must not evict key 1 through a hidden side path or
    // acquire a cache reference for key 2. Once its request reference is
    // released, block 2 is immediately reusable while block 1 remains cached.
    EXPECT_EQ(1u, block_pool->blockCacheRefBlocksNum());
    EXPECT_EQ(total_free - 1, block_pool->freeBlocksNum());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_FALSE(cache.contains(2));

    EXPECT_EQ(1u, cache.evictAndFree(/*min_blocks=*/1));
    EXPECT_EQ(0u, block_pool->blockCacheRefBlocksNum());
    EXPECT_EQ(total_free, block_pool->freeBlocksNum());
}

TEST(SharedBlockCacheBlockPoolTest, AddIsPublishedOnlyAfterCacheReferenceIsEstablished) {
    auto block_pool = std::make_shared<BlockPool>(makeHostBlockPoolConfig(), AllocationType::HOST);
    ASSERT_TRUE(block_pool->init());

    auto blocks = block_pool->malloc(1);
    ASSERT_EQ(1u, blocks.size());
    ASSERT_EQ(0u, block_pool->blockCacheRefBlocksNum());

    auto publisher = std::make_shared<ReferenceObservingPublisher>(block_pool);
    ASSERT_TRUE(publisher->start());

    SharedBlockCache cache;
    cache.init(/*group_num=*/1, {block_pool});
    cache.setEventPublisher(publisher, /*required_group_ids=*/{0});
    cache.put(/*cache_key=*/7, {blocks[0]}, /*is_resident=*/false);

    ASSERT_TRUE(publisher->observedEvent().has_value());
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, publisher->observedEvent()->type);
    EXPECT_EQ(7, publisher->observedEvent()->block_key);
    EXPECT_EQ(1u, publisher->observedCacheRefs());

    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(1u, cache.evictAndFree(/*min_blocks=*/1));
    EXPECT_EQ(0u, block_pool->blockCacheRefBlocksNum());
}

}  // namespace rtp_llm::test
