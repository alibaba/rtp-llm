#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"

#include <atomic>
#include <thread>
#include <vector>

namespace rtp_llm {

class CacheTransferBufferPoolTest: public CacheStoreTestBase {};

TEST_F(CacheTransferBufferPoolTest, ConstructWithZeroSize) {
    CacheTransferBufferPool pool(0, memory_util_);
    EXPECT_EQ(pool.totalBytes(), 0u);
    EXPECT_EQ(pool.freeBytes(), 0u);
    EXPECT_EQ(pool.baseAddr(), nullptr);
    EXPECT_EQ(pool.tryAllocate(64), nullptr);
}

TEST_F(CacheTransferBufferPoolTest, BasicAllocateAndFree) {
    constexpr size_t        kPoolSize = 4096;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);
    EXPECT_EQ(pool.totalBytes(), kPoolSize);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
    EXPECT_NE(pool.baseAddr(), nullptr);

    auto* h1 = pool.tryAllocate(100);
    ASSERT_NE(h1, nullptr);
    EXPECT_EQ(h1->size, alignGatherSize(100));
    EXPECT_EQ(h1->offset, 0u);
    EXPECT_EQ(h1->ptr, pool.baseAddr());
    EXPECT_EQ(pool.freeBytes(), kPoolSize - alignGatherSize(100));

    auto* h2 = pool.tryAllocate(200);
    ASSERT_NE(h2, nullptr);
    EXPECT_EQ(h2->offset, alignGatherSize(100));

    pool.free(h1);
    EXPECT_EQ(pool.freeBytes(), kPoolSize - alignGatherSize(200));

    pool.free(h2);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, AllocateExceedsPool) {
    constexpr size_t        kPoolSize = 256;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    auto* h = pool.tryAllocate(kPoolSize + 1);
    EXPECT_EQ(h, nullptr);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, AllocateExactPoolSize) {
    constexpr size_t        kPoolSize = 256;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    auto* h = pool.tryAllocate(kPoolSize);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->size, kPoolSize);
    EXPECT_EQ(pool.freeBytes(), 0u);

    auto* h2 = pool.tryAllocate(1);
    EXPECT_EQ(h2, nullptr);

    pool.free(h);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, CoalesceAdjacentFreeBlocks) {
    constexpr size_t        kPoolSize = 1024;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    auto* h1 = pool.tryAllocate(256);
    auto* h2 = pool.tryAllocate(256);
    auto* h3 = pool.tryAllocate(256);
    ASSERT_NE(h1, nullptr);
    ASSERT_NE(h2, nullptr);
    ASSERT_NE(h3, nullptr);

    // Free middle first
    pool.free(h2);
    // h2 gap (256) + trailing free (kPoolSize - 768 = 256), but these are not adjacent
    // After h2 freed: [h1:256][free:256][h3:256][free:256]
    EXPECT_EQ(pool.largestFreeBlock(), kPoolSize - 3 * 256);

    pool.free(h1);
    // Now: [free:512][h3:256][free:256] — h1+h2 merged
    EXPECT_EQ(pool.largestFreeBlock(), 512u);

    pool.free(h3);
    // All merged back to full pool
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
    EXPECT_EQ(pool.largestFreeBlock(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, BestFitStrategy) {
    constexpr size_t        kPoolSize = 1024;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    // Create: [h1:128][h2:128][h3:128][free:640]
    auto* h1 = pool.tryAllocate(128);
    auto* h2 = pool.tryAllocate(128);
    auto* h3 = pool.tryAllocate(128);
    ASSERT_NE(h1, nullptr);
    ASSERT_NE(h2, nullptr);
    ASSERT_NE(h3, nullptr);

    // Free h2: [h1:128][free:128][h3:128][free:640]
    pool.free(h2);

    // Allocate 100 — best-fit should pick the 128-byte gap (smallest fit)
    auto* h4 = pool.tryAllocate(100);
    ASSERT_NE(h4, nullptr);
    EXPECT_EQ(h4->offset, 128u);

    pool.free(h1);
    pool.free(h3);
    pool.free(h4);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, AllocateZeroSize) {
    constexpr size_t        kPoolSize = 1024;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);
    EXPECT_EQ(pool.tryAllocate(0), nullptr);
}

TEST_F(CacheTransferBufferPoolTest, Alignment) {
    constexpr size_t        kPoolSize = 4096;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    auto* h = pool.tryAllocate(1);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->size, kGatherAlignBytes);
    EXPECT_EQ(h->size % kGatherAlignBytes, 0u);

    auto* h2 = pool.tryAllocate(17);
    ASSERT_NE(h2, nullptr);
    EXPECT_EQ(h2->size, alignGatherSize(17));
    EXPECT_EQ(h2->size % kGatherAlignBytes, 0u);
    EXPECT_EQ(h2->offset % kGatherAlignBytes, 0u);

    pool.free(h);
    pool.free(h2);
}

TEST_F(CacheTransferBufferPoolTest, ConcurrentAllocFree) {
    constexpr size_t        kPoolSize = 64 * 1024;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    constexpr int            kNumThreads   = 8;
    constexpr int            kOpsPerThread = 100;
    std::vector<std::thread> threads;
    std::atomic<int>         success_count{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&pool, &success_count]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                size_t alloc_size = 64 + (i % 8) * 32;
                auto*  h          = pool.tryAllocate(alloc_size);
                if (h != nullptr) {
                    success_count.fetch_add(1);
                    std::this_thread::yield();
                    pool.free(h);
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_GT(success_count.load(), 0);
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
}

TEST_F(CacheTransferBufferPoolTest, MultipleAllocationsExhaustThenRecover) {
    constexpr size_t        kPoolSize = 512;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    std::vector<CacheTransferBufferPool::BufferHandle*> handles;
    // Allocate in 64-byte chunks until pool is full
    while (true) {
        auto* h = pool.tryAllocate(64);
        if (h == nullptr)
            break;
        handles.push_back(h);
    }
    EXPECT_EQ(pool.freeBytes(), 0u);
    EXPECT_EQ(handles.size(), kPoolSize / alignGatherSize(64));

    // Free all
    for (auto* h : handles) {
        pool.free(h);
    }
    EXPECT_EQ(pool.freeBytes(), kPoolSize);
    EXPECT_EQ(pool.largestFreeBlock(), kPoolSize);
}

}  // namespace rtp_llm
