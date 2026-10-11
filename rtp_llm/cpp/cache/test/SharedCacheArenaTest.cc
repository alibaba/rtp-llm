#include "rtp_llm/cpp/cache/SharedCacheArena.h"

#include <chrono>
#include <random>
#include <thread>
#include <gtest/gtest.h>

namespace rtp_llm {

TEST(SharedCacheArenaTest, WholeBudgetChangesUseWithoutPoolRatio) {
    SharedCacheArena arena(256 * 80, {256, 256 * 8});
    const auto       pages = arena.malloc(0, arena.totalSlots(0));
    ASSERT_EQ(pages.size(), 72u);
    EXPECT_TRUE(arena.malloc(1, 1).empty());
    for (auto page : pages) {
        arena.release(0, page);
    }
    auto states = arena.malloc(1, arena.totalSlots(1));
    ASSERT_EQ(states.size(), 9u);
    EXPECT_TRUE(arena.malloc(0, 1).empty());
    for (auto state : states) {
        arena.release(1, state);
    }
    EXPECT_EQ(arena.freeSlots(0), 72u);
}

TEST(SharedCacheArenaTest, MixedSizesReuseGapsAndCoalesce) {
    SharedCacheArena arena(256 * 120, {256 * 3, 256 * 8, 256});
    auto             state = arena.malloc(1, 1);
    auto             page  = arena.malloc(0, 1);
    auto             tiny  = arena.malloc(2, 1);
    ASSERT_EQ(state.size(), 1u);
    ASSERT_EQ(page.size(), 1u);
    ASSERT_EQ(tiny.size(), 1u);
    EXPECT_EQ(tiny[0], 16);  // Reuse the alignment gap before the 3-unit page.
    arena.release(0, page[0]);
    arena.release(1, state[0]);
    arena.release(2, tiny[0]);
    EXPECT_EQ(arena.freeSlots(2), arena.totalSlots(2));
    EXPECT_EQ(arena.freeSlots(1), arena.totalSlots(1));
}

TEST(SharedCacheArenaTest, AtomicFailureAndJointAdmission) {
    SharedCacheArena arena(256 * 80, {256, 256 * 8});
    EXPECT_TRUE(arena.canAllocate({60, 0}, SharedCacheArena::Capacity::FREE));
    EXPECT_TRUE(arena.canAllocate({0, 8}, SharedCacheArena::Capacity::FREE));
    EXPECT_FALSE(arena.canAllocate({60, 8}, SharedCacheArena::Capacity::FREE));
    EXPECT_TRUE(arena.malloc(1, 10).empty());
    EXPECT_EQ(arena.freeSlots(0), arena.totalSlots(0));
    EXPECT_TRUE(arena.canAllocate({0, 9}, SharedCacheArena::Capacity::FREE));
}

TEST(SharedCacheArenaTest, InFlightAndCacheOnlyCapacity) {
    SharedCacheArena arena(256 * 80, {256, 256 * 8});
    auto             states = arena.malloc(1, 9);
    ASSERT_EQ(states.size(), 9u);
    EXPECT_EQ(arena.availableSlots(0), 0u);
    arena.setReclaimable(1, states[4], true);
    EXPECT_EQ(arena.freeSlots(0), 0u);
    EXPECT_EQ(arena.availableSlots(0), 8u);
    EXPECT_TRUE(arena.canAllocate({8, 0}, SharedCacheArena::Capacity::AVAILABLE));
    EXPECT_FALSE(arena.canAllocate({9, 0}, SharedCacheArena::Capacity::AVAILABLE));
    arena.setReclaimable(1, states[4], false);
    EXPECT_EQ(arena.availableSlots(0), 0u);
    arena.release(1, states[4]);
    auto pages = arena.malloc(0, 8);
    EXPECT_EQ(pages.size(), 8u);
    for (auto page : pages) {
        arena.release(0, page);
    }
    for (size_t i = 0; i < states.size(); ++i) {
        if (i != 4) {
            arena.release(1, states[i]);
        }
    }
    EXPECT_EQ(arena.availableSlots(0), arena.totalSlots(0));
}

TEST(SharedCacheArenaTest, RandomMixedLifetimesNeverAlias) {
    SharedCacheArena                        arena(256 * 4096, {256 * 3, 256 * 17, 256 * 5});
    std::mt19937                            rng(20261010);
    std::vector<std::pair<size_t, int32_t>> live;
    for (int step = 0; step < 20000; ++step) {
        if (!live.empty() && rng() % 3 == 0) {
            const size_t pos = rng() % live.size();
            arena.release(live[pos].first, live[pos].second);
            live.erase(live.begin() + pos);
            continue;
        }
        const size_t group = rng() % 3;
        auto         ids   = arena.malloc(group, 1);
        if (ids.empty()) {
            continue;
        }
        const size_t start = ids[0] * arena.stride(group);
        for (auto old : live) {
            const size_t other = old.second * arena.stride(old.first);
            ASSERT_TRUE(start + arena.stride(group) <= other || other + arena.stride(old.first) <= start);
        }
        arena.setReclaimable(group, ids[0], rng() % 2 == 0);
        live.emplace_back(group, ids[0]);
    }
    for (auto item : live) {
        arena.release(item.first, item.second);
    }
    for (size_t group = 0; group < 3; ++group) {
        EXPECT_EQ(arena.freeSlots(group), arena.totalSlots(group));
        EXPECT_EQ(arena.availableSlots(group), arena.totalSlots(group));
    }
}

TEST(SharedCacheArenaTest, ConcurrentOwnersReturnWholeCapacity) {
    SharedCacheArena         arena(256 * 4096, {256 * 3, 256 * 17});
    std::vector<std::thread> workers;
    for (size_t i = 0; i < 4; ++i) {
        workers.emplace_back([&arena, i]() {
            for (int step = 0; step < 2000; ++step) {
                auto ids = arena.malloc(i % 2, 3);
                for (auto id : ids) {
                    arena.setReclaimable(i % 2, id, true);
                    arena.setReclaimable(i % 2, id, false);
                    arena.release(i % 2, id);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(arena.freeSlots(0), arena.totalSlots(0));
    EXPECT_EQ(arena.availableSlots(1), arena.totalSlots(1));
}

}  // namespace rtp_llm
