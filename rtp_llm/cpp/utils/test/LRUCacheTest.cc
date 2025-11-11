
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/utils/LRUCache.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

using IntPair = std::pair<int, int>;

struct IntPairHash {
    size_t operator()(const IntPair& p) const {
        return std::hash<int>{}(p.first);
    }
};

struct IntPairEqual {
    bool operator()(const IntPair& a, const IntPair& b) const {
        return a.first == b.first && a.second == b.second;
    }
};

class LRUCacheTest: public ::testing::Test {
protected:
    static IntPair key(int first, int second) {
        return std::make_pair(first, second);
    }
};

TEST_F(LRUCacheTest, testSimple) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(3);

    cache.put(key(1, 10), "Item1");
    cache.put(key(2, 20), "Item2");
    cache.put(key(3, 30), "Item3");

    cache.printCache();

    ASSERT_TRUE(cache.contains(key(1, 10)));
    ASSERT_TRUE(cache.contains(key(2, 20)));
    ASSERT_TRUE(cache.contains(key(3, 30)));

    // 插入 (4,40)，应淘汰最久未使用的 (1,10)
    cache.put(key(4, 40), "Item4");

    ASSERT_TRUE(cache.contains(key(4, 40)));
    ASSERT_FALSE(cache.contains(key(1, 10)));      // 被淘汰
    ASSERT_EQ(std::get<1>(cache.pop()), "Item2");  // 弹出 LRU: (2,20)

    cache.printCache();
}

// --------------------------- clear ---------------------------

TEST_F(LRUCacheTest, testClear_OnNonEmptyCache_EmptiesAllItems) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(3);

    cache.put(key(1, 1), "A");
    cache.put(key(2, 2), "B");
    cache.put(key(3, 3), "C");

    ASSERT_FALSE(cache.empty());
    ASSERT_EQ(cache.size(), 3u);

    auto snap_before    = cache.cacheSnapshot(-1);
    auto version_before = snap_before.version;

    cache.clear();

    ASSERT_TRUE(cache.empty());
    ASSERT_EQ(cache.size(), 0u);
    ASSERT_FALSE(cache.contains(key(1, 1)));
    ASSERT_FALSE(cache.contains(key(2, 2)));
    ASSERT_FALSE(cache.contains(key(3, 3)));

    auto snap_after = cache.cacheSnapshot(-1);
    ASSERT_GT(snap_after.version, version_before);
}

TEST_F(LRUCacheTest, testClear_OnEmptyCache_StaysEmptyAndSafe) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(2);

    ASSERT_TRUE(cache.empty());
    auto snap_before = cache.cacheSnapshot(-1);

    cache.clear();

    ASSERT_TRUE(cache.empty());
    ASSERT_EQ(cache.size(), 0u);

    auto snap_after = cache.cacheSnapshot(-1);
    ASSERT_TRUE(cache.empty());  // 确保无崩溃且仍为空
}

// ==================== popWithCond方法测试 ====================

TEST_F(LRUCacheTest, testPopWithCondBasic) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(5);

    // 条件：只弹出 first 为偶数的项
    auto cond = [](const IntPair& k, const std::string& v) { return k.first % 2 == 0; };

    auto [success, item] = cache.popWithCond(cond);
    ASSERT_FALSE(success);
    ASSERT_EQ(item, std::string());

    // 插入数据
    cache.put(key(1, 1), "Item1");
    cache.put(key(2, 2), "Item2");
    cache.put(key(3, 3), "Item3");
    cache.put(key(4, 4), "Item4");
    cache.put(key(5, 5), "Item5");

    // 访问 key(2,2)，使其变为最近使用
    cache.get(key(2, 2));

    // 按 LRU 顺序：front -> back: (2,2), (5,5), (4,4), (3,3), (1,1)
    // 所以最先被淘汰的是 (1,1)，但不符合条件；第一个符合条件的是 (4,4)

    auto [success1, item1] = cache.popWithCond(cond);
    ASSERT_TRUE(success1);
    ASSERT_EQ(item1, "Item4");  // first=4 是偶数

    auto [success2, item2] = cache.popWithCond(cond);
    ASSERT_TRUE(success2);
    ASSERT_EQ(item2, "Item2");  // first=2 是偶数

    auto [success3, item3] = cache.popWithCond(cond);
    ASSERT_FALSE(success3);  // 无更多偶数 first

    // 验证状态
    ASSERT_EQ(cache.size(), 3);
    ASSERT_TRUE(cache.contains(key(1, 1)));
    ASSERT_FALSE(cache.contains(key(2, 2)));
    ASSERT_TRUE(cache.contains(key(3, 3)));
    ASSERT_FALSE(cache.contains(key(4, 4)));
    ASSERT_TRUE(cache.contains(key(5, 5)));
}

// ==================== 边界测试：同 first 不同 second 是否区分 ====================

TEST_F(LRUCacheTest, testSameFirstDifferentSecond_AreDistinctKeys) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(2);

    // 插入两个 first 相同但 second 不同的 key
    cache.put(key(1, 1), "Val1");
    cache.put(key(1, 2), "Val2");

    ASSERT_TRUE(cache.contains(key(1, 1)));
    ASSERT_TRUE(cache.contains(key(1, 2)));
    ASSERT_EQ(cache.size(), 2);

    // 再插入一个，触发淘汰
    cache.put(key(2, 1), "Val3");

    // 因为 LRU 顺序是：(2,1) <- (1,2) <- (1,1)，所以淘汰 (1,1)
    ASSERT_FALSE(cache.contains(key(1, 1)));
    ASSERT_TRUE(cache.contains(key(1, 2)));  // 保留
    ASSERT_TRUE(cache.contains(key(2, 1)));
}

// ==================== get 提升热度测试 ====================

TEST_F(LRUCacheTest, testGetUpdatesLRUOrder) {
    LRUCache<IntPair, std::string, IntPairHash, IntPairEqual> cache(3);

    cache.put(key(1, 1), "A");
    cache.put(key(2, 2), "B");
    cache.put(key(3, 3), "C");

    // 当前 LRU 顺序：front -> back: (3,3), (2,2), (1,1)
    // get(1,1) 后变为：(1,1), (3,3), (2,2)

    cache.get(key(1, 1));

    // 插入新项，应淘汰 (2,2)
    cache.put(key(4, 4), "D");

    ASSERT_FALSE(cache.contains(key(2, 2)));
    ASSERT_TRUE(cache.contains(key(1, 1)));
}

}  // namespace rtp_llm
