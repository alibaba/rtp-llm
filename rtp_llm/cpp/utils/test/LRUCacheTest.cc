
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/utils/LRUCache.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class LRUCacheTest: public ::testing::Test {
protected:

};

TEST_F(LRUCacheTest, testSimple) {
    LRUCache<int, std::string> cache(3);

    cache.put(1, "Item1");
    cache.put(2, "Item2");
    cache.put(3, "Item3");
    cache.printCache();
    ASSERT_TRUE(cache.contains(1));
    ASSERT_TRUE(cache.contains(2));
    ASSERT_TRUE(cache.contains(3));

    cache.put(4, "Item4");  // This will remove Item1 as it is the least recently used item
    ASSERT_TRUE(cache.contains(4));
    ASSERT_FALSE(cache.contains(1));
    ASSERT_EQ(std::get<1>(cache.pop()), "Item2");
    cache.printCache();
}

}  // namespace rtp_llm
